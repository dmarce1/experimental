/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/gravity/stf/m2p_cuda.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>

#include <cmath>
#include <exception>
#include <stdexcept>

namespace cosmictiger::gravity::stf {

namespace detail {

    inline constexpr unsigned kM2pBlockSize = 128;

    template <int order, bool centered>
    __global__ void m2pBatchKernel(M2pCudaRequest request) {
        PROFILE();
        size_t const stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        while (index < request.count) {
            M2pCudaSample const& sample = request.samples[index];
            FpeGuard guard;
            expectNonZero(sample.source);
            expectPositive(sample.sourceScale);
            double const x = sample.separation[0];
            double const y = sample.separation[1];
            double const z = sample.separation[2];
            double const radiusSquared = x * x + y * y + z * z;
            expectPositive(radiusSquared);
            double const inverseRadius = 1.0 / sqrt(radiusSquared);
            double const sourceRatio = sample.sourceScale * inverseRadius;
            expectPositive(sourceRatio);
            double values[NDIM + 1]{};
            M2pKernel<order, centered>::add(values, sample.source, x * inverseRadius, y * inverseRadius, z * inverseRadius, inverseRadius, sourceRatio);
            request.fields[index].potential = values[0];
            for (size_t axis = 0; axis < NDIM; axis++) {
                request.fields[index].acceleration[axis] = values[axis + 1];
            }
            if (request.count - index <= stride) {
                break;
            }
            index += stride;
        }
    }

    template <int order, bool centered>
    hpx::future<void> launchM2pOrder(M2pCudaRequest request, unsigned blocks, cudaStream_t stream, int deviceId) {
        PROFILE();
        return cuda::launch(hpx::annotated_function(
                                [request, blocks](cudaStream_t launchStream) {
                                    PROFILE();
                                    m2pBatchKernel<order, centered><<<blocks, kM2pBlockSize, 0, launchStream>>>(request);
                                },
                                "stf::launchM2pBatch"),
            stream, deviceId);
    }

}    // namespace detail

hpx::future<void> launchM2pCuda(M2pCudaRequest request, bool centered, cudaStream_t stream, int deviceId) {
    size_t constexpr kMaximumBlocks = 65535;
    PROFILE();
    try {
        static_assert(NDIM == 3, "Generated STF M2P operators require three spatial dimensions");
        static_assert(
            COSMICTIGER_MULTIPOLE_ORDER >= kMinOrder && COSMICTIGER_MULTIPOLE_ORDER <= kMaxOrder, "Configured multipole order is outside the generated range");
        if (request.count == 0) {
            return hpx::make_ready_future();
        }
        if (request.samples == nullptr || request.fields == nullptr) {
            throw std::invalid_argument("M2P samples and fields are required for a nonempty batch");
        }
        size_t const neededBlocks = (request.count - 1) / detail::kM2pBlockSize + 1;
        unsigned const blocks = static_cast<unsigned>(neededBlocks < kMaximumBlocks ? neededBlocks : kMaximumBlocks);
        if (centered) {
            return detail::launchM2pOrder<COSMICTIGER_MULTIPOLE_ORDER, true>(request, blocks, stream, deviceId);
        }
        return detail::launchM2pOrder<COSMICTIGER_MULTIPOLE_ORDER, false>(request, blocks, stream, deviceId);
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity::stf
