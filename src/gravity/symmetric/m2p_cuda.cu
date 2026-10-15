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

#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/symmetric/m2p_cuda.hpp>
#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <exception>
#include <stdexcept>

namespace cosmictiger::gravity::symmetric {

namespace detail {

    inline constexpr unsigned kM2pBlockSize = 128;

    template <int order>
    __global__ void m2pBatchKernel(M2pCudaRequest request) {
        PROFILE();
        size_t const stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        size_t sampleIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        while (sampleIndex < request.count) {
            auto const& sample = request.samples[sampleIndex];
            expectNonZero(sample.source);
            Multipole<double, order> source{};
            for (size_t coefficient = 0; coefficient < source.size(); coefficient++) {
                source[coefficient] = sample.source[coefficient];
            }
            std::array<double, NDIM> const separation{sample.separation[0], sample.separation[1], sample.separation[2]};
            auto const field = evaluateM2P<double, order>(source, separation, sample.sourceScale, sample.softening);
            request.fields[sampleIndex].potential = field.potential;
            for (int d = 0; d < NDIM; d++) {
                request.fields[sampleIndex].acceleration[d] = field.acceleration[d];
            }
            if (request.count - sampleIndex <= stride) {
                break;
            }
            sampleIndex += stride;
        }
    }

}    // namespace detail

hpx::future<void> launchM2pCuda(M2pCudaRequest request, cudaStream_t stream, int deviceId) {
    constexpr size_t kMaximumBlocks = 65535;
    PROFILE();
    try {
        if (request.count == 0) {
            return hpx::make_ready_future();
        }
        if (request.samples == nullptr || request.fields == nullptr) {
            throw std::invalid_argument("Full Cartesian M2P requires sample and output buffers for a nonempty batch");
        }
        size_t const neededBlocks = (request.count - 1) / detail::kM2pBlockSize + 1;
        unsigned const blocks = static_cast<unsigned>(neededBlocks < kMaximumBlocks ? neededBlocks : kMaximumBlocks);
        return cuda::launch(hpx::annotated_function(
                                [request, blocks](cudaStream_t launchStream) {
                                    PROFILE();
                                    detail::m2pBatchKernel<kMultipoleOrder><<<blocks, detail::kM2pBlockSize, 0, launchStream>>>(request);
                                },
                                "symmetric::launchM2pBatch"),
            stream, deviceId);
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity::symmetric
