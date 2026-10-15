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

#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/gravity/direct/direct_solve_kernel.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>

#include <exception>
#include <stddef.h>

namespace cosmictiger::gravity {

namespace detail {

    inline constexpr unsigned kDirectSolveBlockSize = 128;

    __global__ void directSolveKernel(DirectSolveRequest request, DirectEwaldParameters parameters, DirectEwaldWeights weights) {
        constexpr unsigned kInitialReductionStride = kDirectSolveBlockSize / 2;
        static_assert((kDirectSolveBlockSize & (kDirectSolveBlockSize - 1)) == 0);
        PROFILE();
        __shared__ double partial[NDIM + 1][kDirectSolveBlockSize];
        size_t sinkOrdinal = blockIdx.x;
        while (sinkOrdinal < request.sinkCount) {
            // Threads cooperate over all sources even when only a few sinks
            // were sampled. Each lane includes its sources' periodic images,
            // reciprocal terms, and uniform-background potential contribution.
            DirectSolveField const field = directSolveSources(request, parameters, sinkOrdinal, threadIdx.x, blockDim.x, &weights);
            partial[0][threadIdx.x] = field.potential;
            for (size_t axis = 0; axis < NDIM; axis++) {
                partial[axis + 1][threadIdx.x] = field.acceleration[axis];
            }
            __syncthreads();
            for (unsigned stride = kInitialReductionStride; stride != 0; stride /= 2) {
                if (threadIdx.x < stride) {
                    for (size_t component = 0; component < NDIM + 1; component++) {
                        partial[component][threadIdx.x] += partial[component][threadIdx.x + stride];
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                request.fields[sinkOrdinal].potential = partial[0][0];
                for (size_t axis = 0; axis < NDIM; axis++) {
                    request.fields[sinkOrdinal].acceleration[axis] = partial[axis + 1][0];
                }
            }
            __syncthreads();
            if (request.sinkCount - sinkOrdinal <= gridDim.x) {
                break;
            }
            sinkOrdinal += gridDim.x;
        }
    }

}    // namespace detail

hpx::future<void> launchDirectSolveCuda(DirectSolveRequest request, DirectEwaldParameters parameters, cudaStream_t stream, int deviceId) {
    constexpr size_t kMaximumBlocks = 65535;
    PROFILE();
    try {
        detail::validateDirectSolveRequest(request, parameters);
        if (request.sinkCount == 0) {
            return hpx::make_ready_future();
        }
        auto const weights = detail::prepareDirectEwaldWeights(request, parameters);
        unsigned const blocks = static_cast<unsigned>(request.sinkCount < kMaximumBlocks ? request.sinkCount : kMaximumBlocks);
        return cuda::launch(hpx::annotated_function(
                                [request, parameters, blocks, weights](cudaStream_t launchStream) {
                                    PROFILE();
                                    detail::directSolveKernel<<<blocks, detail::kDirectSolveBlockSize, 0, launchStream>>>(request, parameters, weights);
                                },
                                "gravity::launchDirectSolveCuda"),
            stream, deviceId);
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity
