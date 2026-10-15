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

#include <exception>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/gravity/fmm/fmm_kernels.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::gravity::fmm {

namespace detail {

    inline constexpr unsigned kWarpSize = 32;
    inline constexpr unsigned kFullWarpMask = 0xffffffffu;
    inline constexpr unsigned kExpansionBlockSize = 128;
    inline constexpr unsigned kExpansionWarpsPerBlock = kExpansionBlockSize / kWarpSize;
    inline constexpr unsigned kParticleBlockSize = 128;
    inline constexpr size_t kMaximumBlocks = 65535;

    // A butterfly leaves the same fixed-order sum in every lane.
    template <int fmmOrder>
    __device__ inline void reduceWarp(float (&local)[localCount<fmmOrder>()]) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        for (size_t coefficient = 0; coefficient < kCount; coefficient++) {
            float value = local[coefficient];
            for (unsigned laneMask = kWarpSize / 2; laneMask > 0; laneMask /= 2) {
                value += __shfl_xor_sync(kFullWarpMask, value, laneMask);
            }
            local[coefficient] = value;
        }
    }

    // After the reduction each lane adds a disjoint subset of coefficients.
    template <int fmmOrder>
    __device__ inline void addLocalFromWarp(float* expansion, float const (&local)[localCount<fmmOrder>()], unsigned lane) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        for (size_t coefficient = lane; coefficient < kCount; coefficient += kWarpSize) {
            expansion[coefficient] += local[coefficient];
        }
    }

    // Warps never diverge across targets, so every lane reaches the shuffles.
    template <int fmmOrder>
    __global__ void m2lKernel(FmmM2lRequest request) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        unsigned const lane = threadIdx.x % kWarpSize;
        size_t const warpStride = static_cast<size_t>(gridDim.x) * kExpansionWarpsPerBlock;
        float const spacing = static_cast<float>(request.spacing);
        for (size_t targetIndex = static_cast<size_t>(blockIdx.x) * kExpansionWarpsPerBlock + threadIdx.x / kWarpSize; targetIndex < request.targetCount;
            targetIndex += warpStride) {
            FmmExpansionTarget const target = request.targets[targetIndex];
            expect(target.sourceBegin <= target.sourceEnd);
            expect(target.sourceEnd <= request.sourceCount);
            float local[kCount]{};
            {
                FpeGuard guard;
                for (size_t sourceIndex = target.sourceBegin + lane; sourceIndex < target.sourceEnd; sourceIndex += kWarpSize) {
                    FmmMultipoleSource const& source = request.sources[sourceIndex];
                    float const separation[NDIM] = {
                        static_cast<float>(source.separation[0]), static_cast<float>(source.separation[1]), static_cast<float>(source.separation[2])};
                    addM2l<float, fmmOrder>(local, expectNonZero(source.multipole), separation, spacing);
                }
            }
            reduceWarp<fmmOrder>(local);
            addLocalFromWarp<fmmOrder>(expectNonZero(target.expansion), local, lane);
        }
    }

    // Lanes stride through each source cell's particles in turn.
    template <int fmmOrder>
    __global__ void p2lKernel(FmmP2lRequest request) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        unsigned const lane = threadIdx.x % kWarpSize;
        size_t const warpStride = static_cast<size_t>(gridDim.x) * kExpansionWarpsPerBlock;
        float const spacing = static_cast<float>(request.spacing);
        float const mass = request.particleMass;
        for (size_t targetIndex = static_cast<size_t>(blockIdx.x) * kExpansionWarpsPerBlock + threadIdx.x / kWarpSize; targetIndex < request.targetCount;
            targetIndex += warpStride) {
            FmmExpansionTarget const target = request.targets[targetIndex];
            expect(target.sourceBegin <= target.sourceEnd);
            expect(target.sourceEnd <= request.sourceCount);
            float local[kCount]{};
            {
                FpeGuard guard;
                for (size_t sourceIndex = target.sourceBegin; sourceIndex < target.sourceEnd; sourceIndex++) {
                    FmmParticleSource const& source = request.sources[sourceIndex];
                    expect(source.particles.begin <= source.particles.end);
                    Coordinate const center[NDIM] = {source.center.x, source.center.y, source.center.z};
                    float const sourceSeparation[NDIM] = {
                        static_cast<float>(source.separation[0]), static_cast<float>(source.separation[1]), static_cast<float>(source.separation[2])};
                    for (ParticleCount particle = source.particles.begin + lane; particle < source.particles.end; particle += kWarpSize) {
                        float separation[NDIM];
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            separation[axis] = sourceSeparation[axis] - minimumImageOffset<float>(request.positions[axis][particle], center[axis]);
                        }
                        addP2l<float, fmmOrder>(local, mass, separation, spacing);
                    }
                }
            }
            reduceWarp<fmmOrder>(local);
            addLocalFromWarp<fmmOrder>(expectNonZero(target.expansion), local, lane);
        }
    }

    // A block owns a target cell; each thread accumulates one sink at a time.
    template <int fmmOrder>
    __global__ void m2pKernel(FmmM2pRequest request) {
        PROFILE();
        float const spacing = static_cast<float>(request.spacing);
        for (size_t targetIndex = blockIdx.x; targetIndex < request.targetCount; targetIndex += gridDim.x) {
            FmmParticleTarget const target = request.targets[targetIndex];
            expect(target.sourceBegin <= target.sourceEnd);
            expect(target.sourceEnd <= request.sourceCount);
            expect(target.sinks.begin <= target.sinks.end);
            Coordinate const center[NDIM] = {target.center.x, target.center.y, target.center.z};
            for (ParticleCount sink = target.sinks.begin + threadIdx.x; sink < target.sinks.end; sink += blockDim.x) {
                FpeGuard guard;
                float offset[NDIM];
                for (size_t axis = 0; axis < NDIM; axis++) {
                    offset[axis] = minimumImageOffset<float>(request.positions[axis][sink], center[axis]);
                }
                float field[NDIM + 1]{};
                for (size_t sourceIndex = target.sourceBegin; sourceIndex < target.sourceEnd; sourceIndex++) {
                    FmmMultipoleSource const& source = request.sources[sourceIndex];
                    float separation[NDIM];
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        separation[axis] = offset[axis] + static_cast<float>(source.separation[axis]);
                    }
                    addM2p<float, fmmOrder>(field, expectNonZero(source.multipole), separation, spacing);
                }
                for (size_t axis = 0; axis < NDIM; axis++) {
                    request.accelerations[axis][sink] += field[axis + 1];
                }
                if (request.potentials != nullptr) {
                    request.potentials[sink] += field[0];
                }
            }
        }
    }

    template <int fmmOrder, typename Function>
    hpx::future<void> dispatchLaunch(int order, Function&& function) {
        PROFILE();
        if (order == fmmOrder) {
            return function(std::integral_constant<int, fmmOrder>{});
        }
        if constexpr (fmmOrder < kMaximumFmmOrder) {
            return dispatchLaunch<fmmOrder + 1>(order, std::forward<Function>(function));
        } else {
            throw std::invalid_argument("FMM order is outside the supported local/source rank range");
        }
    }

    unsigned blockCount(size_t neededBlocks) {
        PROFILE();
        return static_cast<unsigned>(neededBlocks < kMaximumBlocks ? neededBlocks : kMaximumBlocks);
    }

    unsigned expansionBlockCount(size_t targetCount) {
        PROFILE();
        return blockCount((targetCount - 1) / kExpansionWarpsPerBlock + 1);
    }

}    // namespace detail

hpx::future<void> launchM2lCuda(FmmM2lRequest request, cudaStream_t stream, int deviceId) {
    PROFILE();
    try {
        detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
        if (request.targetCount == 0) {
            return hpx::make_ready_future();
        }
        detail::requireTables(request);
        unsigned const blocks = detail::expansionBlockCount(request.targetCount);
        return detail::dispatchLaunch<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
            constexpr int kOrder = decltype(order)::value;
            PROFILE();
            return cuda::launch(hpx::annotated_function(
                                    [request, blocks](cudaStream_t launchStream) {
                                        PROFILE();
                                        detail::m2lKernel<kOrder><<<blocks, detail::kExpansionBlockSize, 0, launchStream>>>(request);
                                    },
                                    "fmm::launchM2lCuda"),
                stream, deviceId);
        });
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

hpx::future<void> launchP2lCuda(FmmP2lRequest request, cudaStream_t stream, int deviceId) {
    PROFILE();
    try {
        detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
        if (request.targetCount == 0) {
            return hpx::make_ready_future();
        }
        detail::requireTables(request);
        detail::requirePositions(request);
        unsigned const blocks = detail::expansionBlockCount(request.targetCount);
        return detail::dispatchLaunch<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
            constexpr int kOrder = decltype(order)::value;
            PROFILE();
            return cuda::launch(hpx::annotated_function(
                                    [request, blocks](cudaStream_t launchStream) {
                                        PROFILE();
                                        detail::p2lKernel<kOrder><<<blocks, detail::kExpansionBlockSize, 0, launchStream>>>(request);
                                    },
                                    "fmm::launchP2lCuda"),
                stream, deviceId);
        });
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

hpx::future<void> launchM2pCuda(FmmM2pRequest request, cudaStream_t stream, int deviceId) {
    PROFILE();
    try {
        detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
        if (request.targetCount == 0) {
            return hpx::make_ready_future();
        }
        detail::requireTables(request);
        detail::requirePositions(request);
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (request.accelerations[axis] == nullptr) {
                throw std::invalid_argument("FMM M2P acceleration outputs are required for a nonempty batch");
            }
        }
        unsigned const blocks = detail::blockCount(request.targetCount);
        return detail::dispatchLaunch<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
            constexpr int kOrder = decltype(order)::value;
            PROFILE();
            return cuda::launch(hpx::annotated_function(
                                    [request, blocks](cudaStream_t launchStream) {
                                        PROFILE();
                                        detail::m2pKernel<kOrder><<<blocks, detail::kParticleBlockSize, 0, launchStream>>>(request);
                                    },
                                    "fmm::launchM2pCuda"),
                stream, deviceId);
        });
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity::fmm
