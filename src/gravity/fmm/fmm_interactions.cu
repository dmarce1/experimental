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
    // Every FMM kernel runs one warp per block, and each block owns one target.
    inline constexpr unsigned kBlockSize = kWarpSize;
    // Source multipoles staged in shared memory at a time by M2P.
    inline constexpr unsigned kM2pTileSize = kWarpSize;
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
        unsigned const lane = threadIdx.x;
        float const spacing = static_cast<float>(request.spacing);
        for (size_t targetIndex = blockIdx.x; targetIndex < request.targetCount; targetIndex += gridDim.x) {
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
        unsigned const lane = threadIdx.x;
        float const spacing = static_cast<float>(request.spacing);
        float const mass = request.particleMass;
        for (size_t targetIndex = blockIdx.x; targetIndex < request.targetCount; targetIndex += gridDim.x) {
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

    // As in the old tree code's pc kernel, the warp stages a tile of source
    // multipoles in shared memory, then every lane applies each staged
    // multipole to its own sinks (lane, lane + 32, ...), whose offsets and
    // fields stay in shared memory. A bucket holds at most kBucketSize sinks;
    // a larger leaf, possible only at fixed-point resolution, takes more passes.
    // Each lane touches only its own sink rows, so __syncwarp guards the tile.
    template <int fmmOrder>
    __global__ void m2pKernel(FmmM2pRequest request) {
        constexpr unsigned kSourceCoefficients = static_cast<unsigned>(stfCount(fmmOrder - 1));
        PROFILE();
        __shared__ float tileMultipoles[kM2pTileSize][kSourceCoefficients];
        __shared__ float tileSeparations[kM2pTileSize][NDIM];
        __shared__ float sinkOffsets[kBucketSize][NDIM];
        __shared__ float sinkFields[kBucketSize][NDIM + 1];
        unsigned const lane = threadIdx.x;
        float const spacing = static_cast<float>(request.spacing);
        for (size_t targetIndex = blockIdx.x; targetIndex < request.targetCount; targetIndex += gridDim.x) {
            FmmParticleTarget const target = request.targets[targetIndex];
            expect(target.sourceBegin <= target.sourceEnd);
            expect(target.sourceEnd <= request.sourceCount);
            expect(target.sinks.begin <= target.sinks.end);
            Coordinate const center[NDIM] = {target.center.x, target.center.y, target.center.z};
            for (ParticleCount passBegin = target.sinks.begin; passBegin < target.sinks.end; passBegin += kBucketSize) {
                ParticleCount const remainingSinks = target.sinks.end - passBegin;
                unsigned const sinkCount = remainingSinks < kBucketSize ? static_cast<unsigned>(remainingSinks) : static_cast<unsigned>(kBucketSize);
                for (unsigned sink = lane; sink < sinkCount; sink += kWarpSize) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        sinkOffsets[sink][axis] = minimumImageOffset<float>(request.positions[axis][passBegin + sink], center[axis]);
                    }
                    for (size_t component = 0; component < NDIM + 1; component++) {
                        sinkFields[sink][component] = 0.0f;
                    }
                }
                for (size_t tileBegin = target.sourceBegin; tileBegin < target.sourceEnd; tileBegin += kM2pTileSize) {
                    size_t const remainingSources = target.sourceEnd - tileBegin;
                    unsigned const tileCount = remainingSources < kM2pTileSize ? static_cast<unsigned>(remainingSources) : kM2pTileSize;
                    // Consecutive lanes copy consecutive coefficients of the tile.
                    for (unsigned element = lane; element < tileCount * kSourceCoefficients; element += kWarpSize) {
                        unsigned const source = element / kSourceCoefficients;
                        unsigned const coefficient = element % kSourceCoefficients;
                        tileMultipoles[source][coefficient] = expectNonZero(request.sources[tileBegin + source].multipole)[coefficient];
                    }
                    for (unsigned element = lane; element < tileCount * NDIM; element += kWarpSize) {
                        unsigned const source = element / NDIM;
                        unsigned const axis = element % NDIM;
                        tileSeparations[source][axis] = static_cast<float>(request.sources[tileBegin + source].separation[axis]);
                    }
                    __syncwarp();
                    {
                        FpeGuard guard;
                        for (unsigned source = 0; source < tileCount; source++) {
                            for (unsigned sink = lane; sink < sinkCount; sink += kWarpSize) {
                                float separation[NDIM];
                                for (size_t axis = 0; axis < NDIM; axis++) {
                                    separation[axis] = sinkOffsets[sink][axis] + tileSeparations[source][axis];
                                }
                                addM2p<float, fmmOrder>(sinkFields[sink], tileMultipoles[source], separation, spacing);
                            }
                        }
                    }
                    // No lane refills the tile while another still reads it.
                    __syncwarp();
                }
                for (unsigned sink = lane; sink < sinkCount; sink += kWarpSize) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        request.accelerations[axis][passBegin + sink] += sinkFields[sink][axis + 1];
                    }
                    if (request.potentials != nullptr) {
                        request.potentials[passBegin + sink] += sinkFields[sink][0];
                    }
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

}    // namespace detail

hpx::future<void> launchM2lCuda(FmmM2lRequest request, cudaStream_t stream, int deviceId) {
    PROFILE();
    try {
        detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
        if (request.targetCount == 0) {
            return hpx::make_ready_future();
        }
        detail::requireTables(request);
        unsigned const blocks = detail::blockCount(request.targetCount);
        return detail::dispatchLaunch<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
            constexpr int kOrder = decltype(order)::value;
            PROFILE();
            return cuda::launch(hpx::annotated_function(
                                    [request, blocks](cudaStream_t launchStream) {
                                        PROFILE();
                                        detail::m2lKernel<kOrder><<<blocks, detail::kBlockSize, 0, launchStream>>>(request);
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
        unsigned const blocks = detail::blockCount(request.targetCount);
        return detail::dispatchLaunch<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
            constexpr int kOrder = decltype(order)::value;
            PROFILE();
            return cuda::launch(hpx::annotated_function(
                                    [request, blocks](cudaStream_t launchStream) {
                                        PROFILE();
                                        detail::p2lKernel<kOrder><<<blocks, detail::kBlockSize, 0, launchStream>>>(request);
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
                                        detail::m2pKernel<kOrder><<<blocks, detail::kBlockSize, 0, launchStream>>>(request);
                                    },
                                    "fmm::launchM2pCuda"),
                stream, deviceId);
        });
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity::fmm
