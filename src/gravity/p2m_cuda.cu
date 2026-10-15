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

#include <cosmictiger/gravity/p2m_cuda.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/stf/combinatorics.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cstring>
#include <limits>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>

namespace cosmictiger::gravity::detail {
namespace {

    inline constexpr int kP2mOrder = kMultipoleOrder;
    inline constexpr size_t kP2mSparseLimit = 8;
    inline constexpr unsigned kP2mWarpSize = 32;
    inline constexpr unsigned kP2mThreadsPerBlock = 128;
    inline constexpr unsigned kP2mWarpsPerBlock = kP2mThreadsPerBlock / kP2mWarpSize;
    inline constexpr stf::Combinatorics<kP2mOrder> kP2mCombinatoricsValues{};
    __device__ __constant__ stf::Combinatorics<kP2mOrder> const p2mCombinatorics = kP2mCombinatoricsValues;

    template <typename T>
    class ManagedBuffer {
    public:
        ManagedBuffer(size_t count, int deviceId)
          : deviceId_(deviceId) {
            static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
            PROFILE();
            if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
                throw std::length_error("CUDA P2M buffer exceeds addressable memory");
            }
            if (count != 0) {
                data_ = static_cast<T*>(cuda::mallocManaged(count * sizeof(T), cudaMemAttachGlobal, deviceId_).get());
                for (size_t index = 0; index < count; index++) {
                    ::new (static_cast<void*>(data_ + index)) T;
                }
            }
        }

        ManagedBuffer(ManagedBuffer const&) = delete;
        ManagedBuffer& operator=(ManagedBuffer const&) = delete;

        ~ManagedBuffer() {
            PROFILE();
            if (data_ != nullptr) {
                try {
                    cuda::free(data_, deviceId_).get();
                } catch (...) {
                    // The successful path releases explicitly so free failures
                    // remain visible without masking an exception in flight.
                }
            }
        }

        T* data() noexcept {
            PROFILE();
            return data_;
        }

        void release() {
            PROFILE();
            if (data_ != nullptr) {
                cuda::free(data_, deviceId_).get();
                data_ = nullptr;
            }
        }

    private:
        int deviceId_;
        T* data_ = nullptr;
    };

    template <bool fullMoments>
    __device__ __forceinline__ void accumulateP2mParticle(P2mCudaCell const& cell, P2mCudaPositions positions, ParticleCount particle, double coordinateUnit,
        double spacing, double particleMass, double* moments, double& secondMomentTrace) {
        PROFILE();
        double powers[NDIM][kP2mOrder + 1];
        double radiusSquared = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const position = double(positions.coordinates[axis][particle].raw()) * coordinateUnit;
            double const normalizedOffset = (position - cell.center[axis]) / spacing;
            powers[axis][0] = 1.0;
            for (int degree = 1; degree <= kP2mOrder; degree++) {
                powers[axis][degree] = powers[axis][degree - 1] * normalizedOffset;
            }
            if constexpr (kP2mOrder >= 2) {
                radiusSquared += powers[axis][2];
            } else {
                radiusSquared += normalizedOffset * normalizedOffset;
            }
        }
        if constexpr (fullMoments) {
            for (int rank = 0; rank <= kP2mOrder; rank++) {
                for (int nz = 0; nz <= rank; nz++) {
                    for (int nx = 0; nx <= rank - nz; nx++) {
                        int const ny = rank - nx - nz;
                        moments[symmetric::index(nx, ny, nz)] += particleMass * powers[0][nx] * powers[1][ny] * powers[2][nz];
                    }
                }
            }
        } else {
            double radiusPowers[kP2mOrder / 2 + 1];
            radiusPowers[0] = 1.0;
            for (int degree = 1; degree <= kP2mOrder / 2; degree++) {
                radiusPowers[degree] = radiusPowers[degree - 1] * radiusSquared;
            }
            for (int rank = 0; rank <= kP2mOrder; rank++) {
                for (int nz = 0; nz <= 1 && nz <= rank; nz++) {
                    for (int nx = 0; nx <= rank - nz; nx++) {
                        int const ny = rank - nx - nz;
                        double value = 0.0;
                        for (int bx = 0; bx <= nx / 2; bx++) {
                            for (int by = 0; by <= ny / 2; by++) {
                                int const k = bx + by;
                                double const sign = k % 2 ? -1.0 : 1.0;
                                double const pairs = p2mCombinatorics.factorial(nx) * p2mCombinatorics.factorial(ny) /
                                    (p2mCombinatorics.factorial(nx - 2 * bx) * p2mCombinatorics.factorial(ny - 2 * by) * p2mCombinatorics.factorial(bx) *
                                        p2mCombinatorics.factorial(by) * p2mCombinatorics.powerOfTwo(k));
                                double const factor =
                                    sign * pairs * p2mCombinatorics.oddFactorial(2 * rank - 2 * k - 1) / p2mCombinatorics.oddFactorial(2 * rank - 1);
                                value += factor * radiusPowers[k] * powers[0][nx - 2 * bx] * powers[1][ny - 2 * by] * powers[2][nz];
                            }
                        }
                        size_t const coefficient = static_cast<size_t>(rank * rank + (nz == 0 ? nx : rank + 1 + nx));
                        moments[coefficient] += particleMass * value;
                    }
                }
            }
        }
        secondMomentTrace += particleMass * radiusSquared;
    }

    template <bool fullMoments>
    __global__ void p2mKernel(P2mCudaCell const* cells, size_t const* tasks, size_t sparseCount, size_t crowdedCount, size_t sparseBlockCount,
        size_t virtualBlockCount, P2mCudaPositions positions, double coordinateUnit, double spacing, double particleMass, P2mCudaResult* results) {
        constexpr size_t kP2mCoefficientCount = fullMoments ? symmetric::coefficientCount(kP2mOrder) : stf::coefficientCount(kP2mOrder);
        PROFILE();
        for (size_t taskBlock = blockIdx.x; taskBlock < virtualBlockCount; taskBlock += gridDim.x) {
            FpeGuard guard;
            if (taskBlock < sparseBlockCount) {
                size_t const task = taskBlock * kP2mThreadsPerBlock + threadIdx.x;
                if (task < sparseCount) {
                    size_t const cellIndex = tasks[task];
                    P2mCudaCell const& cell = cells[cellIndex];
                    double moments[kP2mCoefficientCount]{};
                    double secondMomentTrace = 0.0;
                    for (ParticleCount particle = cell.localParticles.begin; particle < cell.localParticles.end; particle++) {
                        accumulateP2mParticle<fullMoments>(cell, positions, particle, coordinateUnit, spacing, particleMass, moments, secondMomentTrace);
                    }
                    results[cellIndex] = P2mCudaResult{};
                    float* output = fullMoments ? results[cellIndex].fullMultipole : results[cellIndex].multipole;
                    for (size_t coefficient = 0; coefficient < kP2mCoefficientCount; coefficient++) {
                        output[coefficient] = static_cast<float>(moments[coefficient]);
                    }
                    results[cellIndex].secondMomentTrace = secondMomentTrace;
                }
            } else {
                unsigned const lane = threadIdx.x % kP2mWarpSize;
                size_t const warp = threadIdx.x / kP2mWarpSize;
                size_t const task = (taskBlock - sparseBlockCount) * kP2mWarpsPerBlock + warp;
                if (task < crowdedCount) {
                    size_t const cellIndex = tasks[sparseCount + task];
                    P2mCudaCell const& cell = cells[cellIndex];
                    double moments[kP2mCoefficientCount]{};
                    double secondMomentTrace = 0.0;
                    for (ParticleCount particle = cell.localParticles.begin + lane; particle < cell.localParticles.end; particle += kP2mWarpSize) {
                        accumulateP2mParticle<fullMoments>(cell, positions, particle, coordinateUnit, spacing, particleMass, moments, secondMomentTrace);
                    }
                    if (lane == 0) {
                        results[cellIndex] = P2mCudaResult{};
                    }
                    float* output = fullMoments ? results[cellIndex].fullMultipole : results[cellIndex].multipole;
                    for (size_t coefficient = 0; coefficient < kP2mCoefficientCount; coefficient++) {
                        double value = moments[coefficient];
                        for (unsigned offset = kP2mWarpSize / 2; offset != 0; offset /= 2) {
                            value += __shfl_down_sync(0xffffffffU, value, offset);
                        }
                        if (lane == 0) {
                            output[coefficient] = static_cast<float>(value);
                        }
                    }
                    for (unsigned offset = kP2mWarpSize / 2; offset != 0; offset /= 2) {
                        secondMomentTrace += __shfl_down_sync(0xffffffffU, secondMomentTrace, offset);
                    }
                    if (lane == 0) {
                        results[cellIndex].secondMomentTrace = secondMomentTrace;
                    }
                }
            }
        }
    }

}    // namespace

std::vector<P2mCudaResult> computeP2mCuda(std::vector<P2mCudaCell> const& cells, P2mCudaPositions positions, ParticleCount particleCount, double spacing,
    float particleMass, int deviceId, bool fullMoments) {
    constexpr double kCoordinateUnit = [] {
        PROFILE();
        double value = 1.0;
        for (int bit = 0; bit < Coordinate::fractionalBits(); bit++) {
            value *= 0.5;
        }
        return value;
    }();
    static_assert(NDIM == 3, "Cartesian P2M requires three spatial dimensions");
    PROFILE();
    if (!(spacing > 0.0)) {
        throw std::invalid_argument("CUDA P2M requires positive cell spacing");
    }
    if (particleCount != 0) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (positions.coordinates[axis] == nullptr) {
                throw std::invalid_argument("CUDA P2M requires particle position arrays");
            }
        }
    }
    std::vector<P2mCudaResult> results(cells.size());
    if (cells.empty()) {
        return results;
    }
    // Sparse cells retain one lane per cell. A warp cooperates only when a
    // cell has enough particles to amortize the coefficient reductions.
    std::vector<size_t> tasks(cells.size());
    size_t sparseCount = 0;
    size_t crowdedCount = 0;
    for (size_t cellIndex = 0; cellIndex < cells.size(); cellIndex++) {
        auto const range = cells[cellIndex].localParticles;
        if (range.begin > range.end || range.end > particleCount) {
            throw std::out_of_range("CUDA P2M local particle range exceeds particle storage");
        }
        if (range.end - range.begin <= kP2mSparseLimit) {
            tasks[sparseCount] = cellIndex;
            sparseCount++;
        } else {
            tasks[cells.size() - crowdedCount - 1] = cellIndex;
            crowdedCount++;
        }
    }
    size_t const sparseBlockCount = sparseCount == 0 ? 0 : (sparseCount - 1) / kP2mThreadsPerBlock + 1;
    size_t const crowdedBlockCount = crowdedCount == 0 ? 0 : (crowdedCount - 1) / kP2mWarpsPerBlock + 1;
    if (sparseBlockCount > std::numeric_limits<size_t>::max() - crowdedBlockCount) {
        throw std::length_error("CUDA P2M task block count overflows size_t");
    }
    size_t const virtualBlockCount = sparseBlockCount + crowdedBlockCount;
    ManagedBuffer<P2mCudaCell> deviceCells(cells.size(), deviceId);
    ManagedBuffer<size_t> deviceTasks(tasks.size(), deviceId);
    ManagedBuffer<P2mCudaResult> deviceResults(cells.size(), deviceId);
    std::memcpy(deviceCells.data(), cells.data(), cells.size() * sizeof(P2mCudaCell));
    std::memcpy(deviceTasks.data(), tasks.data(), tasks.size() * sizeof(size_t));

    unsigned const blocks = static_cast<unsigned>(virtualBlockCount < 65535 ? virtualBlockCount : 65535);
    cuda::launch(hpx::annotated_function(
                     [deviceCellsPointer = deviceCells.data(), deviceTasksPointer = deviceTasks.data(), sparseCount, crowdedCount, sparseBlockCount,
                         virtualBlockCount, positions, kCoordinateUnit, spacing, mass = static_cast<double>(particleMass),
                         deviceResultsPointer = deviceResults.data(), blocks, fullMoments](cudaStream_t stream) {
                         PROFILE();
                         if (fullMoments) {
                             p2mKernel<true><<<blocks, kP2mThreadsPerBlock, 0, stream>>>(deviceCellsPointer, deviceTasksPointer, sparseCount, crowdedCount,
                                 sparseBlockCount, virtualBlockCount, positions, kCoordinateUnit, spacing, mass, deviceResultsPointer);
                         } else {
                             p2mKernel<false><<<blocks, kP2mThreadsPerBlock, 0, stream>>>(deviceCellsPointer, deviceTasksPointer, sparseCount, crowdedCount,
                                 sparseBlockCount, virtualBlockCount, positions, kCoordinateUnit, spacing, mass, deviceResultsPointer);
                         }
                     },
                     "gravity::computeP2mCuda"),
        nullptr, deviceId)
        .get();
    std::memcpy(results.data(), deviceResults.data(), cells.size() * sizeof(P2mCudaResult));
    deviceResults.release();
    deviceTasks.release();
    deviceCells.release();
    return results;
}

}    // namespace cosmictiger::gravity::detail
