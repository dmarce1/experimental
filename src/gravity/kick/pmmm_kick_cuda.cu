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

#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/plummer.hpp>
#include <cosmictiger/gravity/stf/combinatorics.hpp>
#include <cosmictiger/gravity/symmetric/operators.hpp>

#include <cstring>
#include <limits>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cosmictiger::gravity::detail {
namespace {

    inline constexpr stf::Combinatorics<kExpansionOrder> kPmmmCombinatoricsValues{};
    __device__ __constant__ stf::Combinatorics<kExpansionOrder> const pmmmCombinatorics = kPmmmCombinatoricsValues;

    template <typename T>
    class ManagedBuffer {
    public:
        ManagedBuffer(size_t count, int deviceId)
          : deviceId_(deviceId) {
            static_assert(std::is_trivially_copyable_v<T>);
            PROFILE();
            if (count != 0) {
                if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
                    throw std::length_error("PMMM CUDA buffer exceeds addressable memory");
                }
                data_ = static_cast<T*>(cuda::mallocManaged(count * sizeof(T), cudaMemAttachGlobal, deviceId_).get());
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
                    // Preserve the failure already unwinding; successful paths
                    // release explicitly so free failures are reported.
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

    // STF stores only nz=0,1 derivatives. Expand repeated use of the harmonic
    // trace relation as a finite binomial sum; the constant background
    // Laplacian contributes only to the (0,0,2) component.
    template <int order>
    __device__ double expansionComponent(double const* expansion, int nx, int ny, int nz, bool full) {
        PROFILE();
        if (full) {
            return expansion[symmetric::index(nx, ny, nz)];
        }
        int const pairs = nz / 2;
        int const remainingZ = nz % 2;
        int const rank = nx + ny + nz;
        double value = 0.0;
        for (int xPairs = 0; xPairs <= pairs; xPairs++) {
            int const storedX = nx + 2 * xPairs;
            int const offset = remainingZ == 0 ? storedX : rank + 1 + storedX;
            value += pmmmCombinatorics.binomial(pairs, xPairs) * expansion[rank * rank + offset];
        }
        if ((pairs & 1) != 0) {
            value = -value;
        }
        if (nx == 0 && ny == 0 && nz == 2) {
            value += expansion[(order + 1) * (order + 1)];
        }
        return value;
    }

    template <int order, bool computePotential>
    __device__ void evaluateCellExpansion(
        PmmmCudaCell const& cell, double const displacement[NDIM], double& accelerationX, double& accelerationY, double& accelerationZ, double& potential) {
        PROFILE();
        double powers[NDIM][order + 1];
        for (size_t axis = 0; axis < NDIM; axis++) {
            powers[axis][0] = 1.0;
        }
        for (int degree = 1; degree <= order; degree++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                powers[axis][degree] = powers[axis][degree - 1] * displacement[axis];
            }
        }

        double gradientX = 0.0;
        double gradientY = 0.0;
        double gradientZ = 0.0;
        potential = 0.0;
        for (int nx = 0; nx <= order; nx++) {
            for (int ny = 0; nx + ny <= order; ny++) {
                for (int nz = 0; nx + ny + nz <= order; nz++) {
                    double const weight = powers[0][nx] * powers[1][ny] * powers[2][nz] * pmmmCombinatorics.inverseFactorial(nx) *
                        pmmmCombinatorics.inverseFactorial(ny) * pmmmCombinatorics.inverseFactorial(nz);
                    if constexpr (computePotential) {
                        potential += expansionComponent<order>(cell.expansion, nx, ny, nz, cell.fullExpansion) * weight;
                    }
                    if (nx + ny + nz < order) {
                        gradientX += expansionComponent<order>(cell.expansion, nx + 1, ny, nz, cell.fullExpansion) * weight;
                        gradientY += expansionComponent<order>(cell.expansion, nx, ny + 1, nz, cell.fullExpansion) * weight;
                        gradientZ += expansionComponent<order>(cell.expansion, nx, ny, nz + 1, cell.fullExpansion) * weight;
                    }
                }
            }
        }
        double const inverseSpacing = 1.0 / cell.spacing;
        accelerationX = -gradientX * inverseSpacing;
        accelerationY = -gradientY * inverseSpacing;
        accelerationZ = -gradientZ * inverseSpacing;
    }

    template <int order, bool computePotential>
    __global__ void pmmmKickKernel(PmmmCudaCandidate const* candidates, size_t candidateCount, PmmmCudaCell const* cells, DirectInteractionParticles particles,
        SofteningLength softening, float particleMass, double rungZeroTimestep, std::uint8_t activeRung, DirectKickBackend backend) {
        PROFILE();
        size_t const stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        size_t candidateIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        while (candidateIndex < candidateCount) {
            auto const candidate = candidates[candidateIndex];
            ParticleCount const sink = candidate.sinkIndex;
            Rung const flags = particles.flags[sink];
            if (!flags.isGhost() && flags.rung() >= activeRung) {
                auto const& cell = cells[candidate.cellIndex];
                Position const sinkPosition{particles.positions[0][sink], particles.positions[1][sink], particles.positions[2][sink]};
                double displacement[NDIM];
                for (size_t axis = 0; axis < NDIM; axis++) {
                    // Coordinate is Fixed<uint32_t> in the unit box. Every raw
                    // value converts exactly to double with this binary scale.
                    double const position = double(particles.positions[axis][sink].raw()) * 0x1p-32;
                    displacement[axis] = (position - cell.center[axis]) / cell.spacing;
                }
                double accelerationX;
                double accelerationY;
                double accelerationZ;
                double potential;
                evaluateCellExpansion<kExpansionOrder, computePotential>(cell, displacement, accelerationX, accelerationY, accelerationZ, potential);

                double directX = 0.0;
                double directY = 0.0;
                double directZ = 0.0;
                double directPotential = 0.0;
                for (size_t neighbor = 0; neighbor < cell.nearSourceCount; neighbor++) {
                    auto const range = cell.nearSources[neighbor];
                    for (ParticleCount source = range.begin; source < range.end; source++) {
                        if (source == sink) {
                            continue;
                        }
                        Position const sourcePosition{particles.positions[0][source], particles.positions[1][source], particles.positions[2][source]};
                        if constexpr (computePotential) {
                            auto const pair = softenedAccelerationAndPotential<order>(sinkPosition, sourcePosition, softening);
                            directX += double(pair.acceleration.x);
                            directY += double(pair.acceleration.y);
                            directZ += double(pair.acceleration.z);
                            directPotential += double(pair.potential);
                        } else {
                            auto const pair = softenedAcceleration<order>(sinkPosition, sourcePosition, softening);
                            directX += double(pair.x);
                            directY += double(pair.y);
                            directZ += double(pair.z);
                        }
                    }
                }
                accelerationX += double(particleMass) * directX;
                accelerationY += double(particleMass) * directY;
                accelerationZ += double(particleMass) * directZ;
                if constexpr (computePotential) {
                    potential += double(particleMass) * directPotential;
                }
                backend(sink, {static_cast<float>(accelerationX), static_cast<float>(accelerationY), static_cast<float>(accelerationZ)},
                    static_cast<float>(potential), flags, rungZeroTimestep, particles);
            }
            candidateIndex += stride;
        }
    }

    template <int order, bool computePotential>
    hpx::future<void> launchPmmmKick(PmmmCudaCandidate const* candidates, size_t candidateCount, PmmmCudaCell const* cells,
        DirectInteractionParticles particles, SofteningLength softening, float particleMass, double rungZeroTimestep, std::uint8_t activeRung,
        KickParameters parameters, KickSums* sums, int deviceId) {
        size_t constexpr kThreadsPerBlock = 128;
        PROFILE();
        size_t const neededBlocks = (candidateCount - 1) / kThreadsPerBlock + 1;
        unsigned const blocks = static_cast<unsigned>(neededBlocks < 65535 ? neededBlocks : 65535);
        DirectKickBackend const backend(parameters, sums);
        return cuda::launch(
            hpx::annotated_function(
                [candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, backend, blocks](cudaStream_t stream) {
                    PROFILE();
                    pmmmKickKernel<order, computePotential><<<blocks, kThreadsPerBlock, 0, stream>>>(
                        candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, backend);
                },
                "gravity::launchPmmmKick"),
            nullptr, deviceId);
    }

    template <bool computePotential>
    hpx::future<void> launchPmmmKickOrder(PmmmCudaCandidate const* candidates, size_t candidateCount, PmmmCudaCell const* cells,
        DirectInteractionParticles particles, SofteningLength softening, int ferrersOrder, float particleMass, double rungZeroTimestep, std::uint8_t activeRung,
        KickParameters parameters, KickSums* sums, int deviceId) {
        PROFILE();
        switch (ferrersOrder) {
        case 0:
            return launchPmmmKick<0, computePotential>(
                candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, deviceId);
        case 1:
            return launchPmmmKick<1, computePotential>(
                candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, deviceId);
        case 2:
            return launchPmmmKick<2, computePotential>(
                candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, deviceId);
        case 3:
            return launchPmmmKick<3, computePotential>(
                candidates, candidateCount, cells, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, deviceId);
        default:
            throw std::invalid_argument("PMMM CUDA kick Ferrers order must be between zero and three");
        }
    }

}    // namespace

void applyPmmmKickCuda(PmmmCudaCandidate const* candidates, size_t candidateCount, PmmmCudaCell const* cells, size_t cellCount,
    DirectInteractionParticles particles, SofteningLength softening, int ferrersOrder, bool computePotential, float particleMass, double rungZeroTimestep,
    std::uint8_t activeRung, KickParameters parameters, KickSums* sums, int deviceId) {
    PROFILE();
    if (candidateCount == 0) {
        return;
    }
    if (candidates == nullptr || cells == nullptr || cellCount == 0 || particles.flags == nullptr) {
        throw std::invalid_argument("PMMM CUDA kick requires staged cells and particle arrays");
    }
    ManagedBuffer<PmmmCudaCandidate> deviceCandidates(candidateCount, deviceId);
    ManagedBuffer<PmmmCudaCell> deviceCells(cellCount, deviceId);
    ManagedBuffer<KickSums> deviceSums(sums == nullptr ? 0 : 1, deviceId);
    if (parameters.synchronizedMomentumCount > std::numeric_limits<size_t>::max() / NDIM) {
        throw std::length_error("PMMM snapshot momentum buffer exceeds addressable storage");
    }
    size_t const momentumCount = static_cast<size_t>(parameters.synchronizedMomentumCount);
    ManagedBuffer<double> deviceMomenta(momentumCount * NDIM, deviceId);
    double* hostMomenta[NDIM]{};
    if (momentumCount != 0) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (parameters.synchronizedMomenta[axis] == nullptr) {
                throw std::invalid_argument("PMMM snapshot momentum output requires every component");
            }
            hostMomenta[axis] = parameters.synchronizedMomenta[axis];
            parameters.synchronizedMomenta[axis] = deviceMomenta.data() + axis * momentumCount;
            std::memcpy(parameters.synchronizedMomenta[axis], hostMomenta[axis], momentumCount * sizeof(double));
        }
    }
    std::memcpy(deviceCandidates.data(), candidates, candidateCount * sizeof(PmmmCudaCandidate));
    std::memcpy(deviceCells.data(), cells, cellCount * sizeof(PmmmCudaCell));
    if (sums != nullptr) {
        ::new (static_cast<void*>(deviceSums.data())) KickSums(*sums);
    }
    if (computePotential) {
        launchPmmmKickOrder<true>(deviceCandidates.data(), candidateCount, deviceCells.data(), particles, softening, ferrersOrder, particleMass,
            rungZeroTimestep, activeRung, parameters, deviceSums.data(), deviceId)
            .get();
    } else {
        launchPmmmKickOrder<false>(deviceCandidates.data(), candidateCount, deviceCells.data(), particles, softening, ferrersOrder, particleMass,
            rungZeroTimestep, activeRung, parameters, deviceSums.data(), deviceId)
            .get();
    }
    if (sums != nullptr) {
        *sums = *deviceSums.data();
    }
    if (momentumCount != 0) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            std::memcpy(hostMomenta[axis], parameters.synchronizedMomenta[axis], momentumCount * sizeof(double));
        }
    }
    deviceMomenta.release();
    deviceSums.release();
    deviceCells.release();
    deviceCandidates.release();
}

}    // namespace cosmictiger::gravity::detail
