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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/gravity/p2m_cuda.hpp>
#endif
#include <cosmictiger/parallel.hpp>

#include <array>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger {

namespace detail {

    struct ParticleCellMoments {
        CellMultipole multipole{};
        double secondMomentTrace = 0.0;
    };

    ParticleCellMoments particleMultipole(
        ParticleRange range, ParticleStorage const& particles, std::array<double, NDIM> const& center, double scale, float particleMass) {
        PROFILE();
        FpeGuard guard;
        expectRange(0, range.begin, range.end);
        expectRange(0, range.end, particles.size());
        // Sum in double and round only the stored result. The fixed-point input
        // coordinates and dyadic cell centers are exactly representable here.
        gravity::stf::Multipole<double, gravity::kMultipoleOrder> moment{};
        double secondMomentTrace = 0.0;
        for (ParticleCount particle = range.begin; particle < range.end; particle++) {
            std::array<double, NDIM> offset{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                offset[axis] = static_cast<double>(particles.position(axis, particle)) - center[axis];
            }
            double const radiusSquared = gravity::stf::addPoint<double, gravity::kMultipoleOrder>(moment, static_cast<double>(particleMass), offset, scale);
            secondMomentTrace += static_cast<double>(particleMass) * radiusSquared;
        }
        ParticleCellMoments result{};
        for (size_t coefficient = 0; coefficient < result.multipole.size(); coefficient++) {
            result.multipole[coefficient] = static_cast<float>(moment[coefficient]);
        }
        result.secondMomentTrace = secondMomentTrace;
        return result;
    }

}    // namespace detail

void CartesianGrid::computeMultipole(GridIndices coordinates, ParticleStorage const& particles, float particleMass) {
    PROFILE();
    auto& entry = cells_[geometry_.cellIndex(coordinates)];
    auto const moments = detail::particleMultipole(entry.localParticles, particles, geometry_.cellCenter(coordinates), geometry_.spacing(), particleMass);
    entry.multipole = moments.multipole;
    entry.secondMomentTrace = moments.secondMomentTrace;
}

void CartesianGrid::computeMultipoles(ParticleStorage const& particles, float particleMass, bool fullMoments) {
    PROFILE();
    // Amortize scheduling over many cells while keeping each particle sum in
    // its original order. Empty cells still require replacing their moments.
    detail::ParallelChunkPlan const chunks(cells_.size(), 64);
    detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    auto const coordinates = geometry_.cellCoordinates(index);
                    if (fullMoments) {
                        FpeGuard guard;
                        gravity::symmetric::Multipole<double, gravity::kMultipoleOrder> moments{};
                        double secondMomentTrace = 0.0;
                        auto const center = geometry_.cellCenter(coordinates);
                        auto const range = cells_[index].localParticles;
                        expectRange(range.begin, range.end, particles.size());
                        for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                            std::array<double, NDIM> offset{};
                            for (size_t axis = 0; axis < NDIM; axis++) {
                                offset[axis] = static_cast<double>(particles.position(axis, particle)) - center[axis];
                            }
                            double const radiusSquared = gravity::symmetric::addPoint<double, gravity::kMultipoleOrder>(
                                moments, static_cast<double>(particleMass), offset, geometry_.spacing());
                            secondMomentTrace += static_cast<double>(particleMass) * radiusSquared;
                        }
                        for (size_t coefficient = 0; coefficient < moments.size(); coefficient++) {
                            cells_[index].fullMultipole[coefficient] = static_cast<float>(moments[coefficient]);
                        }
                        cells_[index].secondMomentTrace = secondMomentTrace;
                    } else {
                        auto const moments = detail::particleMultipole(
                            cells_[index].localParticles, particles, geometry_.cellCenter(coordinates), geometry_.spacing(), particleMass);
                        cells_[index].multipole = moments.multipole;
                        cells_[index].secondMomentTrace = moments.secondMomentTrace;
                    }
                }
            },
            "cartesianGrid.parallelForChunks"));
}

#ifdef COSMICTIGER_ENABLE_CUDA
void CartesianGrid::computeMultipolesCuda(ParticleStorage const& particles, float particleMass, int deviceId, bool fullMoments) {
    PROFILE();
    std::vector<gravity::detail::P2mCudaCell> descriptors(cells_.size());
    for (size_t index = 0; index < cells_.size(); index++) {
        auto const range = cells_[index].localParticles;
        if (range.begin > range.end || range.end > particles.size()) {
            throw std::out_of_range("CUDA P2M local particle range exceeds particle storage");
        }
        auto& descriptor = descriptors[index];
        descriptor.localParticles = range;
        auto const center = geometry_.cellCenter(geometry_.cellCoordinates(index));
        for (size_t axis = 0; axis < NDIM; axis++) {
            descriptor.center[axis] = center[axis];
        }
    }
    gravity::detail::P2mCudaPositions positions{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        positions.coordinates[axis] = particles.positionData(axis);
    }
    auto const results = gravity::detail::computeP2mCuda(descriptors, positions, particles.size(), geometry_.spacing(), particleMass, deviceId, fullMoments);
    for (size_t index = 0; index < cells_.size(); index++) {
        auto& destination = cells_[index];
        if (!fullMoments) {
            for (size_t coefficient = 0; coefficient < destination.multipole.size(); coefficient++) {
                destination.multipole[coefficient] = results[index].multipole[coefficient];
            }
        }
        if (fullMoments) {
            for (size_t coefficient = 0; coefficient < destination.fullMultipole.size(); coefficient++) {
                destination.fullMultipole[coefficient] = results[index].fullMultipole[coefficient];
            }
        }
        destination.secondMomentTrace = results[index].secondMomentTrace;
    }
}
#endif

void CartesianGrid::sortParticlesByRung(ParticleStorage& particles, double* const* companionFields) {
    PROFILE();
    // Every cell builds 64 rung ranges even when empty; use coarse cell chunks
    // without adding a pass over ranges or splitting an individual cell's sort.
    detail::ParallelChunkPlan const chunks(cells_.size(), 64);
    detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    sortCellParticlesByRung(cells_[index], particles, companionFields);
                }
            },
            "cartesianGrid.parallelForChunks"));
}

}    // namespace cosmictiger
