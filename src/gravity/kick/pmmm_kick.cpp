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

#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/gravity/tree/kd_tree.hpp>
#include <cosmictiger/gravity/tree/tree_near_field.hpp>
#include <cosmictiger/parallel.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace cosmictiger::gravity::detail {
namespace {

    constexpr size_t kNearSide = 2 * kPmmmMaximumNeighborWidth + 1;
    constexpr size_t kNearCellCount = [] {
        PROFILE();
        size_t count = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            count *= kNearSide;
        }
        return count;
    }();

    struct NearSource {
        ParticleRange range;
        bool localOwner;
    };

    struct NearSources {
        std::array<NearSource, kNearCellCount> values;
        size_t count;
    };

    GridIndices interiorCoordinates(size_t index, size_t side) {
        PROFILE();
        GridIndices coordinates{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            coordinates[axis] = static_cast<std::int64_t>(index % side);
            index /= side;
        }
        return coordinates;
    }

    NearSources nearSources(CartesianGrid const& grid, GridIndices sinkCell, int ghostWidth) {
        PROFILE();
        auto const& layout = grid.layout();
        size_t const nearSide = static_cast<size_t>(2 * ghostWidth + 1);
        NearSources sources{};
        sources.count = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            sources.count *= nearSide;
        }
        for (size_t neighbor = 0; neighbor < sources.count; neighbor++) {
            size_t offsetIndex = neighbor;
            GridIndices coordinates = sinkCell;
            for (size_t axis = 0; axis < NDIM; axis++) {
                coordinates[axis] += static_cast<std::int64_t>(offsetIndex % nearSide) - ghostWidth;
                offsetIndex /= nearSide;
                if (layout.localityCount() == 1) {
                    auto const side = static_cast<std::int64_t>(layout.globalSideLength());
                    coordinates[axis] = (coordinates[axis] + side) % side;
                }
            }
            auto const owner = layout.cellOwner(coordinates);
            bool const localOwner = owner.localityIndex == layout.localityIndex();
            // A same-locality ghost is a copy, even though its ghost flag is
            // clear. Use the exact owned range so index identity excludes self
            // without discarding distinct particles at coincident coordinates.
            auto const range = grid.cell(localOwner ? owner.coordinates : coordinates).localParticles;
            sources.values[neighbor] = {range, localOwner};
        }
        return sources;
    }

    void checkPmmmParameters(CartesianGrid const& grid, SofteningLength softening, int ferrersOrder, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, int ghostWidth) {
        PROFILE();
        auto const& layout = grid.layout();
        if (ghostWidth < 1 || ghostWidth > kPmmmMaximumNeighborWidth) {
            throw std::invalid_argument("PMMM ghost width must be one or two cells");
        }
        if (layout.globalSideLength() < static_cast<size_t>(2 * ghostWidth + 1)) {
            throw std::invalid_argument("PMMM kick requires distinct near cells within the periodic grid");
        }
        if (layout.localityCount() > 1 && layout.ghostWidth() < static_cast<size_t>(ghostWidth)) {
            throw std::invalid_argument("PMMM kick requires a halo at least as wide as its near-cell mask");
        }
        {
            FpeGuard guard;
            expectPositive(softening.inverseQ);
            expectPositive(particleMass);
            expectPositive(rungZeroTimestep);
            expectRange(0, activeRung, Rung::maximumRung());
            double const nearRadius = ghostWidth * layout.spacing();
            if (softening.model == SofteningModel::ferrers && double(softening.qSquared) > nearRadius * nearRadius) {
                throw std::invalid_argument("PMMM softening support extends beyond its direct near-cell region");
            }
        }
        if (ferrersOrder < 0 || ferrersOrder > 3) {
            throw std::invalid_argument("PMMM kick Ferrers order must be between zero and three");
        }
    }

    DirectInteractionParticles nearParticleView(ParticleStorage const& particles) {
        PROFILE();
        DirectInteractionParticles view{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            view.positions[axis] = particles.positionData(axis);
        }
        return view;
    }

    DirectInteractionParticles kickParticleView(ParticleStorage& particles) {
        PROFILE();
        auto view = nearParticleView(particles);
        view.flags = particles.rungData();
        view.potentials = particles.potentialData();
        for (size_t axis = 0; axis < NDIM; axis++) {
            view.velocities[axis] = particles.velocityData(axis);
            view.accelerations[axis] = particles.accelerationData(axis);
        }
        return view;
    }

    ParticleRange candidateSinks(Cell const& cell, [[maybe_unused]] ParticleStorage const& particles, std::uint8_t activeRung) {
        PROFILE();
        auto range = cell.localParticles;
        expectRange(range.begin, range.end, particles.size());
        if (cell.rungRangesValid) {
            range.begin = cell.rungParticles[expectIndex(static_cast<size_t>(activeRung), cell.rungParticles.size())].begin;
            expectRange(cell.localParticles.begin, range.begin, range.end);
        }
        return range;
    }

    template <int order>
    KickParameters softenedKickParameters(SofteningLength softening, KickParameters parameters) {
        PROFILE();
        FpeGuard guard;
        parameters.epsilon = softeningEpsilon<order>(softening);
        return parameters;
    }

    template <int order, bool computePotential>
    void accumulatePmmmNear(ParticleStorage const& particles, ParticleCount sink, NearSources const& sources, DirectInteractionRequest const& request,
        DirectInteractionLanes& lanes) {
        PROFILE();
        Position const position{particles.position(0, sink), particles.position(1, sink), particles.position(2, sink)};
        for (size_t neighbor = 0; neighbor < sources.count; neighbor++) {
            auto const& source = sources.values[neighbor];
            auto const range = source.range;
            expectRange(range.begin, range.end, particles.size());
            if (source.localOwner && sink >= range.begin && sink < range.end) {
                accumulateDirectRange<order, computePotential>(position, range.begin, sink, request, lanes);
                accumulateDirectRange<order, computePotential>(position, sink + 1, range.end, request, lanes);
            } else {
                accumulateDirectRange<order, computePotential>(position, range.begin, range.end, request, lanes);
            }
        }
    }

    struct PmmmField {
        Acceleration acceleration;
        float potential;
    };

    struct PmmmExpansion {
        stf::Expansion<double, kExpansionOrder, true> harmonic;
        symmetric::Expansion<double, kExpansionOrder> full;
        bool isFull;
    };

    PmmmExpansion cellExpansion(Cell const& cell, SofteningModel model) {
        PROFILE();
        PmmmExpansion result{};
        result.isFull = model == SofteningModel::plummer;
        if (result.isFull) {
            for (size_t coefficient = 0; coefficient < result.full.size(); coefficient++) {
                result.full[coefficient] = cell.fullExpansion[coefficient];
            }
        } else {
            for (size_t coefficient = 0; coefficient < result.harmonic.size(); coefficient++) {
                result.harmonic[coefficient] = cell.expansion[coefficient];
            }
        }
        return result;
    }

    template <bool computePotential>
    PmmmField evaluatePmmmField(ParticleStorage const& particles, ParticleCount sink, DirectInteractionLanes const& lanes, PmmmExpansion const& sourceExpansion,
        std::array<double, NDIM> const& center, double spacing, float particleMass) {
        PROFILE();
        Acceleration acceleration{};
        float potential = 0.0f;
        {
            FpeGuard guard;
            std::array<double, NDIM> displacement{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                displacement[axis] = static_cast<double>(particles.position(axis, sink)) - center[axis];
            }
            std::array<double, NDIM> total{};
            double totalPotential = 0.0;
            if (sourceExpansion.isFull) {
                auto const field = symmetric::evaluateL2P<double, kExpansionOrder>(sourceExpansion.full, displacement, spacing);
                total = field.acceleration;
                if constexpr (computePotential) {
                    totalPotential = field.potential;
                }
            } else {
                stf::Expansion<double, kExpansionOrder, true> pointExpansion{};
                stf::addL2L<double, kExpansionOrder, true>(pointExpansion, sourceExpansion.harmonic, displacement, spacing, 1.0);
                total = {-pointExpansion[stf::index(1, 0, 0)], -pointExpansion[stf::index(0, 1, 0)], -pointExpansion[stf::index(0, 0, 1)]};
                if constexpr (computePotential) {
                    totalPotential = pointExpansion[0];
                }
            }
            for (size_t lane = 0; lane < kDirectFloatLanes; lane++) {
                total[0] += double(particleMass) * lanes.x[lane];
                total[1] += double(particleMass) * lanes.y[lane];
                total[2] += double(particleMass) * lanes.z[lane];
                if constexpr (computePotential) {
                    totalPotential += double(particleMass) * lanes.potential[lane];
                }
            }
            acceleration = {static_cast<float>(total[0]), static_cast<float>(total[1]), static_cast<float>(total[2])};
            potential = static_cast<float>(totalPotential);
        }
        return {acceleration, potential};
    }

    template <bool computePotential>
    void applyPmmmField(ParticleStorage const& particles, ParticleCount sink, DirectInteractionLanes const& lanes, PmmmExpansion const& sourceExpansion,
        std::array<double, NDIM> const& center, double spacing, float particleMass, Rung flags, double rungZeroTimestep, DirectInteractionParticles const& view,
        DirectKickBackend const& backend) {
        PROFILE();
        auto const field = evaluatePmmmField<computePotential>(particles, sink, lanes, sourceExpansion, center, spacing, particleMass);
        backend(sink, field.acceleration, field.potential, flags, rungZeroTimestep, view);
    }

    template <int order, bool computePotential>
    void kickPmmm(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, KickParameters parameters, KickSums* sums) {
        PROFILE();
        auto const& layout = grid.layout();
        DirectKickBackend const backend(softenedKickParameters<order>(softening, parameters), sums);
        auto const view = kickParticleView(particles);
        double const spacing = layout.spacing();
        ::cosmictiger::detail::ParallelChunkPlan const plan(layout.interiorCellCount(), 1);
        ::cosmictiger::detail::parallelForChunks(plan,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    for (size_t cellIndex = begin; cellIndex < end; cellIndex++) {
                        auto const coordinates = interiorCoordinates(cellIndex, layout.interiorSideLength());
                        auto const& cell = grid.cell(coordinates);
                        auto const sinks = cell.localParticles;
                        expectRange(sinks.begin, sinks.end, particles.size());
                        if (sinks.begin == sinks.end) {
                            continue;
                        }
                        auto const sources = nearSources(grid, coordinates, parameters.ghostWidth);
                        auto const center = layout.cellCenter(coordinates);
                        auto const sourceExpansion = cellExpansion(cell, softening.model);
                        for (ParticleCount sink = sinks.begin; sink < sinks.end; sink++) {
                            Rung const flags = particles.flags(sink);
                            if (flags.isGhost() || flags.rung() < activeRung) {
                                continue;
                            }
                            DirectInteractionLanes lanes;
                            DirectInteractionRequest request{{}, sinks, view, softening, particleMass, rungZeroTimestep, activeRung};
                            accumulatePmmmNear<order, computePotential>(particles, sink, sources, request, lanes);
                            applyPmmmField<computePotential>(
                                particles, sink, lanes, sourceExpansion, center, spacing, particleMass, flags, rungZeroTimestep, view, backend);
                        }
                    }
                },
                "pmmmKick.parallelForChunks"));
        if (parameters.phase != KickPhase::diagnostics) {
            // All reads of source positions have finished before permutation.
            grid.sortParticlesByRung(particles, parameters.synchronizedMomentumCount == 0 ? nullptr : parameters.synchronizedMomenta);
        }
    }

    template <int order>
    void kickPmmm(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, KickParameters const& parameters, KickSums* sums) {
        PROFILE();
        if (particles.storesPotential() || parameters.sumPotential) {
            kickPmmm<order, true>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        } else {
            kickPmmm<order, false>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        }
    }

    // The prefix stream contains only interior candidate ranges. A crowded
    // cell can span several chunks; empty cells require no copy or task.
    template <typename Function>
    void forEachPmmmCandidateChunk(std::vector<ParticleRange> const& ranges, std::vector<size_t> const& offsets, Function&& function) {
        PROFILE();
        ::cosmictiger::detail::ParallelChunkPlan const plan(offsets.back(), 8);
        ::cosmictiger::detail::parallelForChunks(plan,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    size_t cellIndex = static_cast<size_t>(std::upper_bound(offsets.begin() + 1, offsets.end(), begin) - offsets.begin() - 1);
                    size_t cursor = begin;
                    while (cursor < end) {
                        expectIndex(cellIndex, ranges.size());
                        size_t const stop = std::min(end, offsets[cellIndex + 1]);
                        auto const range = ranges[cellIndex];
                        if (cursor < stop) {
                            ParticleCount const first = range.begin + (cursor - offsets[cellIndex]);
                            ParticleCount const last = range.begin + (stop - offsets[cellIndex]);
                            function(cellIndex, first, last, cursor);
                        }
                        cursor = stop;
                        cellIndex++;
                    }
                },
                "pmmmKick.parallelForChunks"));
    }

    template <int order, bool computePotential>
    void preparePmmmNear(CartesianGrid const& grid, ParticleStorage const& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, std::vector<ParticleRange> const& ranges, std::vector<size_t> const& offsets, std::vector<DirectInteractionLanes>& lanes,
        PmmmProgressObserver const& observer, int ghostWidth) {
        constexpr size_t kProgressBatch = 16;
        PROFILE();

        auto const& layout = grid.layout();
        auto const view = nearParticleView(particles);
        PmmmProgressCounter progress(observer, PmmmProgressStage::directGravity, offsets.back());
        forEachPmmmCandidateChunk(ranges, offsets,
            hpx::annotated_function(
                [&](size_t cellIndex, ParticleCount begin, ParticleCount end, size_t offset) {
                    PROFILE();
                    auto const coordinates = interiorCoordinates(cellIndex, layout.interiorSideLength());
                    auto const sources = nearSources(grid, coordinates, ghostWidth);
                    DirectInteractionRequest const request{{}, ranges[cellIndex], view, softening, particleMass, rungZeroTimestep, activeRung};
                    size_t completed = 0;
                    for (ParticleCount sink = begin; sink < end; sink++) {
                        auto const flags = particles.flags(sink);
                        if (!flags.isGhost() && flags.rung() >= activeRung) {
                            auto& sinkLanes = lanes[expectIndex(offset + static_cast<size_t>(sink - begin), lanes.size())];
                            accumulatePmmmNear<order, computePotential>(particles, sink, sources, request, sinkLanes);
                        }
                        completed++;
                        if (completed == kProgressBatch) {
                            progress.advance(completed);
                            completed = 0;
                        }
                    }
                    if (completed != 0) {
                        progress.advance(completed);
                    }
                },
                "pmmmKick.prepareNearCandidateChunk"));
        progress.finish();
    }

    template <int order>
    void preparePmmmNear(CartesianGrid const& grid, ParticleStorage const& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, bool computePotential, std::vector<ParticleRange> const& ranges, std::vector<size_t> const& offsets,
        std::vector<DirectInteractionLanes>& lanes, PmmmProgressObserver const& observer, int ghostWidth) {
        PROFILE();
        if (computePotential) {
            preparePmmmNear<order, true>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, ranges, offsets, lanes, observer, ghostWidth);
        } else {
            preparePmmmNear<order, false>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, ranges, offsets, lanes, observer, ghostWidth);
        }
    }

    template <int order, bool computePotential>
    void applyPmmmNear(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, KickParameters const& parameters, KickSums* sums, std::vector<ParticleRange> const& ranges, std::vector<size_t> const& offsets,
        std::vector<DirectInteractionLanes> const& lanes, PmmmProgressObserver const& observer) {
        constexpr size_t kProgressBatch = 16;
        PROFILE();

        auto const& layout = grid.layout();
        auto const view = kickParticleView(particles);
        DirectKickBackend const backend(softenedKickParameters<order>(softening, parameters), sums);
        double const spacing = layout.spacing();
        PmmmProgressCounter progress(observer, PmmmProgressStage::particleKick, offsets.back());
        forEachPmmmCandidateChunk(ranges, offsets,
            hpx::annotated_function(
                [&](size_t cellIndex, ParticleCount begin, ParticleCount end, size_t offset) {
                    PROFILE();
                    auto const coordinates = interiorCoordinates(cellIndex, layout.interiorSideLength());
                    auto const& cell = grid.cell(coordinates);
                    auto const center = layout.cellCenter(coordinates);
                    auto const sourceExpansion = cellExpansion(cell, softening.model);
                    size_t completed = 0;
                    for (ParticleCount sink = begin; sink < end; sink++) {
                        auto const flags = particles.flags(sink);
                        if (!flags.isGhost() && flags.rung() >= activeRung) {
                            auto const& sinkLanes = lanes[expectIndex(offset + static_cast<size_t>(sink - begin), lanes.size())];
                            applyPmmmField<computePotential>(
                                particles, sink, sinkLanes, sourceExpansion, center, spacing, particleMass, flags, rungZeroTimestep, view, backend);
                        }
                        completed++;
                        if (completed == kProgressBatch) {
                            progress.advance(completed);
                            completed = 0;
                        }
                    }
                    if (completed != 0) {
                        progress.advance(completed);
                    }
                },
                "pmmmKick.evaluateFarCandidateChunk"));
        progress.finish();
        if (parameters.phase != KickPhase::diagnostics) {
            // Without rung sorting this only invalidates the rung ranges.
            bool const sorting = grid.rungSorting();
            if (sorting) {
                reportPmmmProgress(observer, PmmmProgressStage::particleSort, 0, 1);
            }
            grid.sortParticlesByRung(particles, parameters.synchronizedMomentumCount == 0 ? nullptr : parameters.synchronizedMomenta);
            if (sorting) {
                reportPmmmProgress(observer, PmmmProgressStage::particleSort, 1, 1);
            }
        }
    }

    template <int order>
    void applyPmmmNear(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, bool computePotential, KickParameters const& parameters, KickSums* sums, std::vector<ParticleRange> const& ranges,
        std::vector<size_t> const& offsets, std::vector<DirectInteractionLanes> const& lanes, PmmmProgressObserver const& observer) {
        PROFILE();
        if (computePotential) {
            applyPmmmNear<order, true>(
                grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, ranges, offsets, lanes, observer);
        } else {
            applyPmmmNear<order, false>(
                grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums, ranges, offsets, lanes, observer);
        }
    }

}    // namespace
}    // namespace cosmictiger::gravity::detail

namespace cosmictiger::gravity {

PmmmNearField::PmmmNearField(CartesianGrid const& grid, ParticleStorage const& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, bool sumPotential, bool useCuda, PmmmProgressObserver const& observer, int ghostWidth)
  : grid_(&grid)
  , particles_(&particles)
  , softening_(softening)
  , ferrersOrder_(ferrersOrder)
  , particleMass_(particleMass)
  , rungZeroTimestep_(rungZeroTimestep)
  , activeRung_(activeRung)
  , computePotential_(particles.storesPotential() || sumPotential)
  , cudaEnabled_(useCuda)
  , ghostWidth_(ghostWidth) {
    PROFILE();
    detail::checkPmmmParameters(grid, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, ghostWidth_);
    if (cudaEnabled_) {
        detail::reportPmmmProgress(observer, PmmmProgressStage::cudaPreparation, 0, 1);
    }
    prepareRanges(grid, particles);
#ifdef COSMICTIGER_ENABLE_CUDA
    if (cudaEnabled_) {
        auto const& layout = grid.layout();
        size_t const cellCount = layout.interiorCellCount();
        size_t const candidateCount = offsets_.back();
        if (candidateCount == 0) {
            detail::reportPmmmProgress(observer, PmmmProgressStage::cudaPreparation, 1, 1);
            return;
        }
        cudaCells_.resize(cellCount);
        cudaCandidates_.resize(candidateCount);
        for (size_t cellIndex = 0; cellIndex < cellCount; cellIndex++) {
            auto const range = ranges_[cellIndex];
            if (range.begin == range.end) {
                continue;
            }
            auto const coordinates = detail::interiorCoordinates(cellIndex, layout.interiorSideLength());
            auto const sources = detail::nearSources(grid, coordinates, ghostWidth_);
            auto& cudaCell = cudaCells_[cellIndex];
            cudaCell.nearSourceCount = sources.count;
            for (size_t neighbor = 0; neighbor < sources.count; neighbor++) {
                if (sources.values[neighbor].range.begin > sources.values[neighbor].range.end || sources.values[neighbor].range.end > particles.size()) {
                    throw std::out_of_range("PMMM CUDA source range exceeds particle storage");
                }
                cudaCell.nearSources[neighbor] = sources.values[neighbor].range;
            }
            auto const center = layout.cellCenter(coordinates);
            for (size_t axis = 0; axis < NDIM; axis++) {
                cudaCell.center[axis] = center[axis];
            }
            cudaCell.spacing = layout.spacing();
            for (ParticleCount sink = range.begin; sink < range.end; sink++) {
                cudaCandidates_[offsets_[cellIndex] + static_cast<size_t>(sink - range.begin)] = {sink, cellIndex};
            }
        }
        detail::reportPmmmProgress(observer, PmmmProgressStage::cudaPreparation, 1, 1);
        return;
    }
#else
    if (cudaEnabled_) {
        throw std::invalid_argument("PMMM CUDA kick requires a CUDA-enabled build");
    }
#endif
    lanes_.resize(offsets_.back());
    switch (ferrersOrder_) {
    case 0:
        detail::preparePmmmNear<0>(
            grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, ranges_, offsets_, lanes_, observer, ghostWidth_);
        break;
    case 1:
        detail::preparePmmmNear<1>(
            grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, ranges_, offsets_, lanes_, observer, ghostWidth_);
        break;
    case 2:
        detail::preparePmmmNear<2>(
            grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, ranges_, offsets_, lanes_, observer, ghostWidth_);
        break;
    case 3:
        detail::preparePmmmNear<3>(
            grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, ranges_, offsets_, lanes_, observer, ghostWidth_);
        break;
    default:
        throw std::invalid_argument("PMMM kick Ferrers order must be between zero and three");
    }
}

void PmmmNearField::prepareRanges(CartesianGrid const& grid, ParticleStorage const& particles) {
    PROFILE();
    auto const& layout = grid.layout();
    size_t const cellCount = layout.interiorCellCount();
    if (cellCount >= offsets_.max_size()) {
        throw std::length_error("PMMM near-field offsets exceed addressable memory");
    }
    ranges_.resize(cellCount);
    offsets_.resize(cellCount + 1);
    // Consume ranges to size the compact task output, without examining
    // particle data or allocating lanes for the appended ghost suffix.
    for (size_t cellIndex = 0; cellIndex < cellCount; cellIndex++) {
        auto const coordinates = detail::interiorCoordinates(cellIndex, layout.interiorSideLength());
        auto const range = detail::candidateSinks(grid.cell(coordinates), particles, activeRung_);
        if (cudaEnabled_ && (range.begin > range.end || range.end > particles.size())) {
            throw std::out_of_range("PMMM CUDA sink range exceeds particle storage");
        }
        ranges_[cellIndex] = range;
        ParticleCount const count = range.end - range.begin;
        if (count > offsets_.max_size() - offsets_[cellIndex]) {
            throw std::length_error("PMMM near-field lanes exceed addressable memory");
        }
        offsets_[cellIndex + 1] = offsets_[cellIndex] + static_cast<size_t>(count);
    }
}

// The tree walk writes unit-mass fields per particle; each candidate's first
// lane receives its total, so the CPU kick applies the particle mass and
// combines the far expansion exactly as for the direct near field.
PmmmNearField::PmmmNearField(TreeTag, CartesianGrid const& grid, ParticleStorage const& particles, SofteningLength softening, int ferrersOrder,
    float particleMass, double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, tree::CellKdTrees const& trees,
    PmmmProgressObserver const& observer)
  : grid_(&grid)
  , particles_(&particles)
  , softening_(softening)
  , ferrersOrder_(ferrersOrder)
  , particleMass_(particleMass)
  , rungZeroTimestep_(rungZeroTimestep)
  , activeRung_(activeRung)
  , computePotential_(particles.storesPotential() || parameters.sumPotential)
  , cudaEnabled_(false)
  , ghostWidth_(parameters.ghostWidth) {
    PROFILE();
    detail::checkPmmmParameters(grid, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, ghostWidth_);
    prepareRanges(grid, particles);
    lanes_.resize(offsets_.back());
    detail::reportPmmmProgress(observer, PmmmProgressStage::treeGravity, 0, 1);
    std::array<std::vector<float>, NDIM> accelerations;
    std::vector<float> potentials;
    for (auto& values : accelerations) {
        values.assign(particles.size(), 0.0f);
    }
    if (computePotential_) {
        potentials.assign(particles.size(), 0.0f);
    }
    tree::TreeNearFieldRequest request;
    request.grid = &grid;
    request.particles = &particles;
    request.trees = &trees;
    request.softening = softening;
    request.ferrersOrder = ferrersOrder;
    request.fmmOrder = parameters.fmmOrder;
    request.ghostWidth = ghostWidth_;
    request.openingAngle = parameters.openingAngle;
    request.activeRung = activeRung;
    request.computePotential = computePotential_;
    for (size_t axis = 0; axis < NDIM; axis++) {
        request.accelerations[axis] = accelerations[axis].data();
    }
    request.potentials = computePotential_ ? potentials.data() : nullptr;
    tree::evaluateTreeNearField(request);
    for (size_t cellIndex = 0; cellIndex < ranges_.size(); cellIndex++) {
        auto const range = ranges_[cellIndex];
        for (ParticleCount sink = range.begin; sink < range.end; sink++) {
            auto& lanes = lanes_[offsets_[cellIndex] + static_cast<size_t>(sink - range.begin)];
            lanes.x[0] = accelerations[0][sink];
            lanes.y[0] = accelerations[1][sink];
            lanes.z[0] = accelerations[2][sink];
            if (computePotential_) {
                lanes.potential[0] = potentials[sink];
            }
        }
    }
    detail::reportPmmmProgress(observer, PmmmProgressStage::treeGravity, 1, 1);
}

PmmmNearField PmmmNearField::prepare(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, bool useCuda, PmmmProgressObserver const& observer) {
    PROFILE();
    if (parameters.nearFieldMethod == NearFieldMethod::direct) {
        return PmmmNearField(grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters.sumPotential, useCuda, observer,
            parameters.ghostWidth);
    }
    if (useCuda) {
        throw std::invalid_argument("the tree near field is not yet supported by the CUDA kick");
    }
    detail::checkPmmmParameters(grid, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters.ghostWidth);
    tree::KdTreeParameters treeParameters;
    treeParameters.multipoleCenter = tree::MultipoleCenter::minimumRadius;
    // Unit masses give the raw fields the kick scales by particleMass.
    treeParameters.particleMass = 1.0f;
    auto const trees = tree::CellKdTrees::buildByRungThreshold(grid, particles, activeRung, treeParameters, tree::GeometricMidpointSplitter{},
        parameters.synchronizedMomentumCount == 0 ? nullptr : parameters.synchronizedMomenta);
    return PmmmNearField(TreeTag{}, grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters, trees, observer);
}

PmmmNearField::PmmmNearField(PmmmNearField&& other) noexcept
  : grid_(std::exchange(other.grid_, nullptr))
  , particles_(std::exchange(other.particles_, nullptr))
  , softening_(other.softening_)
  , ferrersOrder_(other.ferrersOrder_)
  , particleMass_(other.particleMass_)
  , rungZeroTimestep_(other.rungZeroTimestep_)
  , activeRung_(other.activeRung_)
  , computePotential_(other.computePotential_)
  , cudaEnabled_(other.cudaEnabled_)
  , ghostWidth_(other.ghostWidth_)
  , ranges_(std::move(other.ranges_))
  , offsets_(std::move(other.offsets_))
  , lanes_(std::move(other.lanes_))
#ifdef COSMICTIGER_ENABLE_CUDA
  , cudaCandidates_(std::move(other.cudaCandidates_))
  , cudaCells_(std::move(other.cudaCells_))
#endif
{
    PROFILE();
}

PmmmNearField& PmmmNearField::operator=(PmmmNearField&& other) noexcept {
    PROFILE();
    if (this != &other) {
        grid_ = std::exchange(other.grid_, nullptr);
        particles_ = std::exchange(other.particles_, nullptr);
        softening_ = other.softening_;
        ferrersOrder_ = other.ferrersOrder_;
        particleMass_ = other.particleMass_;
        rungZeroTimestep_ = other.rungZeroTimestep_;
        activeRung_ = other.activeRung_;
        computePotential_ = other.computePotential_;
        cudaEnabled_ = other.cudaEnabled_;
        ghostWidth_ = other.ghostWidth_;
        ranges_ = std::move(other.ranges_);
        offsets_ = std::move(other.offsets_);
        lanes_ = std::move(other.lanes_);
#ifdef COSMICTIGER_ENABLE_CUDA
        cudaCandidates_ = std::move(other.cudaCandidates_);
        cudaCells_ = std::move(other.cudaCells_);
#endif
    }
    return *this;
}

void PmmmNearField::kick(
    CartesianGrid& grid, ParticleStorage& particles, KickParameters const& parameters, KickSums* sums, PmmmProgressObserver const& observer) {
    PROFILE();
    if (cudaEnabled_) {
        throw std::logic_error("CUDA PMMM near field requires kickCuda");
    }
    if (grid_ != &grid || particles_ != &particles) {
        throw std::logic_error("PMMM near field must be applied once to its original grid and particle storage");
    }
    if (parameters.ghostWidth != ghostWidth_) {
        throw std::invalid_argument("PMMM kick ghost width must match its prepared near field");
    }
    if (!computePotential_ && (particles.storesPotential() || parameters.sumPotential)) {
        throw std::invalid_argument("PMMM near field did not stage the requested potential");
    }
    // Application may partially update particles before a later sink or sort
    // fails. Consume the stage first so retry cannot kick those particles twice.
    grid_ = nullptr;
    particles_ = nullptr;
    switch (ferrersOrder_) {
    case 0:
        detail::applyPmmmNear<0>(grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, parameters, sums, ranges_,
            offsets_, lanes_, observer);
        break;
    case 1:
        detail::applyPmmmNear<1>(grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, parameters, sums, ranges_,
            offsets_, lanes_, observer);
        break;
    case 2:
        detail::applyPmmmNear<2>(grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, parameters, sums, ranges_,
            offsets_, lanes_, observer);
        break;
    case 3:
        detail::applyPmmmNear<3>(grid, particles, softening_, particleMass_, rungZeroTimestep_, activeRung_, computePotential_, parameters, sums, ranges_,
            offsets_, lanes_, observer);
        break;
    default:
        throw std::invalid_argument("PMMM kick Ferrers order must be between zero and three");
    }
}

#ifdef COSMICTIGER_ENABLE_CUDA
void PmmmNearField::kickCuda(
    CartesianGrid& grid, ParticleStorage& particles, KickParameters const& parameters, KickSums* sums, int deviceId, PmmmProgressObserver const& observer) {
    PROFILE();
    if (!cudaEnabled_) {
        throw std::logic_error("CPU PMMM near field cannot be applied on CUDA");
    }
    if (grid_ != &grid || particles_ != &particles) {
        throw std::logic_error("PMMM near field must be applied once to its original grid and particle storage");
    }
    if (parameters.ghostWidth != ghostWidth_) {
        throw std::invalid_argument("PMMM kick ghost width must match its prepared near field");
    }
    if (!computePotential_ && (particles.storesPotential() || parameters.sumPotential)) {
        throw std::invalid_argument("PMMM near field did not stage the requested potential");
    }
    // Once CUDA begins, a failure can leave some particle updates complete.
    grid_ = nullptr;
    particles_ = nullptr;
    detail::reportPmmmProgress(observer, PmmmProgressStage::cudaKick, 0, cudaCandidates_.size());
    if (!cudaCandidates_.empty()) {
        auto const& layout = grid.layout();
        for (size_t cellIndex = 0; cellIndex < ranges_.size(); cellIndex++) {
            if (ranges_[cellIndex].begin == ranges_[cellIndex].end) {
                continue;
            }
            auto const coordinates = detail::interiorCoordinates(cellIndex, layout.interiorSideLength());
            auto const& cell = grid.cell(coordinates);
            auto& descriptor = cudaCells_[cellIndex];
            descriptor.fullExpansion = softening_.model == SofteningModel::plummer;
            if (descriptor.fullExpansion) {
                for (size_t coefficient = 0; coefficient < cell.fullExpansion.size(); coefficient++) {
                    descriptor.expansion[coefficient] = static_cast<double>(cell.fullExpansion[coefficient]);
                }
            } else {
                for (size_t coefficient = 0; coefficient < cell.expansion.size(); coefficient++) {
                    descriptor.expansion[coefficient] = static_cast<double>(cell.expansion[coefficient]);
                }
            }
        }
        KickParameters adjusted = parameters;
        switch (ferrersOrder_) {
        case 0:
            adjusted = detail::softenedKickParameters<0>(softening_, parameters);
            break;
        case 1:
            adjusted = detail::softenedKickParameters<1>(softening_, parameters);
            break;
        case 2:
            adjusted = detail::softenedKickParameters<2>(softening_, parameters);
            break;
        case 3:
            adjusted = detail::softenedKickParameters<3>(softening_, parameters);
            break;
        default:
            throw std::invalid_argument("PMMM kick Ferrers order must be between zero and three");
        }
        detail::applyPmmmKickCuda(cudaCandidates_.data(), cudaCandidates_.size(), cudaCells_.data(), cudaCells_.size(), detail::kickParticleView(particles),
            softening_, ferrersOrder_, computePotential_, particleMass_, rungZeroTimestep_, activeRung_, adjusted, sums, deviceId);
    }
    detail::reportPmmmProgress(observer, PmmmProgressStage::cudaKick, cudaCandidates_.size(), cudaCandidates_.size());
    if (parameters.phase != KickPhase::diagnostics) {
        detail::reportPmmmProgress(observer, PmmmProgressStage::particleSort, 0, 1);
        grid.sortParticlesByRung(particles, parameters.synchronizedMomentumCount == 0 ? nullptr : parameters.synchronizedMomenta);
        detail::reportPmmmProgress(observer, PmmmProgressStage::particleSort, 1, 1);
    }
}
#endif

KickSums kickPmmmParticles(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, std::uint64_t operationId) {
    PROFILE();
    KickSums local;
    std::string localError;
    try {
        kickPmmmParticlesCpu(grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters, &local);
    } catch (std::exception const& error) {
        localError = error.what();
    } catch (...) {
        localError = "unknown local PMMM kick failure";
    }
    return allReduceKickSums(local, parameters.sumPotential, operationId, std::move(localError));
}

void kickPmmmParticlesCpu(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, KickSums* sums) {
    PROFILE();
    detail::checkPmmmParameters(grid, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters.ghostWidth);
    if (parameters.nearFieldMethod == NearFieldMethod::tree) {
        PmmmNearField::prepare(grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters)
            .kick(grid, particles, parameters, sums);
        return;
    }
    switch (ferrersOrder) {
    case 0:
        detail::kickPmmm<0>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        break;
    case 1:
        detail::kickPmmm<1>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        break;
    case 2:
        detail::kickPmmm<2>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        break;
    case 3:
        detail::kickPmmm<3>(grid, particles, softening, particleMass, rungZeroTimestep, activeRung, parameters, sums);
        break;
    default:
        throw std::invalid_argument("PMMM kick Ferrers order must be between zero and three");
    }
}

}    // namespace cosmictiger::gravity
