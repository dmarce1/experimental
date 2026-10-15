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

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/drift/drift.hpp>
#include <cosmictiger/evolution/evolution.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_runtime_convolution.hpp>
#include <cosmictiger/lightcone/lightcone.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <cosmictiger/particle/particle_load_balance.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <hpx/iostream.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cmath>
#include <exception>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmictiger {
namespace detail {

    std::uint64_t evolutionOperation(EvolutionState& state) {
        PROFILE();
        if (state.operationId == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("evolution exhausted collective operation identifiers");
        }
        return state.operationId++;
    }

    template <typename Function>
    void evolutionLocalPhase(EvolutionState& state, Function&& function) {
        PROFILE();
        std::string error;
        try {
            function();
        } catch (std::exception const& failure) {
            error = failure.what();
        } catch (...) {
            error = "unknown local evolution failure";
        }
        gravity::allReduceKickSums({}, false, evolutionOperation(state), std::move(error));
    }

    FriedmannState backgroundState(double a, double derivative) {
        PROFILE();
        return {a, derivative / a, derivative};
    }

    double evolutionClockInterval(EvolutionEvent const& initial, EvolutionState const& state, double rungZeroTimestep) {
        PROFILE();
        if (state.nominalEpoch < initial.nominalEpoch) {
            throw std::runtime_error("evolution snapshot clock moved backwards");
        }
        auto epochs = state.nominalEpoch - initial.nominalEpoch;
        std::uint64_t ticks;
        if (state.timeTick >= initial.timeTick) {
            ticks = state.timeTick - initial.timeTick;
        } else {
            if (epochs == 0) {
                throw std::runtime_error("evolution snapshot clock moved backwards");
            }
            epochs--;
            ticks = gravity::rungTicks(0) - initial.timeTick + state.timeTick;
        }
        // Subtract integer clock coordinates before converting to double.
        // This avoids accumulating rounding over many partial output drifts.
        return rungZeroTimestep * (static_cast<double>(epochs) + std::ldexp(static_cast<double>(ticks), -int(Rung::maximumRung())));
    }

}    // namespace detail

void prepareEvolutionGrid(ParticleStorage& particles, CartesianGrid& grid, size_t meshExponent, size_t ghostWidth, std::uint64_t operationId) {
    constexpr size_t kMinimumExponent = 2;
    constexpr size_t kMaximumExponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
    constexpr size_t kMinimumGhostWidth = 1;
    constexpr size_t kMaximumGhostWidth = gravity::kPmmmMaximumNeighborWidth;
    PROFILE();
    if (meshExponent < kMinimumExponent || meshExponent > kMaximumExponent || ghostWidth < kMinimumGhostWidth || ghostWidth > kMaximumGhostWidth) {
        throw std::invalid_argument("invalid evolution mesh exponent or ghost width");
    }
    if (grid.layout().globalSideLength() == (size_t(1) << meshExponent) && grid.layout().ghostWidth() == ghostWidth) {
        return;
    }
    size_t localityExponent = 0;
    for (size_t side = grid.layout().localitiesPerAxis(); side > 1; side /= 2) {
        localityExponent++;
    }
    std::optional<CartesianGrid> replacement;
    std::string error;
    try {
        replacement.emplace(meshExponent, localityExponent, grid.layout().localityIndex(), ghostWidth);
        // Periodic copies can have their ghost flag unset. Clear using the old
        // grid's ranges before replacing those ranges with the new geometry.
        clearGhostParticles(particles, grid);
    } catch (std::exception const& failure) {
        error = failure.what();
    } catch (...) {
        error = "unknown evolution grid preparation failure";
    }
    gravity::allReduceKickSums({}, false, operationId, std::move(error));
    loadBalanceParticles(particles, *replacement, hpx::find_all_localities(), operationId);
    grid = std::move(*replacement);
}

void extendEvolutionRungs(ParticleStorage& particles, CartesianGrid& grid, EvolutionState& state, unsigned offset) {
    PROFILE();
    auto replacement = state;
    extendEvolutionClock(replacement, offset);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        if (particles.flags(index).isGhost() || unsigned(particles.flags(index).rung()) + offset > Rung::maximumRung()) {
            throw std::invalid_argument("restart rung extension requires owned particles within the available rung range");
        }
        particles.flags(index).setRung(unsigned(particles.flags(index).rung()) + offset);
    }
    state = replacement;
    grid.sortParticlesByRung(particles);
}

void evolveParticles(ParticleStorage& particles, CartesianGrid& grid, GlobalConstantsData const& constants, EvolutionState& state,
    EvolutionObserver const& observer, EvolutionSnapshotObserver const& snapshotObserver, EvolutionProgressObserver const& progressObserver,
    EvolutionInitializationObserver const& initializationObserver, gravity::PmmmProgressObserver const& gravityInitializationObserver) {
    constexpr double kEnergyWeight = 0.5;
    constexpr double kRelativeEventTolerance = 1.0e-10;
    constexpr double kRelativeTimeTolerance = 32.0 * std::numeric_limits<double>::epsilon();
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    constexpr int kImageCutoff = 4;
    constexpr int kWaveCutoff = 4;
    constexpr double kAlphaBox = 1.5;
#ifdef COSMICTIGER_ENABLE_CUDA
    constexpr size_t kDriftFieldsPerDimension = 2;
#endif
    PROFILE();

    auto const localities = hpx::find_all_localities();
    SimulationBackground const background(constants.cosmology);
    if (!(constants.stopPhysicalTime > state.physicalTime) || !(constants.rungZeroTimestep > 0.0)) {
        throw std::invalid_argument("requested stopping time must follow the saved physical time");
    }
    if (constants.outputSnapshotCount == 0) {
        throw std::invalid_argument("output.snapshotCount must be positive");
    }
    auto const initializationStage = [&](EvolutionInitializationStage stage) {
        PROFILE();
        if (initializationObserver) {
            detail::evolutionLocalPhase(state,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        initializationObserver(stage);
                    },
                    "evolution.evolutionLocalPhase"));
        }
    };
    EvolutionEvent const runInitialEvent{state.nominalEpoch, state.timeTick};
    double const runInitialOffset = state.conformalOffset;
    std::optional<SimulationBackgroundTable> backgroundTable;
    initializationStage(EvolutionInitializationStage::backgroundTable);
    detail::evolutionLocalPhase(state,
        hpx::annotated_function(
            [&] {
                PROFILE();
                backgroundTable.emplace(background, detail::backgroundState(state.tickScaleFactor, state.tickConformalDerivative),
                    constants.stopPhysicalTime - state.tickPhysicalTime);
            },
            "evolution.tabulateBackground"));
    double const totalRunConformal = backgroundTable->end().conformalInterval - runInitialOffset;
    if (!(totalRunConformal > 0.0)) {
        throw std::runtime_error("evolution cannot resolve the snapshot conformal interval");
    }
    if (constants.outputSnapshotCount > 1) {
        initializationStage(EvolutionInitializationStage::snapshotCadence);
    }
    std::optional<gravity::PmmmRuntimeConvolution> convolution;
    std::vector<hpx::id_type> fftLocalities;
#ifdef COSMICTIGER_ENABLE_CUDA
    bool prefetchCpuDrift = false;
#endif
    initializationStage(EvolutionInitializationStage::gridAndRungs);
    prepareEvolutionGrid(particles, grid, constants.gravityMeshExponent, static_cast<size_t>(constants.ghostWidth), detail::evolutionOperation(state));
    detail::evolutionLocalPhase(state,
        hpx::annotated_function(
            [&] {
                PROFILE();
                grid.sortParticlesByRung(particles);
#ifdef COSMICTIGER_ENABLE_CUDA
                if (constants.cudaKick && !constants.cudaDrift) {
                    prefetchCpuDrift = cuda::deviceAttribute(cudaDevAttrConcurrentManagedAccess, 0).get() != 0;
                }
#endif
            },
            "evolution.evolutionLocalPhase"));
    units::UnitScales const unitScales(constants.cosmology.hubble, constants.initializationOptions.boxLength);
    std::optional<Lightcone> lightcone;
    if (constants.lightcone.enabled) {
        initializationStage(EvolutionInitializationStage::lightcone);
        detail::evolutionLocalPhase(state,
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    auto options = constants.lightcone;
                    if (!constants.lightconeRunId.empty()) {
                        options.outputPrefix += "." + constants.lightconeRunId;
                    }
                    lightcone.emplace(std::move(options), constants.lightconeObservationConformalTime, constants.lightconeSpeed);
                },
                "evolution.initializeLightcone"));
    }
    initializationStage(EvolutionInitializationStage::periodicConvolution);
    detail::evolutionLocalPhase(state,
        hpx::annotated_function(
            [&] {
                PROFILE();
                convolution.emplace(grid.layout(), constants.fmmOrder, kImageCutoff, kWaveCutoff, kAlphaBox, constants.ghostWidth, constants.softeningModel,
                    constants.softeningLength);
                fftLocalities.resize(localities.size());
                size_t const side = grid.layout().localitiesPerAxis();
                for (size_t index = 0; index < localities.size(); index++) {
                    size_t remainder = index;
                    size_t fftIndex = 0;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        fftIndex = fftIndex * side + remainder % side;
                        remainder /= side;
                    }
                    fftLocalities[fftIndex] = localities[index];
                }
            },
            "evolution.evolutionLocalPhase"));
    std::optional<FftHpxExchange> exchange;
    initializationStage(EvolutionInitializationStage::fftExchange);
    auto const exchangeOperation = detail::evolutionOperation(state);
    detail::evolutionLocalPhase(state,
        hpx::annotated_function(
            [&] {
                PROFILE();
                exchange.emplace(fftLocalities, exchangeOperation, convolution->fftLocalityIndex());
            },
            "evolution.evolutionLocalPhase"));
    EvolutionPhaseTimes phaseTimes;
    std::uint64_t stepStart = 0;
    bool compareDirect = constants.directSinkFraction > 0.0;
    std::array<std::vector<double>, NDIM> synchronizedMomenta;
    ParticleSiloOutput snapshotOutput{};
    auto const evaluate = [&](gravity::KickPhase phase, double elapsedConformalTime, bool reportInitialization = false, bool captureSnapshot = false) {
        PROFILE();
        bool const storedAcceleration = particles.storesAcceleration();
        bool const storedPotential = particles.storesPotential();
        if (compareDirect) {
            detail::evolutionLocalPhase(state,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        particles.configureGravityFields(true, true);
                    },
                    "evolution.prepareDirectComparison"));
        }
        if (reportInitialization) {
            initializationStage(EvolutionInitializationStage::ghostExchange);
        }
        auto const ghostStart = hpx::chrono::high_resolution_clock::now();
        exchangeGhostParticles(particles, grid, localities, detail::evolutionOperation(state));
        phaseTimes.exchange += double(hpx::chrono::high_resolution_clock::now() - ghostStart) * kSecondsPerNanosecond;
        if (reportInitialization) {
            initializationStage(EvolutionInitializationStage::cellMultipoles);
        }
        auto const multipoleStart = hpx::chrono::high_resolution_clock::now();
        detail::evolutionLocalPhase(state,
            hpx::annotated_function(
                [&] {
                    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
                    if (constants.cudaMultipole) {
                        grid.computeMultipolesCuda(
                            particles, static_cast<float>(constants.gravityParticleMass), 0, constants.softeningModel == gravity::SofteningModel::plummer);
                    } else
#endif
                    {
                        grid.computeMultipoles(
                            particles, static_cast<float>(constants.gravityParticleMass), constants.softeningModel == gravity::SofteningModel::plummer);
                    }
                },
                "evolution.evolutionLocalPhase"));
        phaseTimes.multipole = double(hpx::chrono::high_resolution_clock::now() - multipoleStart) * kSecondsPerNanosecond;
        gravity::KickParameters parameters{constants.timestepEta, 0.0, state.scaleFactor, state.timeTick, phase, true};
        parameters.ghostWidth = constants.ghostWidth;
        parameters.minimumKickRung = activeEvolutionRung(state.timeTick);
        parameters.sumKinetic = true;
        parameters.momentaStaggered = state.momentaStaggered;
        parameters.conformalOffset = state.conformalOffset;
        snapshotOutput = {};
        if (captureSnapshot && snapshotObserver) {
            detail::evolutionLocalPhase(state,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        snapshotOutput.particleCount = particles.size();
                        parameters.synchronizedMomentumCount = particles.size();
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            synchronizedMomenta[axis].resize(particles.size());
                            snapshotOutput.synchronizedMomenta[axis] = synchronizedMomenta[axis].data();
                            parameters.synchronizedMomenta[axis] = synchronizedMomenta[axis].data();
                        }
                    },
                    "evolution.prepareSnapshotMomenta"));
        }
        if (reportInitialization) {
            initializationStage(EvolutionInitializationStage::gravityKick);
        }
        gravity::PmmmProgressObserver const noGravityProgress;
        auto const& gravityProgress = reportInitialization ? gravityInitializationObserver : noGravityProgress;
        auto computeFarField = hpx::annotated_function(
            [&] {
                PROFILE();
                convolution->computeExpansions(grid, *exchange, gravityProgress);
            },
            "evolution.computeFarField");
        gravity::KickSums sums;
        gravity::PmmmKickTimings kickTimings;
#ifdef COSMICTIGER_ENABLE_CUDA
        if (constants.cudaKick) {
            sums = gravity::kickPmmmParticlesOverlappedCuda(grid, particles,
                gravity::makeSoftening(static_cast<float>(constants.softeningLength), constants.softeningModel), constants.softeningKernelOrder,
                static_cast<float>(constants.gravityParticleMass), constants.rungZeroTimestep, 0, parameters, computeFarField,
                detail::evolutionOperation(state), 0, gravityProgress, &kickTimings);
        } else
#endif
        {
            sums = gravity::kickPmmmParticlesOverlapped(grid, particles,
                gravity::makeSoftening(static_cast<float>(constants.softeningLength), constants.softeningModel), constants.softeningKernelOrder,
                static_cast<float>(constants.gravityParticleMass), constants.rungZeroTimestep, 0, parameters, computeFarField,
                detail::evolutionOperation(state), gravityProgress, &kickTimings);
        }
        phaseTimes.gravity = kickTimings.gravity;
        phaseTimes.kick = kickTimings.kick;
        if (sums.particleCount != constants.globalParticleCount) {
            throw std::runtime_error("evolution diagnostics did not visit each owned particle exactly once");
        }
        if (reportInitialization) {
            initializationStage(EvolutionInitializationStage::diagnostics);
        }
        if (compareDirect) {
            if (hpx::get_locality_id() == 0) {
                hpx::cout << "Initializing: comparing PMMM with " << (constants.cudaDirect ? "CUDA" : "CPU") << " direct periodic gravity (sink fraction "
                          << constants.directSinkFraction << ")\n"
                          << hpx::iostreams::flush_type{};
            }
            auto const comparison = gravity::compareDirectGravity(particles, grid, constants, detail::evolutionOperation(state));
            if (hpx::get_locality_id() == 0) {
                hpx::cout << gravity::formatDirectComparison(comparison) << hpx::iostreams::flush_type{};
            }
            detail::evolutionLocalPhase(state,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        particles.configureGravityFields(storedAcceleration, storedPotential);
                    },
                    "evolution.releaseDirectComparison"));
            compareDirect = false;
        }
        state.maximumRung = static_cast<std::uint8_t>(sums.maximumRung);
        double const mass = constants.gravityParticleMass;
        double const a = state.scaleFactor;
        updateEnergyHistory(state.energy, a, kEnergyWeight * mass * sums.momentumSquared / (a * a), kEnergyWeight * mass * sums.potential / a,
            state.conformalDerivative, elapsedConformalTime);
        if (phase != gravity::KickPhase::diagnostics) {
            state.momentaStaggered = true;
        }
        detail::evolutionLocalPhase(state,
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    if (observer) {
                        observer(state, sums);
                    }
                },
                "evolution.evolutionLocalPhase"));
        return sums;
    };
    auto const snapshot = [&](gravity::KickSums const& sums) {
        PROFILE();
        detail::evolutionLocalPhase(state,
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    if (snapshotObserver) {
                        snapshotObserver(state, sums, snapshotOutput);
                    }
                },
                "evolution.evolutionLocalPhase"));
    };
    auto const progress = [&](EvolutionProgressKind kind, bool wroteSnapshot, std::uint8_t completedRung) {
        PROFILE();
        if (progressObserver) {
            detail::evolutionLocalPhase(state,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        auto const timingEnd = hpx::chrono::high_resolution_clock::now();
                        double const total = double(timingEnd - stepStart) * kSecondsPerNanosecond;
                        // Attribute reductions, diagnostics, snapshot I/O and
                        // intervening driver bookkeeping to the output phase.
                        phaseTimes.output = total - (phaseTimes.drift + phaseTimes.exchange + phaseTimes.multipole + phaseTimes.gravity + phaseTimes.kick);
                        progressObserver(state, {kind, wroteSnapshot, completedRung, phaseTimes, timingEnd});
                    },
                    "evolution.evolutionLocalPhase"));
        }
    };
    size_t snapshotIndex = 0;
    stepStart = hpx::chrono::high_resolution_clock::now();
    if (!state.momentaStaggered) {
        auto const sums = evaluate(gravity::KickPhase::initial, 0.0, true, constants.outputSnapshotCount > 1);
        if (constants.outputSnapshotCount > 1) {
            if (snapshotObserver) {
                initializationStage(EvolutionInitializationStage::firstSnapshot);
            }
            snapshot(sums);
            snapshotIndex++;
        }
        progress(EvolutionProgressKind::initial, constants.outputSnapshotCount > 1 && bool(snapshotObserver), 0);
    } else {
        if (constants.outputSnapshotCount > 1 || compareDirect) {
            auto const sums = evaluate(gravity::KickPhase::diagnostics, 0.0, true, constants.outputSnapshotCount > 1);
            if (constants.outputSnapshotCount > 1) {
                if (snapshotObserver) {
                    initializationStage(EvolutionInitializationStage::firstSnapshot);
                }
                snapshot(sums);
                snapshotIndex++;
            }
            progress(EvolutionProgressKind::initial, constants.outputSnapshotCount > 1 && bool(snapshotObserver), 0);
        }
    }
    while (state.physicalTime < constants.stopPhysicalTime) {
        phaseTimes = {};
        stepStart = hpx::chrono::high_resolution_clock::now();
        std::uint8_t const completedRung = state.maximumRung;
        auto const event = nextEvolutionEvent(state);
        double const eventInterval = gravity::rungTimestep(constants.rungZeroTimestep, state.maximumRung);
        if (!(state.conformalOffset < eventInterval)) {
            throw std::runtime_error("saved partial drift extends beyond its next rung event");
        }
        auto const tickBackground = detail::backgroundState(state.tickScaleFactor, state.tickConformalDerivative);
        double const tickElapsed = detail::evolutionClockInterval(runInitialEvent, state, constants.rungZeroTimestep);
        double const terminalOffset = backgroundTable->end().conformalInterval - tickElapsed;
        SimulationBackgroundStep next{};
        double interval = 0.0;
        double drift = 0.0;
        bool reachesEvent = true;
        bool finishes = false;
        bool reachesSnapshot = false;
        auto const lightconeOperation = lightcone ? detail::evolutionOperation(state) : 0;
        detail::evolutionLocalPhase(state,
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    // The endpoint was integrated once at initialization. Every
                    // rung and snapshot now samples the same continuous trajectory.
                    double const eventTolerance = kRelativeEventTolerance * eventInterval;
                    bool const terminalAtEvent = std::abs(terminalOffset - eventInterval) <= eventTolerance;
                    finishes = terminalAtEvent || terminalOffset < eventInterval;
                    reachesEvent = terminalAtEvent || terminalOffset > eventInterval;
                    double const targetTime = finishes ? backgroundTable->end().conformalInterval : tickElapsed + eventInterval;
                    next = backgroundTable->interval(tickElapsed, targetTime);
                    next.conformalInterval = finishes ? terminalOffset : eventInterval;
                    if (finishes) {
                        next.physicalInterval = constants.stopPhysicalTime - state.tickPhysicalTime;
                    }
                    if (terminalAtEvent) {
                        next.conformalInterval = eventInterval;
                    }
                    if (snapshotIndex + 1 < constants.outputSnapshotCount) {
                        double const target = totalRunConformal * (static_cast<double>(snapshotIndex) / static_cast<double>(constants.outputSnapshotCount - 1));
                        double const snapshotOffset = target - (tickElapsed - runInitialOffset);
                        if (!(snapshotOffset > state.conformalOffset)) {
                            throw std::runtime_error("evolution cannot resolve further snapshot-time progress");
                        }
                        double const tolerance = kRelativeTimeTolerance * (std::abs(target) + std::abs(tickElapsed) + eventInterval);
                        if (snapshotOffset < next.conformalInterval - tolerance) {
                            next = backgroundTable->interval(tickElapsed, tickElapsed + snapshotOffset);
                            next.conformalInterval = snapshotOffset;
                            if (!(state.tickPhysicalTime + next.physicalInterval < constants.stopPhysicalTime)) {
                                throw std::runtime_error("requested snapshot cadence cannot be resolved before the stopping time");
                            }
                            reachesEvent = false;
                            finishes = false;
                            reachesSnapshot = true;
                        } else {
                            if (std::abs(snapshotOffset - next.conformalInterval) <= tolerance) {
                                reachesSnapshot = true;
                            }
                        }
                    }
                    if (finishes) {
                        if (snapshotIndex + 1 != constants.outputSnapshotCount) {
                            throw std::runtime_error("requested snapshot cadence cannot be resolved before the stopping time");
                        }
                        reachesSnapshot = true;
                    }
                    interval = next.conformalInterval - state.conformalOffset;
                    if (!(interval > 0.0)) {
                        throw std::runtime_error("evolution cannot resolve further conformal-time progress");
                    }
                    if (!finishes && !(state.tickPhysicalTime + next.physicalInterval > state.physicalTime)) {
                        throw std::runtime_error("evolution cannot resolve further physical-time progress");
                    }
                    double const oldDrift = backgroundTable->interval(tickElapsed, tickElapsed + state.conformalOffset).driftFactor;
                    double const driftEnd = finishes ? backgroundTable->end().conformalInterval : tickElapsed + next.conformalInterval;
                    drift = backgroundTable->interval(tickElapsed + state.conformalOffset, driftEnd).driftFactor;
                    if (!(drift > 0.0)) {
                        throw std::runtime_error("evolution cannot resolve further particle drift");
                    }
#ifdef COSMICTIGER_ENABLE_CUDA
                    if (prefetchCpuDrift) {
                        // The completed CUDA kick leaves managed momentum pages
                        // on the GPU. Migrate the owned drift inputs together,
                        // instead of faulting them into each CPU worker on demand.
                        // Keep this transfer in D%, including its completion wait.
                        size_t const owned = static_cast<size_t>(grid.ownedParticleEnd());
                        if (owned != 0) {
                            std::vector<cuda::ManagedMemoryRange> ranges;
                            ranges.reserve(kDriftFieldsPerDimension * NDIM);
                            for (size_t axis = 0; axis < NDIM; axis++) {
                                ranges.push_back({particles.positionData(axis), owned * sizeof(Coordinate)});
                                ranges.push_back({particles.velocityData(axis), owned * sizeof(float)});
                            }
                            cuda::prefetch(std::move(ranges), cudaCpuDeviceId).get();
                        }
                    }
#endif
                    if (lightcone) {
                        double tickConformalTime;
                        {
                            FpeGuard guard;
                            double const fractionalEpoch = std::ldexp(static_cast<double>(state.timeTick), -int(Rung::maximumRung()));
                            tickConformalTime =
                                std::fma(static_cast<double>(state.nominalEpoch), constants.rungZeroTimestep, fractionalEpoch * constants.rungZeroTimestep);
                        }
                        LightconeStep const step{tickBackground, tickConformalTime, state.conformalOffset, driftEnd - tickElapsed, oldDrift, next.driftFactor,
                            &*backgroundTable, tickElapsed};
                        // CPU and CUDA gravity/drift interfaces have completed
                        // their futures here. Sample only the owned grid ranges
                        // before driftParticles removes ghosts and wraps positions.
                        auto const sample = lightcone->sample(particles, grid, background, step);
                        lightcone->write(sample, step, grid.layout().localityIndex(), lightconeOperation, unitScales);
                    }
                    driftParticles(particles, grid, drift, constants.cudaDrift ? DriftExecution::cuda : DriftExecution::cpu);
                },
                "evolution.evolutionLocalPhase"));
        auto const exchangeStart = hpx::chrono::high_resolution_clock::now();
        phaseTimes.drift = double(exchangeStart - stepStart) * kSecondsPerNanosecond;
        loadBalanceParticles(particles, grid, localities, detail::evolutionOperation(state));
        phaseTimes.exchange = double(hpx::chrono::high_resolution_clock::now() - exchangeStart) * kSecondsPerNanosecond;
        state.scaleFactor = next.state.scaleFactor;
        state.conformalDerivative = next.state.conformalDerivative;
        state.physicalTime = finishes ? constants.stopPhysicalTime : state.tickPhysicalTime + next.physicalInterval;
        gravity::KickSums sums;
        if (reachesEvent) {
            state.nominalEpoch = event.nominalEpoch;
            state.timeTick = event.timeTick;
            state.tickScaleFactor = state.scaleFactor;
            state.tickConformalDerivative = state.conformalDerivative;
            state.tickPhysicalTime = state.physicalTime;
            state.conformalOffset = 0.0;
            sums = evaluate(gravity::KickPhase::regular, interval, false, reachesSnapshot);
        } else {
            state.conformalOffset = next.conformalInterval;
            sums = evaluate(gravity::KickPhase::diagnostics, interval, false, reachesSnapshot);
        }
        if (reachesSnapshot) {
            snapshot(sums);
            snapshotIndex++;
        }
        progress(
            reachesEvent ? EvolutionProgressKind::rungStep : EvolutionProgressKind::snapshotSample, reachesSnapshot && bool(snapshotObserver), completedRung);
    }
}

}    // namespace cosmictiger
