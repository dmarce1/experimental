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

#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <stddef.h>

#include <array>
#include <cmath>
#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif
#include <cosmictiger/evolution/evolution.hpp>
#include <cosmictiger/evolution/evolution_checkpoint.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hpx/modules/filesystem.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

cosmictiger::GlobalConstantsData uniformConstants() {
    PROFILE();
    cosmictiger::GlobalConstantsData data;
    data.cosmology.hubble = 50.0;
    data.cosmology.omegaBaryonH2 = 0.05;
    data.cosmology.omegaCdmH2 = 0.20;
    data.cosmology.omegaNeutrinoH2 = 0.0;
    data.cosmology.massiveNeutrinos = 0;
    data.cosmology.masslessNeutrinos = 0;
    data.cosmology.effectiveNeutrinos = 0.0;
    data.cosmology.temperatureCmb = 0.0;
    data.initializationOptions.meshExponent = 3;
    data.gravityMeshExponent = data.initializationOptions.meshExponent;
    data.meshSideLength = data.gridSideLength = 8;
    data.globalParticleCount = 512;
    data.localityCount = 1;
    cosmictiger::initializeSimulationConstants(data, 0.5);
    return data;
}

cosmictiger::EvolutionState initialState(cosmictiger::GlobalConstantsData const& data, std::uint64_t operation) {
    PROFILE();
    cosmictiger::EvolutionState state;
    state.scaleFactor = state.tickScaleFactor = data.initialScaleFactor;
    state.conformalDerivative = state.tickConformalDerivative = data.initialConformalDerivative;
    state.physicalTime = state.tickPhysicalTime = data.initialPhysicalTime;
    state.operationId = operation;
    return state;
}

void uniformParticles(cosmictiger::ParticleStorage& particles, cosmictiger::CartesianGrid& grid) {
    PROFILE();
    for (size_t z = 0; z < 8; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                auto const begin = particles.size();
                particles.append({cosmictiger::Coordinate((x + 0.5) / 8.0), cosmictiger::Coordinate((y + 0.5) / 8.0), cosmictiger::Coordinate((z + 0.5) / 8.0)},
                    {0.001f, -0.0005f, 0.00025f}, 0);
                grid.setInteriorRange({std::int64_t(x), std::int64_t(y), std::int64_t(z)}, {begin, begin + 1});
            }
        }
    }
}

TEST(Evolution, IndependentMeshRebinsOwnedParticlesAndPreservesFields) {
    constexpr size_t kParticleExponent = 3;
    constexpr size_t kCoarseExponent = 2;
    constexpr size_t kFineExponent = 4;
    constexpr size_t kParticleSide = size_t(1) << kParticleExponent;
    constexpr size_t kParticleCount = size_t(1) << (NDIM * kParticleExponent);
    constexpr size_t kCoarseOccupancy = size_t(1) << (NDIM * (kParticleExponent - kCoarseExponent));
    constexpr std::array<float, NDIM> kVelocity{0.001f, -0.0005f, 0.00025f};
    constexpr std::uint8_t kRungCount = 7;
    constexpr std::uint64_t kGhostOperation = 121000;
    constexpr std::uint64_t kCoarseOperation = 121001;
    constexpr std::uint64_t kFineOperation = 121002;
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU remeshing regression uses the CPU-only build to respect the shared GPU budget";
#endif
    cosmictiger::ParticleStorage particles;
    particles.configureGravityFields(true, true);
    cosmictiger::CartesianGrid grid(kParticleExponent, 0, 0, 2);
    uniformParticles(particles, grid);
    for (size_t index = 0; index < particles.size(); index++) {
        particles.flags(index).setRung(index % kRungCount);
        particles.potential(index) = static_cast<float>(index);
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.acceleration(axis, index) = static_cast<float>(index + axis);
        }
    }
    cosmictiger::exchangeGhostParticles(particles, grid, hpx::find_all_localities(), kGhostOperation);
    ASSERT_GT(particles.size(), kParticleCount);
    cosmictiger::prepareEvolutionGrid(particles, grid, kCoarseExponent, 1, kCoarseOperation);
    ASSERT_EQ(particles.size(), kParticleCount);
    EXPECT_EQ(grid.layout().globalSideLength(), 4);
    EXPECT_EQ(grid.layout().ghostWidth(), 1);
    for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
        auto const coordinates = grid.layout().cellCoordinates(cellIndex);
        auto const range = grid.cell(coordinates).localParticles;
        EXPECT_EQ(range.end - range.begin, grid.layout().isInterior(coordinates) ? kCoarseOccupancy : 0);
    }
    cosmictiger::prepareEvolutionGrid(particles, grid, kFineExponent, 2, kFineOperation);
    ASSERT_EQ(particles.size(), kParticleCount);
    EXPECT_EQ(grid.layout().globalSideLength(), 16);
    EXPECT_EQ(grid.layout().ghostWidth(), 2);
    size_t ownedCount = 0;
    for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
        auto const coordinates = grid.layout().cellCoordinates(cellIndex);
        auto const range = grid.cell(coordinates).localParticles;
        ASSERT_LE(range.end - range.begin, 1);
        for (auto index = range.begin; index < range.end; index++) {
            ASSERT_TRUE(grid.layout().isInterior(coordinates));
            ownedCount++;
            size_t originalIndex = 0;
            size_t stride = 1;
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const position = static_cast<double>(particles.position(axis, index));
                originalIndex += static_cast<size_t>(position * kParticleSide) * stride;
                stride *= kParticleSide;
                EXPECT_EQ(coordinates[axis], static_cast<std::int64_t>(position * grid.layout().globalSideLength()));
                EXPECT_FLOAT_EQ(particles.velocity(axis, index), kVelocity[axis]);
            }
            EXPECT_FALSE(particles.flags(index).isGhost());
            EXPECT_EQ(particles.flags(index).rung(), originalIndex % kRungCount);
            EXPECT_FLOAT_EQ(particles.potential(index), static_cast<float>(originalIndex));
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_FLOAT_EQ(particles.acceleration(axis, index), static_cast<float>(originalIndex + axis));
            }
        }
    }
    EXPECT_EQ(ownedCount, kParticleCount);
}

TEST(Evolution, IndependentMeshOffLatticeCheckpointContinuesWithOriginalClockAndLargerRungZero) {
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build to respect the shared GPU budget";
#endif
    auto data = uniformConstants();
    data.gravityMeshExponent = 2;
    data.gridSideLength = size_t(1) << data.gravityMeshExponent;
    data.ghostWidth = 1;
    cosmictiger::initializeSimulationConstants(data, data.initialScaleFactor);
    double const original = data.rungZeroTimestep;
    // Continue beyond the original present-day endpoint.
    data.stopPhysicalTime = 0.05;
    cosmictiger::ParticleStorage fullParticles;
    cosmictiger::CartesianGrid fullGrid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(fullParticles, fullGrid);
    auto full = initialState(data, 100000);
    cosmictiger::evolveParticles(fullParticles, fullGrid, data, full);
    EXPECT_DOUBLE_EQ(full.physicalTime, 0.05);
    EXPECT_GT(full.scaleFactor, 1.0);
    EXPECT_GE(full.nominalEpoch, 1);
    EXPECT_TRUE(full.energy.initialized);

    cosmictiger::ParticleStorage splitParticles;
    cosmictiger::CartesianGrid splitGrid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(splitParticles, splitGrid);
    auto split = initialState(data, 200000);
    auto first = data;
    first.stopPhysicalTime = -0.2;
    cosmictiger::evolveParticles(splitParticles, splitGrid, first, split);
    ASSERT_GT(split.conformalOffset, 0.0);
    double const initialEnergy = split.energy.initialEnergy;
    cosmictiger::InitialConditionsSiloMetadata const metadata{data.cosmology, data.initialScaleFactor, data.initializationOptions.boxLength, 1.0, 0.1, 0.1,
        data.initializationOptions.seed, 3, data.initializationOptions.component, true};
    auto const filename = (hpx::filesystem::temp_directory_path() / ("cosmictiger_evolution_driver_" + std::to_string(getpid()) + ".silo")).string();
    cosmictiger::writeEvolutionCheckpoint(filename, splitParticles, splitGrid, first, split, metadata);
    // Keep an optional fixture for a subsequent, separate executable restart
    // smoke test; running two HPX runtimes concurrently exceeds the local budget.
    if (char const* fixture = std::getenv("COSMICTIGER_EVOLUTION_TEST_CHECKPOINT")) {
        cosmictiger::writeEvolutionCheckpoint(fixture, splitParticles, splitGrid, first, split, metadata);
    }
    auto saved = cosmictiger::readEvolutionCheckpointMetadata(filename);
    cosmictiger::ParticleStorage resumedParticles;
    cosmictiger::CartesianGrid resumedGrid(saved.constants.gravityMeshExponent, 0, 0, saved.constants.ghostWidth);
    cosmictiger::readParticleSilo(filename, resumedParticles, resumedGrid);
    hpx::filesystem::remove(filename);
    auto resumed = saved.state;
    auto resumedData = saved.constants;
    resumedData.stopPhysicalTime = data.stopPhysicalTime;
    cosmictiger::extendEvolutionRungs(resumedParticles, resumedGrid, resumed, 2);
    resumedData.rungZeroTimestep *= 4.0;
    resumedData.rungZeroExtension += 2;
    EXPECT_DOUBLE_EQ(resumedData.originalRungZeroTimestep, original);
    cosmictiger::evolveParticles(resumedParticles, resumedGrid, resumedData, resumed);
    EXPECT_DOUBLE_EQ(resumed.energy.initialEnergy, initialEnergy);
    EXPECT_DOUBLE_EQ(resumed.physicalTime, full.physicalTime);
    EXPECT_NEAR(resumed.scaleFactor, full.scaleFactor, 1.0e-10);
    for (size_t index = 0; index < data.globalParticleCount; index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(double(resumedParticles.position(axis, index)), double(fullParticles.position(axis, index)), 1.0e-6);
        }
    }
    EXPECT_NEAR(resumed.energy.kinetic, full.energy.kinetic, 1.0e-8);
    EXPECT_NEAR(resumed.energy.potential, full.energy.potential, 1.0e-8);
}

TEST(Evolution, DirectComparisonUsesInitialAndRestartEpochAndReleasesTemporaryFields) {
    constexpr double kShortInterval = 1.0e-6;
    constexpr std::uint64_t kOperation = 0x44495245564f0000ULL;
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU direct comparison integration uses the CPU-only build";
#endif
    auto data = uniformConstants();
    data.directSinkFraction = 1.0 / double(data.globalParticleCount);
    data.stopPhysicalTime = data.initialPhysicalTime + kShortInterval;
    data.outputSnapshotCount = 1;
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(particles, grid);
    particles.position(0, 0) = cosmictiger::Coordinate(double(particles.position(0, 0)) + 0.0125);
    auto state = initialState(data, kOperation);
    for (size_t invocation = 0; invocation < 2; invocation++) {
        double const startingTime = state.physicalTime;
        size_t evaluations = 0;
        cosmictiger::evolveParticles(particles, grid, data, state,
            hpx::annotated_function(
                [&](cosmictiger::EvolutionState const& current, cosmictiger::gravity::KickSums const&) {
                    PROFILE();
                    if (evaluations == 0) {
                        EXPECT_DOUBLE_EQ(current.physicalTime, startingTime);
                    }
                    EXPECT_FALSE(particles.storesAcceleration());
                    EXPECT_FALSE(particles.storesPotential());
                    evaluations++;
                },
                "test.evolution.directComparisonEpoch"));
        EXPECT_GE(evaluations, 2);
        EXPECT_FALSE(particles.storesAcceleration());
        EXPECT_FALSE(particles.storesPotential());
        EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
        data.stopPhysicalTime += kShortInterval;
    }
}

TEST(Evolution, ForwardPhysicalTimeCanContractScaleFactor) {
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
    auto data = uniformConstants();
    data.stopPhysicalTime = data.initialPhysicalTime + 0.02;
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(particles, grid);
    auto state = initialState(data, 300000);
    state.conformalDerivative = state.tickConformalDerivative = -data.initialConformalDerivative;
    cosmictiger::evolveParticles(particles, grid, data, state);
    EXPECT_LT(state.scaleFactor, data.initialScaleFactor);
    EXPECT_LT(state.conformalDerivative, 0.0);
    EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
    EXPECT_GT(state.energy.kinetic, 0.0);
    EXPECT_TRUE(state.energy.initialized);
}

TEST(Evolution, ProgressReportsCompletedFineRungWithoutSnapshot) {
    constexpr unsigned kFineRung = 2;
    PROFILE();

#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
    auto data = uniformConstants();
    data.timestepEta = 1.0e6;
    cosmictiger::SimulationBackground const background(data.cosmology);
    double const interval = cosmictiger::gravity::rungTimestep(data.rungZeroTimestep, kFineRung);
    data.stopPhysicalTime = data.initialPhysicalTime + background.advance(background.state(data.initialScaleFactor), interval).physicalInterval;
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(particles, grid);
    for (size_t index = 0; index < particles.size(); index++) {
        particles.flags(index).setRung(kFineRung);
    }
    auto state = initialState(data, 350000);
    state.maximumRung = kFineRung;
    state.momentaStaggered = true;
    std::vector<cosmictiger::EvolutionProgress> progress;
    cosmictiger::evolveParticles(particles, grid, data, state, {}, {},
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::EvolutionProgress const& event) {
                PROFILE();
                progress.push_back(event);
            },
            "test.evolution.observeFineRungProgress"));
    ASSERT_EQ(progress.size(), 1);
    EXPECT_EQ(progress[0].kind, cosmictiger::EvolutionProgressKind::rungStep);
    EXPECT_EQ(progress[0].completedRung, kFineRung);
    EXPECT_FALSE(progress[0].wroteSnapshot);
    EXPECT_GT(progress[0].timingEnd, 0);
    auto const& times = progress[0].phaseTimes;
    EXPECT_GT(times.drift, 0.0);
    EXPECT_GT(times.exchange, 0.0);
    EXPECT_GT(times.multipole, 0.0);
    EXPECT_GT(times.gravity, 0.0);
    EXPECT_GT(times.kick, 0.0);
    EXPECT_GE(times.output, 0.0);
}

TEST(Evolution, OneHundredSnapshotsIncludeEndpointsAtEqualConformalIntervalsWithoutExtraKicks) {
    constexpr double kFinalScaleFactor = 0.51;
    constexpr size_t kPhaseCount = 6;
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    constexpr double kTimingRoundingTolerance = kSecondsPerNanosecond;
    PROFILE();

#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
    auto data = uniformConstants();
    data.outputSnapshotCount = 100;
    // Keep every particle on rung zero, with the short output interval
    // entirely before the first event of the original nominal timestep.
    data.timestepEta = 1.0e6;
    cosmictiger::SimulationBackground const background(data.cosmology);
    data.stopPhysicalTime = data.initialPhysicalTime + background.physicalInterval(data.initialScaleFactor, kFinalScaleFactor);
    double const duration = 2.0 * (std::sqrt(kFinalScaleFactor) - std::sqrt(data.initialScaleFactor));
    ASSERT_LT(duration, data.rungZeroTimestep);
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(particles, grid);
    auto state = initialState(data, 400000);
    std::vector<double> conformalTimes;
    std::vector<double> physicalTimes;
    std::vector<cosmictiger::EvolutionProgress> progress;
    std::vector<std::array<float, NDIM>> initialMomenta;
    size_t diagnosticCount = 0;
    cosmictiger::evolveParticles(particles, grid, data, state,
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::gravity::KickSums const&) {
                PROFILE();
                diagnosticCount++;
            },
            "test.evolution.countSnapshotDiagnostics"),
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const& snapshot, cosmictiger::gravity::KickSums const& sums, cosmictiger::ParticleSiloOutput const& output) {
                PROFILE();
                conformalTimes.push_back(
                    data.rungZeroTimestep * (double(snapshot.nominalEpoch) + std::ldexp(double(snapshot.timeTick), -int(cosmictiger::Rung::maximumRung()))) +
                    snapshot.conformalOffset);
                physicalTimes.push_back(snapshot.physicalTime);
                EXPECT_EQ(output.particleCount, particles.size());
                double outputMomentumSquared = 0.0;
                for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
                    auto const coordinates = grid.layout().cellCoordinates(cellIndex);
                    if (!grid.layout().isInterior(coordinates)) {
                        continue;
                    }
                    auto const range = grid.cell(coordinates).localParticles;
                    for (auto particle = range.begin; particle < range.end; particle++) {
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            double const momentum = output.synchronizedMomenta[axis][particle];
                            outputMomentumSquared += momentum * momentum;
                        }
                    }
                }
                EXPECT_NEAR(outputMomentumSquared, sums.momentumSquared, 1.0e-12 * (1.0 + sums.momentumSquared));
                EXPECT_EQ(sums.particleCount, data.globalParticleCount);
                EXPECT_TRUE(snapshot.momentaStaggered);
                EXPECT_EQ(snapshot.maximumRung, 0);
                EXPECT_EQ(snapshot.nominalEpoch, 0);
                EXPECT_EQ(snapshot.timeTick, 0);
                if (initialMomenta.empty()) {
                    initialMomenta.resize(data.globalParticleCount);
                    for (size_t particle = 0; particle < data.globalParticleCount; particle++) {
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            initialMomenta[particle][axis] = particles.velocity(axis, particle);
                        }
                    }
                }
                for (size_t axis = 0; axis < NDIM; axis++) {
                    EXPECT_EQ(particles.velocity(axis, 0), initialMomenta[0][axis]);
                }
            },
            "test.evolution.observeSnapshot"),
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::EvolutionProgress const& event) {
                PROFILE();
                progress.push_back(event);
            },
            "test.evolution.observeSnapshotProgress"));
    ASSERT_EQ(conformalTimes.size(), 100);
    ASSERT_EQ(physicalTimes.size(), 100);
    EXPECT_EQ(diagnosticCount, 100);
    ASSERT_EQ(progress.size(), 100);
    EXPECT_EQ(progress.front().kind, cosmictiger::EvolutionProgressKind::initial);
    EXPECT_DOUBLE_EQ(progress.front().phaseTimes.drift, 0.0);
    for (size_t index = 0; index < progress.size(); index++) {
        EXPECT_TRUE(progress[index].wroteSnapshot);
        EXPECT_GT(progress[index].timingEnd, 0);
        auto const& times = progress[index].phaseTimes;
        std::array<double, kPhaseCount> const durations{times.drift, times.exchange, times.multipole, times.gravity, times.kick, times.output};
        double total = 0.0;
        for (double duration : durations) {
            EXPECT_GE(duration, 0.0);
            total += duration;
        }
        EXPECT_GT(total, 0.0);
        if (index != 0) {
            EXPECT_EQ(progress[index].kind, cosmictiger::EvolutionProgressKind::snapshotSample);
            EXPECT_GT(times.drift, 0.0);
            ASSERT_GT(progress[index].timingEnd, progress[index - 1].timingEnd);
            double const elapsed = double(progress[index].timingEnd - progress[index - 1].timingEnd) * kSecondsPerNanosecond;
            // A current-row timer cannot cover more than the time since the
            // previous report; accumulated timings would violate this bound.
            EXPECT_LE(total, elapsed + kTimingRoundingTolerance);
        }
    }
    EXPECT_DOUBLE_EQ(conformalTimes.front(), 0.0);
    EXPECT_DOUBLE_EQ(physicalTimes.front(), data.initialPhysicalTime);
    EXPECT_DOUBLE_EQ(physicalTimes.back(), data.stopPhysicalTime);
    EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
    EXPECT_NEAR(state.scaleFactor, kFinalScaleFactor, 1.0e-11);
    for (size_t snapshot = 0; snapshot < conformalTimes.size(); snapshot++) {
        double const expected = duration * double(snapshot) / double(conformalTimes.size() - 1);
        EXPECT_NEAR(conformalTimes[snapshot], expected, 1.0e-10 * duration);
        if (snapshot != 0) {
            EXPECT_GT(physicalTimes[snapshot], physicalTimes[snapshot - 1]);
        }
    }
    // Diagnostic-only output preserves the integration momentum, even when
    // the output time lies between its ordinary half-step kick events.
    for (size_t particle = 0; particle < data.globalParticleCount; particle++) {
        EXPECT_EQ(particles.flags(particle).rung(), 0);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.velocity(axis, particle), initialMomenta[particle][axis]);
        }
    }
    double const exactDrift = 2.0 * (1.0 / std::sqrt(data.initialScaleFactor) - 1.0 / std::sqrt(kFinalScaleFactor));
    double const fixedPointTolerance = double(conformalTimes.size() + 1) * std::ldexp(1.0, -cosmictiger::Coordinate::fractionalBits());
    for (size_t axis = 0; axis < NDIM; axis++) {
        double const expectedPosition = 0.5 / 8.0 + double(initialMomenta[0][axis]) * exactDrift;
        EXPECT_NEAR(double(particles.position(axis, 0)), expectedPosition, fixedPointTolerance);
    }
}

#ifdef COSMICTIGER_ENABLE_CUDA
void checkCudaEvolution(bool cudaDrift) {
    constexpr double kFinalScaleFactor = 0.51;
    PROFILE();

    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    try {
        if (cosmictiger::cuda::deviceCount().get() == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        if (!cosmictiger::cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed particle storage";
        }
    } catch (cosmictiger::cuda::CudaError const& error) {
        if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
            GTEST_SKIP() << error.what();
        }
        throw;
    }

    auto data = uniformConstants();
    data.cudaMultipole = true;
    data.cudaKick = true;
    data.cudaDrift = cudaDrift;
    data.outputSnapshotCount = cudaDrift ? 2 : 1;
    data.timestepEta = 1.0e6;
    cosmictiger::SimulationBackground const background(data.cosmology);
    data.stopPhysicalTime = data.initialPhysicalTime + background.physicalInterval(data.initialScaleFactor, kFinalScaleFactor);
    double const duration = 2.0 * (std::sqrt(kFinalScaleFactor) - std::sqrt(data.initialScaleFactor));
    ASSERT_LT(duration, data.rungZeroTimestep);
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    uniformParticles(particles, grid);
    double const initialPosition = double(particles.position(0, 0));
    auto state = initialState(data, 500000);
    size_t diagnosticCount = 0;
    size_t snapshotCount = 0;
    cosmictiger::evolveParticles(particles, grid, data, state,
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const& diagnosticState, cosmictiger::gravity::KickSums const& sums) {
                PROFILE();
                diagnosticCount++;
                EXPECT_EQ(sums.particleCount, data.globalParticleCount);
                if (diagnosticCount == 1) {
                    EXPECT_DOUBLE_EQ(diagnosticState.physicalTime, data.initialPhysicalTime);
                } else {
                    if (diagnosticCount == 2) {
                        EXPECT_DOUBLE_EQ(diagnosticState.physicalTime, data.stopPhysicalTime);
                        EXPECT_GT(diagnosticState.conformalOffset, 0.0);
                    }
                }
            },
            "test.evolution.observeCudaDrift"),
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::gravity::KickSums const& sums, cosmictiger::ParticleSiloOutput const& output) {
                PROFILE();
                snapshotCount++;
                EXPECT_EQ(output.particleCount, particles.size());
                double momentumSquared = 0.0;
                for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
                    auto const coordinates = grid.layout().cellCoordinates(cellIndex);
                    if (!grid.layout().isInterior(coordinates)) {
                        continue;
                    }
                    auto const range = grid.cell(coordinates).localParticles;
                    for (auto index = range.begin; index < range.end; index++) {
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            double const momentum = output.synchronizedMomenta[axis][index];
                            momentumSquared += momentum * momentum;
                        }
                    }
                }
                EXPECT_NEAR(momentumSquared, sums.momentumSquared, 1.0e-12 * (1.0 + sums.momentumSquared));
            },
            "test.evolution.observeCudaSnapshot"));

    EXPECT_EQ(diagnosticCount, 2);
    EXPECT_EQ(snapshotCount, data.outputSnapshotCount);
    EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
    EXPECT_NEAR(state.scaleFactor, kFinalScaleFactor, 1.0e-11);
    EXPECT_EQ(state.nominalEpoch, 0);
    EXPECT_EQ(state.timeTick, 0);
    EXPECT_GT(state.conformalOffset, 0.0);
    EXPECT_TRUE(state.energy.initialized);
    EXPECT_NE(double(particles.position(0, 0)), initialPosition);
}

TEST(Evolution, CudaMultipoleKickAndDriftCompleteInitialAndFinalDiagnostics) {
    PROFILE();
    checkCudaEvolution(true);
}

TEST(Evolution, CudaKickHandsManagedParticlesToCpuDrift) {
    PROFILE();
    checkCudaEvolution(false);
}
#endif

}    // namespace
