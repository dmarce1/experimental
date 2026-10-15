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

#include <cosmictiger/evolution/evolution.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <vector>

TEST(Evolution, EventEndpointRoundoffProducesAResumableClockWithoutADuplicateKick) {
    constexpr size_t kSide = 8;
    PROFILE();

#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
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
    data.meshSideLength = data.gridSideLength = kSide;
    data.globalParticleCount = 1;
    for (size_t axis = 0; axis < NDIM; axis++) {
        data.globalParticleCount *= kSide;
    }
    data.localityCount = 1;
    data.timestepEta = 1.0e6;
    cosmictiger::initializeSimulationConstants(data, 0.5);
    cosmictiger::SimulationBackground const background(data.cosmology);
    auto const initialBackground = background.state(data.initialScaleFactor);
    double const physicalDuration = background.physicalInterval(data.initialScaleFactor, 0.6);
    data.stopPhysicalTime = data.initialPhysicalTime + physicalDuration;
    auto const terminal = background.advancePhysical(initialBackground, data.stopPhysicalTime - data.initialPhysicalTime);
    // Put the nominal event one representable value below the inverse solve.
    // This reproduces the inconsistent ordering previously saved as an offset
    // beyond the event, or as an unnecessary extra partial drift after it.
    data.rungZeroTimestep = std::nextafter(terminal.conformalInterval, 0.0);
    data.originalRungZeroTimestep = data.rungZeroTimestep;

    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(3, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    for (size_t z = 0; z < kSide; z++) {
        for (size_t y = 0; y < kSide; y++) {
            for (size_t x = 0; x < kSide; x++) {
                auto const begin = particles.size();
                particles.append(
                    {cosmictiger::Coordinate((x + 0.5) / kSide), cosmictiger::Coordinate((y + 0.5) / kSide), cosmictiger::Coordinate((z + 0.5) / kSide)},
                    {0.001f, -0.0005f, 0.00025f}, 0);
                grid.setInteriorRange({std::int64_t(x), std::int64_t(y), std::int64_t(z)}, {begin, begin + 1});
            }
        }
    }
    cosmictiger::EvolutionState state;
    state.scaleFactor = state.tickScaleFactor = data.initialScaleFactor;
    state.conformalDerivative = state.tickConformalDerivative = data.initialConformalDerivative;
    state.physicalTime = state.tickPhysicalTime = data.initialPhysicalTime;
    state.operationId = 600000;
    size_t observations = 0;
    std::vector<cosmictiger::EvolutionProgress> progress;
    cosmictiger::evolveParticles(particles, grid, data, state,
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::gravity::KickSums const&) {
                PROFILE();
                observations++;
            },
            "test.evolution.observeBoundary"),
        {},
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::EvolutionProgress const& event) {
                PROFILE();
                progress.push_back(event);
            },
            "test.evolution.observeBoundaryProgress"));
    EXPECT_EQ(observations, 2);
    ASSERT_EQ(progress.size(), 2);
    EXPECT_EQ(progress[0].kind, cosmictiger::EvolutionProgressKind::initial);
    EXPECT_EQ(progress[1].kind, cosmictiger::EvolutionProgressKind::rungStep);
    EXPECT_EQ(progress[1].completedRung, 0);
    EXPECT_FALSE(progress[0].wroteSnapshot);
    EXPECT_FALSE(progress[1].wroteSnapshot);
    EXPECT_EQ(state.maximumRung, 0);
    EXPECT_EQ(state.nominalEpoch, 1);
    EXPECT_EQ(state.timeTick, 0);
    EXPECT_DOUBLE_EQ(state.conformalOffset, 0.0);
    EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
    EXPECT_DOUBLE_EQ(state.tickPhysicalTime, data.stopPhysicalTime);
    EXPECT_TRUE(state.momentaStaggered);
    EXPECT_NO_THROW(cosmictiger::nextEvolutionEvent(state));

    std::array<float, NDIM> momenta;
    for (size_t axis = 0; axis < NDIM; axis++) {
        momenta[axis] = particles.velocity(axis, 0);
    }
    double const resumedDuration = background.advance(background.state(0.6), 0.125 * data.rungZeroTimestep).physicalInterval;
    data.stopPhysicalTime += resumedDuration;
    observations = 0;
    cosmictiger::evolveParticles(particles, grid, data, state,
        hpx::annotated_function(
            [&](cosmictiger::EvolutionState const&, cosmictiger::gravity::KickSums const&) {
                PROFILE();
                observations++;
            },
            "test.evolution.observeResumedBoundary"));
    EXPECT_EQ(observations, 1);
    EXPECT_EQ(state.nominalEpoch, 1);
    EXPECT_EQ(state.timeTick, 0);
    EXPECT_GT(state.conformalOffset, 0.0);
    EXPECT_LT(state.conformalOffset, data.rungZeroTimestep);
    EXPECT_DOUBLE_EQ(state.physicalTime, data.stopPhysicalTime);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_FLOAT_EQ(particles.velocity(axis, 0), momenta[axis]);
    }
}
