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
#include <cosmictiger/simulation_time.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace {

cosmictiger::CosmologyParameters matterCollapseBackground() {
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.hubble = 100.0;
    parameters.omegaBaryonH2 = 0.0;
    parameters.omegaCdmH2 = 2.0;
    parameters.omegaNeutrinoH2 = 0.0;
    parameters.massiveNeutrinos = 0;
    parameters.masslessNeutrinos = 0;
    parameters.effectiveNeutrinos = 0.0;
    parameters.temperatureCmb = 0.0;
    return parameters;
}

}    // namespace

TEST(EvolutionEnd, ClosedMatterCycloidHasAnalyticCrunchAndContractingCutoff) {
    constexpr double kCutoff = 0.25;
    constexpr double kExpectedEnd = 1.5 * std::numbers::pi + 1.0;
    PROFILE();

    auto parameters = matterCollapseBackground();
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const times = background.collapseTimes(kCutoff);
    double const angle = std::acos(1.0 - kCutoff);
    EXPECT_NEAR(times.endPhysicalTime, kExpectedEnd, 3.0e-11);
    EXPECT_NEAR(times.stopPhysicalTime, kExpectedEnd - angle + std::sin(angle), 3.0e-11);
    EXPECT_LT(times.stopPhysicalTime, times.endPhysicalTime);
    auto const presentCutoff = background.collapseTimes(1.0);
    EXPECT_NEAR(presentCutoff.stopPhysicalTime, std::numbers::pi + 2.0, 3.0e-11);
}

TEST(EvolutionEnd, NegativeLambdaMatterHasAnalyticCrunchAndContractingCutoff) {
    constexpr double kCutoff = 0.2;
    constexpr double kExpectedEnd = 0.5 * std::numbers::pi;
    PROFILE();

    cosmictiger::SimulationBackground const background(matterCollapseBackground());
    auto const times = background.collapseTimes(kCutoff);
    double const earlyTime = (2.0 / 3.0) * std::asin(std::sqrt(0.5 * kCutoff * kCutoff * kCutoff));
    EXPECT_NEAR(times.endPhysicalTime, kExpectedEnd, 3.0e-11);
    EXPECT_NEAR(times.stopPhysicalTime, kExpectedEnd - earlyTime, 3.0e-11);
}

TEST(EvolutionEnd, SmallCutoffStaysBeforeSingularity) {
    constexpr double kCutoff = 1.0e-6;
    PROFILE();

    auto parameters = matterCollapseBackground();
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const times = background.collapseTimes(kCutoff);
    double const expectedRemaining = std::sqrt(2.0) * std::pow(kCutoff, 1.5) / 3.0;
    EXPECT_NEAR(times.endPhysicalTime - times.stopPhysicalTime, expectedRemaining, 3.0e-12);
    EXPECT_LT(times.stopPhysicalTime, times.endPhysicalTime);
}

TEST(EvolutionEnd, FullPresentToDefaultCutoffEvolvesOnContractingBranch) {
    constexpr double kCutoff = 1.0e-6;
    PROFILE();

    auto parameters = matterCollapseBackground();
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const times = background.collapseTimes(kCutoff);
    auto const result = background.advancePhysical(background.state(1.0), times.stopPhysicalTime);
    // At this cutoff only about 5e-10 Hubble times remain before the crunch.
    // Converting an absolute proper-time target into a therefore amplifies
    // accumulated time error; also test the density constraint and drift.
    EXPECT_NEAR(result.state.scaleFactor, kCutoff, 1.0e-8);
    EXPECT_LT(result.state.conformalDerivative, 0.0);
    double const a = result.state.scaleFactor;
    double const expectedSquaredDerivative = 2.0 * a - a * a;
    EXPECT_NEAR(result.state.conformalDerivative * result.state.conformalDerivative, expectedSquaredDerivative, 3.0e-10 * expectedSquaredDerivative);
    EXPECT_NEAR(result.driftFactor, 1.0 + std::sqrt((2.0 - a) / a), 1.0e-4);
    EXPECT_GT(result.conformalInterval, std::numbers::pi);
    EXPECT_DOUBLE_EQ(result.physicalInterval, times.stopPhysicalTime);
}

TEST(EvolutionEnd, FullInitialEpochToDefaultCutoffPreservesNominalInterval) {
    constexpr double kInitialA = 0.01;
    constexpr double kCutoff = 1.0e-6;
    PROFILE();

    auto parameters = matterCollapseBackground();
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const times = background.collapseTimes(kCutoff);
    double const initialPhysicalTime = -background.physicalInterval(kInitialA, 1.0);
    auto const result = background.advancePhysical(background.state(kInitialA), times.stopPhysicalTime - initialPhysicalTime);
    EXPECT_NEAR(result.state.scaleFactor, kCutoff, 1.0e-8);
    EXPECT_LT(result.state.conformalDerivative, 0.0);
    double const a = result.state.scaleFactor;
    double const initialAngle = std::acos(1.0 - kInitialA);
    double const finalAngle = std::acos(1.0 - a);
    EXPECT_NEAR(result.conformalInterval, 2.0 * std::numbers::pi - initialAngle - finalAngle, 2.0e-9);
    EXPECT_NEAR(result.driftFactor, std::sqrt((2.0 - kInitialA) / kInitialA) + std::sqrt((2.0 - a) / a), 1.0e-4);
    EXPECT_DOUBLE_EQ(result.physicalInterval, times.stopPhysicalTime - initialPhysicalTime);
}

TEST(EvolutionEnd, RejectsBackgroundsWithoutFutureRecollapse) {
    PROFILE();
    auto parameters = matterCollapseBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const matter(parameters);
    EXPECT_THROW(matter.collapseTimes(1.0e-6), std::domain_error);
    parameters.omegaCdmH2 = 0.3;
    parameters.omegaCurvature = -0.1;
    cosmictiger::SimulationBackground const closedWithVacuum(parameters);
    EXPECT_THROW(closedWithVacuum.collapseTimes(1.0e-6), std::domain_error);
}

TEST(EvolutionEnd, RejectsInvalidCutoffEpochs) {
    PROFILE();
    auto parameters = matterCollapseBackground();
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_THROW(background.collapseTimes(0.0), std::invalid_argument);
    EXPECT_THROW(background.collapseTimes(-0.1), std::invalid_argument);
    EXPECT_THROW(background.collapseTimes(2.1), std::invalid_argument);
}
