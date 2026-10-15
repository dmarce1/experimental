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

#include <cosmictiger/einstein_boltzmann.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>

#include <gtest/gtest.h>
#include <hpx/timing/high_resolution_clock.hpp>
#include <iostream>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace {

cosmictiger::CosmologyParameters emptyBackground() {
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.hubble = 100.0;
    parameters.omegaBaryonH2 = 0.0;
    parameters.omegaCdmH2 = 0.0;
    parameters.omegaNeutrinoH2 = 0.0;
    parameters.massiveNeutrinos = 0;
    parameters.masslessNeutrinos = 0;
    parameters.effectiveNeutrinos = 0.0;
    parameters.temperatureCmb = 0.0;
    return parameters;
}

TEST(SimulationTime, MatterConformalIntervalAndMomentumConversion) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_NEAR(background.conformalInterval(0.01, 1.0), 1.8, 2.0e-11);
    EXPECT_NEAR(background.conformalDerivative(0.04), 0.2, 2.0e-14);
}

TEST(SimulationTime, VacuumConformalInterval) {
    PROFILE();
    cosmictiger::SimulationBackground const background(emptyBackground());
    EXPECT_NEAR(background.conformalInterval(0.2, 0.8), 3.75, 2.0e-10);
    EXPECT_NEAR(background.conformalDerivative(0.3), 0.09, 2.0e-14);
}

TEST(SimulationTime, ConstantEquationOfState) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.darkEnergyW = -1.0 / 3.0;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_NEAR(background.conformalInterval(0.02, 0.7), std::log(35.0), 2.0e-11);
    EXPECT_NEAR(background.conformalDerivative(0.1), 0.1, 2.0e-14);
}

TEST(SimulationTime, MatterDriftFactorMatchesCanonicalMomentumEvolution) {
    constexpr double kInitial = 0.01;
    constexpr double kFinal = 1.0;
    PROFILE();

    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    double const expected = 2.0 * (1.0 / std::sqrt(kInitial) - 1.0 / std::sqrt(kFinal));
    EXPECT_NEAR(background.driftFactor(kInitial, kFinal), expected, 2.0e-10);
}

TEST(SimulationTime, VacuumDriftFactorMatchesCanonicalMomentumEvolution) {
    constexpr double kInitial = 0.2;
    constexpr double kFinal = 0.8;
    constexpr double kExpected = 0.5 * (1.0 / (kInitial * kInitial) - 1.0 / (kFinal * kFinal));
    PROFILE();

    cosmictiger::SimulationBackground const background(emptyBackground());
    EXPECT_NEAR(background.driftFactor(kInitial, kFinal), kExpected, 2.0e-10);
}

TEST(SimulationTime, CurvatureLikeEquationOfStateDriftFactorMatchesAnalyticIntegral) {
    constexpr double kInitial = 0.02;
    constexpr double kFinal = 0.7;
    PROFILE();

    auto parameters = emptyBackground();
    parameters.darkEnergyW = -1.0 / 3.0;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_NEAR(background.driftFactor(kInitial, kFinal), 1.0 / kInitial - 1.0 / kFinal, 5.0e-10);
}

TEST(SimulationTime, DriftFactorsAreAdditiveSignedAndZeroAtEqualEpochs) {
    constexpr double kFirst = 0.01;
    constexpr double kMiddle = 0.2;
    constexpr double kLast = 1.0;
    PROFILE();

    cosmictiger::CosmologyParameters const parameters;
    cosmictiger::SimulationBackground const background(parameters);
    double const whole = background.driftFactor(kFirst, kLast);
    EXPECT_GT(whole, 0.0);
    EXPECT_NEAR(whole, background.driftFactor(kFirst, kMiddle) + background.driftFactor(kMiddle, kLast), 1.0e-8);
    EXPECT_DOUBLE_EQ(background.driftFactor(kLast, kFirst), -whole);
    EXPECT_DOUBLE_EQ(background.driftFactor(kFirst, kFirst), 0.0);
    EXPECT_DOUBLE_EQ(background.driftFactor(kLast, kLast), 0.0);
}

TEST(SimulationTime, TimeVaryingEquationOfState) {
    constexpr double kA = 0.3;
    PROFILE();

    auto parameters = emptyBackground();
    parameters.darkEnergyW = -0.9;
    parameters.darkEnergyWa = 0.2;
    cosmictiger::SimulationBackground const background(parameters);
    double const expectedHubbleSquared =
        std::pow(kA, -3.0 * (1.0 + parameters.darkEnergyW + parameters.darkEnergyWa)) * std::exp(-3.0 * parameters.darkEnergyWa * (1.0 - kA));
    EXPECT_NEAR(background.conformalDerivative(kA), kA * kA * std::sqrt(expectedHubbleSquared), 2.0e-14);
}

TEST(SimulationTime, ThermalNeutrinosAndRadiationNormalizeToday) {
    constexpr double kEarly = 1.0e-7;
    PROFILE();

    cosmictiger::CosmologyParameters const parameters;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_NEAR(background.conformalDerivative(1.0), 1.0, 2.0e-11);
    auto masslessParameters = parameters;
    masslessParameters.masslessNeutrinos += masslessParameters.massiveNeutrinos;
    masslessParameters.massiveNeutrinos = 0;
    masslessParameters.omegaNeutrinoH2 = 0.0;
    cosmictiger::SimulationBackground const masslessBackground(masslessParameters);
    // Every species becomes relativistic at sufficiently small a.
    EXPECT_NEAR(background.conformalDerivative(kEarly), masslessBackground.conformalDerivative(kEarly), 2.0e-11);
    // At late times the massive species contributes to the matter density.
    EXPECT_GT(background.conformalDerivative(0.1), masslessBackground.conformalDerivative(0.1));
    double const whole = background.conformalInterval(0.01, 1.0);
    EXPECT_GT(whole, 0.0);
    EXPECT_NEAR(whole, background.conformalInterval(0.01, 0.2) + background.conformalInterval(0.2, 1.0), 1.0e-9);
}

TEST(SimulationTime, RejectsReversedEpochsAndImpossibleThermalDensity) {
    PROFILE();
    cosmictiger::SimulationBackground const background(emptyBackground());
    EXPECT_THROW(background.conformalInterval(1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(background.conformalInterval(1.0, 0.01), std::invalid_argument);
    cosmictiger::CosmologyParameters parameters;
    parameters.omegaNeutrinoH2 = 1.0e-10;
    EXPECT_THROW(cosmictiger::SimulationBackground{parameters}, std::invalid_argument);
}

TEST(SimulationTime, EffectiveNeutrinoDensityDeterminesSharedHeating) {
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.masslessNeutrinos = 0;
    parameters.massiveNeutrinos = 1;
    parameters.effectiveNeutrinos = 4.0;
    cosmictiger::SimulationBackground const oneHeatedSpecies(parameters);
    // With the same present density and total N_eff, four colder degenerate
    // species infer a different physical mass but the same m/Tnu and H(a).
    parameters.massiveNeutrinos = 4;
    cosmictiger::SimulationBackground const fourStandardSpecies(parameters);
    for (double const scaleFactor : {1.0e-6, 0.002, 0.03, 0.4, 1.0}) {
        EXPECT_NEAR(oneHeatedSpecies.conformalDerivative(scaleFactor), fourStandardSpecies.conformalDerivative(scaleFactor), 1.0e-13);
    }
}

TEST(SimulationTime, MatterAdvancesBothClocksAndDriftTogether) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(0.25);
    auto const conformal = background.advance(initial, 1.0);
    EXPECT_NEAR(conformal.state.scaleFactor, 1.0, 2.0e-11);
    EXPECT_NEAR(conformal.state.conformalDerivative, 1.0, 2.0e-11);
    EXPECT_NEAR(conformal.state.physicalDerivative, 1.0, 2.0e-11);
    EXPECT_NEAR(conformal.driftFactor, 2.0, 3.0e-11);
    EXPECT_DOUBLE_EQ(conformal.conformalInterval, 1.0);
    EXPECT_NEAR(conformal.physicalInterval, 7.0 / 12.0, 2.0e-11);
    auto const physical = background.advancePhysical(initial, 7.0 / 12.0);
    EXPECT_NEAR(physical.state.scaleFactor, 1.0, 2.0e-11);
    EXPECT_NEAR(physical.conformalInterval, 1.0, 2.0e-11);
    EXPECT_NEAR(physical.driftFactor, 2.0, 3.0e-11);
    EXPECT_DOUBLE_EQ(physical.physicalInterval, 7.0 / 12.0);
}

TEST(SimulationTime, ProperTimeTargetDistinguishesPresentAndFutureEpochs) {
    PROFILE();
    cosmictiger::SimulationBackground const background(emptyBackground());
    auto const present = background.state(1.0);
    auto const future = background.advancePhysical(present, 1.0);
    EXPECT_NEAR(future.state.scaleFactor, std::exp(1.0), 5.0e-11);
    EXPECT_NEAR(future.conformalInterval, 1.0 - std::exp(-1.0), 2.0e-11);
    EXPECT_NEAR(future.driftFactor, 0.5 * (1.0 - std::exp(-2.0)), 2.0e-11);
    auto const past = background.advancePhysical(present, -1.0);
    EXPECT_NEAR(past.state.scaleFactor, std::exp(-1.0), 2.0e-11);
    EXPECT_NEAR(past.conformalInterval, 1.0 - std::exp(1.0), 3.0e-11);
    EXPECT_NEAR(background.physicalInterval(std::exp(-1.0), 1.0), 1.0, 2.0e-11);
    EXPECT_NEAR(background.physicalInterval(1.0, std::exp(-1.0)), -1.0, 2.0e-11);
    EXPECT_DOUBLE_EQ(background.physicalInterval(1.0, 1.0), 0.0);
}

TEST(SimulationTime, ContractingMatterKeepsBothElapsedTimesPositive) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(1.0, cosmictiger::ExpansionBranch::contracting);
    auto const result = background.advancePhysical(initial, 0.5);
    double const expectedA = std::pow(0.25, 2.0 / 3.0);
    EXPECT_NEAR(result.state.scaleFactor, expectedA, 2.0e-11);
    EXPECT_NEAR(result.state.conformalDerivative, -std::sqrt(expectedA), 2.0e-11);
    EXPECT_NEAR(result.conformalInterval, 2.0 * (1.0 - std::sqrt(expectedA)), 2.0e-11);
    EXPECT_NEAR(result.driftFactor, 2.0 * (1.0 / std::sqrt(expectedA) - 1.0), 4.0e-11);
    EXPECT_GT(result.physicalInterval, 0.0);
}

TEST(SimulationTime, ClosedMatterEvolvesThroughTurnaroundIntoContraction) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 2.0;
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(1.0);
    auto const turnaround = background.advance(initial, 0.5 * std::numbers::pi);
    EXPECT_NEAR(turnaround.state.scaleFactor, 2.0, 2.0e-11);
    EXPECT_NEAR(turnaround.state.conformalDerivative, 0.0, 2.0e-11);
    auto const result = background.advance(initial, std::numbers::pi);
    EXPECT_NEAR(result.state.scaleFactor, 1.0, 3.0e-11);
    EXPECT_NEAR(result.state.conformalDerivative, -1.0, 3.0e-11);
    EXPECT_NEAR(result.physicalInterval, std::numbers::pi + 2.0, 4.0e-11);
    EXPECT_NEAR(result.driftFactor, 2.0, 4.0e-11);
    auto const physical = background.advancePhysical(initial, std::numbers::pi + 2.0);
    EXPECT_NEAR(physical.state.scaleFactor, 1.0, 4.0e-11);
    EXPECT_NEAR(physical.state.conformalDerivative, -1.0, 4.0e-11);
    EXPECT_NEAR(physical.conformalInterval, std::numbers::pi, 4.0e-11);
    auto const resumed = background.advance(turnaround.state, 0.5 * std::numbers::pi);
    EXPECT_NEAR(resumed.state.scaleFactor, result.state.scaleFactor, 4.0e-11);
    EXPECT_NEAR(resumed.state.conformalDerivative, result.state.conformalDerivative, 4.0e-11);
}

TEST(SimulationTime, GeneralBackgroundRoundTripMatchesEndpointQuadrature) {
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.darkEnergyW = -0.9;
    parameters.darkEnergyWa = 0.15;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(0.1);
    auto const duration = background.physicalInterval(0.1, 0.8);
    auto const result = background.advancePhysical(initial, duration);
    EXPECT_NEAR(result.state.scaleFactor, 0.8, 2.0e-10);
    EXPECT_NEAR(result.state.conformalDerivative, background.conformalDerivative(0.8), 2.0e-10);
    EXPECT_NEAR(result.conformalInterval, background.conformalInterval(0.1, 0.8), 2.0e-10);
    EXPECT_NEAR(result.driftFactor, background.driftFactor(0.1, 0.8), 5.0e-10);
    auto const restored = background.advancePhysical(result.state, -duration);
    EXPECT_NEAR(restored.state.scaleFactor, initial.scaleFactor, 2.0e-10);
    EXPECT_NEAR(restored.state.conformalDerivative, initial.conformalDerivative, 2.0e-10);
}

TEST(SimulationTime, ZeroDurationRetainsSignedStateAndSingularAdvanceFails) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(1.0, cosmictiger::ExpansionBranch::contracting);
    auto const result = background.advance(initial, 0.0);
    EXPECT_DOUBLE_EQ(result.state.scaleFactor, initial.scaleFactor);
    EXPECT_DOUBLE_EQ(result.state.conformalDerivative, initial.conformalDerivative);
    EXPECT_DOUBLE_EQ(result.driftFactor, 0.0);
    EXPECT_DOUBLE_EQ(result.physicalInterval, 0.0);
    EXPECT_DOUBLE_EQ(result.conformalInterval, 0.0);
    EXPECT_ANY_THROW(background.advancePhysical(initial, 0.7));
}

TEST(SimulationTime, MatterAndCurvaturePresentAgesMatchAnalyticSolutions) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const matter(parameters);
    EXPECT_NEAR(matter.presentAge(), 2.0 / 3.0, 2.0e-12);
    parameters.omegaCdmH2 = 0.0;
    parameters.omegaCurvature = 1.0;
    cosmictiger::SimulationBackground const curvature(parameters);
    EXPECT_NEAR(curvature.presentAge(), 1.0, 2.0e-12);
}

TEST(SimulationTime, FiniteAgeDarkComponentRegularizesItsEarlyEndpoint) {
    PROFILE();
    auto parameters = emptyBackground();
    parameters.darkEnergyW = -0.9;
    cosmictiger::SimulationBackground const background(parameters);
    EXPECT_NEAR(background.presentAge(), 2.0 / (3.0 * (1.0 + parameters.darkEnergyW)), 2.0e-11);
}

TEST(SimulationTime, PresentAgeRejectsInfiniteAgeAndDisconnectedBigBang) {
    PROFILE();
    auto parameters = emptyBackground();
    cosmictiger::SimulationBackground const vacuum(parameters);
    EXPECT_THROW(vacuum.presentAge(), std::domain_error);
    parameters.darkEnergyW = -1.1;
    cosmictiger::SimulationBackground const phantom(parameters);
    EXPECT_THROW(phantom.presentAge(), std::domain_error);
    parameters.darkEnergyW = -1.0;
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const bouncing(parameters);
    EXPECT_THROW(bouncing.presentAge(), std::domain_error);
}

TEST(SimulationTime, PresentAgeIncludesThermalNeutrinoHistory) {
    PROFILE();
    cosmictiger::SimulationBackground const background(cosmictiger::CosmologyParameters{});
    double const age = background.presentAge();
    EXPECT_GT(age, 0.9);
    EXPECT_LT(age, 1.0);
    auto const initial = background.state(0.01);
    double const elapsed = background.physicalInterval(0.01, 1.0);
    EXPECT_GT(age, elapsed);
    EXPECT_LT(age - elapsed, 0.002);
    auto const present = background.advancePhysical(initial, elapsed);
    EXPECT_NEAR(present.state.scaleFactor, 1.0, 2.0e-10);
}

TEST(SimulationTime, DefaultThermalBackgroundEvolvesFromInitialEpochToTenTimesPresentAge) {
    PROFILE();
    cosmictiger::SimulationBackground const background(cosmictiger::CosmologyParameters{});
    auto const initial = background.state(0.01);
    double const toPresent = background.physicalInterval(0.01, 1.0);
    auto const present = background.advancePhysical(initial, toPresent);
    EXPECT_NEAR(present.state.scaleFactor, 1.0, 2.0e-10);
    double const futureDuration = 9.0 * background.presentAge();
    auto const future = background.advancePhysical(initial, toPresent + futureDuration);
    EXPECT_GT(future.state.scaleFactor, 100.0);
    EXPECT_GT(future.state.conformalDerivative, 0.0);
    EXPECT_NEAR(background.physicalInterval(1.0, future.state.scaleFactor), futureDuration, 2.0e-9);
    EXPECT_NEAR(background.conformalInterval(0.01, future.state.scaleFactor), future.conformalInterval, 2.0e-9);
    EXPECT_DOUBLE_EQ(future.physicalInterval, toPresent + futureDuration);
}

TEST(SimulationTime, InterpolatedMatterMatchesAnalyticClocksAndShortDrifts) {
    constexpr size_t kSamples = 257;
    constexpr double kInitialA = 0.01;
    constexpr double kInitialRoot = 0.1;
    constexpr double kDuration = 0.666;
    constexpr double kTolerance = 2.0e-11;
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const background(parameters);
    cosmictiger::SimulationBackgroundTable const table(background, background.state(kInitialA), kDuration);
    EXPECT_GT(table.segmentCount(), 1);
    EXPECT_DOUBLE_EQ(table.end().physicalInterval, kDuration);
    for (size_t index = 0; index <= kSamples; index++) {
        double const tau = table.end().conformalInterval * (double(index) / kSamples);
        double const root = kInitialRoot + 0.5 * tau;
        auto const sample = table.at(tau);
        EXPECT_NEAR(sample.state.scaleFactor, root * root, kTolerance);
        EXPECT_NEAR(sample.state.conformalDerivative, root, kTolerance);
        EXPECT_NEAR(sample.physicalInterval, (2.0 / 3.0) * (root * root * root - kInitialRoot * kInitialRoot * kInitialRoot), kTolerance);
        EXPECT_NEAR(sample.driftFactor, 2.0 * (1.0 / kInitialRoot - 1.0 / root), kTolerance);
        if (index > 0 && index < kSamples) {
            double const next = std::nextafter(tau, table.end().conformalInterval);
            auto const tiny = table.interval(tau, next);
            EXPECT_GT(tiny.driftFactor, 0.0);
            EXPECT_NEAR(tiny.driftFactor / (next - tau), 1.0 / (root * root), kTolerance / (root * root));
        }
    }
    double const middle = 0.47 * table.end().conformalInterval;
    EXPECT_NEAR(
        table.interval(0.0, middle).driftFactor + table.interval(middle, table.end().conformalInterval).driftFactor, table.end().driftFactor, kTolerance);
}

TEST(SimulationTime, InterpolatedClosedMatterCrossesTurnaroundAndRestartsInContraction) {
    constexpr size_t kSamples = 129;
    constexpr double kTolerance = 1.0e-10;
    PROFILE();
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 2.0;
    parameters.omegaCurvature = -1.0;
    cosmictiger::SimulationBackground const background(parameters);
    cosmictiger::SimulationBackgroundTable const table(background, background.state(1.0), std::numbers::pi + 2.0);
    for (size_t index = 0; index <= kSamples; index++) {
        double const tau = table.end().conformalInterval * (double(index) / kSamples);
        auto const sample = table.at(tau);
        EXPECT_NEAR(sample.state.scaleFactor, 1.0 + std::sin(tau), kTolerance);
        EXPECT_NEAR(sample.state.conformalDerivative, std::cos(tau), kTolerance);
        EXPECT_NEAR(sample.physicalInterval, tau + 1.0 - std::cos(tau), kTolerance);
    }
    double const restartTau = 2.0;
    auto const restart = table.at(restartTau);
    cosmictiger::SimulationBackgroundTable const resumed(background, restart.state, table.end().physicalInterval - restart.physicalInterval);
    EXPECT_NEAR(resumed.end().state.scaleFactor, 1.0, kTolerance);
    EXPECT_NEAR(resumed.end().state.conformalDerivative, -1.0, kTolerance);
    EXPECT_NEAR(resumed.end().conformalInterval, table.end().conformalInterval - restartTau, kTolerance);
    EXPECT_NEAR(resumed.end().driftFactor, table.interval(restartTau, table.end().conformalInterval).driftFactor, kTolerance);
}

TEST(SimulationTime, InterpolatedThermalBackgroundMatchesIndependentIntegrations) {
    constexpr double kTolerance = 2.0e-10;
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.darkEnergyW = -0.9;
    parameters.darkEnergyWa = 0.15;
    cosmictiger::SimulationBackground const background(parameters);
    auto const initial = background.state(0.01);
    cosmictiger::SimulationBackgroundTable const table(background, initial, 2.0);
    for (double const fraction : {0.001, 0.013, 0.17, 0.49, 0.83, 0.999}) {
        double const tau = fraction * table.end().conformalInterval;
        auto const interpolated = table.at(tau);
        auto const integrated = background.advance(initial, tau);
        EXPECT_NEAR(interpolated.state.scaleFactor, integrated.state.scaleFactor, kTolerance);
        EXPECT_NEAR(interpolated.state.conformalDerivative, integrated.state.conformalDerivative, kTolerance);
        EXPECT_NEAR(interpolated.physicalInterval, integrated.physicalInterval, kTolerance);
        EXPECT_NEAR(interpolated.driftFactor, integrated.driftFactor, kTolerance);
    }
}

TEST(SimulationTime, BackgroundTableResolvesLateVacuumAndNearCollapse) {
    constexpr double kVacuumDuration = 9.0;
    constexpr double kCollapseA = 1.0e-6;
    constexpr double kStateTolerance = 2.0e-10;
    PROFILE();
    cosmictiger::SimulationBackground const vacuum(emptyBackground());
    cosmictiger::SimulationBackgroundTable const future(vacuum, vacuum.state(1.0), kVacuumDuration);
    for (double const time : {0.01, 1.0, 5.0, 8.9}) {
        double const tau = -std::expm1(-time);
        auto const result = future.at(tau);
        EXPECT_NEAR(result.state.scaleFactor / std::exp(time), 1.0, kStateTolerance);
        EXPECT_NEAR(result.physicalInterval, time, kStateTolerance);
        EXPECT_NEAR(result.driftFactor, -0.5 * std::expm1(-2.0 * time), kStateTolerance);
    }
    auto parameters = emptyBackground();
    parameters.omegaCdmH2 = 1.0;
    cosmictiger::SimulationBackground const matter(parameters);
    double const duration = (2.0 / 3.0) * (1.0 - kCollapseA * std::sqrt(kCollapseA));
    cosmictiger::SimulationBackgroundTable const collapse(matter, matter.state(1.0, cosmictiger::ExpansionBranch::contracting), duration);
    EXPECT_NEAR(collapse.end().state.scaleFactor, kCollapseA, kStateTolerance);
    EXPECT_LT(collapse.end().state.conformalDerivative, 0.0);
    for (double const fraction : {0.01, 0.5, 0.99, 0.99999}) {
        double const tau = fraction * collapse.end().conformalInterval;
        double const root = 1.0 - 0.5 * tau;
        auto const result = collapse.at(tau);
        EXPECT_NEAR(result.state.scaleFactor, root * root, kStateTolerance);
        EXPECT_NEAR(result.driftFactor, 2.0 * (1.0 / root - 1.0), kStateTolerance / root);
    }
}

TEST(SimulationTime, BackgroundTableValidatesBoundsAndZeroDuration) {
    PROFILE();
    cosmictiger::SimulationBackground const background(emptyBackground());
    auto const initial = background.state(1.0);
    cosmictiger::SimulationBackgroundTable const zero(background, initial, 0.0);
    EXPECT_EQ(zero.segmentCount(), 0);
    EXPECT_DOUBLE_EQ(zero.at(0.0).state.scaleFactor, 1.0);
    EXPECT_DOUBLE_EQ(zero.interval(0.0, 0.0).driftFactor, 0.0);
    EXPECT_THROW(zero.at(0.1), std::out_of_range);
    EXPECT_THROW(cosmictiger::SimulationBackgroundTable(background, initial, -1.0), std::invalid_argument);
    cosmictiger::SimulationBackgroundTable const table(background, initial, 1.0);
    EXPECT_THROW(table.at(-0.1), std::out_of_range);
    EXPECT_THROW(table.at(table.end().conformalInterval + 0.1), std::out_of_range);
    EXPECT_THROW(table.interval(0.2, 0.1), std::out_of_range);
    EXPECT_DOUBLE_EQ(table.interval(0.2, 0.2).driftFactor, 0.0);
}

TEST(SimulationTime, DISABLED_BackgroundLookupBenchmark) {
    constexpr size_t kIntegratedQueries = 16;
    constexpr size_t kTableQueries = 100000;
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();
    cosmictiger::SimulationBackground const background(cosmictiger::CosmologyParameters{});
    auto const initial = background.state(0.01);
    auto const buildStart = hpx::chrono::high_resolution_clock::now();
    cosmictiger::SimulationBackgroundTable const table(background, initial, 1.0);
    auto const buildEnd = hpx::chrono::high_resolution_clock::now();
    double checksum = 0.0;
    auto const integratedStart = hpx::chrono::high_resolution_clock::now();
    for (size_t index = 0; index < kIntegratedQueries; index++) {
        double const begin = table.end().conformalInterval * (double(index) / kIntegratedQueries);
        auto const state = table.at(begin);
        checksum += background.advancePhysical(state.state, 1.0 - state.physicalInterval).conformalInterval;
        checksum += background.advance(state.state, table.end().conformalInterval / kTableQueries).driftFactor;
    }
    auto const integratedEnd = hpx::chrono::high_resolution_clock::now();
    for (size_t index = 0; index < kTableQueries; index++) {
        double const begin = table.end().conformalInterval * (double(index) / kTableQueries);
        double const end = table.end().conformalInterval * (double(index + 1) / kTableQueries);
        checksum += table.interval(begin, end).driftFactor;
    }
    auto const queryEnd = hpx::chrono::high_resolution_clock::now();
    std::cout << "Background table: segments=" << table.segmentCount() << " build_seconds=" << double(buildEnd - buildStart) * kSecondsPerNanosecond
              << " old_seconds_per_step=" << double(integratedEnd - integratedStart) * kSecondsPerNanosecond / kIntegratedQueries
              << " table_seconds_per_step=" << double(queryEnd - integratedEnd) * kSecondsPerNanosecond / kTableQueries << " checksum=" << checksum << '\n';
    EXPECT_GT(checksum, 0.0);
}

}    // namespace
