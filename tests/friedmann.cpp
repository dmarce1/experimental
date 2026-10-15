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
#include <cosmictiger/friedmann.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <numbers>
#include <stddef.h>
#include <stdexcept>

namespace {

using cosmictiger::BackgroundCosmologyParameters;
using cosmictiger::CosmologicalTime;
using cosmictiger::ExpansionBranch;
using cosmictiger::FriedmannCosmology;
using cosmictiger::FriedmannOptions;
using cosmictiger::FriedmannState;

BackgroundCosmologyParameters parameters(double radiation, double matter, double darkEnergy, double curvature, double hubble = 1.0) {
    PROFILE();
    BackgroundCosmologyParameters result;
    result.hubbleConstant = hubble;
    result.omegaRadiation = radiation;
    result.omegaMatter = matter;
    result.omegaDarkEnergy = darkEnergy;
    result.omegaCurvature = curvature;
    return result;
}

void expectState(FriedmannState const& actual, double scaleFactor, double physicalDerivative, double conformalDerivative, double tolerance = 1e-10) {
    PROFILE();
    EXPECT_NEAR(actual.scaleFactor, scaleFactor, tolerance * (1.0 + std::abs(scaleFactor)));
    EXPECT_NEAR(actual.physicalDerivative, physicalDerivative, tolerance * (1.0 + std::abs(physicalDerivative)));
    EXPECT_NEAR(actual.conformalDerivative, conformalDerivative, tolerance * (1.0 + std::abs(conformalDerivative)));
}

void expectSameState(FriedmannState const& actual, FriedmannState const& expected, double tolerance = 1e-10) {
    PROFILE();
    expectState(actual, expected.scaleFactor, expected.physicalDerivative, expected.conformalDerivative, tolerance);
}

TEST(Friedmann, DefaultsAndPresentEpochNormalization) {
    PROFILE();
    FriedmannCosmology const cosmology;
    EXPECT_DOUBLE_EQ(cosmology.parameters().hubbleConstant, 1.0);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaRadiation, 0.0);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaMatter, 0.3);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaDarkEnergy, 0.7);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaCurvature, 0.0);
    EXPECT_DOUBLE_EQ(cosmology.options().relativeTolerance, 1e-12);
    EXPECT_DOUBLE_EQ(cosmology.options().absoluteTolerance, 1e-14);
    EXPECT_EQ(cosmology.options().maxSteps, size_t(100000));
    expectState(cosmology.state(1.0), 1.0, 1.0, 1.0, 1e-14);
    expectState(cosmology.state(1.0, ExpansionBranch::contracting), 1.0, -1.0, -1.0, 1e-14);
}

TEST(Friedmann, PureRadiationMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(1.0, 0.0, 0.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = std::sqrt(kInitialScaleFactor * kInitialScaleFactor + 2.0 * sign * kHubble * kElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch), physicalScaleFactor,
            sign * kHubble / physicalScaleFactor, sign * kHubble);
        double const conformalScaleFactor = kInitialScaleFactor + sign * kHubble * kElapsed;
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch), conformalScaleFactor,
            sign * kHubble / conformalScaleFactor, sign * kHubble);
    }
}

TEST(Friedmann, PureMatterMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, 1.0, 0.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = std::pow(std::pow(kInitialScaleFactor, 1.5) + 1.5 * sign * kHubble * kElapsed, 2.0 / 3.0);
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch), physicalScaleFactor,
            sign * kHubble / std::sqrt(physicalScaleFactor), sign * kHubble * std::sqrt(physicalScaleFactor));
        double const conformalRoot = std::sqrt(kInitialScaleFactor) + 0.5 * sign * kHubble * kElapsed;
        double const conformalScaleFactor = conformalRoot * conformalRoot;
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch), conformalScaleFactor, sign * kHubble / conformalRoot,
            sign * kHubble * conformalRoot);
    }
}

TEST(Friedmann, PureCurvatureMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, 0.0, 0.0, 1.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = kInitialScaleFactor + sign * kHubble * kElapsed;
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch), physicalScaleFactor, sign * kHubble,
            sign * kHubble * physicalScaleFactor);
        double const conformalScaleFactor = kInitialScaleFactor * std::exp(sign * kHubble * kElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch), conformalScaleFactor, sign * kHubble,
            sign * kHubble * conformalScaleFactor);
    }
}

TEST(Friedmann, PureDarkEnergyMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, 0.0, 1.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = kInitialScaleFactor * std::exp(sign * kHubble * kElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch), physicalScaleFactor,
            sign * kHubble * physicalScaleFactor, sign * kHubble * physicalScaleFactor * physicalScaleFactor);
        double const conformalScaleFactor = kInitialScaleFactor / (1.0 - sign * kHubble * kInitialScaleFactor * kElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch), conformalScaleFactor,
            sign * kHubble * conformalScaleFactor, sign * kHubble * conformalScaleFactor * conformalScaleFactor);
    }
}

TEST(Friedmann, MatterAndRadiationMatchExactConformalSolutionInBothDirections) {
    constexpr double kRadiation = 0.4;
    constexpr double kMatter = 0.6;
    constexpr double kHubble = 2.1;
    constexpr double kInitialScaleFactor = 0.3;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(kRadiation, kMatter, 0.0, 0.0, kHubble));
    for (double const elapsed : {-0.1, 0.4}) {
        double const scaledElapsed = kHubble * elapsed;
        double const initialVelocity = std::sqrt(kRadiation + kMatter * kInitialScaleFactor);
        double const expectedScaleFactor = kInitialScaleFactor + initialVelocity * scaledElapsed + 0.25 * kMatter * scaledElapsed * scaledElapsed;
        double const expectedConformalDerivative = kHubble * (initialVelocity + 0.5 * kMatter * scaledElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, elapsed, CosmologicalTime::conformal), expectedScaleFactor,
            expectedConformalDerivative / expectedScaleFactor, expectedConformalDerivative);
    }
}

TEST(Friedmann, FlatMatterAndDarkEnergyMatchExactPhysicalSolution) {
    constexpr double kMatter = 0.3;
    constexpr double kDarkEnergy = 0.7;
    constexpr double kHubble = 0.8;
    constexpr double kInitialScaleFactor = 0.2;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, kMatter, kDarkEnergy, 0.0, kHubble));
    double const initialPhase = std::asinh(std::sqrt(kDarkEnergy / kMatter) * std::pow(kInitialScaleFactor, 1.5));
    for (double const elapsed : {-0.05, 1.4}) {
        double const phase = initialPhase + 1.5 * kHubble * std::sqrt(kDarkEnergy) * elapsed;
        double const expectedScaleFactor = std::pow(std::sqrt(kMatter / kDarkEnergy) * std::sinh(phase), 2.0 / 3.0);
        double const expectedPhysicalDerivative = kHubble * std::sqrt(kMatter / expectedScaleFactor + kDarkEnergy * expectedScaleFactor * expectedScaleFactor);
        expectState(
            cosmology.evolve(kInitialScaleFactor, elapsed), expectedScaleFactor, expectedPhysicalDerivative, expectedScaleFactor * expectedPhysicalDerivative);
    }
}

TEST(Friedmann, ClosedMatterPassesThroughTurnaroundAndIntoContractionWithBothClocks) {
    constexpr double kHubble = 1.3;
    constexpr double kPi = std::numbers::pi;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, 2.0, 0.0, -1.0, kHubble));
    // The cycloid is a = 1 - cos(theta), H0 t = theta - sin(theta), H0 eta = theta.
    // Starting theta = pi/2 gives a = 1 and an expanding derivative H0.
    auto const physicalTurnaround = cosmology.evolve(1.0, (0.5 * kPi + 1.0) / kHubble);
    auto const conformalTurnaround = cosmology.evolve(1.0, 0.5 * kPi / kHubble, CosmologicalTime::conformal);
    expectState(physicalTurnaround, 2.0, 0.0, 0.0);
    expectState(conformalTurnaround, 2.0, 0.0, 0.0);
    expectState(cosmology.evolve(1.0, (kPi + 2.0) / kHubble), 1.0, -kHubble, -kHubble);
    expectState(cosmology.evolve(1.0, kPi / kHubble, CosmologicalTime::conformal), 1.0, -kHubble, -kHubble);
    expectState(cosmology.advance(physicalTurnaround, (0.5 * kPi + 1.0) / kHubble), 1.0, -kHubble, -kHubble);
    expectState(cosmology.advance(conformalTurnaround, 0.5 * kPi / kHubble, CosmologicalTime::conformal), 1.0, -kHubble, -kHubble);
    expectState(cosmology.evolve(2.0, (0.5 * kPi + 1.0) / kHubble), 1.0, -kHubble, -kHubble);
}

TEST(Friedmann, NegativeDarkEnergyAllowsRecollapse) {
    constexpr double kHubble = 0.7;
    PROFILE();

    FriedmannCosmology const cosmology(parameters(0.0, 2.0, -1.0, 0.0, kHubble));
    // a^(3/2) = sqrt(2) sin(theta), theta = theta0 + (3/2) H0 t.
    // theta0 = pi/4 and theta = 3pi/4 have a = 1 on opposite branches.
    expectState(cosmology.evolve(1.0, std::numbers::pi / (3.0 * kHubble)), 1.0, -kHubble, -kHubble);
}

TEST(Friedmann, AllFourDensitiesPreserveTheConstraintAndDerivativeClockRelation) {
    PROFILE();
    auto const modelParameters = parameters(0.08, 0.41, 0.57, -0.06, 2.4);
    FriedmannCosmology const cosmology(modelParameters);
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
            auto const initial = cosmology.state(0.8, branch);
            auto const final = cosmology.advance(initial, 0.07, clock);
            double const a = final.scaleFactor;
            double const expectedSquaredVelocity = modelParameters.hubbleConstant * modelParameters.hubbleConstant *
                (modelParameters.omegaRadiation / (a * a) + modelParameters.omegaMatter / a + modelParameters.omegaCurvature +
                    modelParameters.omegaDarkEnergy * a * a);
            EXPECT_NEAR(final.physicalDerivative * final.physicalDerivative, expectedSquaredVelocity, 2e-10 * expectedSquaredVelocity);
            EXPECT_NEAR(final.conformalDerivative, a * final.physicalDerivative, 1e-14 * (1.0 + std::abs(final.conformalDerivative)));
            EXPECT_EQ(final.physicalDerivative > 0.0, branch == ExpansionBranch::expanding);
            expectSameState(cosmology.state(a, branch), final);
        }
    }
}

TEST(Friedmann, HubbleConstantOnlyRescalesTheTimeAndDerivatives) {
    PROFILE();
    auto const firstParameters = parameters(0.02, 0.35, 0.7, -0.07, 0.6);
    auto secondParameters = firstParameters;
    secondParameters.hubbleConstant *= 5.0;
    FriedmannCosmology const first(firstParameters);
    FriedmannCosmology const second(secondParameters);
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        auto const firstState = first.evolve(0.4, 0.3, clock);
        auto const secondState = second.evolve(0.4, 0.3 / 5.0, clock);
        expectState(secondState, firstState.scaleFactor, 5.0 * firstState.physicalDerivative, 5.0 * firstState.conformalDerivative);
    }
}

TEST(Friedmann, SplitStepsAndTimeReversalRetainTheFullState) {
    PROFILE();
    FriedmannCosmology const cosmology(parameters(0.05, 0.33, 0.64, -0.02, 1.2));
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
            auto const initial = cosmology.state(0.9, branch);
            auto const first = cosmology.advance(initial, 0.04, clock);
            auto const split = cosmology.advance(first, 0.06, clock);
            auto const whole = cosmology.advance(initial, 0.1, clock);
            expectSameState(split, whole);
            expectSameState(cosmology.advance(whole, -0.1, clock), initial);
            auto const unchanged = cosmology.advance(initial, 0.0, clock);
            EXPECT_DOUBLE_EQ(unchanged.scaleFactor, initial.scaleFactor);
            EXPECT_DOUBLE_EQ(unchanged.physicalDerivative, initial.physicalDerivative);
            EXPECT_DOUBLE_EQ(unchanged.conformalDerivative, initial.conformalDerivative);
        }
    }
}

TEST(Friedmann, TightAbsoluteToleranceResolvesSmallRadiationScaleFactors) {
    constexpr double kInitialScaleFactor = 1e-7;
    constexpr double kElapsed = 0.25 * kInitialScaleFactor * kInitialScaleFactor;
    PROFILE();

    FriedmannOptions options;
    options.absoluteTolerance = 1e-24;
    FriedmannCosmology const cosmology(parameters(1.0, 0.0, 0.0, 0.0), options);
    double const expectedScaleFactor = std::sqrt(1.5) * kInitialScaleFactor;
    auto const result = cosmology.evolve(kInitialScaleFactor, kElapsed);
    EXPECT_NEAR(result.scaleFactor / expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.physicalDerivative * expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.conformalDerivative, 1.0, 1e-10);
}

TEST(Friedmann, DefaultTolerancesRetainRelativeAccuracyAtSmallScaleFactors) {
    constexpr double kInitialScaleFactor = 1e-12;
    PROFILE();

    FriedmannCosmology const radiation(parameters(1.0, 0.0, 0.0, 0.0));
    double const expectedScaleFactor = std::sqrt(1.5) * kInitialScaleFactor;
    auto const result = radiation.evolve(kInitialScaleFactor, 0.25 * kInitialScaleFactor * kInitialScaleFactor);
    EXPECT_NEAR(result.scaleFactor / expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.physicalDerivative * expectedScaleFactor, 1.0, 1e-10);
}

TEST(Friedmann, ReportsUnrepresentableDerivativesWithoutReturningAFalseStaticState) {
    PROFILE();
    FriedmannCosmology const matter(parameters(0.0, 1.0, 0.0, 0.0, 1e-300));
    EXPECT_THROW(matter.state(1e-200), std::overflow_error);
}

TEST(Friedmann, RejectsInvalidParametersAndOptions) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    auto modelParameters = parameters(0.0, 0.3, 0.7, 0.0);
    for (double const hubble : {0.0, -1.0}) {
        modelParameters.hubbleConstant = hubble;
        EXPECT_THROW((FriedmannCosmology(modelParameters)), std::runtime_error);
    }
    modelParameters = parameters(-0.1, 0.3, 0.8, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::runtime_error);
    modelParameters = parameters(0.1, -0.3, 1.2, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::runtime_error);
    modelParameters = parameters(0.1, 0.3, 0.7, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::runtime_error);

    for (double const invalidTolerance : {-1.0, 1.0}) {
        FriedmannOptions options;
        options.relativeTolerance = invalidTolerance;
        EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::runtime_error);
        options = FriedmannOptions{};
        options.absoluteTolerance = invalidTolerance;
        EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::runtime_error);
    }
    FriedmannOptions options;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;
    EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::runtime_error);
    options = FriedmannOptions{};
    options.maxSteps = 0;
    EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::runtime_error);
}

TEST(Friedmann, RejectsInvalidScaleFactorsCoordinatesAndInconsistentStates) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    FriedmannCosmology const cosmology;
    for (double const scaleFactor : {0.0, -1.0}) {
        EXPECT_THROW(cosmology.state(scaleFactor), std::runtime_error);
        EXPECT_THROW(cosmology.evolve(scaleFactor, 0.1), std::runtime_error);
    }
    EXPECT_THROW(cosmology.state(1.0, static_cast<ExpansionBranch>(-1)), std::runtime_error);
    EXPECT_THROW(cosmology.evolve(1.0, 0.1, static_cast<CosmologicalTime>(-1)), std::runtime_error);
    auto invalidState = cosmology.state(0.5);
    invalidState.scaleFactor = 0.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::runtime_error);
    invalidState = cosmology.state(0.5);
    invalidState.physicalDerivative *= 2.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::runtime_error);
    invalidState = cosmology.state(0.5);
    invalidState.conformalDerivative *= -1.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::runtime_error);
}

TEST(Friedmann, ReportsSingularTimeBoundariesAndStepExhaustion) {
    PROFILE();
    FriedmannCosmology const closed(parameters(0.0, 2.0, 0.0, -1.0));
    EXPECT_ANY_THROW(closed.state(2.1));
    FriedmannCosmology const radiation(parameters(1.0, 0.0, 0.0, 0.0));
    EXPECT_ANY_THROW(radiation.evolve(1.0, -0.6));
    EXPECT_ANY_THROW(radiation.evolve(1.0, 0.6, CosmologicalTime::physical, ExpansionBranch::contracting));
    EXPECT_ANY_THROW(radiation.evolve(1.0, 1.1, CosmologicalTime::conformal, ExpansionBranch::contracting));
    FriedmannCosmology const darkEnergy(parameters(0.0, 0.0, 1.0, 0.0));
    EXPECT_ANY_THROW(darkEnergy.evolve(1.0, 1.1, CosmologicalTime::conformal));
    FriedmannOptions options;
    options.maxSteps = 1;
    FriedmannCosmology const limited(parameters(0.0, 1.0, 0.0, 0.0), options);
    EXPECT_ANY_THROW(limited.evolve(1.0, 100.0));
}

}    // namespace
