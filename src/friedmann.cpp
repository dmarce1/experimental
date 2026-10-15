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

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosmictiger {
namespace detail {

    constexpr double kMinimumRelativeTolerance = 32.0 * std::numeric_limits<double>::epsilon();
    constexpr double kMaximumFriedmannTolerance = 1.0;

    struct FriedmannPolynomial {
        double value;
        double amplitude;
    };

    FriedmannPolynomial friedmannPolynomial(BackgroundCosmologyParameters const& parameters, double a) {
        PROFILE();
        double const matter = parameters.omegaMatter * a;
        double const curvature = parameters.omegaCurvature == 0.0 ? 0.0 : parameters.omegaCurvature * a * a;
        double const darkEnergy = parameters.omegaDarkEnergy == 0.0 ? 0.0 : parameters.omegaDarkEnergy * a * a * a * a;
        double const value = parameters.omegaRadiation + matter + curvature + darkEnergy;
        double const amplitude = parameters.omegaRadiation + matter + std::abs(curvature) + std::abs(darkEnergy);
        if (amplitude == 0.0) {
            throw std::overflow_error("Friedmann component amplitudes are outside the double-precision range");
        }
        return {value, amplitude};
    }

    struct FriedmannVariables {
        double a;
        double u;
    };

    // u=(da/deta)/H0 and s=H0*t or H0*eta. Unlike the positive square
    // root equation, this acceleration system is smooth at a turnaround.
    bool friedmannRhs(BackgroundCosmologyParameters const& parameters, CosmologicalTime time, FriedmannVariables value, FriedmannVariables& result) {
        constexpr double kMatterAccelerationCoefficient = 0.5;
        constexpr double kDarkEnergyAccelerationCoefficient = 2.0;
        PROFILE();
        if (value.a <= 0.0) {
            return false;
        }
        if (time == CosmologicalTime::physical) {
            result.a = value.u / value.a;
            result.u = kMatterAccelerationCoefficient * parameters.omegaMatter / value.a + parameters.omegaCurvature +
                (parameters.omegaDarkEnergy == 0.0 ? 0.0 : kDarkEnergyAccelerationCoefficient * parameters.omegaDarkEnergy * value.a * value.a);
        } else {
            result.a = value.u;
            result.u = kMatterAccelerationCoefficient * parameters.omegaMatter + parameters.omegaCurvature * value.a +
                (parameters.omegaDarkEnergy == 0.0 ? 0.0 : kDarkEnergyAccelerationCoefficient * parameters.omegaDarkEnergy * value.a * value.a * value.a);
        }
        return true;
    }

    constexpr size_t kFriedmannStages = 7;
    using FriedmannStages = std::array<FriedmannVariables, kFriedmannStages>;

    FriedmannVariables friedmannStage(
        FriedmannVariables value, double step, FriedmannStages const& stages, std::array<double, kFriedmannStages> const& weights) {
        PROFILE();
        FriedmannVariables sum{0.0, 0.0};
        for (size_t i = 0; i < kFriedmannStages; i++) {
            sum.a += weights[i] * stages[i].a;
            sum.u += weights[i] * stages[i].u;
        }
        return {value.a + step * sum.a, value.u + step * sum.u};
    }

    // Dormand-Prince 5(4), with a shared seventh stage for the embedded
    // error estimate. Coefficients are dimensionless in either clock.
    bool friedmannStep(BackgroundCosmologyParameters const& parameters, CosmologicalTime time, FriedmannVariables value, double step, FriedmannVariables& next,
        FriedmannVariables& error) {
        constexpr std::array<std::array<double, kFriedmannStages>, kFriedmannStages - 1> kCoefficients{{
            {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
            {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0},
            {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0},
            {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0},
        }};
        constexpr std::array<double, kFriedmannStages> kErrorWeights{35.0 / 384.0 - 5179.0 / 57600.0, 0.0, 500.0 / 1113.0 - 7571.0 / 16695.0,
            125.0 / 192.0 - 393.0 / 640.0, -2187.0 / 6784.0 + 92097.0 / 339200.0, 11.0 / 84.0 - 187.0 / 2100.0, -1.0 / 40.0};
        PROFILE();
        FriedmannStages stages{};
        if (!friedmannRhs(parameters, time, value, stages[0])) {
            return false;
        }
        for (size_t i = 1; i < kFriedmannStages; i++) {
            next = friedmannStage(value, step, stages, kCoefficients[i - 1]);
            if (!friedmannRhs(parameters, time, next, stages[i])) {
                return false;
            }
        }
        error = friedmannStage({0.0, 0.0}, step, stages, kErrorWeights);
        return true;
    }

    double friedmannConstraintError(FriedmannVariables value, FriedmannPolynomial polynomial) {
        PROFILE();
        // Normalize before squaring, avoiding overflow in u*u and allowing
        // for cancellation between signed curvature/Lambda and densities.
        double const normalizedU = value.u / std::sqrt(polynomial.amplitude);
        return std::abs(normalizedU * normalizedU - polynomial.value / polynomial.amplitude);
    }

    double friedmannConstraintTolerance(FriedmannOptions const& options) {
        constexpr double kConstraintToleranceFactor = 256.0;
        PROFILE();
        return kConstraintToleranceFactor * (options.relativeTolerance + options.absoluteTolerance + std::numeric_limits<double>::epsilon());
    }

    FriedmannState friedmannState(BackgroundCosmologyParameters const& parameters, FriedmannVariables value) {
        PROFILE();
        double const conformalDerivative = parameters.hubbleConstant * value.u;
        double const physicalDerivative = conformalDerivative / value.a;
        if (value.u != 0.0 && (conformalDerivative == 0.0 || physicalDerivative == 0.0)) {
            throw std::overflow_error("Friedmann derivative is outside the double-precision range");
        }
        return {value.a, physicalDerivative, conformalDerivative};
    }

}    // namespace detail

FriedmannCosmology::FriedmannCosmology(BackgroundCosmologyParameters parameters, FriedmannOptions options)
  : parameters_{expectPositive(parameters.hubbleConstant), expectNonNegative(parameters.omegaRadiation), expectNonNegative(parameters.omegaMatter),
        parameters.omegaDarkEnergy, parameters.omegaCurvature}
  , options_{expectRange(detail::kMinimumRelativeTolerance, options.relativeTolerance, detail::kMaximumFriedmannTolerance),
        expectNonNegative(options.absoluteTolerance), expectPositive(options.maxSteps)} {
    [[maybe_unused]] constexpr double kDensityClosureTolerance = 1.0e-10;
    PROFILE();
    FpeGuard guard;
    double const sum = parameters.omegaRadiation + parameters.omegaMatter + parameters.omegaDarkEnergy + parameters.omegaCurvature;
    expectRange(-kDensityClosureTolerance, sum - 1.0, kDensityClosureTolerance);
    expect(options_.relativeTolerance < detail::kMaximumFriedmannTolerance);
    expect(options_.absoluteTolerance < detail::kMaximumFriedmannTolerance);
}

BackgroundCosmologyParameters const& FriedmannCosmology::parameters() const noexcept {
    PROFILE();
    return parameters_;
}

FriedmannOptions const& FriedmannCosmology::options() const noexcept {
    PROFILE();
    return options_;
}

FriedmannState FriedmannCosmology::state(double scaleFactor, ExpansionBranch branch) const {
    PROFILE();
    FpeGuard guard;
    expect(branch == ExpansionBranch::expanding || branch == ExpansionBranch::contracting);
    auto const polynomial = detail::friedmannPolynomial(parameters_, expectPositive(scaleFactor));
    if (polynomial.value < 0.0) {
        throw std::domain_error("Friedmann H squared is negative at the requested scale factor");
    }
    double const u = std::sqrt(polynomial.value) * (branch == ExpansionBranch::expanding ? 1.0 : -1.0);
    return detail::friedmannState(parameters_, {scaleFactor, u});
}

FriedmannState FriedmannCosmology::advance(FriedmannState const& initialState, double elapsedTime, CosmologicalTime time) const {
    constexpr double kInitialStepFraction = 0.05;
    constexpr double kFailedStepFactor = 0.25;
    constexpr double kMaximumAcceptedError = 1.0;
    constexpr double kMaximumStepFactor = 5.0;
    constexpr double kMinimumStepFactor = 0.1;
    constexpr double kStepSafetyFactor = 0.9;
    constexpr double kStepErrorExponent = -0.2;
    PROFILE();
    FpeGuard guard;
    expect(time == CosmologicalTime::physical || time == CosmologicalTime::conformal);
    detail::FriedmannVariables value{expectPositive(initialState.scaleFactor), initialState.conformalDerivative / parameters_.hubbleConstant};
    auto polynomial = detail::friedmannPolynomial(parameters_, value.a);
    double const impliedU = initialState.physicalDerivative * value.a / parameters_.hubbleConstant;
    double const consistency = std::abs(impliedU - value.u) / std::sqrt(polynomial.amplitude);
    double const constraintTolerance = detail::friedmannConstraintTolerance(options_);
    expectRange(0.0, consistency, constraintTolerance);
    expectRange(0.0, detail::friedmannConstraintError(value, polynomial), constraintTolerance);
    if (elapsedTime == 0.0) {
        return initialState;
    }
    double const interval = parameters_.hubbleConstant * elapsedTime;
    if (interval == 0.0) {
        throw std::overflow_error("Friedmann elapsed time is outside the dimensionless double-precision range");
    }
    detail::FriedmannVariables rhs{};
    if (!detail::friedmannRhs(parameters_, time, value, rhs)) {
        throw std::overflow_error("initial Friedmann acceleration is outside the double-precision range");
    }
    double const amplitude = std::sqrt(polynomial.amplitude);
    double const frequency = std::fmax(std::abs(rhs.a) / value.a, std::abs(rhs.u) / amplitude);
    double const initialStep = frequency == 0.0 ? std::abs(interval) : std::fmin(std::abs(interval), kInitialStepFraction / frequency);
    double step = std::copysign(initialStep, interval);
    double position = 0.0;
    for (size_t attempt = 0; attempt < options_.maxSteps; attempt++) {
        double const remaining = interval - position;
        if (std::abs(step) > std::abs(remaining)) {
            step = remaining;
        }
        if (step == 0.0 || position + step == position) {
            throw std::domain_error("Friedmann evolution reached a singular boundary or cannot resolve further time steps");
        }
        detail::FriedmannVariables next{};
        detail::FriedmannVariables error{};
        if (!detail::friedmannStep(parameters_, time, value, step, next, error)) {
            step *= kFailedStepFactor;
            continue;
        }
        detail::FriedmannPolynomial nextPolynomial{};
        try {
            nextPolynomial = detail::friedmannPolynomial(parameters_, next.a);
        } catch (std::overflow_error const&) {
            step *= kFailedStepFactor;
            continue;
        }
        double const scaleA = options_.relativeTolerance * std::fmax(value.a, next.a);
        double const scaleU = options_.relativeTolerance * std::fmax(std::abs(value.u), std::abs(next.u)) +
            options_.absoluteTolerance * std::sqrt(std::fmax(polynomial.amplitude, nextPolynomial.amplitude));
        double const errorA = error.a == 0.0 ? 0.0 : std::abs(error.a) / scaleA;
        double const errorU = error.u == 0.0 ? 0.0 : std::abs(error.u) / scaleU;
        double const errorNorm = std::fmax(errorA, errorU);
        if (errorNorm <= kMaximumAcceptedError) {
            if (detail::friedmannConstraintError(next, nextPolynomial) > constraintTolerance) {
                throw std::runtime_error("Friedmann density constraint lost accuracy; use tighter tolerances or shorter intervals");
            }
            value = next;
            polynomial = nextPolynomial;
            if (step == remaining) {
                return detail::friedmannState(parameters_, value);
            }
            position += step;
        }
        double const factor = errorNorm == 0.0 ?
            kMaximumStepFactor :
            std::fmin(kMaximumStepFactor, std::fmax(kMinimumStepFactor, kStepSafetyFactor * std::pow(errorNorm, kStepErrorExponent)));
        step *= factor;
    }
    throw std::runtime_error("Friedmann evolution exceeded maxSteps; check the time interval, singular boundaries, and tolerances");
}

FriedmannState FriedmannCosmology::evolve(double initialScaleFactor, double elapsedTime, CosmologicalTime time, ExpansionBranch branch) const {
    PROFILE();
    return advance(state(initialScaleFactor, branch), elapsedTime, time);
}

}    // namespace cosmictiger
