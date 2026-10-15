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
#include <cosmictiger/simulation_time.hpp>
#include <cosmictiger/units.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace cosmictiger {
namespace detail {

    template <typename Function>
    double integrateSimulationInterval(Function const& function, double left, double right, double leftValue, double middleValue, double rightValue,
        double estimate, double tolerance, unsigned remainingDepth) {
        constexpr double kMidpointWeight = 0.5;
        constexpr double kSimpsonMiddleWeight = 4.0;
        constexpr double kSimpsonDivisor = 6.0;
        constexpr double kErrorCorrectionDivisor = 15.0;
        constexpr double kSubintervalToleranceFraction = 0.5;
        PROFILE();
        double const middle = kMidpointWeight * (left + right);
        double const leftMiddle = kMidpointWeight * (left + middle);
        double const rightMiddle = kMidpointWeight * (middle + right);
        double const leftMiddleValue = function(leftMiddle);
        double const rightMiddleValue = function(rightMiddle);
        double const leftEstimate = (middle - left) * (leftValue + kSimpsonMiddleWeight * leftMiddleValue + middleValue) / kSimpsonDivisor;
        double const rightEstimate = (right - middle) * (middleValue + kSimpsonMiddleWeight * rightMiddleValue + rightValue) / kSimpsonDivisor;
        double const correction = leftEstimate + rightEstimate - estimate;
        if (std::abs(correction) <= kErrorCorrectionDivisor * tolerance) {
            return leftEstimate + rightEstimate + correction / kErrorCorrectionDivisor;
        }
        if (remainingDepth == 0 || leftMiddle == left || rightMiddle == right) {
            throw std::runtime_error("simulation background quadrature did not converge");
        }
        return integrateSimulationInterval(function, left, middle, leftValue, leftMiddleValue, middleValue, leftEstimate,
                   kSubintervalToleranceFraction * tolerance, remainingDepth - 1) +
            integrateSimulationInterval(function, middle, right, middleValue, rightMiddleValue, rightValue, rightEstimate,
                kSubintervalToleranceFraction * tolerance, remainingDepth - 1);
    }

    template <typename Function>
    double integrateSimulationInterval(Function const& function, double left, double right) {
        constexpr double kMidpointWeight = 0.5;
        constexpr double kSimpsonMiddleWeight = 4.0;
        constexpr double kSimpsonDivisor = 6.0;
        constexpr double kRelativeTolerance = 1.0e-11;
        constexpr double kToleranceScaleFloor = 1.0;
        constexpr int kMaximumDepth = 24;
        PROFILE();
        double const middle = kMidpointWeight * (left + right);
        double const leftValue = function(left);
        double const middleValue = function(middle);
        double const rightValue = function(right);
        double const estimate = (right - left) * (leftValue + kSimpsonMiddleWeight * middleValue + rightValue) / kSimpsonDivisor;
        double const absoluteEstimate = std::abs(estimate);
        return integrateSimulationInterval(function, left, right, leftValue, middleValue, rightValue, estimate,
            kRelativeTolerance * (absoluteEstimate > kToleranceScaleFloor ? absoluteEstimate : kToleranceScaleFloor), kMaximumDepth);
    }

    // Thermal energy density relative to one massless species. The momentum
    // q is in units of its temperature; y=a*m/Tnu0. Normalization and density
    // inversion follow CAMB's massive_neutrinos.f90 and results.f90, but this
    // independent quadrature does not link the Einstein-Boltzmann solver.
    double thermalNeutrinoDensity(double y) {
        constexpr double kMinimumMomentum = 0.0;
        constexpr double kMaximumMomentum = 64.0;
        constexpr double kPi = std::numbers::pi;
        constexpr double kMasslessIntegral = 7.0 * kPi * kPi * kPi * kPi / 120.0;
        PROFILE();
        if (y == 0.0) {
            return 1.0;
        }
        // In the nonrelativistic limit the integral grows linearly with y.
        // Its first wide-interval Simpson estimate samples only the thermal
        // tail and cannot set a reliable absolute tolerance at large y.
        // Normalize that amplitude before quadrature, keeping the resolved
        // thermal shape and its roundoff at order unity at every epoch.
        double const normalization = std::hypot(1.0, y);
        auto const integrand = [y, normalization](double q) {
            PROFILE();
            return q * q * (std::hypot(q, y) / normalization) / (1.0 + std::exp(q));
        };

        // The omitted exponential tail is less than 1e-19 relative to the
        // relativistic or nonrelativistic integral.
        return (integrateSimulationInterval(integrand, kMinimumMomentum, kMaximumMomentum) / kMasslessIntegral) * normalization;
    }

    double thermalNeutrinoDensityDerivative(double y) {
        constexpr double kMinimumMomentum = 0.0;
        constexpr double kMaximumMomentum = 64.0;
        constexpr double kPi = std::numbers::pi;
        constexpr double kMasslessIntegral = 7.0 * kPi * kPi * kPi * kPi / 120.0;
        PROFILE();
        if (y == 0.0) {
            return 0.0;
        }
        auto const integrand = [y](double q) {
            PROFILE();
            return q * q * (y / std::hypot(q, y)) / (1.0 + std::exp(q));
        };

        return integrateSimulationInterval(integrand, kMinimumMomentum, kMaximumMomentum) / kMasslessIntegral;
    }

    double thermalNeutrinoMass(double densityRatio) {
        constexpr double kMinimumDensityRatio = 1.0;
        constexpr double kInitialUpperFactor = 4.0;
        constexpr int kMaximumIterations = 64;
        constexpr double kMidpointWeight = 0.5;
        constexpr double kRelativeMassTolerance = 1.0e-12;
        PROFILE();
        if (densityRatio < kMinimumDensityRatio) {
            throw std::invalid_argument("massive neutrino density is below the thermal massless minimum");
        }
        if (densityRatio == kMinimumDensityRatio) {
            return 0.0;
        }
        double lower = 0.0;
        double upper = kInitialUpperFactor * densityRatio;
        for (unsigned iteration = 0; iteration < kMaximumIterations; iteration++) {
            double const middle = kMidpointWeight * (lower + upper);
            if (thermalNeutrinoDensity(middle) < densityRatio) {
                lower = middle;
            } else {
                upper = middle;
            }
            if (upper - lower <= kRelativeMassTolerance * upper) {
                return kMidpointWeight * (lower + upper);
            }
        }
        throw std::runtime_error("thermal neutrino mass inversion did not converge");
    }

    struct SimulationVariables {
        double a;
        double u;
        double drift;
        double otherTime;
    };

    constexpr size_t kSimulationStages = 7;
    using SimulationStages = std::array<SimulationVariables, kSimulationStages>;

    SimulationVariables simulationStage(
        SimulationVariables value, double step, SimulationStages const& stages, std::array<double, kSimulationStages> const& weights) {
        PROFILE();
        SimulationVariables sum{};
        for (size_t index = 0; index < kSimulationStages; index++) {
            sum.a += weights[index] * stages[index].a;
            sum.u += weights[index] * stages[index].u;
            sum.drift += weights[index] * stages[index].drift;
            sum.otherTime += weights[index] * stages[index].otherTime;
        }
        return {value.a + step * sum.a, value.u + step * sum.u, value.drift + step * sum.drift, value.otherTime + step * sum.otherTime};
    }

    template <typename Function>
    bool simulationStep(
        Function const& rhs, SimulationVariables value, double step, SimulationVariables& next, SimulationVariables& error, SimulationStages& stages) {
        constexpr std::array<std::array<double, kSimulationStages>, kSimulationStages - 1> kCoefficients{{
            {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
            {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
            {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0},
            {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0},
            {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0},
        }};
        constexpr std::array<double, kSimulationStages> kErrorWeights{35.0 / 384.0 - 5179.0 / 57600.0, 0.0, 500.0 / 1113.0 - 7571.0 / 16695.0,
            125.0 / 192.0 - 393.0 / 640.0, -2187.0 / 6784.0 + 92097.0 / 339200.0, 11.0 / 84.0 - 187.0 / 2100.0, -1.0 / 40.0};
        PROFILE();
        if (!rhs(value, stages[0])) {
            return false;
        }
        for (size_t index = 1; index < kSimulationStages; index++) {
            next = simulationStage(value, step, stages, kCoefficients[index - 1]);
            if (!rhs(next, stages[index])) {
                return false;
            }
        }
        error = simulationStage({}, step, stages, kErrorWeights);
        return true;
    }

}    // namespace detail

SimulationBackground::SimulationBackground(CosmologyParameters const& parameters) {
    constexpr double kReferenceHubble = 100.0;
    constexpr double kCriticalDensityNumerator = 8.0;
    constexpr double kCriticalDensityDenominator = 3.0;
    constexpr double kPhotonDensityCoefficient = 4.0;
    constexpr int kPhotonTemperatureExponent = 4;
    constexpr double kNeutrinoStatisticalWeight = 7.0 / 8.0;
    constexpr double kNeutrinoTemperatureRatio = 4.0 / 11.0;
    constexpr double kNeutrinoTemperatureExponent = 4.0 / 3.0;
    constexpr double kSpeedOfLight = 2.99792458e8;
    constexpr double kStefanBoltzmann = 5.670374419e-8;
    constexpr double kCgsGravityToSi = 1.0e-3;
    constexpr double kGravitationalConstantSi = units::kGravitationalConstant.value() * kCgsGravityToSi;
    PROFILE();
    FpeGuard guard;
    double const h = expectPositive(parameters.hubble) / kReferenceHubble;
    double const inverseH2 = 1.0 / (h * h);
    omegaCold_ = (expectNonNegative(parameters.omegaBaryonH2) + expectNonNegative(parameters.omegaCdmH2)) * inverseH2;
    omegaCurvature_ = parameters.omegaCurvature;
    darkEnergyW_ = parameters.darkEnergyW;
    darkEnergyWa_ = parameters.darkEnergyWa;
    // SI constants match CAMB's constants.f90. Photon mass density is
    // 4*sigma_SB*T^4/c^3 and rho_critical = 3*H0^2/(8*pi*G).

    double const hubbleSi = units::hubbleRate(parameters.hubble).value();
    double const temperature = expectNonNegative(parameters.temperatureCmb);
    double const omegaPhoton = (kCriticalDensityNumerator * std::numbers::pi * kGravitationalConstantSi / (kCriticalDensityDenominator * hubbleSi * hubbleSi)) *
        (kPhotonDensityCoefficient * kStefanBoltzmann / (kSpeedOfLight * kSpeedOfLight * kSpeedOfLight)) * std::pow(temperature, kPhotonTemperatureExponent);
    double const count = static_cast<double>(expectNonNegative(parameters.masslessNeutrinos)) + expectNonNegative(parameters.massiveNeutrinos);
    double const effective = expectNonNegative(parameters.effectiveNeutrinos);
    expectEqual(count == 0.0, effective == 0.0);
    double const degeneracy = count == 0.0 ? 0.0 : effective / count;
    double const omegaNeutrinoUnit = kNeutrinoStatisticalWeight * std::pow(kNeutrinoTemperatureRatio, kNeutrinoTemperatureExponent) * omegaPhoton * degeneracy;
    omegaRadiation_ = omegaPhoton + parameters.masslessNeutrinos * omegaNeutrinoUnit;
    omegaMassiveRelativistic_ = parameters.massiveNeutrinos * omegaNeutrinoUnit;
    double const omegaMassive = expectNonNegative(parameters.omegaNeutrinoH2) * inverseH2;
    expectEqual(omegaMassive > 0.0, parameters.massiveNeutrinos > 0);
    // This inversion gives m/Tnu0 including the species' effective heating.
    // Degeneracy^(1/4) is needed only for an independent physical-mass input;
    // applying it again here would disagree with CAMB's density inversion.
    neutrinoMassRatio_ = omegaMassive == 0.0 ? 0.0 : detail::thermalNeutrinoMass(omegaMassive / expectPositive(omegaMassiveRelativistic_));
    omegaDarkEnergy_ = 1.0 - omegaCold_ - omegaMassive - omegaRadiation_ - omegaCurvature_;
}

void SimulationBackground::evaluate(double scaleFactor, double& squaredDerivative, double& amplitude, double& derivative) const {
    constexpr double kEquationOfStateCoefficient = 3.0;
    constexpr double kCurvatureDerivativeCoefficient = 2.0;
    PROFILE();
    double const a = expectPositive(scaleFactor);
    double const darkExponent = 1.0 - kEquationOfStateCoefficient * (darkEnergyW_ + darkEnergyWa_);
    double const darkEnergy =
        omegaDarkEnergy_ == 0.0 ? 0.0 : omegaDarkEnergy_ * std::exp(darkExponent * std::log(a) + kEquationOfStateCoefficient * darkEnergyWa_ * (a - 1.0));
    double const massive = omegaMassiveRelativistic_ == 0.0 ? 0.0 : omegaMassiveRelativistic_ * detail::thermalNeutrinoDensity(a * neutrinoMassRatio_);
    double const massiveDerivative = omegaMassiveRelativistic_ == 0.0 ?
        0.0 :
        omegaMassiveRelativistic_ * neutrinoMassRatio_ * detail::thermalNeutrinoDensityDerivative(a * neutrinoMassRatio_);
    double const cold = omegaCold_ * a;
    double const curvature = omegaCurvature_ == 0.0 ? 0.0 : omegaCurvature_ * a * a;
    squaredDerivative = omegaRadiation_ + cold + curvature + darkEnergy + massive;
    amplitude = omegaRadiation_ + cold + std::abs(curvature) + std::abs(darkEnergy) + massive;
    derivative = omegaCold_ + kCurvatureDerivativeCoefficient * omegaCurvature_ * a +
        darkEnergy * (darkExponent / a + kEquationOfStateCoefficient * darkEnergyWa_) + massiveDerivative;
}

double SimulationBackground::conformalDerivative(double scaleFactor) const {
    PROFILE();
    FpeGuard guard;
    double squared;
    double amplitude;
    double derivative;
    evaluate(scaleFactor, squared, amplitude, derivative);
    if (!(squared > 0.0)) {
        throw std::domain_error("simulation interval reaches a non-expanding cosmological background");
    }
    return std::sqrt(squared);
}

double SimulationBackground::conformalInterval(double initialScaleFactor, double finalScaleFactor) const {
    PROFILE();
    FpeGuard guard;
    double const initial = expectPositive(initialScaleFactor);
    double const final = expectPositive(finalScaleFactor);
    if (!(final > initial)) {
        throw std::invalid_argument("conformal quadrature requires increasing scale-factor endpoints");
    }
    auto const integrand = [this](double logScaleFactor) {
        PROFILE();
        double const a = std::exp(logScaleFactor);
        return a / conformalDerivative(a);
    };
    return detail::integrateSimulationInterval(integrand, std::log(initial), std::log(final));
}

FriedmannState SimulationBackground::state(double scaleFactor, ExpansionBranch branch) const {
    PROFILE();
    FpeGuard guard;
    expect(branch == ExpansionBranch::expanding || branch == ExpansionBranch::contracting);
    double squared;
    double amplitude;
    double derivative;
    evaluate(scaleFactor, squared, amplitude, derivative);
    if (squared < 0.0) {
        throw std::domain_error("simulation background H squared is negative at the requested scale factor");
    }
    double const u = std::sqrt(squared) * (branch == ExpansionBranch::expanding ? 1.0 : -1.0);
    return {scaleFactor, u / scaleFactor, u};
}

SimulationBackgroundStep SimulationBackground::advanceInTime(
    FriedmannState const& initialState, double elapsedTime, CosmologicalTime time, SimulationBackgroundTable* table) const {
    constexpr double kInitialStepFraction = 0.05;
    constexpr double kFailedStepFactor = 0.25;
    constexpr double kMaximumAcceptedError = 1.0;
    constexpr double kMaximumStepFactor = 5.0;
    constexpr double kMinimumStepFactor = 0.1;
    constexpr double kStepSafetyFactor = 0.9;
    constexpr double kStepErrorExponent = -0.2;
    constexpr double kAccelerationCoefficient = 0.5;
    constexpr double kProjectionMinimumNormalizedDerivative = 0.125;
    constexpr double kRelativeTolerance = 1.0e-12;
    constexpr double kAbsoluteTolerance = 1.0e-14;
    constexpr double kDenseToleranceFactor = 0.01;
    constexpr double kConstraintTolerance = 256.0 * (kRelativeTolerance + kAbsoluteTolerance + std::numeric_limits<double>::epsilon());
    constexpr double kLocalConstraintTolerance = 16.0 * (kRelativeTolerance + kAbsoluteTolerance + std::numeric_limits<double>::epsilon());
    constexpr size_t kMaximumAttempts = 100000;
    PROFILE();
    FpeGuard guard;
    expectPositive(initialState.scaleFactor);
    if (elapsedTime == 0.0) {
        return {initialState, 0.0, 0.0, 0.0};
    }

    auto const rhs = [this, time](detail::SimulationVariables value, detail::SimulationVariables& result) {
        PROFILE();
        if (value.a <= 0.0) {
            return false;
        }
        double squared;
        double amplitude;
        double derivative;
        evaluate(value.a, squared, amplitude, derivative);
        double const clockMultiplier = time == CosmologicalTime::conformal ? 1.0 : 1.0 / value.a;
        result = {clockMultiplier * value.u, kAccelerationCoefficient * clockMultiplier * derivative, clockMultiplier / value.a,
            time == CosmologicalTime::conformal ? value.a : clockMultiplier};
        return true;
    };
    auto const constraintResidual = [](double u, double squared, double amplitude) {
        PROFILE();
        double const normalized = u / std::sqrt(amplitude);
        return normalized * normalized - squared / amplitude;
    };
    detail::SimulationVariables value{initialState.scaleFactor, initialState.conformalDerivative, 0.0, 0.0};
    double squared;
    double amplitude;
    double derivative;
    evaluate(value.a, squared, amplitude, derivative);
    if (!(amplitude > 0.0)) {
        throw std::domain_error("simulation background component amplitude is zero");
    }
    expectRange(0.0, std::abs(initialState.physicalDerivative * value.a - value.u) / std::sqrt(amplitude), kConstraintTolerance);
    expectRange(0.0, std::abs(constraintResidual(value.u, squared, amplitude)), kConstraintTolerance);
    detail::SimulationVariables initialRhs{};
    rhs(value, initialRhs);
    double const frequency = std::fmax(std::abs(initialRhs.a) / value.a, std::abs(initialRhs.u) / std::sqrt(amplitude));
    double step = std::copysign(frequency == 0.0 ? std::abs(elapsedTime) : std::fmin(std::abs(elapsedTime), kInitialStepFraction / frequency), elapsedTime);
    double position = 0.0;
    for (size_t attempt = 0; attempt < kMaximumAttempts; attempt++) {
        double const remaining = elapsedTime - position;
        if (std::abs(step) > std::abs(remaining)) {
            step = remaining;
        }
        if (step == 0.0 || position + step == position) {
            throw std::domain_error("simulation background reached a singular boundary or cannot resolve further time steps");
        }
        detail::SimulationVariables next{};
        detail::SimulationVariables error{};
        detail::SimulationStages stages{};
        if (!detail::simulationStep(rhs, value, step, next, error, stages)) {
            step *= kFailedStepFactor;
            continue;
        }
        double nextSquared;
        double nextAmplitude;
        double nextDerivative;
        evaluate(next.a, nextSquared, nextAmplitude, nextDerivative);
        if (!(nextAmplitude > 0.0)) {
            throw std::domain_error("simulation background component amplitude is outside the representable range");
        }
        double const scaleA = kRelativeTolerance * std::fmax(value.a, next.a);
        double const scaleU =
            kRelativeTolerance * std::fmax(std::abs(value.u), std::abs(next.u)) + kAbsoluteTolerance * std::sqrt(std::fmax(amplitude, nextAmplitude));
        double const scaleDrift = kRelativeTolerance * std::fmax(std::abs(next.drift), std::abs(step * initialRhs.drift));
        double const scaleClock = kRelativeTolerance * std::fmax(std::abs(next.otherTime), std::abs(step * initialRhs.otherTime));
        double const errorA = error.a == 0.0 ? 0.0 : std::abs(error.a) / scaleA;
        double const errorU = error.u == 0.0 ? 0.0 : std::abs(error.u) / scaleU;
        double const errorDrift = error.drift == 0.0 ? 0.0 : std::abs(error.drift) / scaleDrift;
        double const errorClock = error.otherTime == 0.0 ? 0.0 : std::abs(error.otherTime) / scaleClock;
        // u*u-F(a) is a first integral of the acceleration equation. Its
        // change over a trial step measures a local defect independently of
        // the embedded solution estimate. Normalize both endpoints by the
        // same component amplitude before subtracting; a projection must
        // never conceal a poorly resolved step.
        double const commonAmplitude = std::fmax(amplitude, nextAmplitude);
        double const localConstraintError =
            std::abs(constraintResidual(next.u, nextSquared, commonAmplitude) - constraintResidual(value.u, squared, commonAmplitude));

        double const errorConstraint = localConstraintError / kLocalConstraintTolerance;
        double const errorNorm = std::fmax(errorConstraint, std::fmax(std::fmax(errorA, errorU), std::fmax(errorDrift, errorClock)));
        double const controlledError = table ? errorNorm / kDenseToleranceFactor : errorNorm;
        if (controlledError <= kMaximumAcceptedError) {
            if (std::abs(constraintResidual(next.u, nextSquared, nextAmplitude)) > kConstraintTolerance) {
                throw std::runtime_error("simulation background density constraint lost accuracy");
            }
            // Project an accepted step onto its known first integral where
            // solving for u is well conditioned. This removes the absolute
            // energy error which otherwise dominates as matter density
            // shrinks near a crunch. Preserve the integrated sign, and use
            // the smooth acceleration system without projection near u=0
            // so the turnaround and change of branch remain continuous.
            if (nextSquared > 0.0 && std::abs(next.u) >= kProjectionMinimumNormalizedDerivative * std::sqrt(nextAmplitude)) {
                next.u = std::copysign(std::sqrt(nextSquared), next.u);
            }
            if (table) {
                // Shampine continuous extension of Dormand-Prince 5(4):
                // Hairer/Wanner DOPRI5, CDOPRI and CONTD5.
                // https://www.unige.ch/~hairer/prog/nonstiff/dopri5.f
                constexpr std::array<double, detail::kSimulationStages> kDenseWeights{-12715105075.0 / 11282082432.0, 0.0, 87487479700.0 / 32700410799.0,
                    -10690763975.0 / 1880347072.0, 701980252875.0 / 199316789632.0, -1453857185.0 / 822651844.0, 69997945.0 / 29380423.0};
                constexpr std::array<double, detail::kSimulationStages> kEndpointWeights{
                    35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
                constexpr double kTwice = 2.0;
                constexpr double kInterpolationDerivativeTolerance = 1.0e-12;
                constexpr double kRefinementFactor = 0.5;
                constexpr size_t kInterpolationChecks = 3;
                constexpr std::array<double, kInterpolationChecks> kCheckFractions{0.25, 0.5, 0.75};
                auto const dense = detail::simulationStage({}, step, stages, kDenseWeights);
                auto const delta = detail::simulationStage({}, step, stages, kEndpointWeights);
                using Values = std::array<double, SimulationBackgroundTable::kVariableCount_>;
                Values const begin{value.a, value.u, 0.0, 0.0};
                Values const change{next.a - value.a, next.u - value.u, delta.drift, delta.otherTime};
                Values const first{stages.front().a, stages.front().u, stages.front().drift, stages.front().otherTime};
                Values const last{stages.back().a, stages.back().u, stages.back().drift, stages.back().otherTime};
                Values const extension{dense.a, dense.u, dense.drift, dense.otherTime};
                SimulationBackgroundTable::Segment segment{position, step, value.otherTime, next.otherTime, value.drift, {}};
                if (!(segment.conformalEnd > segment.conformalBegin)) {
                    throw std::runtime_error("background table cannot resolve conformal progress");
                }
                for (size_t component = 0; component < begin.size(); component++) {
                    double const slope = step * first[component];
                    double const bend = slope - change[component];
                    double const cubic = change[component] - step * last[component] - bend;
                    double const quartic = extension[component];
                    segment.coefficients[component] = {begin[component], slope, cubic + quartic - bend, -cubic - kTwice * quartic, quartic};
                }
                // Endpoint error control alone does not bound the derivative
                // of dense output. Short drifts depend on that derivative, so
                // refine until both dtau/dt=1/a and dD/dt=1/a^2 agree locally.
                bool accurate = true;
                for (double const fraction : kCheckFractions) {
                    double const a = SimulationBackgroundTable::polynomial(segment.coefficients[SimulationBackgroundTable::kScaleFactor_], fraction);
                    double const clockRate =
                        SimulationBackgroundTable::dividedDifference(segment.coefficients[SimulationBackgroundTable::kConformal_], fraction, fraction) / step;
                    double const driftRate =
                        SimulationBackgroundTable::dividedDifference(segment.coefficients[SimulationBackgroundTable::kDrift_], fraction, fraction) / step;
                    if (!(a > 0.0) || std::abs(a * clockRate - 1.0) > kInterpolationDerivativeTolerance ||
                        std::abs(a * a * driftRate - 1.0) > kInterpolationDerivativeTolerance) {
                        accurate = false;
                        break;
                    }
                }
                if (!accurate) {
                    step *= kRefinementFactor;
                    continue;
                }
                table->segments_.push_back(segment);
            }
            value = next;
            squared = nextSquared;
            amplitude = nextAmplitude;
            if (step == remaining) {
                FriedmannState const finalState{value.a, value.u / value.a, value.u};
                return {finalState, value.drift, time == CosmologicalTime::conformal ? elapsedTime : value.otherTime,
                    time == CosmologicalTime::physical ? elapsedTime : value.otherTime};
            }
            position += step;
        }
        double const factor = controlledError == 0.0 ?
            kMaximumStepFactor :
            std::fmin(kMaximumStepFactor, std::fmax(kMinimumStepFactor, kStepSafetyFactor * std::pow(controlledError, kStepErrorExponent)));
        step *= factor;
    }
    throw std::runtime_error("simulation background evolution exceeded the integration step limit");
}

SimulationBackgroundStep SimulationBackground::advance(FriedmannState const& initialState, double elapsedConformalTime) const {
    PROFILE();
    return advanceInTime(initialState, elapsedConformalTime, CosmologicalTime::conformal);
}

SimulationBackgroundStep SimulationBackground::advancePhysical(FriedmannState const& initialState, double elapsedPhysicalTime) const {
    PROFILE();
    return advanceInTime(initialState, elapsedPhysicalTime, CosmologicalTime::physical);
}

double SimulationBackground::physicalInterval(double initialScaleFactor, double finalScaleFactor) const {
    PROFILE();
    FpeGuard guard;
    double const initial = expectPositive(initialScaleFactor);
    double const final = expectPositive(finalScaleFactor);
    if (initial == final) {
        return 0.0;
    }
    auto const integrand = [this](double logScaleFactor) {
        PROFILE();
        double const a = std::exp(logScaleFactor);
        return a * a / conformalDerivative(a);
    };
    if (final > initial) {
        return detail::integrateSimulationInterval(integrand, std::log(initial), std::log(final));
    }
    return -detail::integrateSimulationInterval(integrand, std::log(final), std::log(initial));
}

double SimulationBackground::presentAge() const {
    constexpr double kEquationOfStateCoefficient = 3.0;
    constexpr double kRadiationPower = 0.0;
    constexpr double kMatterPower = 1.0;
    constexpr double kCurvaturePower = 2.0;
    constexpr double kInfiniteAgeThresholdPower = 4.0;
    constexpr double kRegularizationPower = 2.0;
    constexpr double kLowPowerIntegrandExponent = 3.0;
    constexpr double kMinimumQuadratureCoordinate = 0.0;
    constexpr double kMaximumQuadratureCoordinate = 1.0;
    PROFILE();
    FpeGuard guard;
    double const darkPower = 1.0 - kEquationOfStateCoefficient * (darkEnergyW_ + darkEnergyWa_);
    double leadingPower = omegaDarkEnergy_ == 0.0 ? std::numeric_limits<double>::max() : darkPower;
    if (omegaRadiation_ > 0.0 || omegaMassiveRelativistic_ > 0.0) {
        leadingPower = std::fmin(leadingPower, kRadiationPower);
    }
    if (omegaCold_ > 0.0) {
        leadingPower = std::fmin(leadingPower, kMatterPower);
    }
    if (omegaCurvature_ != 0.0) {
        leadingPower = std::fmin(leadingPower, kCurvaturePower);
    }
    // If F(a) vanishes at least as a^4, integral a/sqrt(F) da has
    // infinite age. This includes a pure Lambda background.
    if (!(leadingPower < kInfiniteAgeThresholdPower)) {
        throw std::domain_error("simulation background has no finite Big Bang age");
    }
    // a=s^power regularizes radiation, matter, curvature and any dominant
    // finite-age dark component. Divide F by its leading power analytically,
    // so neither F nor a must remain representable close to the endpoint.
    double const power = leadingPower <= kCurvaturePower ? kRegularizationPower : kInfiniteAgeThresholdPower / (kInfiniteAgeThresholdPower - leadingPower);
    double const integrandPower = leadingPower <= kCurvaturePower ? kLowPowerIntegrandExponent - leadingPower : 1.0;
    auto const integrand = [this, darkPower, leadingPower, power, integrandPower](double coordinate) {
        PROFILE();
        if (coordinate == 0.0) {
            return 0.0;
        }
        double const logCoordinate = std::log(coordinate);
        double const logA = power * logCoordinate;
        double const a = std::exp(logA);
        double const radiationScale = omegaRadiation_ == 0.0 && omegaMassiveRelativistic_ == 0.0 ? 0.0 : std::exp(-leadingPower * logA);
        double const radiation = omegaRadiation_ * radiationScale;
        double const cold = omegaCold_ == 0.0 ? 0.0 : omegaCold_ * std::exp((kMatterPower - leadingPower) * logA);
        double const curvature = omegaCurvature_ == 0.0 ? 0.0 : omegaCurvature_ * std::exp((kCurvaturePower - leadingPower) * logA);
        double const darkEnergy = omegaDarkEnergy_ == 0.0 ?
            0.0 :
            omegaDarkEnergy_ * std::exp((darkPower - leadingPower) * logA + kEquationOfStateCoefficient * darkEnergyWa_ * (a - 1.0));
        double const massive =
            omegaMassiveRelativistic_ == 0.0 ? 0.0 : omegaMassiveRelativistic_ * radiationScale * detail::thermalNeutrinoDensity(a * neutrinoMassRatio_);
        double const scaledSquared = radiation + cold + curvature + darkEnergy + massive;
        if (!(scaledSquared > 0.0)) {
            throw std::domain_error("simulation background cannot reach a Big Bang on the expanding present branch");
        }
        return power * std::exp(integrandPower * logCoordinate) / std::sqrt(scaledSquared);
    };
    return detail::integrateSimulationInterval(integrand, kMinimumQuadratureCoordinate, kMaximumQuadratureCoordinate);
}

double SimulationBackground::driftFactor(double initialScaleFactor, double finalScaleFactor) const {
    PROFILE();
    FpeGuard guard;
    double const initial = expectPositive(initialScaleFactor);
    double const final = expectPositive(finalScaleFactor);
    if (initial == final) {
        return 0.0;
    }
    auto const integrand = [this](double logScaleFactor) {
        PROFILE();
        return 1.0 / conformalDerivative(std::exp(logScaleFactor));
    };
    if (final > initial) {
        return detail::integrateSimulationInterval(integrand, std::log(initial), std::log(final));
    }
    return -detail::integrateSimulationInterval(integrand, std::log(final), std::log(initial));
}

SimulationBackgroundTable::SimulationBackgroundTable(SimulationBackground const& background, FriedmannState const& initial, double physicalDuration)
  : initial_(initial) {
    PROFILE();
    if (!(physicalDuration >= 0.0)) {
        throw std::invalid_argument("background table requires a nonnegative duration");
    }
    end_ = background.advanceInTime(initial, physicalDuration, CosmologicalTime::physical, this);
}

SimulationBackgroundStep const& SimulationBackgroundTable::end() const {
    PROFILE();
    return end_;
}

size_t SimulationBackgroundTable::segmentCount() const {
    PROFILE();
    return segments_.size();
}

double SimulationBackgroundTable::polynomial(Polynomial const& coefficients, double fraction) {
    PROFILE();
    double result = coefficients.back();
    for (size_t index = coefficients.size() - 1; index > 0; index--) {
        result = std::fma(result, fraction, coefficients[index - 1]);
    }
    return result;
}

double SimulationBackgroundTable::dividedDifference(Polynomial const& coefficients, double left, double right) {
    PROFILE();
    // (p(right)-p(left))/(right-left), without subtracting close values.
    // For equal arguments this is exactly the polynomial's derivative.
    double value = coefficients.back();
    double difference = 0.0;
    for (size_t index = coefficients.size() - 1; index > 0; index--) {
        difference = std::fma(difference, left, value);
        value = std::fma(value, right, coefficients[index - 1]);
    }
    return difference;
}

SimulationBackgroundTable::Location SimulationBackgroundTable::locate(double conformalTime) const {
    constexpr size_t kMaximumIterations = 64;
    constexpr double kTolerance = 8.0 * std::numeric_limits<double>::epsilon();
    constexpr double kHalf = 0.5;
    PROFILE();
    if (!(conformalTime >= 0.0 && conformalTime <= end_.conformalInterval) || segments_.empty()) {
        throw std::out_of_range("conformal time outside the background table");
    }
    size_t first = 0;
    size_t last = segments_.size() - 1;
    while (first < last) {
        size_t const middle = first + (last - first) / 2;
        if (conformalTime < segments_[middle].conformalEnd) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    auto const& segment = segments_[first];
    if (conformalTime == segment.conformalBegin) {
        return {first, 0.0};
    }
    if (conformalTime == segment.conformalEnd) {
        return {first, 1.0};
    }
    auto const& clock = segment.coefficients[kConformal_];
    double const target = conformalTime - segment.conformalBegin;
    double const width = segment.conformalEnd - segment.conformalBegin;
    double lo = 0.0;
    double hi = 1.0;
    double fraction = target / width;
    for (size_t iteration = 0; iteration < kMaximumIterations; iteration++) {
        double const residual = polynomial(clock, fraction) - target;
        if (std::abs(residual) <= kTolerance * std::fmax(width, std::abs(conformalTime))) {
            return {first, fraction};
        }
        if (residual < 0.0) {
            lo = fraction;
        } else {
            hi = fraction;
        }
        double const derivative = dividedDifference(clock, fraction, fraction);
        if (!(derivative > 0.0)) {
            throw std::runtime_error("background interpolant lost clock monotonicity");
        }
        double const candidate = fraction - residual / derivative;
        fraction = candidate > lo && candidate < hi ? candidate : lo + kHalf * (hi - lo);
    }
    throw std::runtime_error("background interpolation clock inversion did not converge");
}

SimulationBackgroundStep SimulationBackgroundTable::at(double conformalTime) const {
    PROFILE();
    FpeGuard guard;
    if (conformalTime == 0.0) {
        return {initial_, 0.0, 0.0, 0.0};
    }
    if (conformalTime == end_.conformalInterval) {
        return end_;
    }
    auto const location = locate(conformalTime);
    auto const& segment = segments_[location.index];
    double const a = expectPositive(polynomial(segment.coefficients[kScaleFactor_], location.fraction));
    double const u = polynomial(segment.coefficients[kDerivative_], location.fraction);
    return {{a, u / a, u}, segment.driftBegin + polynomial(segment.coefficients[kDrift_], location.fraction), conformalTime,
        std::fma(segment.physicalWidth, location.fraction, segment.physicalBegin)};
}

SimulationBackgroundStep SimulationBackgroundTable::interval(double begin, double end) const {
    PROFILE();
    FpeGuard guard;
    if (!(begin >= 0.0 && end >= begin && end <= end_.conformalInterval)) {
        throw std::out_of_range("invalid background table interval");
    }
    auto result = at(end);
    result.conformalInterval = end - begin;
    result.physicalInterval = 0.0;
    result.driftFactor = 0.0;
    if (begin == end) {
        return result;
    }
    auto const left = locate(begin);
    auto const right = locate(end);
    for (size_t index = left.index; index <= right.index; index++) {
        auto const& segment = segments_[index];
        double const lo = index == left.index ? left.fraction : 0.0;
        double const hi = index == right.index ? right.fraction : 1.0;
        double const tauLo = index == left.index ? begin : segment.conformalBegin;
        double const tauHi = index == right.index ? end : segment.conformalEnd;
        double const clockSlope = expectPositive(dividedDifference(segment.coefficients[kConformal_], lo, hi));
        double const fractionWidth = (tauHi - tauLo) / clockSlope;
        result.driftFactor += fractionWidth * dividedDifference(segment.coefficients[kDrift_], lo, hi);
        result.physicalInterval += fractionWidth * segment.physicalWidth;
    }
    if (begin == 0.0 && end == end_.conformalInterval) {
        result.physicalInterval = end_.physicalInterval;
    }
    return result;
}

}    // namespace cosmictiger
