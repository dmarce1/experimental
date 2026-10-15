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

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stddef.h>
#include <stdexcept>

namespace cosmictiger {
namespace detail {

    // Open quadrature never evaluates the singular cosmological endpoint a=0.
    template <typename Function>
    double collapseQuadrature(Function const& function, double lower, double upper) {
        constexpr double kHalf = 0.5;

        constexpr size_t kPositiveNodes = 4;
        constexpr std::array<double, kPositiveNodes> kNodes{0.18343464249564980494, 0.52553240991632898582, 0.79666647741362673959, 0.96028985649753623168};
        constexpr std::array<double, kPositiveNodes> kWeights{0.36268378337836198297, 0.31370664587788728734, 0.22238103445337447054, 0.10122853629037625915};
        PROFILE();
        double const middle = kHalf * (lower + upper);
        double const halfWidth = kHalf * (upper - lower);
        double sum = 0.0;
        for (size_t index = 0; index < kPositiveNodes; index++) {
            double const offset = halfWidth * kNodes[index];
            sum += kWeights[index] * (function(middle - offset) + function(middle + offset));
        }
        return halfWidth * sum;
    }

    template <typename Function>
    double collapseQuadrature(Function const& function, double lower, double upper, double estimate, double tolerance, unsigned depth) {
        constexpr double kHalf = 0.5;
        PROFILE();

        double const middle = kHalf * (lower + upper);
        double const left = collapseQuadrature(function, lower, middle);
        double const right = collapseQuadrature(function, middle, upper);
        double const combined = left + right;
        if (std::abs(combined - estimate) <= tolerance) {
            return combined;
        }
        if (depth == 0 || middle == lower || middle == upper) {
            throw std::runtime_error("cosmological collapse-time quadrature did not converge");
        }
        return collapseQuadrature(function, lower, middle, left, kHalf * tolerance, depth - 1) +
            collapseQuadrature(function, middle, upper, right, kHalf * tolerance, depth - 1);
    }

    template <typename Function>
    double collapseQuadrature(Function const& function) {
        constexpr double kRelativeTolerance = 2.0e-12;
        constexpr unsigned kMaximumDepth = 32;
        PROFILE();

        double const estimate = collapseQuadrature(function, 0.0, 1.0);
        double const scale = std::abs(estimate) > 1.0 ? std::abs(estimate) : 1.0;
        return collapseQuadrature(function, 0.0, 1.0, estimate, kRelativeTolerance * scale, kMaximumDepth);
    }

}    // namespace detail

SimulationCollapseTimes SimulationBackground::collapseTimes(double minimumScaleFactor) const {
    constexpr unsigned kMaximumBracketIterations = 1024;
    constexpr unsigned kMaximumRefinements = 64;
    constexpr double kDarkEnergyFactor = 3.0;
    constexpr double kCurvaturePower = 2.0;
    constexpr double kInitialStepFraction = 0.125;
    constexpr double kCrossingFactor = 2.0;
    constexpr double kHalf = 0.5;
    PROFILE();

    if (!(minimumScaleFactor > 0.0)) {
        throw std::invalid_argument("cosmological collapse cutoff scale factor must be positive");
    }
    FpeGuard guard;
    if (omegaCurvature_ >= 0.0 && omegaDarkEnergy_ >= 0.0) {
        throw std::domain_error("cosmological background has no future turnaround");
    }

    double lower = 1.0;
    double lowerValue;
    double amplitude;
    double lowerDerivative;
    evaluate(lower, lowerValue, amplitude, lowerDerivative);
    if (!(lowerValue > 0.0)) {
        throw std::domain_error("cosmological collapse time requires an expanding present epoch");
    }
    double upper = lower;
    bool bracketed = false;
    double const darkPower = 1.0 - kDarkEnergyFactor * (darkEnergyW_ + darkEnergyWa_);
    for (unsigned iteration = 0; iteration < kMaximumBracketIterations; iteration++) {
        // A positive dark component that already outweighs negative curvature
        // and whose ratio can only grow excludes all later turnarounds.
        if (omegaCurvature_ < 0.0 && omegaDarkEnergy_ > 0.0 && darkEnergyWa_ >= 0.0 &&
            (darkPower - kCurvaturePower) / lower + kDarkEnergyFactor * darkEnergyWa_ >= 0.0) {
            double const logRatio = std::log(omegaDarkEnergy_) - std::log(-omegaCurvature_) + (darkPower - kCurvaturePower) * std::log(lower) +
                kDarkEnergyFactor * darkEnergyWa_ * (lower - 1.0);
            if (logRatio >= 0.0) {
                throw std::domain_error("cosmological background has no future turnaround");
            }
        }
        double step = kInitialStepFraction * lower;
        if (lowerDerivative < 0.0) {
            double const crossingStep = kCrossingFactor * lowerValue / (-lowerDerivative);
            if (crossingStep < step) {
                step = crossingStep;
            }
        }
        if (omegaDarkEnergy_ != 0.0) {
            double const darkSlope = std::abs(darkPower / lower + kDarkEnergyFactor * darkEnergyWa_);
            if (darkSlope > 0.0 && 1.0 / darkSlope < step) {
                step = 1.0 / darkSlope;
            }
        }
        upper = lower + step;
        if (!(upper > lower)) {
            upper = std::nextafter(lower, std::numeric_limits<double>::max());
        }
        double upperValue;
        double upperDerivative;
        evaluate(upper, upperValue, amplitude, upperDerivative);
        if (upperValue <= 0.0) {
            bracketed = true;
            break;
        }
        if (lowerDerivative < 0.0 && upperDerivative > 0.0) {
            // Check the intervening minimum, even when the two sampled values
            // are positive, so a shallow forbidden interval is not skipped.
            double minimumLower = lower;
            double minimumUpper = upper;
            for (unsigned refinement = 0; refinement < kMaximumRefinements; refinement++) {
                double const middle = kHalf * (minimumLower + minimumUpper);
                if (middle == minimumLower || middle == minimumUpper) {
                    break;
                }
                double value;
                double derivative;
                evaluate(middle, value, amplitude, derivative);
                if (derivative < 0.0) {
                    minimumLower = middle;
                } else {
                    minimumUpper = middle;
                }
            }
            double const minimum = kHalf * (minimumLower + minimumUpper);
            double value;
            double derivative;
            evaluate(minimum, value, amplitude, derivative);
            if (value <= 0.0) {
                upper = minimum;
                bracketed = true;
                break;
            }
        }
        lower = upper;
        lowerValue = upperValue;
        lowerDerivative = upperDerivative;
    }
    if (!bracketed) {
        throw std::domain_error("no future cosmological turnaround found within the bounded search");
    }
    for (unsigned iteration = 0; iteration < kMaximumRefinements; iteration++) {
        double const middle = kHalf * (lower + upper);
        if (middle == lower || middle == upper) {
            break;
        }
        double value;
        double derivative;
        evaluate(middle, value, amplitude, derivative);
        if (value > 0.0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    double const maximumScaleFactor = kHalf * (lower + upper);
    double rootValue;
    double rootDerivative;
    evaluate(maximumScaleFactor, rootValue, amplitude, rootDerivative);
    if (!(rootDerivative < 0.0)) {
        throw std::domain_error("future cosmological turnaround is not a finite simple root");
    }
    if (!(minimumScaleFactor < maximumScaleFactor)) {
        throw std::invalid_argument("cosmological collapse cutoff must be below the turnaround scale factor");
    }

    auto const toTurnaround = [&](double endpoint) {
        constexpr double kRootFactor = 2.0;
        PROFILE();

        double const width = maximumScaleFactor - endpoint;
        double const rootLimit = kRootFactor * maximumScaleFactor * std::sqrt(width / (-rootDerivative));
        auto const integrand = [&](double coordinate) {
            constexpr double kCancellationThreshold = 1.0e-3;
            constexpr double kSimpsonMiddleWeight = 4.0;
            constexpr double kSimpsonNormalization = 6.0;
            constexpr double kCoordinateJacobianFactor = 2.0;

            constexpr double kHalfPi = kHalf * std::numbers::pi;
            PROFILE();
            if (coordinate == 0.0) {
                return kHalfPi * rootLimit;
            }
            // a=aMax-width*s*s removes the turnaround singularity. Taking
            // s=sin(pi*u/2) also smooths the matter/radiation endpoint at a=0.
            // The cosine form retains tiny a without cancellation.
            double const sine = std::sin(kHalfPi * coordinate);
            double const cosine = std::cos(kHalfPi * coordinate);
            double const distance = width * sine * sine;
            double const a = endpoint + width * cosine * cosine;
            double value;
            double derivative;
            evaluate(a, value, amplitude, derivative);
            if (distance < kCancellationThreshold * maximumScaleFactor) {
                // Integrating F' avoids cancellation in F close to its root.
                // Simpson's rule is fourth order in the distance from the root.
                double middleValue;
                double middleDerivative;
                evaluate(maximumScaleFactor - kHalf * distance, middleValue, amplitude, middleDerivative);
                value = -distance * (rootDerivative + kSimpsonMiddleWeight * middleDerivative + derivative) / kSimpsonNormalization;
            }
            if (!(value > 0.0)) {
                throw std::domain_error("cosmological contraction cannot reach the requested endpoint");
            }
            return kCoordinateJacobianFactor * width * sine * a * kHalfPi * cosine / std::sqrt(value);
        };
        return detail::collapseQuadrature(integrand);
    };

    double const expanding = toTurnaround(1.0);
    double const contracting = toTurnaround(0.0);
    double const cutoff = toTurnaround(minimumScaleFactor);
    return {expanding + contracting, expanding + cutoff};
}

}    // namespace cosmictiger
