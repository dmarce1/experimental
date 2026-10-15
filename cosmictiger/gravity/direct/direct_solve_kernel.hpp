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

#pragma once

#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_plummer_ewald.hpp>
#include <cosmictiger/profiler.hpp>

#include <cmath>
#include <cstdint>
#include <stddef.h>

#if defined(__CUDACC__)
#define COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE
#endif

namespace cosmictiger::gravity::detail {

constexpr double kDirectPi = 3.141592653589793238462643383279502884;
constexpr double kDirectInverseRootPi = 0.564189583547756286948079451560772586;

constexpr size_t kDirectCachedWaveCutoff = 8;
constexpr size_t kDirectCachedWaveSquared = NDIM * kDirectCachedWaveCutoff * kDirectCachedWaveCutoff;

// Kept below the CUDA kernel argument limit. Default and usual convergence
// cutoffs use only these cached weights; larger custom cutoffs remain valid.
struct DirectEwaldWeights {
    double reciprocal[kDirectCachedWaveSquared + 1]{};
    double zeroMode = 1.0;
};

inline DirectEwaldWeights prepareDirectEwaldWeights(DirectSolveRequest const& request, DirectEwaldParameters const& parameters) {
    PROFILE();
    DirectEwaldWeights weights;
    if (request.softeningModel != SofteningModel::plummer) {
        return weights;
    }
    FpeGuard fpeGuard;
    double const alphaSquared = parameters.alpha * parameters.alpha;
    double const z = request.softeningLength * request.softeningLength * alphaSquared;
    double const waveScale = kDirectPi * kDirectPi / alphaSquared;
    weights.zeroMode = plummerZeroModeIntegral(z);
    size_t const cutoff =
        static_cast<size_t>(parameters.waveCutoff) < kDirectCachedWaveCutoff ? static_cast<size_t>(parameters.waveCutoff) : kDirectCachedWaveCutoff;
    size_t const maximumSquared = NDIM * cutoff * cutoff;
    for (size_t waveSquared = 1; waveSquared <= maximumSquared; waveSquared++) {
        weights.reciprocal[waveSquared] = plummerReciprocalIntegral(waveScale * static_cast<double>(waveSquared), z);
    }
    return weights;
}

COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline double directDisplacement(Coordinate target, Coordinate source) noexcept {
    constexpr double kCoordinateScale = 0x1p-32;
    PROFILE();
    std::uint32_t const wrapped = target.raw() - source.raw();
    return static_cast<double>(static_cast<std::int32_t>(wrapped)) * kCoordinateScale;
}

COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline double directFerrersForcePolynomial(double t, int order) noexcept {
    constexpr double kLinearCoefficient = 1.5;
    constexpr double kQuadraticCoefficient = 1.875;
    constexpr double kCubicCoefficient = 2.1875;
    PROFILE();
    double result = 0.0;
    if (order >= 3) {
        result = kCubicCoefficient;
    }
    if (order >= 2) {
        result = kQuadraticCoefficient + t * result;
    }
    if (order >= 1) {
        result = kLinearCoefficient + t * result;
    }
    return 1.0 + t * result;
}

COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline double directFerrersPotentialPolynomial(double t, int order) noexcept {
    constexpr double kLinearCoefficient = 0.5;
    constexpr double kQuadraticCoefficient = 0.375;
    constexpr double kCubicCoefficient = 0.3125;
    constexpr double kQuarticCoefficient = 0.2734375;
    PROFILE();
    double result = 0.0;
    if (order >= 3) {
        result = kQuarticCoefficient;
    }
    if (order >= 2) {
        result = kCubicCoefficient + t * result;
    }
    if (order >= 1) {
        result = kQuadraticCoefficient + t * result;
    }
    return 1.0 + t * (kLinearCoefficient + t * result);
}

// Add the regular primary Ewald correction erf(alpha*r)/r. Positive lower
// incomplete-gamma series avoid subtracting nearly equal terms in its gradient
// near zero. They also supply the exact regular self limit without division.
COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline void directPrimaryCorrection(
    double radiusSquared, double alpha, double& potential, double& accelerationScale) noexcept {
    constexpr double kSeriesThreshold = 0.25;
    constexpr int kSeriesTerms = 18;
    constexpr double kPotentialDenominator = 0.5;
    constexpr double kAccelerationDenominator = 1.5;
    constexpr double kForceCoefficient = 4.0 / 3.0;
    PROFILE();
    double const alphaSquared = alpha * alpha;
    double const argument = alphaSquared * radiusSquared;
    if (argument <= kSeriesThreshold) {
        double potentialTerm = 1.0;
        double potentialSum = 1.0;
        double forceTerm = 1.0;
        double forceSum = 1.0;
        for (int termIndex = 1; termIndex < kSeriesTerms; termIndex++) {
            potentialTerm *= argument / (static_cast<double>(termIndex) + kPotentialDenominator);
            forceTerm *= argument / (static_cast<double>(termIndex) + kAccelerationDenominator);
            potentialSum += potentialTerm;
            forceSum += forceTerm;
        }
        double const damping = ::exp(-argument);
        potential += 2.0 * alpha * kDirectInverseRootPi * damping * potentialSum;
        accelerationScale += kForceCoefficient * alpha * alphaSquared * kDirectInverseRootPi * damping * forceSum;
    } else {
        double const radius = ::sqrt(radiusSquared);
        double const inverseRadius = 1.0 / radius;
        double const alphaRadius = alpha * radius;
        double const errorFunction = ::erf(alphaRadius);
        potential += errorFunction * inverseRadius;
        accelerationScale += (errorFunction - 2.0 * alphaRadius * kDirectInverseRootPi * ::exp(-argument)) * inverseRadius * inverseRadius * inverseRadius;
    }
}

COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline void directRealImage(
    double const displacement[NDIM], double const image[NDIM], double alpha, DirectSolveField& field, double epsilonSquared = 0.0) {
    PROFILE();
    double shifted[NDIM]{};
    double radiusSquared = epsilonSquared;
    for (size_t axis = 0; axis < NDIM; axis++) {
        shifted[axis] = displacement[axis] + image[axis];
        radiusSquared += shifted[axis] * shifted[axis];
    }
    double const inverseRadius = 1.0 / ::sqrt(expectPositive(radiusSquared));
    double const alphaRadius = alpha / inverseRadius;
    double const complement = ::erfc(alphaRadius);
    double const damping = ::exp(-alphaRadius * alphaRadius);
    field.potential -= complement * inverseRadius;
    double const scale = -(complement * inverseRadius + 2.0 * alpha * kDirectInverseRootPi * damping) * inverseRadius * inverseRadius;
    for (size_t axis = 0; axis < NDIM; axis++) {
        field.acceleration[axis] += scale * shifted[axis];
    }
}

// Unit source mass. The real-space sum is evaluated in image pairs; Fourier
// waves are likewise paired. The neutralizing background is already included
// by the reciprocal sum and pi/alpha^2 constant, so no extra quadratic term is
// added here. Unlike STF PMMM this evaluator retains the complete radial field.
COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline DirectSolveField directPeriodicPair(Position target, Position source, bool self, double softeningLength,
    int ferrersOrder, DirectEwaldParameters const& parameters, SofteningModel model = SofteningModel::ferrers, DirectEwaldWeights const* weights = nullptr) {
    static_assert(NDIM == 3, "The cubic direct Ewald sum requires three spatial dimensions");
    PROFILE();
    double const displacement[NDIM] = {directDisplacement(target.x, source.x), directDisplacement(target.y, source.y), directDisplacement(target.z, source.z)};
    double radiusSquared = 0.0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        radiusSquared += displacement[axis] * displacement[axis];
    }
    double const epsilonSquared = model == SofteningModel::plummer ? softeningLength * softeningLength : 0.0;
    double const softenedRadiusSquared = radiusSquared + epsilonSquared;
    DirectSolveField field{};
    double accelerationScale = 0.0;
    if (!self) {
        double const inverseSoftening = 1.0 / softeningLength;
        double const inverseSofteningSquared = inverseSoftening * inverseSoftening;
        if (model == SofteningModel::plummer) {
            double const inverseRadius = 1.0 / ::sqrt(softenedRadiusSquared);
            field.potential = -inverseRadius;
            accelerationScale = -inverseRadius * inverseRadius * inverseRadius;
        } else if (radiusSquared < softeningLength * softeningLength) {
            double const t = 1.0 - radiusSquared * inverseSofteningSquared;
            field.potential = -inverseSoftening * directFerrersPotentialPolynomial(t, ferrersOrder);
            accelerationScale = -inverseSoftening * inverseSofteningSquared * directFerrersForcePolynomial(t, ferrersOrder);
        } else {
            double const inverseRadius = 1.0 / ::sqrt(radiusSquared);
            field.potential = -inverseRadius;
            accelerationScale = -inverseRadius * inverseRadius * inverseRadius;
        }
    }
    directPrimaryCorrection(softenedRadiusSquared, parameters.alpha, field.potential, accelerationScale);
    for (size_t axis = 0; axis < NDIM; axis++) {
        field.acceleration[axis] = accelerationScale * displacement[axis];
    }
    for (int imageZ = -parameters.imageCutoff; imageZ <= parameters.imageCutoff; imageZ++) {
        for (int imageY = -parameters.imageCutoff; imageY <= parameters.imageCutoff; imageY++) {
            for (int imageX = -parameters.imageCutoff; imageX <= parameters.imageCutoff; imageX++) {
                if (imageX < 0 || (imageX == 0 && imageY < 0) || (imageX == 0 && imageY == 0 && imageZ <= 0)) {
                    continue;
                }
                double const positiveImage[NDIM] = {static_cast<double>(imageX), static_cast<double>(imageY), static_cast<double>(imageZ)};
                double const negativeImage[NDIM] = {-positiveImage[0], -positiveImage[1], -positiveImage[2]};
                DirectSolveField pair{};
                directRealImage(displacement, positiveImage, parameters.alpha, pair, epsilonSquared);
                directRealImage(displacement, negativeImage, parameters.alpha, pair, epsilonSquared);
                field.potential += pair.potential;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    field.acceleration[axis] += pair.acceleration[axis];
                }
            }
        }
    }
    double const inverseAlphaSquared = 1.0 / (parameters.alpha * parameters.alpha);
    double const z = epsilonSquared / inverseAlphaSquared;
    double const zeroMode = model == SofteningModel::plummer ? (weights == nullptr ? plummerZeroModeIntegral(z) : weights->zeroMode) : 1.0;
    field.potential += kDirectPi * inverseAlphaSquared * zeroMode;
    for (int waveZ = -parameters.waveCutoff; waveZ <= parameters.waveCutoff; waveZ++) {
        for (int waveY = -parameters.waveCutoff; waveY <= parameters.waveCutoff; waveY++) {
            for (int waveX = -parameters.waveCutoff; waveX <= parameters.waveCutoff; waveX++) {
                if (waveX < 0 || (waveX == 0 && waveY < 0) || (waveX == 0 && waveY == 0 && waveZ <= 0)) {
                    continue;
                }
                double const wave[NDIM] = {static_cast<double>(waveX), static_cast<double>(waveY), static_cast<double>(waveZ)};
                double waveSquared = 0.0;
                double phase = 0.0;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    waveSquared += wave[axis] * wave[axis];
                    phase += wave[axis] * displacement[axis];
                }
                phase *= 2.0 * kDirectPi;
                double const b = kDirectPi * kDirectPi * waveSquared * inverseAlphaSquared;
                double damping = ::exp(-b) / waveSquared;
                if (model == SofteningModel::plummer) {
                    damping *= weights != nullptr && waveSquared <= static_cast<double>(kDirectCachedWaveSquared) ?
                        weights->reciprocal[static_cast<size_t>(waveSquared)] :
                        plummerReciprocalIntegral(b, z);
                }
                field.potential -= (2.0 / kDirectPi) * damping * ::cos(phase);
                double const scale = -4.0 * damping * ::sin(phase);
                for (size_t axis = 0; axis < NDIM; axis++) {
                    field.acceleration[axis] += scale * wave[axis];
                }
            }
        }
    }
    // Cubic symmetry makes the regular field exactly zero at the origin.
    // Fused operations may otherwise leave tiny remnants after image pairs
    // cancel, which would give a spurious normalization for a zero-force case.
    if (radiusSquared == 0.0) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            field.acceleration[axis] = 0.0;
        }
    }
    return field;
}

// Each partial result includes source mass and the complete potential gauge,
// allowing CUDA to distribute sources across a cooperating block of threads.
COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline DirectSolveField directSolveSources(DirectSolveRequest const& request, DirectEwaldParameters const& parameters,
    size_t sinkOrdinal, size_t sourceBegin, size_t sourceStride, DirectEwaldWeights const* weights = nullptr) {
    PROFILE();
    FpeGuard fpeGuard;
    expectIndex(sinkOrdinal, request.sinkCount);
    expectPositive(sourceStride);
    size_t const sinkIndex = expectIndex(request.sinks[sinkOrdinal], request.sourceCount);
    Position const target = request.sources[sinkIndex];
    DirectSolveField result{};
    DirectSolveField compensation{};
    for (size_t sourceIndex = sourceBegin; sourceIndex < request.sourceCount;) {
        DirectSolveField const pair = directPeriodicPair(target, request.sources[sourceIndex], sourceIndex == sinkIndex, request.softeningLength,
            request.ferrersOrder, parameters, request.softeningModel, weights);
        double const potentialTerm = pair.potential - compensation.potential;
        double const potentialSum = result.potential + potentialTerm;
        compensation.potential = (potentialSum - result.potential) - potentialTerm;
        result.potential = potentialSum;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const accelerationTerm = pair.acceleration[axis] - compensation.acceleration[axis];
            double const accelerationSum = result.acceleration[axis] + accelerationTerm;
            compensation.acceleration[axis] = (accelerationSum - result.acceleration[axis]) - accelerationTerm;
            result.acceleration[axis] = accelerationSum;
        }
        if (sourceStride >= request.sourceCount - sourceIndex) {
            break;
        }
        sourceIndex += sourceStride;
    }
    result.potential *= request.particleMass;
    for (size_t axis = 0; axis < NDIM; axis++) {
        result.acceleration[axis] *= request.particleMass;
    }
    return result;
}

COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE inline DirectSolveField directSolveSink(
    DirectSolveRequest const& request, DirectEwaldParameters const& parameters, size_t sinkOrdinal, DirectEwaldWeights const* weights = nullptr) {
    PROFILE();
    return directSolveSources(request, parameters, sinkOrdinal, 0, 1, weights);
}

}    // namespace cosmictiger::gravity::detail

#undef COSMICTIGER_DIRECT_SOLVE_HOST_DEVICE
