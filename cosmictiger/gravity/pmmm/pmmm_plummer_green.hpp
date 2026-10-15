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

#include <cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_plummer_ewald.hpp>
#include <cosmictiger/gravity/symmetric/multipole.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <vector>

namespace cosmictiger::gravity {

template <int maxRank>
using PmmmPlummerGreenCoefficients = symmetric::Expansion<double, maxRank>;

namespace detail {
    struct PmmmPlummerRadialTerm {
        size_t output;
        std::array<int, NDIM> powers;
        int radialRank;
        double factor;
    };

    template <int maxRank>
    constexpr size_t pmmmPlummerRadialTermCount() {
        constexpr auto kExponents = symmetric::exponents<maxRank>();
        PROFILE();
        size_t result = 0;
        for (auto const& exponent : kExponents) {
            result += static_cast<size_t>((exponent.nx / 2 + 1) * (exponent.ny / 2 + 1) * (exponent.nz / 2 + 1));
        }
        return result;
    }

    template <int maxRank>
    constexpr auto pmmmPlummerRadialTerms() {
        constexpr auto kExponents = symmetric::exponents<maxRank>();
        constexpr size_t kTermCount = pmmmPlummerRadialTermCount<maxRank>();
        constexpr auto kFactorials = [] {
            PROFILE();
            std::array<double, maxRank + 1> values{};
            values[0] = 1.0;
            for (int rank = 1; rank <= maxRank; rank++) {
                values[rank] = static_cast<double>(rank) * values[rank - 1];
            }
            return values;
        }();
        constexpr auto kPowersOfTwo = [] {
            PROFILE();
            std::array<double, maxRank + 1> values{};
            values[0] = 1.0;
            for (int rank = 1; rank <= maxRank; rank++) {
                values[rank] = 2.0 * values[rank - 1];
            }
            return values;
        }();
        PROFILE();
        std::array<PmmmPlummerRadialTerm, kTermCount> result{};
        size_t term = 0;
        for (size_t output = 0; output < kExponents.size(); output++) {
            auto const& exponent = kExponents[output];
            for (int bz = 0; bz <= exponent.nz / 2; bz++) {
                for (int by = 0; by <= exponent.ny / 2; by++) {
                    for (int bx = 0; bx <= exponent.nx / 2; bx++) {
                        int const nx = exponent.nx - 2 * bx;
                        int const ny = exponent.ny - 2 * by;
                        int const nz = exponent.nz - 2 * bz;
                        double const factor = kFactorials[exponent.nx] * kFactorials[exponent.ny] * kFactorials[exponent.nz] * kPowersOfTwo[nx + ny + nz] /
                            (kFactorials[bx] * kFactorials[by] * kFactorials[bz] * kFactorials[nx] * kFactorials[ny] * kFactorials[nz]);
                        result[term++] = {output, {nx, ny, nz}, exponent.rank - bx - by - bz, factor};
                    }
                }
            }
        }
        return result;
    }

    template <int maxRank>
    inline constexpr auto kPmmmPlummerRadialTerms = pmmmPlummerRadialTerms<maxRank>();

    // Derivatives with respect to s=r^2+epsilon^2 of -erfc(alpha*sqrt(s))/sqrt(s).
    // For an excluded primary use +erf(alpha*sqrt(s))/sqrt(s) directly. The
    // lower-integral series remains regular even at s=0; evaluating a tiny
    // incomplete-gamma ratio times an enormous inverse power would not.
    template <int maxRank>
    std::array<double, maxRank + 1> pmmmPlummerRadialDerivatives(double squaredRadius, double alpha, bool subtractPrimary) {
        constexpr double kInverseRootPi = 0.564189583547756286948079451560772586;
        constexpr double kSeriesThreshold = 32.0;
        constexpr double kTolerance = std::numeric_limits<double>::epsilon();
        constexpr int kMaximumTerms = 256;
        PROFILE();
        std::array<double, maxRank + 1> result{};
        double const alphaSquared = alpha * alpha;
        double const argument = alphaSquared * squaredRadius;
        if (subtractPrimary && argument <= kSeriesThreshold) {
            double prefactor = alpha * kInverseRootPi * std::exp(-argument);
            for (int rank = 0; rank <= maxRank; rank++) {
                double const gammaRank = static_cast<double>(rank) + 0.5;
                double term = 1.0;
                double sum = 1.0;
                for (int n = 1; n <= kMaximumTerms; n++) {
                    term *= argument / (gammaRank + static_cast<double>(n));
                    sum += term;
                    if (term <= kTolerance * sum) {
                        break;
                    }
                }
                result[rank] = prefactor * sum / gammaRank;
                prefactor *= -alphaSquared;
            }
            return result;
        }
        expectPositive(squaredRadius);
        auto const upper = pmmmUpperGammaRatios<maxRank>(argument);
        double const inverseSquaredRadius = 1.0 / squaredRadius;
        double derivative = -std::sqrt(inverseSquaredRadius);
        for (int rank = 0; rank <= maxRank; rank++) {
            result[rank] = derivative * (subtractPrimary ? upper[rank] - 1.0 : upper[rank]);
            derivative *= -(static_cast<double>(rank) + 0.5) * inverseSquaredRadius;
        }
        return result;
    }

    template <int maxRank>
    PmmmPlummerGreenCoefficients<maxRank> pmmmPlummerRealDerivatives(
        std::array<double, NDIM> const& separation, double epsilon, double alpha, bool subtractPrimary) {
        PROFILE();
        std::array<std::array<double, maxRank + 1>, NDIM> powers{};
        double squaredRadius = epsilon * epsilon;
        for (int axis = 0; axis < NDIM; axis++) {
            powers[axis][0] = 1.0;
            for (int rank = 1; rank <= maxRank; rank++) {
                powers[axis][rank] = powers[axis][rank - 1] * separation[axis];
            }
            squaredRadius += separation[axis] * separation[axis];
        }
        auto const radial = pmmmPlummerRadialDerivatives<maxRank>(squaredRadius, alpha, subtractPrimary);
        PmmmPlummerGreenCoefficients<maxRank> result{};
        for (auto const& term : kPmmmPlummerRadialTerms<maxRank>) {
            result[term.output] += term.factor * radial[term.radialRank] * powers[0][term.powers[0]] * powers[1][term.powers[1]] * powers[2][term.powers[2]];
        }
        return result;
    }
}    // namespace detail

// Full symmetric Cartesian derivatives of the zero-mean periodic Plummer
// Green function. The Laplace representation of -1/sqrt(r^2+epsilon^2) is
// split at t=alpha^2 and Poisson-summed below that point. Both sums therefore
// retain the Plummer kernel, including its noncompact tails and nonzero trace.
// No STF background polynomial is added to these coefficients.
template <int maxRank, int neighborWidth = kPmmmNeighborWidth>
class PmmmPlummerGreenEvaluator {
public:
    PmmmPlummerGreenEvaluator(size_t sideLength, double spacing, double epsilon, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5)
      : sideLength_(sideLength)
      , spacing_(spacing)
      , epsilonCells_(epsilon / expectPositive(spacing))
      , alphaCells_(alphaBox / static_cast<double>(expectPositive(sideLength)))
      , imageCutoff_(imageCutoff) {
        static constexpr auto kExponents = symmetric::exponents<maxRank>();
        static_assert(NDIM == 3, "The cubic Plummer Ewald sum requires three spatial dimensions");
        static_assert(maxRank >= 0 && maxRank <= 18, "The full Cartesian Ewald rank must lie in [0,18]");
        static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
        PROFILE();
        FpeGuard fpeGuard;
        expectRange(static_cast<size_t>(2 * neighborWidth + 1), sideLength, static_cast<size_t>(std::numeric_limits<int>::max()));
        expectNonNegative(epsilon);
        expectNonNegative(imageCutoff);
        expectNonNegative(waveCutoff);
        expectPositive(alphaBox);
        double const period = spacing * static_cast<double>(sideLength);
        double const z = epsilonCells_ * epsilonCells_ * alphaCells_ * alphaCells_;
        zeroMode_ = detail::kPmmmPi * detail::plummerZeroModeIntegral(z) / (period * alphaBox * alphaBox);
        double const waveRatio = 2.0 * detail::kPmmmPi / static_cast<double>(sideLength);
        std::vector<double> reciprocalWeights(static_cast<size_t>(NDIM * waveCutoff * waveCutoff + 1));
        for (size_t squaredWave = 1; squaredWave < reciprocalWeights.size(); squaredWave++) {
            double const b = detail::kPmmmPi * detail::kPmmmPi * static_cast<double>(squaredWave) / (alphaBox * alphaBox);
            reciprocalWeights[squaredWave] =
                -2.0 * std::exp(-b) * detail::plummerReciprocalIntegral(b, z) / (detail::kPmmmPi * period * static_cast<double>(squaredWave));
        }
        for (int nz = -waveCutoff; nz <= waveCutoff; nz++) {
            for (int ny = -waveCutoff; ny <= waveCutoff; ny++) {
                for (int nx = -waveCutoff; nx <= waveCutoff; nx++) {
                    if (nx < 0 || (nx == 0 && ny < 0) || (nx == 0 && ny == 0 && nz <= 0)) {
                        continue;
                    }
                    Wave wave{};
                    wave.vector = {waveRatio * static_cast<double>(nx), waveRatio * static_cast<double>(ny), waveRatio * static_cast<double>(nz)};
                    std::array<std::array<double, maxRank + 1>, NDIM> powers{};
                    for (int axis = 0; axis < NDIM; axis++) {
                        powers[axis][0] = 1.0;
                        for (int rank = 1; rank <= maxRank; rank++) {
                            powers[axis][rank] = powers[axis][rank - 1] * wave.vector[axis];
                        }
                    }
                    double const weight = reciprocalWeights[static_cast<size_t>(nx * nx + ny * ny + nz * nz)];
                    for (size_t index = 0; index < kExponents.size(); index++) {
                        auto const& exponent = kExponents[index];
                        wave.coefficients[index] = weight * powers[0][exponent.nx] * powers[1][exponent.ny] * powers[2][exponent.nz];
                    }
                    waves_.push_back(wave);
                }
            }
        }
    }

    // Continuous cell-coordinate version is also useful for independent
    // derivative checks. The caller chooses whether to remove the primary.
    PmmmPlummerGreenCoefficients<maxRank> evaluateCells(std::array<double, NDIM> const& separation, bool subtractPrimary) const {
        static constexpr auto kExponents = symmetric::exponents<maxRank>();
        PROFILE();
        FpeGuard fpeGuard;
        PmmmPlummerGreenCoefficients<maxRank> result{};
        PmmmPlummerGreenCoefficients<maxRank> compensation{};
        double const inverseSpacing = 1.0 / spacing_;
        for (int nz = -imageCutoff_; nz <= imageCutoff_; nz++) {
            for (int ny = -imageCutoff_; ny <= imageCutoff_; ny++) {
                for (int nx = -imageCutoff_; nx <= imageCutoff_; nx++) {
                    bool const primary = nx == 0 && ny == 0 && nz == 0;
                    std::array<double, NDIM> const shifted = {separation[0] + static_cast<double>(nx) * static_cast<double>(sideLength_),
                        separation[1] + static_cast<double>(ny) * static_cast<double>(sideLength_),
                        separation[2] + static_cast<double>(nz) * static_cast<double>(sideLength_)};
                    auto const real = detail::pmmmPlummerRealDerivatives<maxRank>(shifted, epsilonCells_, alphaCells_, primary && subtractPrimary);
                    for (size_t index = 0; index < result.size(); index++) {
                        double const increment = inverseSpacing * real[index] - compensation[index];
                        double const sum = result[index] + increment;
                        compensation[index] = (sum - result[index]) - increment;
                        result[index] = sum;
                    }
                }
            }
        }
        result[0] += zeroMode_;
        for (auto const& wave : waves_) {
            double phase = 0.0;
            for (int axis = 0; axis < NDIM; axis++) {
                phase += wave.vector[axis] * separation[axis];
            }
            double const cosine = std::cos(phase);
            double const sine = std::sin(phase);
            for (size_t index = 0; index < result.size(); index++) {
                result[index] += wave.coefficients[index] * detail::pmmmFourierPhase(kExponents[index].rank, cosine, sine);
            }
        }
        return result;
    }

    PmmmPlummerGreenCoefficients<maxRank> evaluate(std::array<int, NDIM> const& displacement) const {
        static constexpr auto kExponents = symmetric::exponents<maxRank>();
        PROFILE();
        std::array<int, NDIM> offset{};
        std::array<double, NDIM> separation{};
        int64_t const integerSide = static_cast<int64_t>(sideLength_);
        for (int axis = 0; axis < NDIM; axis++) {
            int64_t wrapped = static_cast<int64_t>(displacement[axis]) % integerSide;
            if (wrapped < 0) {
                wrapped += integerSide;
            }
            if (wrapped > integerSide / 2) {
                wrapped -= integerSide;
            }
            offset[axis] = static_cast<int>(wrapped);
            separation[axis] = static_cast<double>(wrapped);
        }
        auto result = evaluateCells(separation, pmmmNear<neighborWidth>(offset));
        for (size_t index = 0; index < result.size(); index++) {
            auto const& exponent = kExponents[index];
            std::array<int, NDIM> const powers = {exponent.nx, exponent.ny, exponent.nz};
            for (int axis = 0; axis < NDIM; axis++) {
                if (powers[axis] % 2 != 0 && (offset[axis] == 0 || 2 * static_cast<int64_t>(offset[axis]) == integerSide)) {
                    result[index] = 0.0;
                    break;
                }
            }
        }
        return result;
    }

private:
    struct Wave {
        std::array<double, NDIM> vector;
        PmmmPlummerGreenCoefficients<maxRank> coefficients;
    };

    size_t sideLength_;
    double spacing_;
    double epsilonCells_;
    double alphaCells_;
    int imageCutoff_;
    double zeroMode_;
    std::vector<Wave> waves_;
};

template <int maxRank, int neighborWidth = kPmmmNeighborWidth>
PmmmPlummerGreenCoefficients<maxRank> pmmmPlummerPeriodicGreen(std::array<int, NDIM> const& displacement, size_t sideLength, double spacing, double epsilon,
    int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5) {
    PROFILE();
    PmmmPlummerGreenEvaluator<maxRank, neighborWidth> const evaluator(sideLength, spacing, epsilon, imageCutoff, waveCutoff, alphaBox);
    return evaluator.evaluate(displacement);
}

template <int maxRank, int neighborWidth = kPmmmNeighborWidth, size_t batchSize = 1>
std::vector<std::array<double, batchSize>> pmmmPlummerPeriodicGreenFilterBlock(size_t sideLength, size_t localitySide, size_t localityId, double spacing,
    double epsilon, size_t firstCoefficient, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5, PmmmProgressObserver const& observer = {}) {
    constexpr size_t kEwaldChunkSize = 4;
    static_assert(batchSize > 0, "PMMM FFT batch must have at least one lane");
    PROFILE();
    expectRange(static_cast<size_t>(2 * neighborWidth + 1), sideLength, static_cast<size_t>(std::numeric_limits<int>::max()));
    expect(std::has_single_bit(sideLength));
    expectRange(size_t(1), localitySide, sideLength);
    expect(std::has_single_bit(localitySide));
    expect(sideLength % localitySide == 0);
    expect(localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expect(localitySide * localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expectIndex(localityId, localitySide * localitySide * localitySide);
    expectIndex(firstCoefficient, symmetric::coefficientCount(maxRank));
    size_t const blockSide = sideLength / localitySide;
    expect(blockSide <= std::numeric_limits<size_t>::max() / blockSide);
    expect(blockSide * blockSide <= std::numeric_limits<size_t>::max() / blockSide);
    std::vector<std::array<double, batchSize>> block(blockSide * blockSide * blockSide);
    size_t const blockX = localityId / (localitySide * localitySide);
    size_t const blockY = (localityId / localitySide) % localitySide;
    size_t const blockZ = localityId % localitySide;
    PmmmPlummerGreenEvaluator<maxRank, neighborWidth> const evaluator(sideLength, spacing, epsilon, imageCutoff, waveCutoff, alphaBox);
    detail::PmmmProgressCounter progress(observer, PmmmProgressStage::ewaldFilters, block.size());
    ::cosmictiger::detail::ParallelChunkPlan const chunks(block.size(), kEwaldChunkSize);
    ::cosmictiger::detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    size_t const x = index % blockSide;
                    size_t const y = (index / blockSide) % blockSide;
                    size_t const z = index / (blockSide * blockSide);
                    std::array<int, NDIM> const offset = {
                        static_cast<int>(blockX * blockSide + x), static_cast<int>(blockY * blockSide + y), static_cast<int>(blockZ * blockSide + z)};
                    auto const coefficients = evaluator.evaluate(offset);
                    for (size_t lane = 0; lane < batchSize && lane < coefficients.size() - firstCoefficient; lane++) {
                        block[index][lane] = coefficients[firstCoefficient + lane];
                    }
                    progress.advance(1);
                }
            },
            "pmmmPlummerPeriodicGreen.parallelForChunks"));
    progress.finish();
    return block;
}

}    // namespace cosmictiger::gravity
