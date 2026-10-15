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

#include <cosmictiger/gravity/pmmm/pmmm_plummer_green.hpp>
#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/profiler.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <stddef.h>
#include <vector>

namespace {

namespace gravity = cosmictiger::gravity;
namespace symmetric = gravity::symmetric;

// An independent unsplit Fourier sum. The exact 3D Fourier transform of the
// Plummer kernel is -4*pi*epsilon*K_1(epsilon*|k|)/|k| (DLMF 10.32.10).
// A fairly large epsilon makes this reference exponentially convergent.
template <int maxRank>
gravity::PmmmPlummerGreenCoefficients<maxRank> plummerFourierReference(std::array<double, NDIM> const& separation, double epsilon, double spacing) {
    constexpr int kCutoff = 28;
    constexpr double kTwoPi = 2.0 * std::numbers::pi;
    constexpr auto kExponents = symmetric::exponents<maxRank>();
    PROFILE();
    std::vector<double> weights(NDIM * kCutoff * kCutoff + 1);
    for (size_t squaredWave = 1; squaredWave < weights.size(); squaredWave++) {
        double const waveMagnitude = std::sqrt(static_cast<double>(squaredWave));
        weights[squaredWave] = -4.0 * epsilon * std::cyl_bessel_k(1.0, kTwoPi * epsilon * waveMagnitude) / waveMagnitude;
    }
    gravity::PmmmPlummerGreenCoefficients<maxRank> result{};
    for (int nz = -kCutoff; nz <= kCutoff; nz++) {
        for (int ny = -kCutoff; ny <= kCutoff; ny++) {
            for (int nx = -kCutoff; nx <= kCutoff; nx++) {
                if (nx < 0 || (nx == 0 && ny < 0) || (nx == 0 && ny == 0 && nz <= 0)) {
                    continue;
                }
                std::array<int, NDIM> const wave = {nx, ny, nz};
                double phase = 0.0;
                std::array<std::array<double, maxRank + 1>, NDIM> powers{};
                for (int axis = 0; axis < NDIM; axis++) {
                    phase += kTwoPi * static_cast<double>(wave[axis]) * separation[axis];
                    powers[axis][0] = 1.0;
                    for (int rank = 1; rank <= maxRank; rank++) {
                        powers[axis][rank] = powers[axis][rank - 1] * kTwoPi * spacing * static_cast<double>(wave[axis]);
                    }
                }
                double const weight = weights[static_cast<size_t>(nx * nx + ny * ny + nz * nz)];
                for (size_t index = 0; index < kExponents.size(); index++) {
                    auto const& exponent = kExponents[index];
                    double const derivativePhase = std::cos(phase + static_cast<double>(exponent.rank) * std::numbers::pi / 2.0);
                    result[index] += weight * derivativePhase * powers[0][exponent.nx] * powers[1][exponent.ny] * powers[2][exponent.nz];
                }
            }
        }
    }
    return result;
}

TEST(PlummerGreen, ZeroModeIntegralMatchesIndependentExponentialIntegral) {
    constexpr double kRelativeTolerance = 2.0e-13;
    constexpr std::array<double, 11> kArguments = {1e-12, 1e-6, 0.01, 0.25, 0.9, 1.0, 1.01, 2.0, 4.0, 10.0, 32.0};
    PROFILE();
    EXPECT_EQ(gravity::detail::plummerZeroModeIntegral(0.0), 1.0);
    for (double argument : kArguments) {
        double const reference = std::exp(-argument) + argument * std::expint(-argument);
        EXPECT_NEAR(gravity::detail::plummerZeroModeIntegral(argument), reference, kRelativeTolerance * reference) << argument;
    }
}

TEST(PlummerGreen, AllCartesianDerivativesMatchIndependentUnsplitFourierSum) {
    constexpr int kRank = 4;
    constexpr double kEpsilon = 0.3;
    constexpr double kSpacing = 0.125;
    constexpr double kTolerance = 3e-12;
    constexpr std::array<int, NDIM> kOffset = {3, 1, -2};
    constexpr std::array<double, NDIM> kSeparation = {0.375, 0.125, -0.25};
    PROFILE();
    auto const actual = gravity::pmmmPlummerPeriodicGreen<kRank, 1>(kOffset, 8, kSpacing, kEpsilon, 5, 5, 1.8);
    auto const reference = plummerFourierReference<kRank>(kSeparation, kEpsilon, kSpacing);
    for (size_t index = 0; index < actual.size(); index++) {
        EXPECT_NEAR(actual[index], reference[index], kTolerance * (1.0 + std::abs(reference[index]))) << index;
    }
    // A softened periodic Green function is not harmonic even in far cells.
    double const trace = actual[symmetric::index(2, 0, 0)] + actual[symmetric::index(0, 2, 0)] + actual[symmetric::index(0, 0, 2)];
    EXPECT_GT(std::abs(trace), 1e-3);
}

TEST(PlummerGreen, HighOrderDerivativesAreIndependentOfEwaldSplit) {
    constexpr int kRank = 8;
    constexpr double kTolerance = 2e-11;
    constexpr std::array<std::array<int, NDIM>, 3> kOffsets = {{{0, 0, 0}, {1, -1, 1}, {3, 1, -2}}};
    PROFILE();
    gravity::PmmmPlummerGreenEvaluator<kRank, 1> const first(8, 0.125, 0.02, 6, 6, 1.4);
    gravity::PmmmPlummerGreenEvaluator<kRank, 1> const second(8, 0.125, 0.02, 6, 6, 2.8);
    for (auto const& offset : kOffsets) {
        auto const a = first.evaluate(offset);
        auto const b = second.evaluate(offset);
        for (size_t index = 0; index < a.size(); index++) {
            EXPECT_NEAR(a[index], b[index], kTolerance * (1.0 + std::abs(a[index]))) << index;
        }
    }
}

TEST(PlummerGreen, NearMaskRemovesOnlyPrimaryPlummerPotentialAndAllDerivatives) {
    constexpr int kRank = 5;
    constexpr double kEpsilon = 0.035;
    constexpr double kSpacing = 0.125;
    constexpr std::array<double, NDIM> kCells = {1.0, -1.0, 1.0};
    constexpr std::array<double, NDIM> kSeparation = {0.125, -0.125, 0.125};
    constexpr double kTolerance = 3e-12;
    PROFILE();
    gravity::PmmmPlummerGreenEvaluator<kRank, 1> const evaluator(8, kSpacing, kEpsilon);
    auto const complete = evaluator.evaluateCells(kCells, false);
    auto const excluded = evaluator.evaluateCells(kCells, true);
    auto const primary = symmetric::greenDerivatives<double, kRank>(kSeparation, kEpsilon, kSpacing);
    for (size_t index = 0; index < complete.size(); index++) {
        EXPECT_NEAR(complete[index] - excluded[index], primary[index], kTolerance * (1.0 + std::abs(primary[index]))) << index;
    }
    EXPECT_GT(std::abs(excluded[0]), 1.0);
}

TEST(PlummerGreen, OriginLimitRetainsImagesAndIsRegularAsSofteningVanishes) {
    constexpr int kRank = 6;
    constexpr double kSpacing = 0.125;
    constexpr double kTinyEpsilon = 1e-14;
    constexpr double kTolerance = 2e-12;
    constexpr auto kExponents = symmetric::exponents<kRank>();
    PROFILE();
    auto const unsoftened = gravity::pmmmPlummerPeriodicGreen<kRank, 1>({0, 0, 0}, 8, kSpacing, 0.0);
    auto const tiny = gravity::pmmmPlummerPeriodicGreen<kRank, 1>({0, 0, 0}, 8, kSpacing, kTinyEpsilon);
    for (size_t index = 0; index < tiny.size(); index++) {
        EXPECT_NEAR(tiny[index], unsoftened[index], kTolerance * (1.0 + std::abs(unsoftened[index]))) << index;
        auto const& exponent = kExponents[index];
        if (exponent.nx % 2 != 0 || exponent.ny % 2 != 0 || exponent.nz % 2 != 0) {
            EXPECT_EQ(tiny[index], 0.0);
        }
    }
    double const trace = unsoftened[symmetric::index(2, 0, 0)] + unsoftened[symmetric::index(0, 2, 0)] + unsoftened[symmetric::index(0, 0, 2)];
    EXPECT_NEAR(trace, -4.0 * std::numbers::pi * kSpacing * kSpacing, kTolerance);
    EXPECT_GT(std::abs(tiny[0]), 1.0);
}

TEST(PlummerGreen, MixedDerivativeAgreesWithContinuousPotentialDifferences) {
    constexpr double kDelta = 1e-4;
    constexpr double kTolerance = 2e-8;
    constexpr std::array<double, NDIM> kCells = {1.15, -0.8, 0.7};
    constexpr auto kExponents = symmetric::exponents<3>();
    PROFILE();
    gravity::PmmmPlummerGreenEvaluator<4, 1> const evaluator(8, 0.125, 0.03);
    auto const center = evaluator.evaluateCells(kCells, true);
    for (int axis = 0; axis < NDIM; axis++) {
        auto plus = kCells;
        auto minus = kCells;
        plus[axis] += kDelta;
        minus[axis] -= kDelta;
        auto const upper = evaluator.evaluateCells(plus, true);
        auto const lower = evaluator.evaluateCells(minus, true);
        for (size_t index = 0; index < kExponents.size(); index++) {
            auto const& exponent = kExponents[index];
            size_t const derivative = symmetric::index(exponent.nx + (axis == 0), exponent.ny + (axis == 1), exponent.nz + (axis == 2));
            double const difference = (upper[index] - lower[index]) / (2.0 * kDelta);
            EXPECT_NEAR(difference, center[derivative], kTolerance * (1.0 + std::abs(center[derivative]))) << axis << ":" << index;
        }
    }
}

TEST(PlummerGreen, PeriodicFilterBlockPreservesBatchAndLocalityLayout) {
    constexpr size_t kSide = 8;
    constexpr size_t kLocalitySide = 2;
    constexpr size_t kLocality = 1;
    constexpr size_t kBatch = 4;
    constexpr double kSpacing = 0.125;
    constexpr double kEpsilon = 0.04;
    constexpr size_t kFirst = 6;
    constexpr size_t kBlockSide = kSide / kLocalitySide;
    PROFILE();
    auto const block = gravity::pmmmPlummerPeriodicGreenFilterBlock<2, 1, kBatch>(kSide, kLocalitySide, kLocality, kSpacing, kEpsilon, kFirst, 3, 3);
    ASSERT_EQ(block.size(), kBlockSide * kBlockSide * kBlockSide);
    // Locality one occupies z=[4,8), with x fastest inside the local block.
    auto const expected = gravity::pmmmPlummerPeriodicGreen<2, 1>({2, 1, 7}, kSide, kSpacing, kEpsilon, 3, 3);
    size_t const index = (3 * kBlockSide + 1) * kBlockSide + 2;
    for (size_t lane = 0; lane < kBatch; lane++) {
        EXPECT_NEAR(block[index][lane], expected[kFirst + lane], 2e-13);
    }
}

}    // namespace
