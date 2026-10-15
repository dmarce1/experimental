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

#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stddef.h>

namespace {

namespace symmetric = cosmictiger::gravity::symmetric;

struct Particle {
    double mass;
    std::array<double, NDIM> offset;
};

inline constexpr std::array<Particle, 4> kParticles{
    {{0.7, {0.011, -0.023, 0.014}}, {1.1, {-0.025, 0.017, 0.008}}, {0.3, {0.031, 0.004, -0.019}}, {0.9, {-0.016, -0.027, -0.021}}}};

template <int order>
constexpr auto binomialCoefficients() {
    PROFILE();
    std::array<double, order + 1> result{};
    result[0] = 1;
    for (int rank = 1; rank <= order; rank++) {
        result[rank] = result[rank - 1] * (-0.5 - double(rank - 1)) / double(rank);
    }
    return result;
}

// Independent oracle: multiply multivariate polynomials in the scalar binomial
// expansion -(1+q)^(-1/2)/rho, where q=(2 r.h+h.h)/rho^2.
// The production implementation instead differentiates a first-order identity.
template <int order>
symmetric::Expansion<double, order> binomialGreen(std::array<double, NDIM> const& separation, double epsilon, double spacing) {
    constexpr auto kExponents = symmetric::exponents<order>();
    constexpr auto kBinomials = binomialCoefficients<order>();
    constexpr symmetric::Combinatorics<order> kCombinatorics{};
    PROFILE();
    double radiusSquared = epsilon * epsilon;
    for (int d = 0; d < NDIM; d++) {
        radiusSquared += separation[d] * separation[d];
    }
    double const inverseRadius = 1.0 / std::sqrt(radiusSquared);
    symmetric::Expansion<double, order> power{};
    symmetric::Expansion<double, order> result{};
    std::array<double, order + 1> spacingPowers{};
    spacingPowers[0] = 1;
    for (int rank = 1; rank <= order; rank++) {
        spacingPowers[rank] = spacingPowers[rank - 1] * spacing;
    }
    power[0] = 1;
    for (int term = 0; term <= order; term++) {
        symmetric::Expansion<double, order> next{};
        for (size_t coefficient = 0; coefficient < result.size(); coefficient++) {
            result[coefficient] -= inverseRadius * kBinomials[term] * power[coefficient];
            auto const a = kExponents[coefficient];
            if (a.rank < order) {
                for (int d = 0; d < NDIM; d++) {
                    std::array<int, NDIM> destination{a.nx, a.ny, a.nz};
                    destination[d]++;
                    next[symmetric::index(destination[0], destination[1], destination[2])] += power[coefficient] * 2.0 * separation[d] / radiusSquared;
                    if (a.rank + 2 <= order) {
                        destination[d]++;
                        next[symmetric::index(destination[0], destination[1], destination[2])] += power[coefficient] / radiusSquared;
                    }
                }
            }
        }
        power = next;
    }
    for (size_t coefficient = 0; coefficient < result.size(); coefficient++) {
        auto const a = kExponents[coefficient];
        result[coefficient] *= kCombinatorics.factorial(a.nx) * kCombinatorics.factorial(a.ny) * kCombinatorics.factorial(a.nz) * spacingPowers[a.rank];
    }
    return result;
}

TEST(SymmetricOperators, LayoutIncludesEveryIndependentSymmetricComponent) {
    constexpr int kOrder = 9;
    constexpr auto kExponents = symmetric::exponents<kOrder>();
    static_assert(symmetric::coefficientCount(0) == 1);
    static_assert(symmetric::coefficientCount(2) == 10);
    static_assert(symmetric::coefficientCount(kOrder) == 220);
    PROFILE();
    std::array<bool, symmetric::coefficientCount(kOrder)> seen{};
    for (int rank = 0; rank <= kOrder; rank++) {
        for (int nz = 0; nz <= rank; nz++) {
            for (int nx = 0; nx <= rank - nz; nx++) {
                int const ny = rank - nx - nz;
                auto const coefficient = symmetric::index(nx, ny, nz);
                ASSERT_LT(coefficient, seen.size());
                EXPECT_FALSE(seen[coefficient]);
                seen[coefficient] = true;
                EXPECT_EQ(kExponents[coefficient].nx, nx);
                EXPECT_EQ(kExponents[coefficient].ny, ny);
                EXPECT_EQ(kExponents[coefficient].nz, nz);
                EXPECT_EQ(kExponents[coefficient].rank, rank);
            }
        }
    }
    for (bool present : seen) {
        EXPECT_TRUE(present);
    }
}

TEST(SymmetricOperators, NewtonianAndPlummerDerivativesMatchIndependentSeries) {
    constexpr int kOrder = 8;
    constexpr double kSpacing = 0.7;
    constexpr std::array<double, NDIM> kSeparation{0.8, -1.2, 0.6};
    constexpr std::array<double, 3> kSoftenings{0.0, 0.35, 1.7};
    constexpr double kTolerance = 3e-9;
    PROFILE();
    for (double softening : kSoftenings) {
        auto const actual = symmetric::greenDerivatives<double, kOrder>(kSeparation, softening, kSpacing);
        auto const expected = binomialGreen<kOrder>(kSeparation, softening, kSpacing);
        for (size_t coefficient = 0; coefficient < actual.size(); coefficient++) {
            EXPECT_NEAR(actual[coefficient], expected[coefficient], kTolerance * (1.0 + std::abs(expected[coefficient])))
                << "coefficient=" << coefficient << " epsilon=" << softening;
        }
    }
}

TEST(SymmetricOperators, PlummerTraceAndCentralDerivativesAreRetained) {
    constexpr double kEpsilon = 0.5;
    constexpr double kScale = 0.25;
    constexpr double kMass = 2.0;
    constexpr std::array<double, NDIM> kZero{};
    constexpr double kTolerance = 2e-13;
    PROFILE();
    auto const green = symmetric::greenDerivatives<double, 4>(kZero, kEpsilon, kScale);
    EXPECT_DOUBLE_EQ(green[0], -1.0 / kEpsilon);
    double trace = 0;
    for (int d = 0; d < NDIM; d++) {
        std::array<int, NDIM> exponent{};
        exponent[d] = 1;
        EXPECT_DOUBLE_EQ(green[symmetric::index(exponent[0], exponent[1], exponent[2])], 0.0);
        exponent[d] = 2;
        double const second = green[symmetric::index(exponent[0], exponent[1], exponent[2])];
        EXPECT_NEAR(second, kScale * kScale / (kEpsilon * kEpsilon * kEpsilon), kTolerance);
        trace += second;
    }
    EXPECT_GT(trace, 0.0);
    EXPECT_NEAR(
        green[symmetric::index(4, 0, 0)], -9.0 * kScale * kScale * kScale * kScale / (kEpsilon * kEpsilon * kEpsilon * kEpsilon * kEpsilon), kTolerance);
    EXPECT_NEAR(green[symmetric::index(2, 2, 0)], green[symmetric::index(4, 0, 0)] / 3.0, kTolerance);
    auto const field = symmetric::evaluateP2P(kMass, kZero, kEpsilon);
    EXPECT_DOUBLE_EQ(field.potential, -kMass / kEpsilon);
    for (auto const acceleration : field.acceleration) {
        EXPECT_DOUBLE_EQ(acceleration, 0.0);
    }
}

TEST(SymmetricOperators, RawMomentTracesAndCenterTranslationPreserveParticleCloud) {
    constexpr int kOrder = 6;
    constexpr double kSourceScale = 0.1;
    constexpr double kTargetScale = 0.2;
    constexpr std::array<double, NDIM> kShift{0.07, -0.03, 0.04};
    constexpr double kTolerance = 2e-13;
    PROFILE();
    symmetric::Multipole<double, kOrder> source{};
    symmetric::Multipole<double, kOrder> shifted{};
    symmetric::Multipole<double, kOrder> expected{};
    double radiusSum = 0;
    for (auto const& particle : kParticles) {
        radiusSum += particle.mass * symmetric::addPoint<double, kOrder>(source, particle.mass, particle.offset, kSourceScale);
        auto offset = particle.offset;
        for (int d = 0; d < NDIM; d++) {
            offset[d] += kShift[d];
        }
        symmetric::addPoint<double, kOrder>(expected, particle.mass, offset, kTargetScale);
    }
    double const trace = source[symmetric::index(2, 0, 0)] + source[symmetric::index(0, 2, 0)] + source[symmetric::index(0, 0, 2)];
    EXPECT_NEAR(trace, radiusSum, kTolerance);
    EXPECT_GT(trace, 0.0);
    symmetric::addM2M<double, kOrder>(shifted, source, kShift, kSourceScale, kTargetScale);
    for (size_t coefficient = 0; coefficient < shifted.size(); coefficient++) {
        EXPECT_NEAR(shifted[coefficient], expected[coefficient], kTolerance * (1.0 + std::abs(expected[coefficient])));
    }
}

TEST(SymmetricOperators, LocalTranslationPreservesGeneralNonharmonicPolynomial) {
    constexpr int kOrder = 5;
    constexpr double kSourceScale = 0.4;
    constexpr double kTargetScale = 0.7;
    constexpr std::array<double, NDIM> kShift{0.12, -0.17, 0.11};
    constexpr std::array<double, NDIM> kOffset{-0.02, 0.035, 0.05};
    constexpr double kTolerance = 2e-12;
    PROFILE();
    symmetric::Expansion<double, kOrder> source{};
    symmetric::Expansion<double, kOrder> shifted{};
    for (size_t coefficient = 0; coefficient < source.size(); coefficient++) {
        source[coefficient] = double(int(coefficient % 11) - 5) / double(coefficient + 1);
    }
    auto totalOffset = kOffset;
    for (int d = 0; d < NDIM; d++) {
        totalOffset[d] += kShift[d];
    }
    symmetric::addL2L<double, kOrder>(shifted, source, kShift, kSourceScale, kTargetScale);
    auto const actual = symmetric::evaluateL2P<double, kOrder>(shifted, kOffset, kTargetScale);
    auto const expected = symmetric::evaluateL2P<double, kOrder>(source, totalOffset, kSourceScale);
    EXPECT_NEAR(actual.potential, expected.potential, kTolerance);
    for (int d = 0; d < NDIM; d++) {
        EXPECT_NEAR(actual.acceleration[d], expected.acceleration[d], kTolerance);
    }
}

TEST(SymmetricOperators, MultipoleAndLocalFieldsConvergeToSoftenedDirectCloud) {
    constexpr int kOrder = 7;
    constexpr double kSourceScale = 0.1;
    constexpr double kTargetScale = 0.2;
    constexpr std::array<double, NDIM> kSeparation{1.2, -0.8, 0.9};
    constexpr std::array<double, NDIM> kTargetOffset{0.014, -0.008, 0.017};
    constexpr std::array<double, 2> kSoftenings{0.0, 0.3};
    constexpr double kTolerance = 2e-11;
    PROFILE();
    symmetric::Multipole<double, kOrder> source{};
    for (auto const& particle : kParticles) {
        symmetric::addPoint<double, kOrder>(source, particle.mass, particle.offset, kSourceScale);
    }
    for (double softening : kSoftenings) {
        symmetric::Expansion<double, kOrder> local{};
        symmetric::Expansion<double, kOrder> particleLocal{};
        symmetric::Field<double> expected{};
        auto target = kSeparation;
        for (int d = 0; d < NDIM; d++) {
            target[d] += kTargetOffset[d];
        }
        for (auto const& particle : kParticles) {
            auto distance = target;
            auto centerDistance = kSeparation;
            for (int d = 0; d < NDIM; d++) {
                distance[d] -= particle.offset[d];
                centerDistance[d] -= particle.offset[d];
            }
            auto const field = symmetric::evaluateP2P(particle.mass, distance, softening);
            expected.potential += field.potential;
            for (int d = 0; d < NDIM; d++) {
                expected.acceleration[d] += field.acceleration[d];
            }
            symmetric::addP2L<double, kOrder>(particleLocal, particle.mass, centerDistance, kTargetScale, softening);
        }
        symmetric::addM2L<double, kOrder>(local, source, kSeparation, kSourceScale, kTargetScale, softening);
        auto const m2l = symmetric::evaluateL2P<double, kOrder>(local, kTargetOffset, kTargetScale);
        auto const m2p = symmetric::evaluateM2P<double, kOrder>(source, target, kSourceScale, softening);
        auto const p2l = symmetric::evaluateL2P<double, kOrder>(particleLocal, kTargetOffset, kTargetScale);
        for (auto const& actual : {m2l, m2p, p2l}) {
            EXPECT_NEAR(actual.potential, expected.potential, kTolerance);
            for (int d = 0; d < NDIM; d++) {
                EXPECT_NEAR(actual.acceleration[d], expected.acceleration[d], kTolerance);
            }
        }
    }
}

TEST(SymmetricOperators, HighestSourceRankContributesToHighestLocalRank) {
    constexpr int kSourceOrder = 2;
    constexpr int kLocalOrder = 3;
    constexpr double kSourceScale = 0.5;
    constexpr double kTargetScale = 0.25;
    constexpr double kSoftening = 0.3;
    constexpr std::array<double, NDIM> kSeparation{0.8, -0.6, 1.1};
    constexpr double kTolerance = 2e-13;
    PROFILE();
    symmetric::Multipole<double, kSourceOrder> source{};
    symmetric::Expansion<double, kLocalOrder> local{};
    source[symmetric::index(0, 0, kSourceOrder)] = 2.0;
    symmetric::addM2L<double, kSourceOrder, kLocalOrder>(local, source, kSeparation, kSourceScale, kTargetScale, kSoftening);
    auto const green = binomialGreen<kSourceOrder + kLocalOrder>(kSeparation, kSoftening, kTargetScale);
    double const expected = (kSourceScale / kTargetScale) * (kSourceScale / kTargetScale) * green[symmetric::index(0, 0, kSourceOrder + kLocalOrder)];
    EXPECT_NEAR(local[symmetric::index(0, 0, kLocalOrder)], expected, kTolerance);
    EXPECT_NE(local[symmetric::index(0, 0, kLocalOrder)], 0.0);
}

}    // namespace
