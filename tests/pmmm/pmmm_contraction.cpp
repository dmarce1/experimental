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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_contraction.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_green.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/profiler.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stddef.h>

namespace {

namespace gravity = cosmictiger::gravity;
namespace stf = cosmictiger::gravity::stf;

template <int sourceOrder, int expansionOrder>
stf::Expansion<double, expansionOrder> contract(
    stf::Multipole<double, sourceOrder> const& source, gravity::PmmmGreenCoefficients<sourceOrder + expansionOrder> const& green) {
    PROFILE();
    stf::Expansion<double, expansionOrder> local{};
    for (auto const& term : gravity::pmmmContractionTerms<sourceOrder, expansionOrder>()) {
        local[term.localCoefficient] += term.weight * source[term.sourceCoefficient] * green[term.greenCoefficient];
    }
    return local;
}

TEST(PmmmContraction, MonopoleMappingHasOneUnitTerm) {
    PROFILE();
    auto const terms = gravity::pmmmContractionTerms<0, 0>();
    ASSERT_EQ(terms.size(), 1);
    EXPECT_EQ(terms[0].localCoefficient, 0);
    EXPECT_EQ(terms[0].sourceCoefficient, 0);
    EXPECT_EQ(terms[0].greenCoefficient, 0);
    EXPECT_DOUBLE_EQ(terms[0].weight, 1.0);
}

TEST(PmmmContraction, AgreesWithFmmWhereRanksOverlap) {
    constexpr double kSpacing = 0.25;
    constexpr std::array<double, NDIM> kSeparation{1.25, -0.75, 2.0};
    PROFILE();

    auto const green = gravity::pmmmFreeSpaceGreen<4>(kSeparation, kSpacing);
    for (int sourceRank = 0; sourceRank <= 2; sourceRank++) {
        size_t const overlappingCount = static_cast<size_t>((3 - sourceRank) * (3 - sourceRank));
        for (int sourceIndex = sourceRank * sourceRank; sourceIndex < (sourceRank + 1) * (sourceRank + 1); sourceIndex++) {
            stf::Multipole<double, 2> source{};
            source[static_cast<size_t>(sourceIndex)] = 1.0;
            stf::Expansion<double, 2> reference{};
            stf::addM2L<double, 2, 2>(reference, source, kSeparation, kSpacing, kSpacing);
            auto const actual = contract<2, 2>(source, green);
            for (size_t localIndex = 0; localIndex < overlappingCount; localIndex++) {
                EXPECT_NEAR(actual[localIndex], reference[localIndex], 2e-12 * std::max(1.0, std::abs(reference[localIndex])))
                    << "source=" << sourceIndex << " local=" << localIndex;
            }
        }
    }
}

TEST(PmmmContraction, DipoleContributesToFullOrderLocalGradient) {
    constexpr double kSpacing = 0.5;
    constexpr double kMass = 3.0;
    constexpr std::array<double, NDIM> kSeparation{2.0, 1.0, 0.0};
    constexpr std::array<double, NDIM> kSourceOffset{0.125, -0.25, 0.0};
    PROFILE();

    stf::Multipole<double, 1> source{};
    source[0] = kMass;
    source[stf::index(0, 1, 0)] = kMass * kSourceOffset[1] / kSpacing;
    source[stf::index(1, 0, 0)] = kMass * kSourceOffset[0] / kSpacing;
    auto const green = gravity::pmmmFreeSpaceGreen<2>(kSeparation, kSpacing);
    auto const actual = contract<1, 1>(source, green);

    double const inverseRadius = 1.0 / std::sqrt(5.0);
    double const inverseRadiusCubed = inverseRadius / 5.0;
    double const inverseRadiusFifth = inverseRadiusCubed / 5.0;
    double const gx = 2.0 * inverseRadiusCubed;
    double const gy = inverseRadiusCubed;
    double const gxx = inverseRadiusCubed - 12.0 * inverseRadiusFifth;
    double const gxy = -6.0 * inverseRadiusFifth;
    double const gyy = inverseRadiusCubed - 3.0 * inverseRadiusFifth;
    double const potential = kMass * (-inverseRadius - kSourceOffset[0] * gx - kSourceOffset[1] * gy);
    double const localX = kMass * kSpacing * (gx - kSourceOffset[0] * gxx - kSourceOffset[1] * gxy);
    double const localY = kMass * kSpacing * (gy - kSourceOffset[0] * gxy - kSourceOffset[1] * gyy);
    EXPECT_NEAR(actual[0], potential, 2e-14);
    EXPECT_NEAR(actual[stf::index(1, 0, 0)], localX, 2e-14);
    EXPECT_NEAR(actual[stf::index(0, 1, 0)], localY, 2e-14);
    EXPECT_NEAR(actual[stf::index(0, 0, 1)], 0.0, 2e-14);
}

}    // namespace
