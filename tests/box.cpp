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

#include <cosmictiger/box.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>

namespace {

using cosmictiger::Box;
using cosmictiger::Coordinate;

template <typename Real>
using Point = std::array<Real, NDIM>;

static_assert(std::is_same_v<decltype(Box<double>(Point<double>{}, Point<double>{})), Box<double>>);
static_assert(std::is_same_v<decltype(Box<Coordinate>(Point<Coordinate>{}, Point<Coordinate>{})), Box<Coordinate>>);

TEST(Box, HalfOpenPointMembership) {
    PROFILE();
    Box<double> const box({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0});
    EXPECT_EQ(box.min(0), 1.0);
    EXPECT_EQ(box.max(2), 6.0);
    EXPECT_TRUE(box.contains({1.0, 2.0, 3.0}));
    EXPECT_TRUE(box.contains({3.5, 4.5, 5.5}));
    EXPECT_FALSE(box.contains({4.0, 3.0, 4.0}));
    EXPECT_FALSE(box.contains({2.0, 5.0, 4.0}));
    EXPECT_FALSE(box.contains({2.0, 3.0, 6.0}));
    EXPECT_FALSE(box.contains({0.9, 3.0, 4.0}));
    EXPECT_FALSE(box.contains({2.0, 1.9, 4.0}));
    EXPECT_FALSE(box.contains({2.0, 3.0, 2.9}));
}

TEST(Box, OverlapContactAndSeparation) {
    PROFILE();
    Box<double> const box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    Box<double> const overlap({0.5, 0.5, 0.5}, {1.5, 1.5, 1.5});
    Box<double> const face({1.0, 0.25, 0.25}, {2.0, 0.75, 0.75});
    Box<double> const edge({1.0, 1.0, 0.25}, {2.0, 2.0, 0.75});
    Box<double> const corner({1.0, 1.0, 1.0}, {2.0, 2.0, 2.0});
    Box<double> const gap({1.1, 0.25, 0.25}, {2.0, 0.75, 0.75});
    Box<double> const offset({1.0, 2.0, 0.25}, {2.0, 3.0, 0.75});

    EXPECT_TRUE(box.intersects(overlap));
    EXPECT_TRUE(overlap.intersects(box));
    EXPECT_FALSE(box.touches(overlap));
    EXPECT_FALSE(box.isDisjoint(overlap));
    for (auto const& contact : {face, edge, corner}) {
        EXPECT_FALSE(box.intersects(contact));
        EXPECT_FALSE(contact.intersects(box));
        EXPECT_TRUE(box.touches(contact));
        EXPECT_TRUE(contact.touches(box));
        EXPECT_FALSE(box.isDisjoint(contact));
    }
    for (auto const& separated : {gap, offset}) {
        EXPECT_FALSE(box.intersects(separated));
        EXPECT_FALSE(box.touches(separated));
        EXPECT_TRUE(box.isDisjoint(separated));
    }
}

TEST(Box, ScalarAndAxiswiseResize) {
    PROFILE();
    Box<double> const box({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0});
    auto const expanded = box.expanded(0.5);
    EXPECT_EQ(expanded.min(0), 0.5);
    EXPECT_EQ(expanded.max(0), 4.5);
    EXPECT_EQ(expanded.min(1), 1.5);
    EXPECT_EQ(expanded.max(2), 6.5);
    auto const restored = expanded.contracted(0.5);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(restored.min(axis), box.min(axis));
        EXPECT_EQ(restored.max(axis), box.max(axis));
    }

    Point<double> const margin{0.25, 0.5, 0.75};
    auto const axisExpanded = box.expanded(margin);
    EXPECT_EQ(axisExpanded.min(0), 0.75);
    EXPECT_EQ(axisExpanded.max(0), 4.25);
    EXPECT_EQ(axisExpanded.min(1), 1.5);
    EXPECT_EQ(axisExpanded.max(1), 5.5);
    EXPECT_EQ(axisExpanded.min(2), 2.25);
    EXPECT_EQ(axisExpanded.max(2), 6.75);
    auto const axisContracted = box.contracted(margin);
    EXPECT_EQ(axisContracted.min(0), 1.25);
    EXPECT_EQ(axisContracted.max(0), 3.75);
    EXPECT_EQ(axisContracted.min(1), 2.5);
    EXPECT_EQ(axisContracted.max(1), 4.5);
    EXPECT_EQ(axisContracted.min(2), 3.75);
    EXPECT_EQ(axisContracted.max(2), 5.25);
}

TEST(Box, SplitsAtInteriorCutWithRightHalfOwningTheCut) {
    PROFILE();
    Box<double> const box({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0});
    auto const [left, right] = box.split(0, 2.0);
    EXPECT_EQ(left.min(0), 1.0);
    EXPECT_EQ(left.max(0), 2.0);
    EXPECT_EQ(right.min(0), 2.0);
    EXPECT_EQ(right.max(0), 4.0);
    for (size_t axis = 1; axis < NDIM; axis++) {
        EXPECT_EQ(left.min(axis), box.min(axis));
        EXPECT_EQ(left.max(axis), box.max(axis));
        EXPECT_EQ(right.min(axis), box.min(axis));
        EXPECT_EQ(right.max(axis), box.max(axis));
    }
    EXPECT_TRUE(left.contains({1.5, 3.0, 4.0}));
    EXPECT_FALSE(left.contains({2.0, 3.0, 4.0}));
    EXPECT_TRUE(right.contains({2.0, 3.0, 4.0}));
    EXPECT_FALSE(right.contains({4.0, 3.0, 4.0}));
    EXPECT_TRUE(left.touches(right));
}

TEST(Box, RejectsInvalidBoundsCutsAndMargins) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation is disabled in Release";
    }
    EXPECT_THROW((Box<double>({2.0, 0.0, 0.0}, {1.0, 1.0, 1.0})), std::runtime_error);
    Box<double> const box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    EXPECT_THROW(box.min(NDIM), std::runtime_error);
    EXPECT_THROW(box.max(NDIM), std::runtime_error);
    EXPECT_THROW(box.split(NDIM, 0.5), std::runtime_error);
    EXPECT_THROW(box.split(0, 0.0), std::runtime_error);
    EXPECT_THROW(box.split(0, 1.0), std::runtime_error);
    EXPECT_THROW(box.split(0, -0.1), std::runtime_error);
    EXPECT_THROW(box.split(0, 1.1), std::runtime_error);
    EXPECT_THROW(box.expanded(-0.1), std::runtime_error);
    EXPECT_THROW(box.contracted(-0.1), std::runtime_error);
    EXPECT_THROW(box.contracted(0.6), std::runtime_error);
    Point<double> const negativeMargin{0.1, -0.1, 0.1};
    Point<double> const excessContraction{0.1, 0.6, 0.1};
    EXPECT_THROW(box.expanded(negativeMargin), std::runtime_error);
    EXPECT_THROW(box.contracted(excessContraction), std::runtime_error);
}

TEST(Box, FixedPeriodicUpperEndpointAndWholeAxis) {
    PROFILE();
    Coordinate const zero = Coordinate::min();
    Coordinate const eighth = Coordinate::fromRaw(std::uint32_t(1) << (Coordinate::fractionalBits() - 3));
    Coordinate const quarter = Coordinate::fromRaw(std::uint32_t(1) << (Coordinate::fractionalBits() - 2));
    Coordinate const threeEighths = Coordinate::fromRaw(std::uint32_t(3) << (Coordinate::fractionalBits() - 3));
    Coordinate const half = Coordinate::fromRaw(std::uint32_t(1) << (Coordinate::fractionalBits() - 1));
    Coordinate const threeQuarters = Coordinate::fromRaw(std::uint32_t(3) << (Coordinate::fractionalBits() - 2));
    Coordinate const sevenEighths = Coordinate::fromRaw(std::uint32_t(7) << (Coordinate::fractionalBits() - 3));
    Coordinate const last = Coordinate::max();

    auto const lastCell = Box<Coordinate>::fromPeriodicBounds({threeQuarters, zero, zero}, {zero, quarter, quarter});
    EXPECT_TRUE(lastCell.hasPeriodicUpper(0));
    EXPECT_FALSE(lastCell.hasPeriodicUpper(1));
    EXPECT_TRUE(lastCell.contains({threeQuarters, zero, zero}));
    EXPECT_TRUE(lastCell.contains({last, zero, zero}));
    EXPECT_FALSE(lastCell.contains({zero, zero, zero}));
    EXPECT_FALSE(lastCell.contains({half, zero, zero}));
    EXPECT_FALSE(lastCell.contains({last, quarter, zero}));
    auto const [firstHalf, secondHalf] = lastCell.split(0, sevenEighths);
    EXPECT_FALSE(firstHalf.hasPeriodicUpper(0));
    EXPECT_TRUE(secondHalf.hasPeriodicUpper(0));
    EXPECT_TRUE(firstHalf.contains({threeQuarters, zero, zero}));
    EXPECT_FALSE(firstHalf.contains({sevenEighths, zero, zero}));
    EXPECT_TRUE(secondHalf.contains({sevenEighths, zero, zero}));
    EXPECT_TRUE(secondHalf.contains({last, zero, zero}));
    EXPECT_TRUE(firstHalf.touches(secondHalf));

    Box<Coordinate> const interior({quarter, quarter, quarter}, {half, half, half});
    auto const enlarged = interior.expanded(eighth);
    EXPECT_EQ(enlarged.min(0), eighth);
    EXPECT_EQ(enlarged.max(0), threeQuarters - eighth);
    auto const restored = enlarged.contracted(eighth);
    EXPECT_EQ(restored, interior);
    auto const [lowerInterior, upperInterior] = interior.split(0, threeEighths);
    EXPECT_EQ(lowerInterior.max(0), threeEighths);
    EXPECT_EQ(upperInterior.min(0), threeEighths);

    auto const wholePeriod = Box<Coordinate>::fromPeriodicBounds({zero, zero, zero}, {zero, zero, zero});
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_TRUE(wholePeriod.hasPeriodicUpper(axis));
    }
    EXPECT_TRUE(wholePeriod.contains({zero, zero, zero}));
    EXPECT_TRUE(wholePeriod.contains({last, last, last}));
    EXPECT_TRUE(wholePeriod.contains({quarter, half, threeQuarters}));
}

TEST(Box, PeriodicMembershipPreservesScaledFixedEndpoints) {
    PROFILE();
    Box<double> const left({-0.125, 0.0, 0.0}, {0.0, 0.125, 0.125});
    Box<double> const right({1.0, 0.0, 0.0}, {1.125, 0.125, 0.125});
    using Twice = cosmictiger::Fixed<std::uint32_t, 2>;
    using Tenfold = cosmictiger::Fixed<std::uint32_t, 10>;
    using Signed = cosmictiger::Fixed<std::int32_t, 2>;
    using SignedTenth = cosmictiger::Fixed<std::int32_t, 0.1>;
    Twice const justBelowOne = Twice::fromRaw((std::uint32_t(1) << (Twice::fractionalBits() - 1)) - 1);
    Twice const exactlyOne = Twice::fromRaw(std::uint32_t(1) << (Twice::fractionalBits() - 1));
    EXPECT_TRUE(left.containsPeriodic(std::array<Twice, NDIM>{justBelowOne, Twice::min(), Twice::min()}));
    EXPECT_FALSE(left.containsPeriodic(std::array<Twice, NDIM>{exactlyOne, Twice::min(), Twice::min()}));
    EXPECT_TRUE(right.containsPeriodic(std::array<Twice, NDIM>{exactlyOne, Twice::min(), Twice::min()}));
    Tenfold const tenfoldNearOne = Tenfold::fromRaw(std::numeric_limits<std::uint32_t>::max() / 10);
    EXPECT_TRUE(left.containsPeriodic(std::array<Tenfold, NDIM>{tenfoldNearOne, Tenfold::min(), Tenfold::min()}));
    Signed const negativeEpsilon = Signed::fromRaw(-1);
    EXPECT_TRUE(left.containsPeriodic(std::array<Signed, NDIM>{negativeEpsilon, Signed::fromRaw(0), Signed::fromRaw(0)}));
    SignedTenth const negativeTenthEpsilon = SignedTenth::fromRaw(-1);
    EXPECT_TRUE(left.containsPeriodic(std::array<SignedTenth, NDIM>{negativeTenthEpsilon, SignedTenth::fromRaw(0), SignedTenth::fromRaw(0)}));
}

TEST(Box, PeriodicMembershipNearZeroAndExtremeDoubleBounds) {
    constexpr double kTinyPositive = 0x1p-60;
    constexpr double kTinyNegative = -1e-20;
    constexpr double kLargest = std::numeric_limits<double>::max();
    PROFILE();

    Coordinate const zero = Coordinate::min();
    Coordinate const last = Coordinate::max();
    Box<double> const positive({kTinyPositive, 0.0, 0.0}, {1.0, 1.0, 1.0});
    EXPECT_FALSE(positive.containsPeriodic(std::array<Coordinate, NDIM>{zero, zero, zero}));
    EXPECT_TRUE(positive.containsPeriodic(std::array<Coordinate, NDIM>{Coordinate::epsilon(), zero, zero}));
    EXPECT_TRUE(positive.containsPeriodic(std::array<Coordinate, NDIM>{last, zero, zero}));
    EXPECT_FALSE(positive.containsPeriodic(std::array<double, NDIM>{std::nextafter(kTinyPositive, 0.0), 0.0, 0.0}));
    EXPECT_TRUE(positive.containsPeriodic(std::array<double, NDIM>{kTinyPositive, 0.0, 0.0}));
    EXPECT_TRUE(positive.containsPeriodic(std::array<double, NDIM>{std::nextafter(1.0, 0.0), 0.0, 0.0}));
    EXPECT_FALSE(positive.containsPeriodic(std::array<double, NDIM>{1.0, 0.0, 0.0}));

    Box<double> const negative({kTinyNegative, 0.0, 0.0}, {0.0, 1.0, 1.0});
    EXPECT_TRUE(negative.containsPeriodic(std::array<double, NDIM>{kTinyNegative, 0.0, 0.0}));
    EXPECT_TRUE(negative.containsPeriodic(std::array<double, NDIM>{-5e-21, 0.0, 0.0}));
    EXPECT_FALSE(negative.containsPeriodic(std::array<double, NDIM>{-2e-20, 0.0, 0.0}));
    EXPECT_FALSE(negative.containsPeriodic(std::array<double, NDIM>{0.0, 0.0, 0.0}));
    EXPECT_FALSE(negative.containsPeriodic(std::array<Coordinate, NDIM>{last, zero, zero}));
    using TinyScaled = cosmictiger::Fixed<std::int32_t, 1e-12>;
    TinyScaled const scaledZero = TinyScaled::fromRaw(0);
    EXPECT_TRUE(negative.containsPeriodic(std::array<TinyScaled, NDIM>{TinyScaled::fromRaw(-1), scaledZero, scaledZero}));
    EXPECT_FALSE(negative.containsPeriodic(std::array<TinyScaled, NDIM>{TinyScaled::fromRaw(-100), scaledZero, scaledZero}));

    Box<double> const seam({-0.1, 0.0, 0.0}, {0.1, 1.0, 1.0});
    EXPECT_TRUE(seam.containsPeriodic(std::array<double, NDIM>{-0.1, 0.0, 0.0}));
    EXPECT_FALSE(seam.containsPeriodic(std::array<double, NDIM>{0.1, 0.0, 0.0}));
    EXPECT_TRUE(seam.containsPeriodic(std::array<Coordinate, NDIM>{Coordinate::fromRaw(429496729U), zero, zero}));
    EXPECT_FALSE(seam.containsPeriodic(std::array<Coordinate, NDIM>{Coordinate::fromRaw(429496730U), zero, zero}));
    EXPECT_TRUE(seam.containsPeriodic(std::array<Coordinate, NDIM>{Coordinate::fromRaw(3865470567U), zero, zero}));
    EXPECT_FALSE(seam.containsPeriodic(std::array<Coordinate, NDIM>{Coordinate::fromRaw(3865470566U), zero, zero}));

    Box<double> const broad({-kLargest, -kLargest, -kLargest}, {kLargest, kLargest, kLargest});
    EXPECT_TRUE(broad.containsPeriodic(std::array<double, NDIM>{kLargest, -kLargest, 0.0}));
    EXPECT_TRUE(broad.containsPeriodic(std::array<Coordinate, NDIM>{last, zero, Coordinate::epsilon()}));
}

}    // namespace
