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

#include <cosmictiger/profiler.hpp>
#include <cosmictiger/units.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

namespace {

using namespace cosmictiger::units;

TEST(Units, DimensionsAndCosmologicalFramesRemainDistinct) {
    constexpr Length kDistance = Length::from_value(12.0);
    constexpr ComovingPosition kComoving(kDistance);

    static_assert(std::is_convertible_v<decltype(Length{} / Time{}), Velocity>);
    static_assert(std::is_convertible_v<decltype(Velocity{} / Time{}), Acceleration>);
    static_assert(std::is_convertible_v<decltype(Mass{} * Velocity{} * Velocity{}), Energy>);
    static_assert(std::is_convertible_v<decltype(Mass{} * Acceleration{}), Force>);
    static_assert(std::is_convertible_v<decltype(Length{} * Velocity{} * Velocity{} / GravitationalConstant{}), Mass>);
    static_assert(!std::is_convertible_v<Length, Time>);
    static_assert(!std::is_convertible_v<ComovingPosition, ProperPosition>);
    static_assert(!std::is_convertible_v<CanonicalMomentumPerMass, PeculiarVelocity>);
    static_assert(!std::is_convertible_v<PhysicalTime, ConformalTime>);
    static_assert(!std::is_convertible_v<PeculiarVelocity, double>);
    static_assert(sizeof(ComovingPosition) == sizeof(double));
    static_assert(sizeof(PeculiarVelocity) == sizeof(double));
    PROFILE();

    EXPECT_DOUBLE_EQ(kComoving.quantity().value(), kDistance.value());
}

TEST(Units, ConvertsInputLittleHToPhysicalScales) {
    constexpr double kHubble = 50.0;
    constexpr double kBoxLength = 100.0;
    constexpr double kMegaparsecCgs = 3.085677581e24;
    constexpr double kExpectedLength = 200.0 * kMegaparsecCgs;
    constexpr double kExpectedVelocity = 1.0e9;
    constexpr double kExpectedTime = kMegaparsecCgs / 5.0e6;
    constexpr double kExpectedMass = 1.0e18 * kExpectedLength / 6.67430e-8;
    PROFILE();
    UnitScales const scales(kHubble, kBoxLength);

    EXPECT_DOUBLE_EQ(scales.lengthMpc(), 200.0);
    EXPECT_DOUBLE_EQ(scales.lengthCgs(), kExpectedLength);
    EXPECT_DOUBLE_EQ(scales.timeSeconds(), kExpectedTime);
    EXPECT_DOUBLE_EQ(scales.velocityKmPerSecond(), 10000.0);
    EXPECT_DOUBLE_EQ(scales.velocityCgs(), kExpectedVelocity);
    EXPECT_DOUBLE_EQ(scales.massCgs(), kExpectedMass);
    EXPECT_DOUBLE_EQ(scales.massSolar(), kExpectedMass / 1.98847e33);
    EXPECT_DOUBLE_EQ(scales.accelerationCgs(), kExpectedVelocity / kExpectedTime);
    EXPECT_DOUBLE_EQ(scales.potentialCgs(), 1.0e18);
    EXPECT_DOUBLE_EQ(scales.energyCgs(), kExpectedMass * 1.0e18);
    EXPECT_DOUBLE_EQ(scales.forceCgs(), kExpectedMass * (kExpectedVelocity / kExpectedTime));
    EXPECT_DOUBLE_EQ(scales.densityCgs(), kExpectedMass / (kExpectedLength * kExpectedLength * kExpectedLength));
}

TEST(Units, FixedInputMpcOverHHasExpectedHubbleDependence) {
    constexpr double kLowHubble = 50.0;
    constexpr double kHighHubble = 100.0;
    constexpr double kBoxLength = 100.0;
    PROFILE();
    UnitScales const low(kLowHubble, kBoxLength);
    UnitScales const high(kHighHubble, kBoxLength);

    EXPECT_DOUBLE_EQ(low.lengthMpc(), 2.0 * high.lengthMpc());
    EXPECT_DOUBLE_EQ(low.timeSeconds(), 2.0 * high.timeSeconds());
    EXPECT_DOUBLE_EQ(low.massCgs(), 2.0 * high.massCgs());
    EXPECT_DOUBLE_EQ(low.velocityCgs(), high.velocityCgs());
    EXPECT_DOUBLE_EQ(low.potentialCgs(), high.potentialCgs());
    EXPECT_DOUBLE_EQ(low.accelerationCgs(), 0.5 * high.accelerationCgs());
    EXPECT_DOUBLE_EQ(low.densityCgs(), 0.25 * high.densityCgs());
}

TEST(Units, ConvertsFramesAtTheSpecifiedEpochWithoutHubbleFlow) {
    constexpr double kHubble = 50.0;
    constexpr double kBoxLength = 100.0;
    constexpr double kScaleFactor = 0.25;
    constexpr double kCoordinate = 0.75;
    constexpr double kMomentum = -0.02;
    constexpr double kField = -0.5;
    constexpr double kPotential = -0.125;
    PROFILE();
    UnitScales const scales(kHubble, kBoxLength);

    EXPECT_DOUBLE_EQ(scales.comovingPosition(kCoordinate).value(), kCoordinate * scales.lengthCgs());
    EXPECT_DOUBLE_EQ(scales.properPosition(kCoordinate, kScaleFactor).value(), kScaleFactor * scales.comovingPosition(kCoordinate).value());
    EXPECT_DOUBLE_EQ(scales.canonicalMomentumPerMass(kMomentum).value(), -2.0e7);
    EXPECT_DOUBLE_EQ(scales.peculiarVelocity(kMomentum, kScaleFactor).value(), -8.0e7);
    EXPECT_DOUBLE_EQ(scales.peculiarAcceleration(kField, kScaleFactor).value(), -8.0 * scales.accelerationCgs());
    EXPECT_DOUBLE_EQ(scales.peculiarPotential(kPotential, kScaleFactor).value(), -0.5 * scales.potentialCgs());

    // A particle at rest in the expanding frame has zero velocity at every
    // position and epoch. Conversion has no position or expansion-rate input.
    EXPECT_DOUBLE_EQ(scales.peculiarVelocity(0.0, kScaleFactor).value(), 0.0);
    EXPECT_DOUBLE_EQ(scales.peculiarVelocity(0.0, 1.0).value(), 0.0);
    EXPECT_DOUBLE_EQ(scales.peculiarVelocity(0.0, 2.0).value(), 0.0);
}

TEST(Units, ConvertsSignedTimesAndRetainsDoubleRange) {
    constexpr double kHubble = 67.4;
    constexpr double kBoxLength = 100.0;
    constexpr double kTime = -0.125;
    PROFILE();
    UnitScales const scales(kHubble, kBoxLength);

    EXPECT_DOUBLE_EQ(scales.physicalTime(kTime).value(), kTime * scales.timeSeconds());
    EXPECT_DOUBLE_EQ(scales.conformalTime(kTime).value(), kTime * scales.timeSeconds());
    EXPECT_DOUBLE_EQ(kYear.value(), 31557600.0);
    EXPECT_GT(scales.massCgs(), double(std::numeric_limits<float>::max()));
    EXPECT_GT(scales.energyCgs(), double(std::numeric_limits<float>::max()));
    EXPECT_LT(scales.energyCgs(), std::numeric_limits<double>::max());
}

}    // namespace
