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

#include <cosmictiger/gravity/ferrers.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using cosmictiger::Coordinate;
using cosmictiger::Position;

struct ExpectedField {
    double x;
    double y;
    double z;
    double potential;
};

struct Sample {
    Position target;
    Position source;
};

Position position(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    PROFILE();
    return {Coordinate::fromRaw(x), Coordinate::fromRaw(y), Coordinate::fromRaw(z)};
}

template <typename Function>
double integrateSimpson(double begin, double end, Function const& function) {
    constexpr int kIntervals = 2048;
    PROFILE();

    double const spacing = (end - begin) / kIntervals;
    double sum = function(begin) + function(end);
    for (int index = 1; index < kIntervals; index++) {
        sum += (index % 2 == 0 ? 2.0 : 4.0) * function(begin + index * spacing);
    }
    return sum * spacing / 3.0;
}

template <int order>
double densityShape(double radius) {
    PROFILE();
    double const factor = 1.0 - radius * radius;
    double value = 1.0;
    for (int power = 0; power < order; power++) {
        value *= factor;
    }
    return value;
}

template <int order>
ExpectedField reference(Position target, Position source, double softeningLength) {
    PROFILE();
    // The uint32_t subtraction is exact in the periodic fixed-32 coordinate;
    // interpreting its result as int32_t selects the nearest image.
    double const x = static_cast<double>(static_cast<std::int32_t>(target.x.raw() - source.x.raw())) * 0x1p-32;
    double const y = static_cast<double>(static_cast<std::int32_t>(target.y.raw() - source.y.raw())) * 0x1p-32;
    double const z = static_cast<double>(static_cast<std::int32_t>(target.z.raw() - source.z.raw())) * 0x1p-32;
    double const distance = std::sqrt(x * x + y * y + z * z);

    auto const radialMassDensity = [](double radius) {
        PROFILE();
        return radius * radius * densityShape<order>(radius);
    };
    double const normalization = integrateSimpson(0.0, 1.0, radialMassDensity);
    if (distance >= softeningLength) {
        double const factor = -1.0 / (distance * distance * distance);
        return {factor * x, factor * y, factor * z, -1.0 / distance};
    }

    double const scaledRadius = distance / softeningLength;
    double const enclosedMass = integrateSimpson(0.0, scaledRadius, radialMassDensity) / normalization;
    double const outerShell = integrateSimpson(scaledRadius, 1.0, [](double radius) {
        PROFILE();
        return radius * densityShape<order>(radius);
    }) / normalization;
    double const potential = -1.0 / softeningLength * (distance == 0.0 ? outerShell : enclosedMass / scaledRadius + outerShell);
    double const factor = distance == 0.0 ? 0.0 : -enclosedMass / (distance * distance * distance);
    return {factor * x, factor * y, factor * z, potential};
}

void expectNear(float actual, double expected) {
    PROFILE();
    double const scale = std::fabs(expected) > 1.0 ? std::fabs(expected) : 1.0;
    EXPECT_NEAR(static_cast<double>(actual), expected, 2.0e-5 * scale);
}

template <int order>
void checkProfile(float q, cosmictiger::gravity::SofteningLength const& softening) {
    PROFILE();
    auto const center = position(0x80000000U, 0x80000000U, 0x80000000U);
    std::array<Sample, 10> const samples{{
        {center, center},
        {position(0x82000000U, 0x80000000U, 0x80000000U), center},
        {position(0x90000000U, 0x80000000U, 0x80000000U), center},
        {position(0xa0000000U, 0x80000000U, 0x80000000U), center},
        {position(0xc0000000U, 0x80000000U, 0x80000000U), center},
        {position(0xd0000000U, 0x80000000U, 0x80000000U), center},
        {position(0x90000000U, 0x88000000U, 0x84000000U), center},
        {position(0x01000000U, 0x80000000U, 0x80000000U), position(0xff000000U, 0x80000000U, 0x80000000U)},
        {position(0xff000000U, 0x80000000U, 0x80000000U), position(0x01000000U, 0x80000000U, 0x80000000U)},
        {position(0x70000000U, 0x68000000U, 0x60000000U), center},
    }};

    for (Sample const& sample : samples) {
        ExpectedField const expected = reference<order>(sample.target, sample.source, q);
        auto const acceleration = cosmictiger::gravity::ferrersAcceleration<order>(sample.target, sample.source, softening);
        auto const combined = cosmictiger::gravity::ferrersAccelerationAndPotential<order>(sample.target, sample.source, softening);
        expectNear(acceleration.x, expected.x);
        expectNear(acceleration.y, expected.y);
        expectNear(acceleration.z, expected.z);
        expectNear(combined.acceleration.x, expected.x);
        expectNear(combined.acceleration.y, expected.y);
        expectNear(combined.acceleration.z, expected.z);
        expectNear(combined.potential, expected.potential);
        EXPECT_FLOAT_EQ(acceleration.x, combined.acceleration.x);
        EXPECT_FLOAT_EQ(acceleration.y, combined.acceleration.y);
        EXPECT_FLOAT_EQ(acceleration.z, combined.acceleration.z);
    }
}

TEST(FerrersGravity, NormalizedDensityOrdersZeroThroughThree) {
    constexpr float kQ = 0.25f;
    PROFILE();

    auto const softening = cosmictiger::gravity::makeFerrersSoftening(kQ);
    checkProfile<0>(kQ, softening);
    checkProfile<1>(kQ, softening);
    checkProfile<2>(kQ, softening);
    checkProfile<3>(kQ, softening);
}

TEST(FerrersGravity, RuntimeSofteningCanBeReusedAcrossProfilesAndSeparations) {
    constexpr float kQ = 0.125f;
    PROFILE();

    auto const softening = cosmictiger::gravity::makeFerrersSoftening(kQ);
    checkProfile<0>(kQ, softening);
    checkProfile<1>(kQ, softening);
    checkProfile<2>(kQ, softening);
    checkProfile<3>(kQ, softening);
}

TEST(FerrersGravity, RuntimeSofteningRejectsInvalidLengths) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    EXPECT_THROW(cosmictiger::gravity::makeFerrersSoftening(0.0f), std::runtime_error);
    EXPECT_THROW(cosmictiger::gravity::makeFerrersSoftening(-0.25f), std::runtime_error);
}

TEST(FerrersGravity, NearestImageWrapsInBothDirections) {
    PROFILE();
    auto const low = position(0x01000000U, 0x80000000U, 0x80000000U);
    auto const high = position(0xff000000U, 0x80000000U, 0x80000000U);
    auto const softening = cosmictiger::gravity::makeFerrersSoftening(0.25f);
    auto const leftAcceleration = cosmictiger::gravity::ferrersAcceleration<2>(low, high, softening);
    auto const rightAcceleration = cosmictiger::gravity::ferrersAcceleration<2>(high, low, softening);
    EXPECT_LT(leftAcceleration.x, 0.0f);
    EXPECT_GT(rightAcceleration.x, 0.0f);
    EXPECT_FLOAT_EQ(leftAcceleration.x, -rightAcceleration.x);
    EXPECT_FLOAT_EQ(leftAcceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(leftAcceleration.z, 0.0f);
}

TEST(FerrersGravity, HalfBoxTieUsesNegativeNearestImage) {
    PROFILE();
    auto const target = position(0x80000000U, 0U, 0U);
    auto const source = position(0U, 0U, 0U);
    auto const softening = cosmictiger::gravity::makeFerrersSoftening(0.25f);
    auto const result = cosmictiger::gravity::ferrersAccelerationAndPotential<3>(target, source, softening);
    EXPECT_FLOAT_EQ(result.acceleration.x, 4.0f);
    EXPECT_FLOAT_EQ(result.acceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(result.acceleration.z, 0.0f);
    EXPECT_FLOAT_EQ(result.potential, -2.0f);
}

TEST(FerrersGravity, CoincidentPositionsRemainFiniteForTinySupport) {
    PROFILE();
    auto const center = position(0U, 0U, 0U);
    auto const softening = cosmictiger::gravity::makeFerrersSoftening(1.0e-30f);
    auto const result = cosmictiger::gravity::ferrersAccelerationAndPotential<3>(center, center, softening);
    EXPECT_FLOAT_EQ(result.acceleration.x, 0.0f);
    EXPECT_FLOAT_EQ(result.acceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(result.acceleration.z, 0.0f);
    EXPECT_FLOAT_EQ(result.potential, -(315.0f / 128.0f) / 1.0e-30f);
}

}    // namespace
