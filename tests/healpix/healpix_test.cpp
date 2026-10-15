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

#include <cosmictiger/healpix/healpix.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using cosmictiger::healpix::Healpix;
using cosmictiger::healpix::Ordering;
using Direction = std::array<double, NDIM>;

TEST(Healpix, KnownAxisPixels) {
    constexpr std::array<Direction, 6> kDirections{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}}};
    constexpr std::array<std::uint64_t, 6> kBasePixels{4, 5, 6, 7, 0, 8};
    constexpr std::array<std::uint64_t, 6> kNestedPixels{17, 21, 25, 29, 3, 32};
    constexpr std::array<std::uint64_t, 6> kRingPixels{20, 22, 24, 26, 0, 44};
    PROFILE();
    Healpix const base(1);
    Healpix const nested(2);
    Healpix const ring(2, Ordering::ring);
    for (size_t index = 0; index < kDirections.size(); index++) {
        EXPECT_EQ(base.vectorToPixel(kDirections[index]), kBasePixels[index]);
        EXPECT_EQ(nested.vectorToPixel(kDirections[index]), kNestedPixels[index]);
        EXPECT_EQ(ring.vectorToPixel(kDirections[index]), kRingPixels[index]);
    }
}

TEST(Healpix, BasePixelCentersHaveKnownLatitudeAndLongitude) {
    constexpr std::uint32_t kNside = 1;
    constexpr double kLatitude = 2.0 / 3.0;
    constexpr double kTolerance = 2.0e-15;
    constexpr std::uint64_t kPolarFaces = 4;
    constexpr std::uint64_t kSouthStart = 8;
    constexpr double kQuarterPi = std::numbers::pi / 4.0;
    constexpr double kHalfPi = std::numbers::pi / 2.0;
    PROFILE();
    Healpix const map(kNside);
    for (std::uint64_t pixel = 0; pixel < map.pixelCount(); pixel++) {
        auto const direction = map.pixelToVector(pixel);
        double const expectedZ = pixel < kPolarFaces ? kLatitude : (pixel < kSouthStart ? 0.0 : -kLatitude);
        double const phi = static_cast<double>(pixel % kPolarFaces) * kHalfPi + (pixel >= kPolarFaces && pixel < kSouthStart ? 0.0 : kQuarterPi);
        double const transverse = std::sqrt(1.0 - expectedZ * expectedZ);
        EXPECT_NEAR(direction[0], transverse * std::cos(phi), kTolerance);
        EXPECT_NEAR(direction[1], transverse * std::sin(phi), kTolerance);
        EXPECT_NEAR(direction[2], expectedZ, kTolerance);
    }
}

TEST(Healpix, AllCentersRoundTripAndOrderingsAgree) {
    constexpr std::array<std::uint32_t, 5> kSides{1, 2, 4, 8, 16};
    constexpr double kTolerance = 2.0e-14;
    PROFILE();
    for (auto const side : kSides) {
        Healpix const nested(side);
        Healpix const ring(side, Ordering::ring);
        for (std::uint64_t pixel = 0; pixel < nested.pixelCount(); pixel++) {
            auto const ringPixel = nested.convertOrdering(pixel, Ordering::ring);
            auto const direction = nested.pixelToVector(pixel);
            auto const ringDirection = ring.pixelToVector(ringPixel);
            EXPECT_EQ(nested.vectorToPixel(direction), pixel);
            EXPECT_EQ(ring.vectorToPixel(direction), ringPixel);
            EXPECT_EQ(ring.convertOrdering(ringPixel, Ordering::nested), pixel);
            for (size_t dimension = 0; dimension < NDIM; dimension++) {
                EXPECT_NEAR(direction[dimension], ringDirection[dimension], kTolerance);
            }
        }
    }
}

TEST(Healpix, NonPowerOfTwoRingMapsRoundTrip) {
    constexpr std::array<std::uint32_t, 3> kSides{3, 5, 7};
    PROFILE();
    for (auto const side : kSides) {
        Healpix const map(side, Ordering::ring);
        for (std::uint64_t pixel = 0; pixel < map.pixelCount(); pixel++) {
            EXPECT_EQ(map.vectorToPixel(map.pixelToVector(pixel)), pixel);
        }
        EXPECT_THROW(map.convertOrdering(0, Ordering::nested), std::invalid_argument);
    }
}

TEST(Healpix, LargestMapKeepsPolarRadiusAndUsesExactOrderingConversion) {
    constexpr std::uint32_t kSide = std::uint32_t{1} << 29;
    constexpr std::uint64_t kCount = std::uint64_t{12} * kSide * kSide;
    constexpr std::uint64_t kCap = std::uint64_t{2} * kSide * (kSide - 1);
    constexpr std::array<std::uint64_t, 10> kPixels{0, 3, kCap - 1, kCap, kCount / 2, kCount / 2 + 1, kCount - kCap - 1, kCount - kCap, kCount - 4, kCount - 1};
    PROFILE();
    Healpix const ring(kSide, Ordering::ring);
    Healpix const nested(kSide);
    EXPECT_EQ(ring.pixelCount(), kCount);
    for (auto const pixel : kPixels) {
        auto const direction = ring.pixelToVector(pixel);
        EXPECT_EQ(ring.vectorToPixel(direction), pixel);
        auto const nestedPixel = ring.convertOrdering(pixel, Ordering::nested);
        EXPECT_EQ(nested.convertOrdering(nestedPixel, Ordering::ring), pixel);
        EXPECT_EQ(nested.vectorToPixel(direction), nestedPixel);
    }
    auto const north = ring.pixelToVector(0);
    EXPECT_GT(north[0], 0.0);
    EXPECT_GT(north[1], 0.0);
    EXPECT_EQ(north[2], 1.0);
}

TEST(Healpix, ScalarAndNativeBatchesAgreeAtSeamsPolesAndTails) {
    constexpr size_t kRegularSamples = 259;
    constexpr std::uint32_t kSide = 1024;
    constexpr double kBoundary = 2.0 / 3.0;
    constexpr double kStablePoleBoundary = 0.99;
    constexpr double kTiny = 1.0e-150;
    constexpr double kHuge = 1.0e150;
    constexpr double kPhase = 0.731;
    constexpr double kHalfPi = std::numbers::pi / 2.0;
    PROFILE();
    std::vector<Direction> directions{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}, {1.0, -0.0, 0.0}, {1.0, -kTiny, 0.0},
        {kTiny, kTiny, 1.0}, {kTiny, 0.0, 0.0}, {kHuge, -kHuge, kHuge}};
    for (size_t index = 0; index < kRegularSamples; index++) {
        double const z = 2.0 * (static_cast<double>(index) + 0.5) / kRegularSamples - 1.0;
        double const phi = index * kPhase;
        double const transverse = std::sqrt((1.0 - z) * (1.0 + z));
        directions.push_back({transverse * std::cos(phi), transverse * std::sin(phi), z});
    }
    for (double const latitude : {kBoundary, -kBoundary, kStablePoleBoundary, -kStablePoleBoundary}) {
        for (double const z : {std::nextafter(latitude, -1.0), latitude, std::nextafter(latitude, 1.0)}) {
            for (int quadrant = 0; quadrant < 4; quadrant++) {
                double const phi = quadrant * kHalfPi;
                double const transverse = std::sqrt((1.0 - z) * (1.0 + z));
                directions.push_back({transverse * std::cos(phi), transverse * std::sin(phi), z});
            }
        }
    }
    for (auto const ordering : {Ordering::nested, Ordering::ring}) {
        Healpix const map(kSide, ordering);
        for (size_t tail = 0; tail <= Healpix::nativeLaneCount(); tail++) {
            size_t const count = directions.size() - tail;
            std::vector<std::uint64_t> pixels(count);
            map.vectorsToPixels(std::span<Direction const>(directions.data(), count), pixels);
            for (size_t index = 0; index < count; index++) {
                EXPECT_EQ(pixels[index], map.vectorToPixel(directions[index])) << "lane " << index;
            }
        }
        map.vectorsToPixels({}, {});
    }
}

TEST(Healpix, ResolutionAndBufferErrorsStayOperational) {
    constexpr std::uint32_t kTooLarge = (std::uint32_t{1} << 29) + 1;
    PROFILE();
    Healpix const map(4);
    std::array<Direction, 1> directions{{{1.0, 0.0, 0.0}}};
    EXPECT_THROW(Healpix(0), std::invalid_argument);
    EXPECT_THROW(Healpix(3), std::invalid_argument);
    EXPECT_THROW((void) Healpix(kTooLarge), std::invalid_argument);
    EXPECT_THROW(Healpix(1, static_cast<Ordering>(99)), std::invalid_argument);
    EXPECT_THROW(map.pixelToVector(map.pixelCount()), std::out_of_range);
    EXPECT_THROW(map.convertOrdering(map.pixelCount(), Ordering::ring), std::out_of_range);
    EXPECT_THROW(map.vectorsToPixels(directions, {}), std::invalid_argument);
}

}    // namespace
