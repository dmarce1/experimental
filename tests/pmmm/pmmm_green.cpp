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
#include <cosmictiger/gravity/pmmm/pmmm_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/profiler.hpp>
#include <gtest/gtest.h>
#include <hpx/type_support/bit_cast.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stddef.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

namespace gravity = cosmictiger::gravity;

static_assert(std::tuple_size_v<gravity::PmmmGreenCoefficients<0>> == 1);
static_assert(std::tuple_size_v<gravity::PmmmGreenCoefficients<2>> == 9);

template <int neighborWidth>
void checkNearCube() {
    PROFILE();
    int nearCount = 0;
    for (int z = -3; z <= 3; z++) {
        for (int y = -3; y <= 3; y++) {
            for (int x = -3; x <= 3; x++) {
                bool const expected = std::abs(x) <= neighborWidth && std::abs(y) <= neighborWidth && std::abs(z) <= neighborWidth;
                bool const actual = gravity::pmmmNear<neighborWidth>({x, y, z});
                EXPECT_EQ(actual, expected) << "offset=(" << x << "," << y << "," << z << ")";
                nearCount += actual;
            }
        }
    }
    EXPECT_EQ(nearCount, (2 * neighborWidth + 1) * (2 * neighborWidth + 1) * (2 * neighborWidth + 1));
}

TEST(PmmmGreen, WidthOneAndTwoExcludeCompleteNearCubes) {
    PROFILE();
    checkNearCube<1>();
    checkNearCube<2>();
    EXPECT_TRUE(gravity::pmmmNear<1>({0, 0, 0}));
    EXPECT_FALSE(gravity::pmmmNear<1>({2, 0, 0}));
    EXPECT_TRUE(gravity::pmmmNear<2>({2, -2, 2}));
    EXPECT_FALSE(gravity::pmmmNear<2>({3, 0, 0}));
}

TEST(PmmmGreen, FreeSpaceAxialMonopoleDipoleAndQuadrupole) {
    PROFILE();
    // The compact STF order is [1, y, x, z, yy, xy, xx, yz, xz].
    // R=(2,0,0), spacing=1/2; each rank n is spacing^n d^n(-1/r).
    auto const green = gravity::pmmmFreeSpaceGreen<2>({2.0, 0.0, 0.0}, 0.5);
    EXPECT_NEAR(green[0], -0.5, 1e-15);
    EXPECT_NEAR(green[1], 0.0, 1e-15);
    EXPECT_NEAR(green[2], 0.125, 1e-15);
    EXPECT_NEAR(green[3], 0.0, 1e-15);
    EXPECT_NEAR(green[4], 0.03125, 1e-15);
    EXPECT_NEAR(green[5], 0.0, 1e-15);
    EXPECT_NEAR(green[6], -0.0625, 1e-15);
    EXPECT_NEAR(green[7], 0.0, 1e-15);
    EXPECT_NEAR(green[8], 0.0, 1e-15);
    EXPECT_NEAR(green[4] + green[6], -green[4], 1e-15);
}

TEST(PmmmGreen, FreeSpaceCoefficientsHaveDerivativeParity) {
    constexpr std::array<double, NDIM> kR{1.25, -0.75, 2.0};
    PROFILE();

    auto const positive = gravity::pmmmFreeSpaceGreen<3>(kR, 0.4);
    auto const negative = gravity::pmmmFreeSpaceGreen<3>({-kR[0], -kR[1], -kR[2]}, 0.4);
    for (int rank = 0; rank <= 3; rank++) {
        double const sign = rank % 2 == 0 ? 1.0 : -1.0;
        for (int index = rank * rank; index < (rank + 1) * (rank + 1); index++) {
            EXPECT_NEAR(negative[index], sign * positive[index], 2e-13 * std::max(1.0, std::abs(positive[index]))) << index;
        }
    }
}

TEST(PmmmGreen, FreeSpaceHigherRanksMatchIndependentMonopoleM2L) {
    constexpr double kSpacing = 0.25;
    constexpr std::array<double, NDIM> kSeparation{1.25, -0.75, 2.0};
    PROFILE();

    cosmictiger::gravity::stf::Multipole<double, 5> source{};
    source[0] = 1.0;
    cosmictiger::gravity::stf::Expansion<double, 5> reference{};
    cosmictiger::gravity::stf::addM2L<double, 5, 5>(reference, source, kSeparation, kSpacing, kSpacing);
    auto const green = gravity::pmmmFreeSpaceGreen<5>(kSeparation, kSpacing);
    for (size_t index = 0; index < green.size(); index++) {
        EXPECT_NEAR(green[index], reference[index], 5e-13 * std::max(1.0, std::abs(reference[index]))) << index;
    }
}

TEST(PmmmGreen, OpenKernelZerosNearCellsAndUsesPhysicalCellSeparationElsewhere) {
    PROFILE();
    auto const edgeForWidthOne = gravity::pmmmOpenGreen<0, 1>({2, 0, 0}, 0.25);
    auto const edgeForWidthTwo = gravity::pmmmOpenGreen<0, 2>({2, 0, 0}, 0.25);
    EXPECT_NE(edgeForWidthOne[0], 0.0);
    EXPECT_EQ(edgeForWidthTwo[0], 0.0);
    for (auto const value : gravity::pmmmOpenGreen<2, 1>({1, -1, 1}, 0.25)) {
        EXPECT_EQ(value, 0.0);
    }
    for (auto const value : gravity::pmmmOpenGreen<2, 2>({2, -2, 2}, 0.25)) {
        EXPECT_EQ(value, 0.0);
    }
    for (auto const value : gravity::pmmmOpenGreen<2, 2>({0, 0, 0}, 0.25)) {
        EXPECT_EQ(value, 0.0);
    }

    auto const widthOne = gravity::pmmmOpenGreen<2, 1>({2, -1, 0}, 0.25);
    auto const expectedOne = gravity::pmmmFreeSpaceGreen<2>({0.5, -0.25, 0.0}, 0.25);
    auto const widthTwo = gravity::pmmmOpenGreen<2, 2>({3, 0, 1}, 0.25);
    auto const expectedTwo = gravity::pmmmFreeSpaceGreen<2>({0.75, 0.0, 0.25}, 0.25);
    for (size_t index = 0; index < widthOne.size(); index++) {
        EXPECT_NEAR(widthOne[index], expectedOne[index], 2e-14 * std::max(1.0, std::abs(expectedOne[index])));
        EXPECT_NEAR(widthTwo[index], expectedTwo[index], 2e-14 * std::max(1.0, std::abs(expectedTwo[index])));
    }
}

TEST(PmmmGreen, OpenFilterBlockUsesFftLocalityAndAliasPlaneLayout) {
    PROFILE();
    auto const positiveX = gravity::pmmmOpenGreenFilterBlock<1, 1, 4>(4, 2, 0, 0.25, 0);
    auto const negativeX = gravity::pmmmOpenGreenFilterBlock<1, 1, 4>(4, 2, 4, 0.25, 0);
    auto const negativeZ = gravity::pmmmOpenGreenFilterBlock<1, 1, 4>(4, 2, 1, 0.25, 0);
    ASSERT_EQ(positiveX.size(), 64);
    ASSERT_EQ(negativeX.size(), 64);
    ASSERT_EQ(negativeZ.size(), 64);

    // The padded side is eight and the block side is four. Global +x=2 is
    // local index 2 on locality 0; global -x=6 is index 2 on locality 4.
    auto const expectedPositive = gravity::pmmmOpenGreen<1, 1>({2, 0, 0}, 0.25);
    auto const expectedNegative = gravity::pmmmOpenGreen<1, 1>({-2, 0, 0}, 0.25);
    for (size_t lane = 0; lane < 4; lane++) {
        EXPECT_NEAR(positiveX[2][lane], expectedPositive[lane], 1e-14);
        EXPECT_NEAR(negativeX[2][lane], expectedNegative[lane], 1e-14);
        EXPECT_EQ(negativeX[0][lane], 0.0);    // global x=4 alias plane
        EXPECT_EQ(positiveX[0][lane], 0.0);    // excluded self cell
    }
    // Locality 1 is the negative-z block because FFT locality IDs are z-fastest.
    auto const expectedNegativeZ = gravity::pmmmOpenGreen<1, 1>({0, 0, -2}, 0.25);
    for (size_t lane = 0; lane < 4; lane++) {
        EXPECT_NEAR(negativeZ[2 * 16][lane], expectedNegativeZ[lane], 1e-14);
    }
}

TEST(PmmmGreen, PeriodicOriginHasFiniteImagesAndNearCellsRetainPeriodicContribution) {
    PROFILE();
    // Direct images in the central near cube are removed; periodic images and
    // the reciprocal-space term still contribute at the same mesh offsets.
    auto const origin = gravity::pmmmPeriodicGreen<2, 1>({0, 0, 0}, 8, 0.125, 2, 2, 1.5);
    auto const nearOne = gravity::pmmmPeriodicGreen<2, 1>({1, 0, 0}, 8, 0.125, 2, 2, 1.5);
    auto const nearTwo = gravity::pmmmPeriodicGreen<2, 2>({2, 0, 0}, 8, 0.125, 2, 2, 1.5);
    EXPECT_NE(origin[0], 0.0);
    EXPECT_NE(nearOne[0], 0.0);
    EXPECT_NE(nearTwo[0], 0.0);
    EXPECT_NEAR(origin[1], 0.0, 1e-11);
    EXPECT_NEAR(origin[2], 0.0, 1e-11);
    EXPECT_NEAR(origin[3], 0.0, 1e-11);
}

TEST(PmmmGreen, PeriodicRankEightMatchesReferenceAndEwaldSplit) {
    PROFILE();
    // Independent OctoTigerII periodic correction sample: primary near-cell
    // Newtonian interaction removed, box length eight, unit cell spacing.
    auto const unit = gravity::pmmmPeriodicGreen<8, 1>({1, 0, 0}, 8, 1.0, 6, 6, 1.5);
    EXPECT_NEAR(unit[0], 0.3504764161589791, 3e-13);
    EXPECT_NEAR(unit[2], -0.008562488886943283, 3e-13);
    EXPECT_NEAR(unit[6], -0.001147839306255122, 3e-13);

    auto const firstSplit = gravity::pmmmPeriodicGreen<8, 1>({1, 0, 0}, 8, 0.125, 6, 6, 1.5);
    auto const secondSplit = gravity::pmmmPeriodicGreen<8, 1>({1, 0, 0}, 8, 0.125, 6, 6, 2.0);
    for (size_t index = 0; index < firstSplit.size(); index++) {
        double const tolerance = 2e-12 * std::max(1.0, std::abs(firstSplit[index]));
        EXPECT_NEAR(firstSplit[index], secondSplit[index], tolerance) << index;
        EXPECT_NEAR(0.125 * firstSplit[index], unit[index], tolerance) << index;
    }
}

TEST(PmmmGreen, PeriodicBackgroundLaplacianUsesPhysicalBoxDensity) {
    constexpr double kPi = std::numbers::pi;
    PROFILE();

    EXPECT_NEAR(gravity::pmmmPeriodicBackgroundLaplacian(8, 1.0), -kPi / 128.0, 2e-15);
    EXPECT_NEAR(gravity::pmmmPeriodicBackgroundLaplacian(8, 0.125), -kPi / 16.0, 2e-15);
    if (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(gravity::pmmmPeriodicBackgroundLaplacian(0, 1.0), std::runtime_error);
        EXPECT_THROW(gravity::pmmmPeriodicBackgroundLaplacian(8, 0.0), std::runtime_error);
    }
}

TEST(PmmmGreen, PeriodicBackgroundCompletesTraceAndCancelsSingleSourceSelfGradient) {
    constexpr double kMass = 2.5;
    constexpr std::array<double, NDIM> kSourceOffset{0.25, -0.125, 0.375};
    constexpr std::array<double, NDIM> kDipole{kMass * kSourceOffset[0], kMass * kSourceOffset[1], kMass * kSourceOffset[2]};
    constexpr double kSecondMoment = kMass * (kSourceOffset[0] * kSourceOffset[0] + kSourceOffset[1] * kSourceOffset[1] + kSourceOffset[2] * kSourceOffset[2]);
    PROFILE();

    double const b = gravity::pmmmPeriodicBackgroundLaplacian(8, 0.125);
    cosmictiger::gravity::stf::Expansion<double, 2, true> local{};
    gravity::pmmmAddPeriodicBackground<double, 2>(local, kMass, kDipole, kSecondMoment, 8, 0.125);

    EXPECT_NEAR(local[0], b * kSecondMoment / (2.0 * NDIM), 1e-14);
    EXPECT_NEAR(local[1], -b * kDipole[1] / static_cast<double>(NDIM), 1e-14);    // y
    EXPECT_NEAR(local[2], -b * kDipole[0] / static_cast<double>(NDIM), 1e-14);    // x
    EXPECT_NEAR(local[3], -b * kDipole[2] / static_cast<double>(NDIM), 1e-14);    // z
    EXPECT_NEAR(local[4], b * kMass / static_cast<double>(NDIM), 1e-14);          // yy
    EXPECT_NEAR(local[6], b * kMass / static_cast<double>(NDIM), 1e-14);          // xx
    EXPECT_NEAR(local[9], b * kMass, 1e-14);                                      // appended trace
    EXPECT_EQ(local[5], 0.0);                                                     // xy
    EXPECT_EQ(local[7], 0.0);                                                     // yz
    EXPECT_EQ(local[8], 0.0);                                                     // xz

    double const zz = cosmictiger::gravity::stf::component<double, 2, true>(local, 0, 0, 2);
    EXPECT_NEAR(zz, local[9] - local[6] - local[4], 1e-14);
    EXPECT_NEAR(zz, b * kMass / static_cast<double>(NDIM), 1e-14);

    // The completed quadratic is (b/6)*mass*|t-sourceOffset|^2. Its value
    // and normalized gradient vanish when evaluated at that source offset.
    double const x = kSourceOffset[0], y = kSourceOffset[1], z = kSourceOffset[2];
    double const value = local[0] + local[2] * x + local[1] * y + local[3] * z + 0.5 * (local[6] * x * x + local[4] * y * y + zz * z * z) + local[5] * x * y +
        local[8] * x * z + local[7] * y * z;
    std::array<double, NDIM> const gradient{local[2] + local[6] * x + local[5] * y + local[8] * z, local[1] + local[5] * x + local[4] * y + local[7] * z,
        local[3] + local[8] * x + local[7] * y + zz * z};
    EXPECT_NEAR(value, 0.0, 1e-14);
    for (double const component : gradient) {
        EXPECT_NEAR(component, 0.0, 1e-14);
    }

    cosmictiger::gravity::stf::Expansion<double, 2, true> preexisting{};
    preexisting.fill(0.1);
    gravity::pmmmAddPeriodicBackground<double, 2>(preexisting, kMass, kDipole, kSecondMoment, 8, 0.125);
    for (size_t index = 0; index < local.size(); index++) {
        EXPECT_NEAR(preexisting[index], local[index] + 0.1, 1e-14) << index;
    }
}

TEST(PmmmGreen, PeriodicFilterBlockUsesUnpaddedFftLayout) {
    PROFILE();
    auto const originBlock = gravity::pmmmPeriodicGreenFilterBlock<1, 1, 4>(8, 2, 0, 0.125, 0, 2, 2, 1.5);
    auto const negativeXBlock = gravity::pmmmPeriodicGreenFilterBlock<1, 1, 4>(8, 2, 4, 0.125, 0, 2, 2, 1.5);
    auto const negativeZBlock = gravity::pmmmPeriodicGreenFilterBlock<1, 1, 4>(8, 2, 1, 0.125, 0, 2, 2, 1.5);
    ASSERT_EQ(originBlock.size(), 64);
    ASSERT_EQ(negativeXBlock.size(), 64);
    ASSERT_EQ(negativeZBlock.size(), 64);

    auto const origin = gravity::pmmmPeriodicGreen<1, 1>({0, 0, 0}, 8, 0.125, 2, 2, 1.5);
    auto const positiveX = gravity::pmmmPeriodicGreen<1, 1>({1, 0, 0}, 8, 0.125, 2, 2, 1.5);
    auto const negativeX = gravity::pmmmPeriodicGreen<1, 1>({7, 0, 0}, 8, 0.125, 2, 2, 1.5);
    auto const negativeZ = gravity::pmmmPeriodicGreen<1, 1>({0, 0, 7}, 8, 0.125, 2, 2, 1.5);
    for (size_t lane = 0; lane < 4; lane++) {
        EXPECT_NEAR(originBlock[0][lane], origin[lane], 1e-13);
        EXPECT_NEAR(originBlock[1][lane], positiveX[lane], 1e-13);
        EXPECT_NEAR(negativeXBlock[3][lane], negativeX[lane], 1e-13);
        EXPECT_NEAR(negativeZBlock[3 * 16][lane], negativeZ[lane], 1e-13);
    }
    EXPECT_NE(originBlock[0][0], 0.0);    // no zero-filled central alias plane
}

TEST(PmmmGreen, PeriodicFilterBlockMatchesSerialCellsBitwiseIncludingWrappedCellsAndTail) {
    constexpr size_t kSideLength = 8;
    constexpr size_t kBlockSide = kSideLength / 2;
    constexpr size_t kBatchSize = 6;
    PROFILE();

    // Opposite corner blocks include the origin, the half-period boundary,
    // and wrapping in every axis. Build the reference with serial cell calls
    // and signed offsets; the block builder receives unwrapped FFT indices.
    for (size_t const localityId : {size_t(0), size_t(7)}) {
        int const firstCoordinate = localityId == 0 ? 0 : static_cast<int>(kBlockSide);
        std::vector<gravity::PmmmGreenCoefficients<2>> serial;
        serial.reserve(kBlockSide * kBlockSide * kBlockSide);
        for (size_t z = 0; z < kBlockSide; z++) {
            for (size_t y = 0; y < kBlockSide; y++) {
                for (size_t x = 0; x < kBlockSide; x++) {
                    std::array<int, NDIM> offset{
                        firstCoordinate + static_cast<int>(x), firstCoordinate + static_cast<int>(y), firstCoordinate + static_cast<int>(z)};
                    for (int& coordinate : offset) {
                        if (coordinate > static_cast<int>(kSideLength / 2)) {
                            coordinate -= static_cast<int>(kSideLength);
                        }
                    }
                    serial.push_back(gravity::pmmmPeriodicGreen<2, 1>(offset, kSideLength, 0.125, 2, 2, 1.7));
                }
            }
        }
        // Together these batches cover all nine coefficients and the three
        // unused lanes in the final batch. With multiple HPX workers the
        // 64-cell block exercises concurrent tasks against the serial oracle.
        for (size_t const firstCoefficient : {size_t(0), kBatchSize}) {
            auto const block = gravity::pmmmPeriodicGreenFilterBlock<2, 1, kBatchSize>(kSideLength, 2, localityId, 0.125, firstCoefficient, 2, 2, 1.7);
            ASSERT_EQ(block.size(), serial.size());
            for (size_t index = 0; index < block.size(); index++) {
                for (size_t lane = 0; lane < kBatchSize; lane++) {
                    size_t const coefficient = firstCoefficient + lane;
                    double const expected = coefficient < serial[index].size() ? serial[index][coefficient] : 0.0;
                    EXPECT_EQ(hpx::bit_cast<std::uint64_t>(block[index][lane]), hpx::bit_cast<std::uint64_t>(expected))
                        << "locality=" << localityId << " cell=" << index << " coefficient=" << coefficient;
                }
            }
        }
    }
}

TEST(PmmmGreen, RejectsInvalidInputs) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    EXPECT_THROW((gravity::pmmmFreeSpaceGreen<0>({0.0, 0.0, 0.0}, 1.0)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmFreeSpaceGreen<0>({1.0, 0.0, 0.0}, 0.0)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmOpenGreen<0, 1>({3, 0, 0}, -1.0)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmPeriodicGreen<0, 1>({0, 0, 0}, 0, 1.0)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmPeriodicGreen<0, 1>({0, 0, 0}, 8, 0.0)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmPeriodicGreen<0, 1>({0, 0, 0}, 8, 0.125, -1, 2)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmPeriodicGreen<0, 1>({0, 0, 0}, 8, 0.125, 2, -1)), std::runtime_error);
    EXPECT_THROW((gravity::pmmmPeriodicGreen<0, 1>({0, 0, 0}, 8, 0.125, 2, 2, 0.0)), std::runtime_error);
}

}    // namespace
