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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <gtest/gtest.h>
#include <hpx/type_support/bit_cast.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::CellExpansion;
using cosmictiger::CellMultipole;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleStorage;
namespace gravity = cosmictiger::gravity;
namespace stf = cosmictiger::gravity::stf;

class CellMultipolesTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        int deviceCount = 0;
        try {
            deviceCount = cosmictiger::cuda::deviceCount().get();
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA device unavailable: " << error.what();
            }
            throw;
        }
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        int const device = cosmictiger::cuda::device().get();
        auto const properties = cosmictiger::cuda::deviceProperties(device).get();
        if (!properties.managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
#endif
    }
};

void appendRaw(ParticleStorage& particles, std::array<std::uint32_t, NDIM> const& raw) {
    PROFILE();
    particles.append({Coordinate::fromRaw(raw[0]), Coordinate::fromRaw(raw[1]), Coordinate::fromRaw(raw[2])}, {0.0f, 0.0f, 0.0f}, 0);
}

void expectZero(CellMultipole const& multipole, CellExpansion const& expansion) {
    PROFILE();
    for (auto const value : multipole) {
        EXPECT_EQ(value, 0.0f);
    }
    for (auto const value : expansion) {
        EXPECT_EQ(value, 0.0f);
    }
}

TEST(CellMultipoles, ConfiguredFloatStorageIsInitiallyZeroInEveryCell) {
    static_assert(std::is_same_v<CellMultipole::value_type, float>);
    static_assert(std::is_same_v<CellExpansion::value_type, float>);
    static_assert(std::tuple_size_v<CellMultipole> == size_t((gravity::kMultipoleOrder + 1) * (gravity::kMultipoleOrder + 1)));
    static_assert(std::tuple_size_v<CellExpansion> == size_t((gravity::kExpansionOrder + 1) * (gravity::kExpansionOrder + 1) + 1));
    static_assert(std::is_same_v<decltype(std::declval<CartesianGrid&>().expansion(GridIndices{})), CellExpansion&>);
    PROFILE();
    CartesianGrid const grid(1, 0, 0);
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const& cell = grid.cell(grid.layout().cellCoordinates(index));
        expectZero(cell.multipole, cell.expansion);
        EXPECT_DOUBLE_EQ(cell.secondMomentTrace, 0.0);
    }
}

TEST_F(CellMultipolesTest, GeometricCenterKeepsDipolesAndAnalyticQuadrupoles) {
    PROFILE();
    ParticleStorage particles;
    // Unit-cell offsets: (1/4,1/8,-1/8), (-1/8,1/4,1/8).
    // Both radii squared are 3/32. Q_ij = sum m*(r_i*r_j-delta_ij*r^2/3).
    appendRaw(particles, {0xc0000000U, 0xa0000000U, 0x60000000U});
    appendRaw(particles, {0x60000000U, 0xc0000000U, 0xa0000000U});
    CartesianGrid grid(0, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 2});
    grid.computeMultipole({0, 0, 0}, particles, 2.0f);
    auto const& moment = grid.cell({0, 0, 0}).multipole;
    EXPECT_FLOAT_EQ(moment[stf::index(0, 0, 0)], 4.0f);
    EXPECT_DOUBLE_EQ(grid.cell({0, 0, 0}).secondMomentTrace, 3.0 / 8.0);
    if constexpr (gravity::kMultipoleOrder >= 1) {
        EXPECT_FLOAT_EQ(moment[stf::index(1, 0, 0)], 0.25f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 1, 0)], 0.75f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 0, 1)], 0.0f);
    }
    if constexpr (gravity::kMultipoleOrder >= 2) {
        EXPECT_FLOAT_EQ(moment[stf::index(2, 0, 0)], 1.0f / 32.0f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 2, 0)], 1.0f / 32.0f);
        EXPECT_FLOAT_EQ(moment[stf::index(1, 1, 0)], 0.0f);
        EXPECT_FLOAT_EQ(moment[stf::index(1, 0, 1)], -3.0f / 32.0f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 1, 1)], 1.0f / 32.0f);
        EXPECT_FLOAT_EQ(-moment[stf::index(2, 0, 0)] - moment[stf::index(0, 2, 0)], -1.0f / 16.0f);
    }
}

TEST_F(CellMultipolesTest, HigherRanksMatchAxialHarmonicCoefficients) {
    PROFILE();
    ParticleStorage particles;
    appendRaw(particles, {0xc0000000U, 0x80000000U, 0x80000000U});
    CartesianGrid grid(0, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.computeMultipole({0, 0, 0}, particles, 3.0f);
    auto const& moment = grid.cell({0, 0, 0}).multipole;
    // STF(x^n) on the x axis has all-x coefficient n!/(2n-1)!!.
    // Mixed coefficients follow the traceless condition and transverse symmetry.
    if constexpr (gravity::kMultipoleOrder >= 3) {
        EXPECT_FLOAT_EQ(moment[stf::index(3, 0, 0)], float(3.0 * (2.0 / 5.0) / 64.0));
        EXPECT_FLOAT_EQ(moment[stf::index(1, 2, 0)], float(-3.0 / (5.0 * 64.0)));
    }
    if constexpr (gravity::kMultipoleOrder >= 4) {
        EXPECT_FLOAT_EQ(moment[stf::index(4, 0, 0)], float(3.0 * (8.0 / 35.0) / 256.0));
        EXPECT_FLOAT_EQ(moment[stf::index(2, 2, 0)], float(-3.0 * (4.0 / 35.0) / 256.0));
    }
    if constexpr (gravity::kMultipoleOrder >= 5) {
        EXPECT_FLOAT_EQ(moment[stf::index(5, 0, 0)], float(3.0 * (8.0 / 63.0) / 1024.0));
        EXPECT_FLOAT_EQ(moment[stf::index(3, 2, 0)], float(-3.0 * (4.0 / 63.0) / 1024.0));
        EXPECT_FLOAT_EQ(moment[stf::index(1, 4, 0)], float(3.0 / (21.0 * 1024.0)));
    }
}

TEST_F(CellMultipolesTest, NormalizationUsesCellWidth) {
    PROFILE();
    ParticleStorage particles;
    // Width=1/2, center=(1/4,1/4,1/4), physical x offset=1/8.
    appendRaw(particles, {0x60000000U, 0x40000000U, 0x40000000U});
    CartesianGrid grid(1, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.computeMultipole({0, 0, 0}, particles);
    auto const& moment = grid.cell({0, 0, 0}).multipole;
    EXPECT_FLOAT_EQ(moment[0], 1.0f);
    if constexpr (gravity::kMultipoleOrder >= 1) {
        EXPECT_FLOAT_EQ(moment[stf::index(1, 0, 0)], 0.25f);
    }
    if constexpr (gravity::kMultipoleOrder >= 2) {
        EXPECT_FLOAT_EQ(moment[stf::index(2, 0, 0)], float(1.0 / 24.0));
    }
}

TEST_F(CellMultipolesTest, PeriodicGhostUsesLocalCopiesAndCanonicalCenter) {
    constexpr GridIndices kGhost{-1, 0, 0};
    PROFILE();

    ParticleStorage particles;
    appendRaw(particles, {0x20000000U, 0x20000000U, 0x20000000U});
    appendRaw(particles, {0xf0000000U, 0x30000000U, 0x10000000U});
    CartesianGrid grid(2, 1, 0);
    grid.setGhostRanges(kGhost, {1, 2}, {1000, 1001});
    ASSERT_NE(grid.layout().cellOwner(kGhost).localityIndex, grid.layout().localityIndex());
    grid.computeMultipoles(particles);
    auto const& moment = grid.cell(kGhost).multipole;
    EXPECT_FLOAT_EQ(moment[0], 1.0f);
    if constexpr (gravity::kMultipoleOrder >= 1) {
        EXPECT_FLOAT_EQ(moment[stf::index(1, 0, 0)], 0.25f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 1, 0)], 0.25f);
        EXPECT_FLOAT_EQ(moment[stf::index(0, 0, 1)], -0.25f);
    }
    EXPECT_FLOAT_EQ(grid.cell({0, 0, 0}).multipole[0], 0.0f);
}

TEST_F(CellMultipolesTest, FineCellsSubtractFixedPositionsBeforeFloatConversion) {
    size_t constexpr kSide = size_t(1) << 20;
    size_t constexpr kHalf = kSide / 2;
    size_t constexpr kRank = kHalf + kSide * (kHalf + kSide * kHalf);
    std::uint32_t constexpr kRawCenter = 0x80000800U;
    PROFILE();

    if constexpr (gravity::kMultipoleOrder < 1) {
        GTEST_SKIP() << "dipole storage is disabled by this configuration";
    }
    CartesianGrid grid(20, 20, kRank);
    ParticleStorage particles;
    appendRaw(particles, {kRawCenter + 1, kRawCenter + 17, kRawCenter - 19});
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.computeMultipole({0, 0, 0}, particles);
    auto const& moment = grid.cell({0, 0, 0}).multipole;
    // The cell is 4096 raw units wide; absolute float positions discard these offsets.
    EXPECT_FLOAT_EQ(moment[stf::index(1, 0, 0)], 1.0f / 4096.0f);
    EXPECT_FLOAT_EQ(moment[stf::index(0, 1, 0)], 17.0f / 4096.0f);
    EXPECT_FLOAT_EQ(moment[stf::index(0, 0, 1)], -19.0f / 4096.0f);
}

TEST_F(CellMultipolesTest, RecomputeReplacesMomentsAndPreservesExpansion) {
    PROFILE();
    ParticleStorage particles;
    appendRaw(particles, {0xc0000000U, 0x80000000U, 0x80000000U});
    CartesianGrid grid(0, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.expansion({0, 0, 0}).fill(7.25f);
    grid.computeMultipole({0, 0, 0}, particles, 2.0f);
    particles.position(0, 0) = Coordinate::fromRaw(0x60000000U);
    grid.computeMultipoles(particles, 3.0f);
    auto const& cell = grid.cell({0, 0, 0});
    EXPECT_FLOAT_EQ(cell.multipole[0], 3.0f);
    if constexpr (gravity::kMultipoleOrder >= 1) {
        EXPECT_FLOAT_EQ(cell.multipole[stf::index(1, 0, 0)], -3.0f / 8.0f);
    }
    for (auto const value : cell.expansion) {
        EXPECT_EQ(value, 7.25f);
    }
    grid.setInteriorRange({0, 0, 0}, {1, 1});
    grid.expansion({0, 0, 0}).fill(-2.0f);
    grid.computeMultipole({0, 0, 0}, particles);
    for (auto const value : cell.multipole) {
        EXPECT_EQ(value, 0.0f);
    }
    EXPECT_DOUBLE_EQ(cell.secondMomentTrace, 0.0);
    for (auto const value : cell.expansion) {
        EXPECT_EQ(value, -2.0f);
    }
}

TEST_F(CellMultipolesTest, BulkMatchesSingleCellBitwiseForGhostsEmptyAndSkewedCells) {
    size_t constexpr kCellExponent = 2;
    std::int64_t constexpr kRawWidth = std::int64_t(1) << (Coordinate::fractionalBits() - kCellExponent);
    PROFILE();

    CartesianGrid bulk(kCellExponent, 0, 0);
    ParticleStorage particles;
    particles.reserve(4096);
    for (size_t cellIndex = 0; cellIndex < bulk.layout().storedCellCount(); cellIndex++) {
        auto const coordinates = bulk.layout().cellCoordinates(cellIndex);
        auto const global = bulk.layout().globalCoordinates(coordinates);
        size_t const count = cellIndex == 103 ? 1025 : (cellIndex % 7 == 0 ? 0 : cellIndex % 9 + 1);
        auto const begin = particles.size();
        for (size_t particle = 0; particle < count; particle++) {
            std::array<std::uint32_t, NDIM> raw{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                std::int64_t const displacement = static_cast<std::int64_t>((particle + 3 * axis) % 15) - 7;
                raw[axis] = static_cast<std::uint32_t>(global[axis] * kRawWidth + kRawWidth / 2 + displacement * (kRawWidth / 16));
            }
            appendRaw(particles, raw);
        }
        cosmictiger::ParticleRange const range{begin, particles.size()};
        if (bulk.layout().isInterior(coordinates)) {
            bulk.setInteriorRange(coordinates, range);
        } else {
            bulk.setGhostRanges(coordinates, range, {begin + 100000, particles.size() + 100000});
        }
        bulk.expansion(coordinates).fill(static_cast<float>(int(cellIndex % 17) - 8));
    }
    CartesianGrid reference = bulk;
    for (size_t index = 0; index < reference.layout().storedCellCount(); index++) {
        reference.computeMultipole(reference.layout().cellCoordinates(index), particles, 0.75f);
    }
    bulk.computeMultipoles(particles, 0.75f);
    for (size_t index = 0; index < bulk.layout().storedCellCount(); index++) {
        auto const coordinates = bulk.layout().cellCoordinates(index);
        auto const& actual = bulk.cell(coordinates);
        auto const& expected = reference.cell(coordinates);
        for (size_t coefficient = 0; coefficient < actual.multipole.size(); coefficient++) {
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(actual.multipole[coefficient]), hpx::bit_cast<std::uint32_t>(expected.multipole[coefficient]))
                << "cell " << index << ", coefficient " << coefficient;
        }
        EXPECT_EQ(hpx::bit_cast<std::uint64_t>(actual.secondMomentTrace), hpx::bit_cast<std::uint64_t>(expected.secondMomentTrace)) << "cell " << index;
        EXPECT_EQ(actual.expansion, expected.expansion);
        EXPECT_EQ(actual.localParticles.begin, expected.localParticles.begin);
        EXPECT_EQ(actual.localParticles.end, expected.localParticles.end);
        EXPECT_EQ(actual.ownerParticles.begin, expected.ownerParticles.begin);
        EXPECT_EQ(actual.ownerParticles.end, expected.ownerParticles.end);
    }
}

TEST_F(CellMultipolesTest, RangeChangesInvalidateBothGravityArrays) {
    constexpr GridIndices kGhost{-1, -1, -1};
    PROFILE();

    ParticleStorage particles;
    appendRaw(particles, {0x80000000U, 0x80000000U, 0x80000000U});
    CartesianGrid grid(0, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.computeMultipole({0, 0, 0}, particles);
    grid.expansion({0, 0, 0}).fill(4.0f);
    grid.fullMultipole({0, 0, 0}).fill(6.0f);
    grid.fullExpansion({0, 0, 0}).fill(7.0f);
    grid.setInteriorRange({0, 0, 0}, {0, 0});
    expectZero(grid.cell({0, 0, 0}).multipole, grid.cell({0, 0, 0}).expansion);
    for (auto const value : grid.cell({0, 0, 0}).fullMultipole) {
        EXPECT_EQ(value, 0.0f);
    }
    for (auto const value : grid.cell({0, 0, 0}).fullExpansion) {
        EXPECT_EQ(value, 0.0f);
    }
    EXPECT_DOUBLE_EQ(grid.cell({0, 0, 0}).secondMomentTrace, 0.0);
    grid.setGhostRanges(kGhost, {0, 1}, {9, 10});
    grid.computeMultipole(kGhost, particles);
    grid.expansion(kGhost).fill(5.0f);
    grid.fullMultipole(kGhost).fill(6.0f);
    grid.fullExpansion(kGhost).fill(7.0f);
    grid.setGhostRanges(kGhost, {0, 0}, {9, 9});
    expectZero(grid.cell(kGhost).multipole, grid.cell(kGhost).expansion);
    for (auto const value : grid.cell(kGhost).fullMultipole) {
        EXPECT_EQ(value, 0.0f);
    }
    for (auto const value : grid.cell(kGhost).fullExpansion) {
        EXPECT_EQ(value, 0.0f);
    }
    EXPECT_DOUBLE_EQ(grid.cell(kGhost).secondMomentTrace, 0.0);
}

TEST_F(CellMultipolesTest, InvalidBulkRangeIsCheckedWhenItsCellIsReached) {
    constexpr GridIndices kFirst{-1, -1, -1};
    PROFILE();

    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    ParticleStorage particles;
    appendRaw(particles, {0xd0000000U, 0xd0000000U, 0xd0000000U});
    CartesianGrid grid(1, 1, 0);
    grid.setGhostRanges(kFirst, {0, 1}, {99, 100});
    grid.computeMultipole(kFirst, particles);
    grid.expansion(kFirst).fill(8.0f);
    auto const previous = grid.cell(kFirst);
    grid.setInteriorRange({0, 0, 0}, {0, 2});
    particles.position(0, 0) = Coordinate::fromRaw(0xe0000000U);
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.computeMultipoles(particles), std::runtime_error);
    }
    EXPECT_NE(grid.cell(kFirst).multipole, previous.multipole);
    EXPECT_NE(grid.cell(kFirst).secondMomentTrace, previous.secondMomentTrace);
    EXPECT_EQ(grid.cell(kFirst).expansion, previous.expansion);
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.computeMultipole({0, 0, 0}, particles), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.computeMultipole({2, 0, 0}, particles), std::runtime_error);
    }
}

class LocalStorageScope {
public:
    LocalStorageScope()
      : previous_(std::move(cosmictiger::localParticleStorage())) {
        PROFILE();
    }

    ~LocalStorageScope() {
        PROFILE();
        cosmictiger::localParticleStorage() = std::move(previous_);
    }

private:
    ParticleStorage previous_;
};

TEST_F(CellMultipolesTest, ConvenienceFunctionUsesLocalParticleStorage) {
    PROFILE();
    LocalStorageScope scope;
    auto& particles = cosmictiger::localParticleStorage();
    appendRaw(particles, {0x80000000U, 0x80000000U, 0x80000000U});
    CartesianGrid grid(0, 0, 0);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    grid.computeMultipoles(4.0f);
    EXPECT_FLOAT_EQ(grid.cell({0, 0, 0}).multipole[0], 4.0f);
    for (size_t index = 1; index < grid.cell({0, 0, 0}).multipole.size(); index++) {
        EXPECT_EQ(grid.cell({0, 0, 0}).multipole[index], 0.0f);
    }
}

}    // namespace
