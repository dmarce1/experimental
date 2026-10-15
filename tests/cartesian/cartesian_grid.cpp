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
#include <cosmictiger/spatial_types.hpp>

#include <gtest/gtest.h>
#include <hpx/type_support/bit_cast.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::CartesianGridLayout;
using cosmictiger::GridIndices;
using cosmictiger::ParticleCount;

TEST(CartesianGrid, CubicDecompositionAndRegularSpacing) {
    PROFILE();
    CartesianGridLayout const layout(4, 2, 57);    // block (1,2,3)
    EXPECT_EQ(layout.globalSideLength(), 16);
    EXPECT_EQ(layout.globalCellCount(), 4096);
    EXPECT_EQ(layout.localitiesPerAxis(), 4);
    EXPECT_EQ(layout.localityCount(), 64);
    EXPECT_EQ(layout.localityIndex(), 57);
    EXPECT_EQ(layout.localityCoordinates(), (GridIndices{1, 2, 3}));
    EXPECT_EQ(layout.globalOrigin(), (GridIndices{4, 8, 12}));
    EXPECT_EQ(layout.interiorSideLength(), 4);
    EXPECT_EQ(layout.interiorCellCount(), 64);
    EXPECT_EQ(layout.storedSideLength(), 6);
    EXPECT_EQ(layout.storedCellCount(), 216);
    EXPECT_DOUBLE_EQ(layout.spacing(), 1.0 / 16);
    EXPECT_EQ(layout.cellCenter({0, 0, 0}), (std::array<double, NDIM>{4.5 / 16, 8.5 / 16, 12.5 / 16}));
    EXPECT_EQ(layout.cellCenter({1, 2, 3}), (std::array<double, NDIM>{5.5 / 16, 10.5 / 16, 15.5 / 16}));
}

TEST(CartesianGrid, ConfigurableTwoCellHaloPreservesPeriodicOwnershipAndIndexing) {
    PROFILE();
    CartesianGridLayout const layout(3, 1, 0, 2);
    EXPECT_EQ(layout.ghostWidth(), 2);
    EXPECT_EQ(layout.storedSideLength(), 8);
    EXPECT_EQ(layout.storedCellCount(), 512);
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        EXPECT_EQ(layout.cellIndex(coordinates), index);
        auto const owner = layout.cellOwner(coordinates);
        CartesianGridLayout const ownerLayout(3, 1, owner.localityIndex, 2);
        EXPECT_TRUE(ownerLayout.isInterior(owner.coordinates));
        EXPECT_EQ(layout.globalCoordinates(coordinates), ownerLayout.globalCoordinates(owner.coordinates));
    }
    EXPECT_EQ(layout.globalCoordinates({-2, -2, -2}), (GridIndices{6, 6, 6}));
    EXPECT_EQ(layout.globalCoordinates({5, 5, 5}), (GridIndices{5, 5, 5}));
    EXPECT_DOUBLE_EQ(layout.physicalBox({-2, 0, 0}).min(0), -0.25);
    EXPECT_DOUBLE_EQ(layout.physicalBox({5, 0, 0}).max(0), 0.75);
    EXPECT_THROW((CartesianGridLayout(3, 1, 0, 0)), std::invalid_argument);
    EXPECT_THROW((CartesianGridLayout(3, 1, 0, 9)), std::invalid_argument);
}

TEST(CartesianGrid, CompleteShellAndIndexRoundTrips) {
    PROFILE();
    CartesianGrid const grid(3, 1, 0);
    auto const& layout = grid.layout();
    auto const n = static_cast<std::int64_t>(layout.interiorSideLength());
    std::vector<bool> seen(layout.storedCellCount());
    size_t interiorCount = 0;
    // Count by the number of boundary coordinates: interior, faces, edges, corners.
    std::array<size_t, NDIM + 1> counts{};
    for (std::int64_t z = -1; z <= n; z++) {
        for (std::int64_t y = -1; y <= n; y++) {
            for (std::int64_t x = -1; x <= n; x++) {
                GridIndices const coordinates{x, y, z};
                size_t const index = layout.cellIndex(coordinates);
                ASSERT_LT(index, seen.size());
                EXPECT_FALSE(seen[index]);
                seen[index] = true;
                EXPECT_EQ(layout.cellCoordinates(index), coordinates);
                size_t boundaryAxes = 0;
                for (auto const coordinate : coordinates) {
                    boundaryAxes += coordinate == -1 || coordinate == n;
                }
                counts[boundaryAxes]++;
                interiorCount += layout.isInterior(coordinates);
                auto const& ranges = grid.cell(coordinates);
                EXPECT_EQ(ranges.localParticles.begin, 0);
                EXPECT_EQ(ranges.localParticles.end, 0);
                EXPECT_EQ(ranges.ownerParticles.begin, 0);
                EXPECT_EQ(ranges.ownerParticles.end, 0);
            }
        }
    }
    EXPECT_EQ(interiorCount, 64);
    EXPECT_EQ(counts, (std::array<size_t, NDIM + 1>{64, 96, 48, 8}));
    EXPECT_EQ(layout.cellIndex({-1, -1, -1}), 0);
    EXPECT_EQ(layout.cellIndex({0, -1, -1}), 1);
    EXPECT_EQ(layout.cellIndex({-1, 0, -1}), 6);
    EXPECT_EQ(layout.cellIndex({-1, -1, 0}), 36);
    EXPECT_EQ(layout.cellIndex({4, 4, 4}), 215);
}

TEST(CartesianGrid, EveryCellMapsToItsUniqueInteriorOwnerAcrossPeriodicSeams) {
    PROFILE();
    // Cover every locality, every stored cell, and n=1 as well as p=0.
    for (size_t p = 0; p <= 3; p++) {
        CartesianGridLayout const base(3, p, 0);
        std::vector<size_t> globalVisits(base.globalCellCount());
        for (size_t rank = 0; rank < base.localityCount(); rank++) {
            CartesianGridLayout const layout(3, p, rank);
            auto const block = layout.localityCoordinates();
            auto const n = static_cast<std::int64_t>(layout.interiorSideLength());
            for (size_t index = 0; index < layout.storedCellCount(); index++) {
                auto const coordinates = layout.cellCoordinates(index);
                GridIndices expected{};
                GridIndices expectedInterior{};
                GridIndices expectedBlock{};
                for (size_t axis = 0; axis < NDIM; axis++) {
                    expected[axis] = (block[axis] * n + coordinates[axis] + 8) % 8;
                    expectedBlock[axis] = expected[axis] / n;
                    expectedInterior[axis] = expected[axis] % n;
                }
                size_t const q = layout.localitiesPerAxis();
                size_t const expectedRank = expectedBlock[0] + q * (expectedBlock[1] + q * expectedBlock[2]);
                auto const owner = layout.cellOwner(coordinates);
                EXPECT_EQ(layout.globalCoordinates(coordinates), expected);
                EXPECT_EQ(owner.localityIndex, expectedRank);
                EXPECT_EQ(owner.coordinates, expectedInterior);
                CartesianGridLayout const ownerLayout(3, p, owner.localityIndex);
                EXPECT_TRUE(ownerLayout.isInterior(owner.coordinates));
                EXPECT_EQ(ownerLayout.globalCoordinates(owner.coordinates), expected);
                if (layout.isInterior(coordinates)) {
                    EXPECT_EQ(owner.localityIndex, rank);
                    EXPECT_EQ(owner.coordinates, coordinates);
                    globalVisits[expected[0] + 8 * (expected[1] + 8 * expected[2])]++;
                }
            }
        }
        for (auto const visits : globalVisits) {
            EXPECT_EQ(visits, 1);
        }
    }
}

TEST(CartesianGrid, All26GhostDirectionsResolveToNeighboringBlocks) {
    PROFILE();
    // Four localities per axis makes the two periodic neighbors distinct.
    CartesianGridLayout const layout(3, 2, 0);
    size_t directions = 0;
    for (std::int64_t dz = -1; dz <= 1; dz++) {
        for (std::int64_t dy = -1; dy <= 1; dy++) {
            for (std::int64_t dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                GridIndices const direction{dx, dy, dz};
                GridIndices cell{};
                GridIndices peer{};
                for (size_t axis = 0; axis < NDIM; axis++) {
                    cell[axis] = direction[axis] < 0 ? -1 : (direction[axis] > 0 ? 2 : 0);
                    peer[axis] = direction[axis] < 0 ? 3 : (direction[axis] > 0 ? 1 : 0);
                }
                EXPECT_FALSE(layout.isInterior(cell));
                EXPECT_EQ(layout.cellOwner(cell).localityIndex, peer[0] + 4 * (peer[1] + 4 * peer[2]));
                directions++;
            }
        }
    }
    EXPECT_EQ(directions, 26);
}

TEST(CartesianGrid, SingleCellSingleLocalityWrapsAllGhostsToTheInterior) {
    PROFILE();
    CartesianGrid grid(0, 0, 0);
    EXPECT_EQ(grid.layout().storedCellCount(), 27);
    EXPECT_EQ(grid.layout().interiorCellCount(), 1);
    for (size_t index = 0; index < 27; index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        EXPECT_EQ(grid.layout().globalCoordinates(coordinates), (GridIndices{0, 0, 0}));
        EXPECT_EQ(grid.layout().cellOwner(coordinates).localityIndex, 0);
        EXPECT_EQ(grid.layout().cellOwner(coordinates).coordinates, (GridIndices{0, 0, 0}));
    }
    grid.setGhostRanges({-1, 1, -1}, {3, 5}, {9, 11});
    EXPECT_EQ(grid.cell({-1, 1, -1}).localParticles.begin, 3);
    EXPECT_EQ(grid.cell({-1, 1, -1}).ownerParticles.begin, 9);
}

TEST(CartesianGrid, RangePairsStayIdenticalInInteriorAndIndependentInGhosts) {
    static_assert(std::is_same_v<decltype(cosmictiger::ParticleRange::begin), cosmictiger::ParticleStorage::Size>);
    static_assert(std::is_same_v<decltype(cosmictiger::Cell::physicalBox), cosmictiger::Box<double>>);
    static_assert(sizeof(cosmictiger::CellMultipole) == cosmictiger::gravity::stf::coefficientCount(cosmictiger::gravity::kMultipoleOrder) * sizeof(float));
    static_assert(
        sizeof(cosmictiger::CellExpansion) == (cosmictiger::gravity::stf::coefficientCount(cosmictiger::gravity::kExpansionOrder) + 1) * sizeof(float));
    PROFILE();
    CartesianGrid grid(2, 1, 0);
    ParticleCount const large = ParticleCount(1) << 40;
    grid.setInteriorRange({0, 1, 0}, {large, large + 17});
    auto const& interior = grid.cell({0, 1, 0});
    EXPECT_EQ(interior.localParticles.begin, large);
    EXPECT_EQ(interior.localParticles.end, large + 17);
    EXPECT_EQ(interior.ownerParticles.begin, large);
    EXPECT_EQ(interior.ownerParticles.end, large + 17);
    grid.setGhostRanges({-1, 2, -1}, {100, 105}, {large, large + 5});
    auto const& ghost = grid.cell({-1, 2, -1});
    EXPECT_EQ(ghost.localParticles.begin, 100);
    EXPECT_EQ(ghost.localParticles.end, 105);
    EXPECT_EQ(ghost.ownerParticles.begin, large);
    EXPECT_EQ(ghost.ownerParticles.end, large + 5);
    grid.setInteriorRange({0, 1, 0}, {7, 7});
    EXPECT_EQ(interior.localParticles.begin, 7);
    EXPECT_EQ(interior.ownerParticles.begin, 7);
    EXPECT_TRUE(interior.rungRangesValid);
    for (auto const range : interior.rungParticles) {
        EXPECT_EQ(range.begin, 7);
        EXPECT_EQ(range.end, 7);
    }
    EXPECT_EQ(grid.cell({1, 1, 1}).localParticles.end, 0);
}

TEST(CartesianGrid, SortsWholeParticlesWithinEachCellAndBuildsAllRungRanges) {
    constexpr std::array<std::uint8_t, 7> kRungs{63, 2, 0, 2, 1, 63, 0};

    static_assert(std::tuple_size_v<decltype(cosmictiger::Cell::rungParticles)> == 64);
    PROFILE();
    CartesianGrid grid(1, 0, 0);
    cosmictiger::ParticleStorage particles;
    for (size_t id = 0; id < kRungs.size(); id++) {
        auto const raw = static_cast<std::uint32_t>(id + 1);
        particles.append({cosmictiger::Coordinate::fromRaw(raw), cosmictiger::Coordinate::fromRaw(raw + 10), cosmictiger::Coordinate::fromRaw(raw + 20)},
            {float(raw), -float(raw), float(raw) + 0.5f}, kRungs[id], id >= 5);
    }
    grid.setInteriorRange({0, 0, 0}, {0, 5});
    grid.setGhostRanges({-1, 0, 0}, {5, 7}, {100, 102});
    EXPECT_FALSE(grid.cell({0, 0, 0}).rungRangesValid);
    EXPECT_FALSE(grid.cell({-1, 0, 0}).rungRangesValid);

    std::array<std::array<double, kRungs.size()>, NDIM> companion{};
    std::array<double*, NDIM> companionPointers{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        companionPointers[axis] = companion[axis].data();
        for (size_t id = 0; id < kRungs.size(); id++) {
            companion[axis][id] = double(id) + double(axis) / 4.0;
        }
    }
    grid.sortParticlesByRung(particles, companionPointers.data());
    for (size_t index = 0; index < particles.size(); index++) {
        auto const id = particles.position(0, index).raw() - 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_DOUBLE_EQ(companion[axis][index], double(id) + double(axis) / 4.0);
        }
    }
    for (auto const coordinates : {GridIndices{0, 0, 0}, GridIndices{-1, 0, 0}}) {
        auto const& cell = grid.cell(coordinates);
        ASSERT_TRUE(cell.rungRangesValid);
        ParticleCount cursor = cell.localParticles.begin;
        for (size_t rung = 0; rung < cell.rungParticles.size(); rung++) {
            auto const range = cell.rungParticles[rung];
            EXPECT_EQ(range.begin, cursor);
            for (ParticleCount index = range.begin; index < range.end; index++) {
                EXPECT_EQ(particles.flags(index).rung(), rung);
            }
            cursor = range.end;
        }
        EXPECT_EQ(cursor, cell.localParticles.end);
    }
    EXPECT_EQ(grid.cell({0, 0, 0}).rungParticles[0].end - grid.cell({0, 0, 0}).rungParticles[0].begin, 1);
    EXPECT_EQ(grid.cell({0, 0, 0}).rungParticles[2].end - grid.cell({0, 0, 0}).rungParticles[2].begin, 2);
    EXPECT_EQ(grid.cell({0, 0, 0}).rungParticles[63].end - grid.cell({0, 0, 0}).rungParticles[63].begin, 1);
    EXPECT_EQ(grid.cell({-1, 0, 0}).ownerParticles.begin, 100);
    EXPECT_EQ(grid.cell({-1, 0, 0}).ownerParticles.end, 102);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        auto const id = particles.position(0, index).raw() - 1;
        ASSERT_LT(id, kRungs.size());
        EXPECT_EQ(particles.position(1, index).raw(), id + 11);
        EXPECT_EQ(particles.position(2, index).raw(), id + 21);
        EXPECT_EQ(particles.velocity(0, index), float(id + 1));
        EXPECT_EQ(particles.velocity(1, index), -float(id + 1));
        EXPECT_EQ(particles.velocity(2, index), float(id + 1) + 0.5f);
        EXPECT_EQ(particles.flags(index).rung(), kRungs[id]);
        EXPECT_EQ(particles.flags(index).isGhost(), id >= 5);
    }

    grid.setInteriorRange({0, 0, 0}, {0, 5});
    EXPECT_FALSE(grid.cell({0, 0, 0}).rungRangesValid);
    grid.sortCellParticlesByRung({0, 0, 0}, particles);
    EXPECT_TRUE(grid.cell({0, 0, 0}).rungRangesValid);
    grid.setGhostRanges({-1, 0, 0}, {7, 8}, {100, 101});
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.sortCellParticlesByRung({-1, 0, 0}, particles), std::runtime_error);
    }
    EXPECT_FALSE(grid.cell({-1, 0, 0}).rungRangesValid);
}

TEST(CartesianGrid, BulkRungSortMatchesSingleCellForGhostsEmptyAndSkewedCells) {
    PROFILE();
    CartesianGrid bulk(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    cosmictiger::ParticleStorage referenceParticles;
    particles.reserve(8192);
    referenceParticles.reserve(8192);
    for (size_t cellIndex = 0; cellIndex < bulk.layout().storedCellCount(); cellIndex++) {
        auto const coordinates = bulk.layout().cellCoordinates(cellIndex);
        bool const ghost = !bulk.layout().isInterior(coordinates);
        size_t const count = cellIndex == 103 ? 1025 : (cellIndex % 7 == 0 ? 0 : cellIndex % 23 + 1);
        ParticleCount const begin = particles.size();
        for (size_t particle = 0; particle < count; particle++) {
            auto const id = static_cast<std::uint32_t>(particles.size() + 1);
            std::array<cosmictiger::Coordinate, NDIM> position{};
            std::array<float, NDIM> velocity{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                position[axis] = cosmictiger::Coordinate::fromRaw(id + 10000 * static_cast<std::uint32_t>(axis));
                velocity[axis] = static_cast<float>(id) + static_cast<float>(axis) * 0.25f;
            }
            std::uint64_t const rung = (id * 37 + cellIndex * 11) % (cosmictiger::Rung::maximumRung() + 1);
            particles.append(position, velocity, rung, ghost);
            referenceParticles.append(position, velocity, rung, ghost);
        }
        cosmictiger::ParticleRange const range{begin, particles.size()};
        if (ghost) {
            bulk.setGhostRanges(coordinates, range, {begin + 100000, particles.size() + 100000});
        } else {
            bulk.setInteriorRange(coordinates, range);
        }
        bulk.expansion(coordinates).fill(static_cast<float>(cellIndex % 17));
    }
    CartesianGrid reference = bulk;
    for (size_t index = 0; index < reference.layout().storedCellCount(); index++) {
        reference.sortCellParticlesByRung(reference.layout().cellCoordinates(index), referenceParticles);
    }
    bulk.sortParticlesByRung(particles);
    for (size_t index = 0; index < bulk.layout().storedCellCount(); index++) {
        auto const coordinates = bulk.layout().cellCoordinates(index);
        auto const& actual = bulk.cell(coordinates);
        auto const& expected = reference.cell(coordinates);
        EXPECT_TRUE(actual.rungRangesValid);
        EXPECT_EQ(actual.localParticles.begin, expected.localParticles.begin);
        EXPECT_EQ(actual.localParticles.end, expected.localParticles.end);
        EXPECT_EQ(actual.ownerParticles.begin, expected.ownerParticles.begin);
        EXPECT_EQ(actual.ownerParticles.end, expected.ownerParticles.end);
        EXPECT_EQ(actual.expansion, expected.expansion);
        for (size_t rung = 0; rung < actual.rungParticles.size(); rung++) {
            EXPECT_EQ(actual.rungParticles[rung].begin, expected.rungParticles[rung].begin) << "cell " << index << ", rung " << rung;
            EXPECT_EQ(actual.rungParticles[rung].end, expected.rungParticles[rung].end) << "cell " << index << ", rung " << rung;
        }
    }
    ASSERT_EQ(particles.size(), referenceParticles.size());
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.position(axis, index).raw(), referenceParticles.position(axis, index).raw());
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index)), hpx::bit_cast<std::uint32_t>(referenceParticles.velocity(axis, index)));
        }
        EXPECT_EQ(hpx::bit_cast<std::uint8_t>(particles.flags(index)), hpx::bit_cast<std::uint8_t>(referenceParticles.flags(index)));
    }
}

TEST(CartesianGrid, RejectsInvalidRangesAndCellKindsWithoutChangingCells) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    CartesianGrid grid(2, 1, 0);
    grid.setInteriorRange({0, 0, 0}, {4, 8});
    grid.setGhostRanges({-1, 0, 0}, {10, 12}, {20, 22});
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.setInteriorRange({0, 0, 0}, {9, 8}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.setInteriorRange({-1, 0, 0}, {0, 1}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.setGhostRanges({0, 0, 0}, {0, 1}, {0, 1}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.setGhostRanges({-1, 0, 0}, {12, 10}, {20, 22}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(grid.setGhostRanges({-1, 0, 0}, {10, 12}, {22, 20}), std::runtime_error);
    }
    EXPECT_EQ(grid.cell({0, 0, 0}).ownerParticles.end, 8);
    EXPECT_EQ(grid.cell({-1, 0, 0}).localParticles.begin, 10);
    EXPECT_EQ(grid.cell({-1, 0, 0}).ownerParticles.begin, 20);
}

TEST(CartesianGrid, RejectsInvalidTopologyAndOutOfBoundsCoordinates) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(CartesianGridLayout(1, 2, 0), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(CartesianGridLayout(2, 1, 8), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(CartesianGridLayout(0, 0, 1), std::runtime_error);
    }
    size_t const maximumExponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
    EXPECT_THROW(CartesianGridLayout(maximumExponent + 1, 0, 0), std::length_error);
    EXPECT_THROW(CartesianGridLayout(1, std::numeric_limits<size_t>::max(), 0), std::length_error);
    EXPECT_THROW(CartesianGrid(maximumExponent, 0, 0), std::length_error);
    CartesianGrid const grid(2, 1, 0);
    auto const& layout = grid.layout();
    for (size_t axis = 0; axis < NDIM; axis++) {
        for (auto const invalid : {std::int64_t(-2), std::int64_t(3), std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()}) {
            GridIndices coordinates{};
            coordinates[axis] = invalid;
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.cellIndex(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.isInterior(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.globalCoordinates(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.cellOwner(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.cellCenter(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(layout.physicalBox(coordinates), std::runtime_error);
            }
            if constexpr (cosmictiger::validationChecksEnabled()) {
                EXPECT_THROW(grid.cell(coordinates), std::runtime_error);
            }
        }
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(layout.cellCoordinates(layout.storedCellCount()), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(layout.cellCoordinates(std::numeric_limits<size_t>::max()), std::runtime_error);
    }
}

TEST(CartesianGrid, LargeGlobalCountsNeedOnlyLocalRangeStorage) {
    PROFILE();
    if constexpr (std::numeric_limits<size_t>::digits >= 64) {
        CartesianGrid grid(21, 21, (size_t(1) << 63) - 1);
        auto const& layout = grid.layout();
        EXPECT_EQ(layout.globalCellCount(), size_t(1) << 63);
        EXPECT_EQ(layout.localityCount(), size_t(1) << 63);
        EXPECT_EQ(layout.storedCellCount(), 27);
        EXPECT_EQ(layout.globalOrigin(), (GridIndices{2097151, 2097151, 2097151}));
        EXPECT_EQ(layout.cellOwner({1, 1, 1}).localityIndex, 0);
        EXPECT_EQ(layout.globalCoordinates({1, 1, 1}), (GridIndices{0, 0, 0}));
    }
}

TEST(CartesianGrid, PhysicalRangesPartitionTheDomainAndKeepGhostCoordinatesUnwrapped) {
    PROFILE();
    for (size_t rank = 0; rank < 8; rank++) {
        CartesianGrid grid(3, 1, rank);
        auto const& layout = grid.layout();
        auto const origin = layout.globalOrigin();
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            auto const coordinates = layout.cellCoordinates(index);
            auto const owner = layout.cellOwner(coordinates);
            CartesianGridLayout const ownerLayout(3, 1, owner.localityIndex);
            auto const& cell = grid.cell(coordinates);
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto const unwrappedIndex = origin[axis] + coordinates[axis];
                EXPECT_DOUBLE_EQ(cell.physicalBox.min(axis), static_cast<double>(unwrappedIndex) / 8.0);
                EXPECT_DOUBLE_EQ(cell.physicalBox.max(axis), static_cast<double>(unwrappedIndex + 1) / 8.0);
            }
            EXPECT_EQ(cell.physicalBox, layout.physicalBox(coordinates));
            EXPECT_TRUE(cell.physicalBox.containsPeriodic(ownerLayout.cellCenter(owner.coordinates)));
            if (layout.isInterior(coordinates)) {
                grid.setInteriorRange(coordinates, {7, 12});
            } else {
                grid.setGhostRanges(coordinates, {1, 2}, {3, 4});
            }
            EXPECT_EQ(cell.physicalBox, layout.physicalBox(coordinates));
        }
    }
}

TEST(CartesianGrid, PeriodicMembershipCoversBothGhostFacesAndCornerImages) {
    PROFILE();
    CartesianGrid const first(3, 1, 0);
    CartesianGrid const lastX(3, 1, 1);
    auto const& left = first.cell({-1, 0, 0}).physicalBox;
    auto const& right = lastX.cell({4, 0, 0}).physicalBox;
    EXPECT_EQ(left.min(0), -1.0 / 8.0);
    EXPECT_EQ(left.max(0), 0.0);
    EXPECT_EQ(right.min(0), 1.0);
    EXPECT_EQ(right.max(0), 1.0 + 1.0 / 8.0);

    EXPECT_TRUE(left.containsPeriodic(std::array<double, NDIM>{15.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0}));
    EXPECT_TRUE(left.containsPeriodic(std::array<cosmictiger::Coordinate, NDIM>{
        cosmictiger::Coordinate::max(), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28)}));
    EXPECT_TRUE(left.containsPeriodic(cosmictiger::Position{
        cosmictiger::Coordinate::max(), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28)}));
    EXPECT_TRUE(right.containsPeriodic(std::array<double, NDIM>{0.0, 1.0 / 16.0, 1.0 / 16.0}));
    EXPECT_TRUE(right.containsPeriodic(std::array<cosmictiger::Coordinate, NDIM>{
        cosmictiger::Coordinate::min(), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28), cosmictiger::Coordinate::fromRaw(std::uint32_t(1) << 28)}));
    EXPECT_FALSE(left.containsPeriodic(std::array<double, NDIM>{0.0, 1.0 / 16.0, 1.0 / 16.0}));
    EXPECT_FALSE(right.containsPeriodic(std::array<double, NDIM>{1.0 / 8.0, 1.0 / 16.0, 1.0 / 16.0}));

    CartesianGrid const lastZ(3, 1, 4);
    auto const& corner = lastZ.cell({-1, -1, 4}).physicalBox;
    EXPECT_TRUE(corner.containsPeriodic(std::array<double, NDIM>{15.0 / 16.0, 15.0 / 16.0, 1.0 / 16.0}));
    EXPECT_FALSE(corner.containsPeriodic(std::array<double, NDIM>{15.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0}));
}

TEST(CartesianGrid, UnwrappedBoundsCoverFullDomainAndFinestSupportedGrid) {
    PROFILE();
    CartesianGrid const single(0, 0, 0);
    for (size_t index = 0; index < single.layout().storedCellCount(); index++) {
        auto const coordinates = single.layout().cellCoordinates(index);
        auto const& box = single.cell(coordinates).physicalBox;
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_DOUBLE_EQ(box.min(axis), static_cast<double>(coordinates[axis]));
            EXPECT_DOUBLE_EQ(box.max(axis), static_cast<double>(coordinates[axis] + 1));
        }
        EXPECT_TRUE(box.containsPeriodic(std::array<double, NDIM>{0.25, 0.5, 0.75}));
    }
    size_t const exponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
    size_t const ranks = size_t(1) << (NDIM * exponent);
    CartesianGrid const last(exponent, exponent, ranks - 1);
    double const width = 1.0 / static_cast<double>(size_t(1) << exponent);
    auto const& terminal = last.cell({0, 0, 0}).physicalBox;
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_DOUBLE_EQ(terminal.min(axis), 1.0 - width);
        EXPECT_DOUBLE_EQ(terminal.max(axis), 1.0);
    }
    EXPECT_TRUE(terminal.containsPeriodic(
        std::array<cosmictiger::Coordinate, NDIM>{cosmictiger::Coordinate::max(), cosmictiger::Coordinate::max(), cosmictiger::Coordinate::max()}));
    EXPECT_FALSE(terminal.containsPeriodic(
        std::array<cosmictiger::Coordinate, NDIM>{cosmictiger::Coordinate::min(), cosmictiger::Coordinate::min(), cosmictiger::Coordinate::min()}));
    auto const& first = last.cell({1, 1, 1}).physicalBox;
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_DOUBLE_EQ(first.min(axis), 1.0);
        EXPECT_DOUBLE_EQ(first.max(axis), 1.0 + width);
    }
    EXPECT_TRUE(first.containsPeriodic(
        std::array<cosmictiger::Coordinate, NDIM>{cosmictiger::Coordinate::min(), cosmictiger::Coordinate::min(), cosmictiger::Coordinate::min()}));
}

}    // namespace
