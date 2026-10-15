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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <gtest/gtest.h>
#include <hpx/bit.hpp>
#include <hpx/future.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleStorage;
using ParticleRecord =
    hpx::tuple<std::array<std::uint32_t, NDIM>, std::array<std::uint32_t, NDIM>, std::uint8_t, bool, std::array<std::uint32_t, NDIM>, std::uint32_t>;
using CellRecord = hpx::tuple<ParticleCount, ParticleCount, ParticleCount, ParticleCount, cosmictiger::CellBox>;

struct OwnedState {
    std::vector<ParticleRecord> records;
    std::vector<CellRecord> cells;
};

struct StorageState {
    std::vector<ParticleRecord> records;
    std::vector<CellRecord> cells;
    ParticleCount capacity;
};

std::uint64_t nextOperationId() {
    PROFILE();
    static std::atomic<std::uint64_t> sequence{0x5047450000000000ULL};
    return sequence.fetch_add(1);
}

void require(bool condition, std::string const& message) {
    PROFILE();
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<ParticleRecord> particleRecords(ParticleStorage const& particles, ParticleCount count) {
    PROFILE();
    std::vector<ParticleRecord> records;
    records.reserve(static_cast<size_t>(count));
    for (ParticleCount index = 0; index < count; index++) {
        std::array<std::uint32_t, NDIM> position{};
        std::array<std::uint32_t, NDIM> velocity{};
        std::array<std::uint32_t, NDIM> acceleration{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            position[axis] = particles.position(axis, index).raw();
            velocity[axis] = hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index));
            if (particles.storesAcceleration()) {
                acceleration[axis] = hpx::bit_cast<std::uint32_t>(particles.acceleration(axis, index));
            }
        }
        auto const potential = particles.storesPotential() ? hpx::bit_cast<std::uint32_t>(particles.potential(index)) : std::uint32_t(0);
        records.emplace_back(position, velocity, particles.flags(index).rung(), particles.flags(index).isGhost(), acceleration, potential);
    }
    return records;
}

std::vector<CellRecord> cellRecords(CartesianGrid const& grid) {
    PROFILE();
    std::vector<CellRecord> records;
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const& cell = grid.cell(grid.layout().cellCoordinates(index));
        records.emplace_back(cell.localParticles.begin, cell.localParticles.end, cell.ownerParticles.begin, cell.ownerParticles.end, cell.physicalBox);
    }
    return records;
}

OwnedState ownedState(ParticleStorage const& particles, CartesianGrid const& grid) {
    PROFILE();
    auto const side = static_cast<std::int64_t>(grid.layout().interiorSideLength());
    auto const ownedCount = grid.cell({side - 1, side - 1, side - 1}).localParticles.end;
    return {particleRecords(particles, ownedCount), cellRecords(grid)};
}

StorageState storageState(ParticleStorage const& particles, CartesianGrid const& grid) {
    PROFILE();
    return {particleRecords(particles, particles.size()), cellRecords(grid), particles.capacity()};
}

void expectUnchanged(ParticleStorage const& particles, CartesianGrid const& grid, StorageState const& previous) {
    PROFILE();
    EXPECT_EQ(particleRecords(particles, particles.size()), previous.records);
    EXPECT_EQ(cellRecords(grid), previous.cells);
    EXPECT_EQ(particles.capacity(), previous.capacity);
}

void requireSameParticles(std::vector<ParticleRecord> actual, std::vector<ParticleRecord> expected, std::string const& message) {
    PROFILE();
    hpx::sort(actual.begin(), actual.end());
    hpx::sort(expected.begin(), expected.end());
    require(actual == expected, message);
}

void verifyRungRanges(ParticleStorage const& particles, cosmictiger::Cell const& cell) {
    static_assert(cosmictiger::Rung::maximumRung() == 63);
    static_assert(std::tuple_size_v<decltype(cosmictiger::Cell::rungParticles)> == 64);
    PROFILE();
    require(cell.rungRangesValid, "cell rung ranges are invalid after ghost exchange");
    require(cell.localParticles.end <= particles.size(), "cell range exceeds particle storage");
    ParticleCount cursor = cell.localParticles.begin;
    for (size_t rung = 0; rung < cell.rungParticles.size(); rung++) {
        auto const range = cell.rungParticles[rung];
        require(range.begin == cursor && range.end <= cell.localParticles.end, "cell rung ranges do not partition its particle range");
        for (ParticleCount index = range.begin; index < range.end; index++) {
            require(particles.flags(index).rung() == rung, "particle lies outside its rung range");
        }
        cursor = range.end;
    }
    require(cursor == cell.localParticles.end, "cell rung ranges omit particles");
}

// Seed ranges directly in the documented x-fastest owned-prefix order. This
// keeps the reference independent of both exchange and load balancing.
template <typename CellCount>
void seedOwned(ParticleStorage& particles, CartesianGrid& grid, CellCount cellCount) {
    PROFILE();
    auto const& layout = grid.layout();
    auto const origin = layout.globalOrigin();
    auto const side = static_cast<std::int64_t>(layout.interiorSideLength());
    auto const globalSide = static_cast<std::uint64_t>(layout.globalSideLength());
    auto const rawWidth = (std::uint64_t(1) << 32) / globalSide;
    for (std::int64_t z = 0; z < side; z++) {
        for (std::int64_t y = 0; y < side; y++) {
            for (std::int64_t x = 0; x < side; x++) {
                GridIndices const global{origin[0] + x, origin[1] + y, origin[2] + z};
                auto const count = cellCount(global);
                auto const begin = particles.size();
                auto const globalIndex = std::uint64_t(global[0]) + globalSide * (std::uint64_t(global[1]) + globalSide * std::uint64_t(global[2]));
                for (size_t index = 0; index < count; index++) {
                    auto const identity = globalIndex * 4096 + index;
                    std::array<Coordinate, NDIM> position{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        auto const raw = std::uint64_t(global[axis]) * rawWidth + rawWidth / 2 + index;
                        position[axis] = Coordinate::fromRaw(static_cast<std::uint32_t>(raw));
                    }
                    std::array<float, NDIM> const velocity{identity == 0 ? -0.0f : float(identity) + 0.25f, -float(identity) - 0.5f, float(identity) + 0.75f};
                    particles.append(position, velocity, static_cast<std::uint8_t>((identity * 37) % 64));
                }
                grid.setInteriorRange({x, y, z}, {begin, particles.size()});
            }
        }
    }
}

// Compute periodic ownership with an integer reference rather than cellOwner().
std::pair<size_t, GridIndices> referenceOwner(CartesianGrid const& grid, GridIndices coordinates) {
    PROFILE();
    auto const& layout = grid.layout();
    auto const origin = layout.globalOrigin();
    auto const globalSide = static_cast<std::int64_t>(layout.globalSideLength());
    auto const side = static_cast<std::int64_t>(layout.interiorSideLength());
    auto const q = layout.localitiesPerAxis();
    std::array<size_t, NDIM> block{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        auto const global = (origin[axis] + coordinates[axis] + globalSide) % globalSide;
        block[axis] = static_cast<size_t>(global / side);
        coordinates[axis] = global % side;
    }
    return {block[0] + q * (block[1] + q * block[2]), coordinates};
}

void verifyGhosts(ParticleStorage const& particles, CartesianGrid const& grid, std::vector<OwnedState> const& expected) {
    PROFILE();
    auto const& layout = grid.layout();
    auto const rank = layout.localityIndex();
    auto const& owned = expected[rank];
    auto const actualCells = cellRecords(grid);
    auto const actualParticles = particleRecords(particles, particles.size());
    requireSameParticles(particleRecords(particles, owned.records.size()), owned.records, "exchange changed an owned particle field");
    ParticleCount cursor = owned.records.size();
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        auto const& cell = grid.cell(coordinates);
        verifyRungRanges(particles, cell);
        require(hpx::get<4>(actualCells[index]) == hpx::get<4>(owned.cells[index]), "exchange changed physical geometry");
        if (layout.isInterior(coordinates)) {
            require(actualCells[index] == owned.cells[index], "exchange changed an interior range");
            auto const range = cell.localParticles;
            auto const first = static_cast<size_t>(range.begin);
            auto const last = static_cast<size_t>(range.end);
            requireSameParticles({actualParticles.begin() + first, actualParticles.begin() + last},
                {owned.records.begin() + first, owned.records.begin() + last}, "exchange changed an interior cell's particles");
            continue;
        }
        auto const [owner, ownerCoordinates] = referenceOwner(grid, coordinates);
        auto const& sourceCell = expected[owner].cells[layout.cellIndex(ownerCoordinates)];
        auto const sourceBegin = hpx::get<0>(sourceCell);
        auto const sourceEnd = hpx::get<1>(sourceCell);
        require(cell.ownerParticles.begin == sourceBegin && cell.ownerParticles.end == sourceEnd, "ghost owner range differs from the source interior range");
        require(cell.localParticles.begin == cursor && cell.localParticles.end == cursor + sourceEnd - sourceBegin,
            "ghost ranges do not partition the appended suffix in stored x-fastest order");
        std::vector<ParticleRecord> expectedGhosts;
        expectedGhosts.reserve(static_cast<size_t>(sourceEnd - sourceBegin));
        for (ParticleCount sourceIndex = sourceBegin; sourceIndex < sourceEnd; sourceIndex++) {
            auto record = expected[owner].records[static_cast<size_t>(sourceIndex)];
            hpx::get<3>(record) = owner != rank;
            expectedGhosts.push_back(record);
        }
        requireSameParticles({actualParticles.begin() + static_cast<size_t>(cursor), actualParticles.begin() + static_cast<size_t>(cell.localParticles.end)},
            std::move(expectedGhosts), "ghost copy changed a particle field");
        cursor = cell.localParticles.end;
    }
    require(cursor == particles.size(), "exchange left unreferenced particles after the ghost suffix");
}

// These tests sort owned cells before taking the snapshot, so the final rung
// sort leaves copied source order unchanged within every target cell.
void verifyOrderedGhosts(ParticleStorage const& particles, CartesianGrid const& grid, std::vector<OwnedState> const& expected) {
    PROFILE();
    auto const actual = particleRecords(particles, particles.size());
    auto const& layout = grid.layout();
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        if (layout.isInterior(coordinates)) {
            continue;
        }
        auto const [owner, ownerCoordinates] = referenceOwner(grid, coordinates);
        auto const& sourceCell = expected[owner].cells[layout.cellIndex(ownerCoordinates)];
        auto const sourceBegin = hpx::get<0>(sourceCell);
        auto const sourceEnd = hpx::get<1>(sourceCell);
        auto const destination = grid.cell(coordinates).localParticles;
        require(destination.end - destination.begin == sourceEnd - sourceBegin, "ordered ghost copy has the wrong length");
        for (ParticleCount offset = 0; offset < sourceEnd - sourceBegin; offset++) {
            auto record = expected[owner].records[static_cast<size_t>(sourceBegin + offset)];
            hpx::get<3>(record) = owner != layout.localityIndex();
            require(actual[static_cast<size_t>(destination.begin + offset)] == record, "ghost packing or unpacking changed source order");
        }
    }
}

void verifyCleared(ParticleStorage const& particles, CartesianGrid const& grid, OwnedState const& owned, ParticleCount capacity) {
    PROFILE();
    requireSameParticles(particleRecords(particles, particles.size()), owned.records, "clear changed owned particle values or retained appended copies");
    require(particles.capacity() == capacity, "clear changed particle capacity");
    auto const actual = cellRecords(grid);
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        verifyRungRanges(particles, grid.cell(coordinates));
        if (grid.layout().isInterior(coordinates)) {
            require(actual[index] == owned.cells[index], "clear changed an interior range or physical geometry");
        } else {
            require(hpx::get<0>(actual[index]) == 0 && hpx::get<1>(actual[index]) == 0 && hpx::get<2>(actual[index]) == 0 && hpx::get<3>(actual[index]) == 0,
                "clear retained ghost local or owner ranges");
            require(hpx::get<4>(actual[index]) == hpx::get<4>(owned.cells[index]), "clear changed ghost geometry");
        }
    }
}

class LogicalLocalities {
public:
    LogicalLocalities(size_t cellExponent, size_t localityExponent, size_t ghostWidth = 1)
      : localityIds_(size_t(1) << (NDIM * localityExponent), hpx::find_here()) {
        PROFILE();
        particles_.reserve(localityIds_.size());
        grids_.reserve(localityIds_.size());
        for (size_t rank = 0; rank < localityIds_.size(); rank++) {
            particles_.emplace_back();
            grids_.emplace_back(cellExponent, localityExponent, rank, ghostWidth);
        }
    }

    template <typename CellCount>
    void seed(CellCount cellCount) {
        PROFILE();
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            seedOwned(particles_[rank], grids_[rank], cellCount);
        }
    }

    std::vector<OwnedState> ownedStates() const {
        PROFILE();
        std::vector<OwnedState> states;
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            states.push_back(ownedState(particles_[rank], grids_[rank]));
        }
        return states;
    }

    std::vector<StorageState> storageStates() const {
        PROFILE();
        std::vector<StorageState> states;
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            states.push_back(storageState(particles_[rank], grids_[rank]));
        }
        return states;
    }

    // Drain every future even when a participant fails.
    std::vector<std::string> exchange() {
        PROFILE();
        auto const operationId = nextOperationId();
        std::vector<hpx::future<void>> futures;
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            futures.emplace_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, rank, operationId] {
                        PROFILE();
                        cosmictiger::exchangeGhostParticles(particles_[rank], grids_[rank], localityIds_, operationId);
                    },
                    "test.particle.exchangeLogicalGhosts")));
        }
        std::vector<std::string> failures;
        for (auto& future : futures) {
            try {
                future.get();
            } catch (std::exception const& error) {
                failures.emplace_back(error.what());
            } catch (...) {
                failures.emplace_back("nonstandard exception");
            }
        }
        return failures;
    }

    void verify(std::vector<OwnedState> const& expected) const {
        PROFILE();
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            verifyGhosts(particles_[rank], grids_[rank], expected);
        }
    }

    void expectUnchanged(std::vector<StorageState> const& previous) const {
        PROFILE();
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            ::expectUnchanged(particles_[rank], grids_[rank], previous[rank]);
        }
    }

    std::vector<hpx::id_type> const& localityIds() const noexcept {
        PROFILE();
        return localityIds_;
    }
    std::vector<ParticleStorage>& particles() noexcept {
        PROFILE();
        return particles_;
    }
    std::vector<ParticleStorage> const& particles() const noexcept {
        PROFILE();
        return particles_;
    }
    std::vector<CartesianGrid>& grids() noexcept {
        PROFILE();
        return grids_;
    }
    std::vector<CartesianGrid> const& grids() const noexcept {
        PROFILE();
        return grids_;
    }

private:
    std::vector<hpx::id_type> localityIds_;
    std::vector<ParticleStorage> particles_;
    std::vector<CartesianGrid> grids_;
};

class ParticleGhostExchangeTest : public ::testing::Test {
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

void exerciseLargeGhostRefresh(LogicalLocalities& simulation) {
    PROFILE();
    std::vector<OwnedState> expected;
    for (size_t refresh = 0; refresh < 2; refresh++) {
        for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
            auto& particles = simulation.particles()[rank];
            auto& grid = simulation.grids()[rank];
            auto const owned = ownedState(particles, grid).records.size();
            for (ParticleCount index = 0; index < owned; index++) {
                particles.velocity(0, index) = float(rank * 65536 + index) + float(refresh) + 0.125f;
                particles.flags(index).setRung((index * 29 + rank + refresh) % 64);
                for (size_t axis = 0; axis < NDIM; axis++) {
                    particles.acceleration(axis, index) = -particles.velocity(axis, index) + float(refresh);
                }
                particles.potential(index) = -particles.velocity(0, index) - 3.0f - float(refresh);
            }
            grid.sortParticlesByRung(particles);
        }
        expected = simulation.ownedStates();
        require(simulation.exchange().empty(), "large ghost refresh failed");
        simulation.verify(expected);
        for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
            verifyOrderedGhosts(simulation.particles()[rank], simulation.grids()[rank], expected);
        }
    }
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        auto& particles = simulation.particles()[rank];
        auto& grid = simulation.grids()[rank];
        auto const capacity = particles.capacity();
        cosmictiger::clearGhostParticles(particles, grid);
        verifyCleared(particles, grid, expected[rank], capacity);
        cosmictiger::clearGhostParticles(particles, grid);
        verifyCleared(particles, grid, expected[rank], capacity);
    }
}

TEST_F(ParticleGhostExchangeTest, PeriodicLocalFacesEdgesAndCornersPreserveAllFields) {
    PROFILE();
    LogicalLocalities simulation(2, 0);
    simulation.seed([](GridIndices global) {
        PROFILE();
        return size_t(1 + (global[0] + 2 * global[1] + 3 * global[2]) % 3);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (ParticleCount index = 0; index < simulation.particles()[0].size(); index++) {
        EXPECT_FALSE(simulation.particles()[0].flags(index).isGhost());
    }
}

TEST_F(ParticleGhostExchangeTest, EightLogicalLocalitiesExchangePeriodicFacesEdgesAndCorners) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices global) {
        PROFILE();
        return size_t(1 + (global[0] + global[1] + global[2]) % 3);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        for (ParticleCount index = expected[rank].records.size(); index < simulation.particles()[rank].size(); index++) {
            EXPECT_TRUE(simulation.particles()[rank].flags(index).isGhost());
        }
    }
}

TEST_F(ParticleGhostExchangeTest, GravityOutputFieldsFollowRemoteAndPeriodicGhostCopies) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    for (auto& particles : simulation.particles()) {
        particles.configureGravityFields(true, true);
    }
    simulation.seed([](GridIndices const&) {
        PROFILE();
        return size_t(1);
    });
    for (auto& particles : simulation.particles()) {
        for (ParticleCount index = 0; index < particles.size(); index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                particles.acceleration(axis, index) = -particles.velocity(axis, index);
            }
            particles.potential(index) = -particles.velocity(0, index) - 3.0f;
        }
    }
    auto const owned = simulation.ownedStates();
    ASSERT_TRUE(simulation.exchange().empty());
    EXPECT_NO_THROW(simulation.verify(owned));
    for (auto const& particles : simulation.particles()) {
        for (ParticleCount index = 0; index < particles.size(); index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_EQ(particles.acceleration(axis, index), -particles.velocity(axis, index));
            }
            EXPECT_EQ(particles.potential(index), -particles.velocity(0, index) - 3.0f);
        }
    }
}

TEST_F(ParticleGhostExchangeTest, TwoCellHaloExchangesAllLayersAcrossLogicalLocalities) {
    PROFILE();
    LogicalLocalities simulation(3, 1, 2);
    simulation.seed([](GridIndices const&) {
        PROFILE();
        return size_t(1);
    });
    auto const owned = simulation.ownedStates();
    ASSERT_TRUE(simulation.exchange().empty());
    EXPECT_NO_THROW(simulation.verify(owned));
    for (auto const& grid : simulation.grids()) {
        EXPECT_EQ(grid.layout().ghostWidth(), 2);
    }
}

TEST_F(ParticleGhostExchangeTest, InconsistentGravityOutputFieldsFailEveryParticipant) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.particles().back().configureGravityFields(false, true);
    EXPECT_EQ(simulation.exchange().size(), simulation.particles().size());
}

TEST_F(ParticleGhostExchangeTest, SingletonGridAppendsTwentySixIndependentLocalCopies) {
    PROFILE();
    LogicalLocalities simulation(0, 0);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(2);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(simulation.particles()[0].size(), 54);
    auto const capacity = simulation.particles()[0].capacity();
    cosmictiger::clearGhostParticles(simulation.particles()[0], simulation.grids()[0]);
    EXPECT_NO_THROW(verifyCleared(simulation.particles()[0], simulation.grids()[0], expected[0], capacity));
}

TEST_F(ParticleGhostExchangeTest, OneCellPerRankRetainsDuplicatePeriodicGhostCells) {
    PROFILE();
    LogicalLocalities simulation(1, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (auto const& particles : simulation.particles()) {
        EXPECT_EQ(particles.size(), 27);
    }
}

TEST_F(ParticleGhostExchangeTest, LargeSingleCellPeriodicCopiesPreserveOrderAcrossRefreshAndClear) {
    PROFILE();
    // One cell exceeds several coarse chunks; every halo cell requests the
    // same owner range, so chunk boundaries also cross repeated requests.
    LogicalLocalities simulation(0, 0);
    simulation.particles()[0].configureGravityFields(true, true);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(10003);
    });
    EXPECT_NO_THROW(exerciseLargeGhostRefresh(simulation));
    EXPECT_EQ(simulation.particles()[0].size(), 10003);
}

TEST_F(ParticleGhostExchangeTest, SkewedDuplicatePeriodicGhostsPreserveAllFieldsAndRanges) {
    PROFILE();
    LogicalLocalities simulation(1, 1);
    for (auto& particles : simulation.particles()) {
        particles.configureGravityFields(true, true);
    }
    simulation.seed([](GridIndices global) {
        PROFILE();
        if (global == GridIndices{0, 0, 0}) {
            return size_t(10003);
        }
        if (global == GridIndices{1, 1, 1}) {
            return size_t(13);
        }
        return size_t(0);
    });
    EXPECT_NO_THROW(exerciseLargeGhostRefresh(simulation));
    EXPECT_EQ(simulation.particles()[0].size(), 10003);
    EXPECT_EQ(simulation.particles()[7].size(), 13);
    for (size_t rank = 1; rank < 7; rank++) {
        EXPECT_EQ(simulation.particles()[rank].size(), 0);
    }
}

TEST_F(ParticleGhostExchangeTest, EmptyCellsAndOwnedEmptyRanksReceiveOnlyRelevantCopies) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices global) {
        PROFILE();
        return global == GridIndices{2, 2, 0} ? size_t(3) : size_t(0);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(expected[3].records.size(), 3);
    EXPECT_GT(simulation.particles()[0].size(), 0);
    auto const capacities = simulation.storageStates();
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        cosmictiger::clearGhostParticles(simulation.particles()[rank], simulation.grids()[rank]);
        EXPECT_NO_THROW(verifyCleared(simulation.particles()[rank], simulation.grids()[rank], expected[rank], capacities[rank].capacity));
        cosmictiger::clearGhostParticles(simulation.particles()[rank], simulation.grids()[rank]);
        EXPECT_NO_THROW(verifyCleared(simulation.particles()[rank], simulation.grids()[rank], expected[rank], capacities[rank].capacity));
    }
}

TEST_F(ParticleGhostExchangeTest, AllEmptyRanksExchangeAndClear) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(0);
    });
    auto const expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        cosmictiger::clearGhostParticles(simulation.particles()[rank], simulation.grids()[rank]);
        EXPECT_NO_THROW(verifyCleared(simulation.particles()[rank], simulation.grids()[rank], expected[rank], 0));
    }
}

TEST_F(ParticleGhostExchangeTest, RepeatedRefreshReplacesCopiesAndClearPreservesOwnedPrefixAndCapacity) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(2);
    });
    auto expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    auto const firstExchange = simulation.storageStates();
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        for (ParticleCount index = 0; index < expected[rank].records.size(); index++) {
            simulation.particles()[rank].velocity(1, index) = float(rank * 1024 + index) + 0.125f;
            simulation.particles()[rank].flags(index).setRung((rank + index + 17) % 64);
        }
    }
    expected = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        EXPECT_EQ(simulation.particles()[rank].size(), firstExchange[rank].records.size());
        auto const capacity = simulation.particles()[rank].capacity();
        cosmictiger::clearGhostParticles(simulation.particles()[rank], simulation.grids()[rank]);
        EXPECT_NO_THROW(verifyCleared(simulation.particles()[rank], simulation.grids()[rank], expected[rank], capacity));
    }
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleGhostExchangeTest, ClearPreservesOwnedParticlesAfterPositionsMoveOutsideTheirCells) {
    PROFILE();
    LogicalLocalities simulation(1, 0);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    auto const initial = simulation.ownedStates();
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    auto& particles = simulation.particles()[0];
    for (ParticleCount index = 0; index < initial[0].records.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, index) = Coordinate::fromRaw(0xffffffffU);
        }
    }
    auto const moved = simulation.ownedStates();
    auto const capacity = particles.capacity();
    cosmictiger::clearGhostParticles(particles, simulation.grids()[0]);
    EXPECT_NO_THROW(verifyCleared(particles, simulation.grids()[0], moved[0], capacity));
}

TEST_F(ParticleGhostExchangeTest, MovedOwnedParticleFailsEveryRankWithoutMutationAndCanBeCleared) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    simulation.particles()[5].position(0, 0) = Coordinate::fromRaw(0xffffffffU);
    auto const moved = simulation.ownedStates();
    auto const previous = simulation.storageStates();
    EXPECT_EQ(simulation.exchange().size(), simulation.particles().size());
    simulation.expectUnchanged(previous);
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        cosmictiger::clearGhostParticles(simulation.particles()[rank], simulation.grids()[rank]);
        EXPECT_NO_THROW(verifyCleared(simulation.particles()[rank], simulation.grids()[rank], moved[rank], previous[rank].capacity));
    }
}

TEST_F(ParticleGhostExchangeTest, LargePackingFailureReachesEveryRankWithoutMutationAndCanRetry) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(1, 1);
    simulation.seed([](GridIndices global) {
        PROFILE();
        return global == GridIndices{0, 0, 0} ? size_t(10003) : size_t(0);
    });
    auto& particles = simulation.particles()[0];
    auto const index = particles.size() - 1;
    auto const original = particles.position(0, index);
    particles.position(0, index) = Coordinate::fromRaw(0xffffffffU);
    auto const previous = simulation.storageStates();
    EXPECT_EQ(simulation.exchange().size(), simulation.particles().size());
    simulation.expectUnchanged(previous);
    particles.position(0, index) = original;
    auto const expected = simulation.ownedStates();
    ASSERT_TRUE(simulation.exchange().empty());
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleGhostExchangeTest, InvalidInteriorPartitionFailsEveryRankWithoutMutationAndCanRetry) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    auto const expected = simulation.ownedStates();
    simulation.grids()[5].setInteriorRange({0, 0, 0}, {1, 2});
    auto const previous = simulation.storageStates();
    EXPECT_EQ(simulation.exchange().size(), simulation.particles().size());
    simulation.expectUnchanged(previous);
    simulation.grids()[5].setInteriorRange({0, 0, 0}, {0, 1});
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleGhostExchangeTest, RefreshReplacesUnusedPreviousGhostRanges) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    auto const expected = simulation.ownedStates();
    simulation.grids()[5].setGhostRanges({-1, 0, 0}, {0, 1}, {0, 1});
    ASSERT_EQ(simulation.exchange(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleGhostExchangeTest, InconsistentTopologyFailsEveryRankWithoutMutation) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(2, 1);
    simulation.grids()[5] = CartesianGrid(3, 1, 5);
    simulation.seed([](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    auto const previous = simulation.storageStates();
    EXPECT_EQ(simulation.exchange().size(), simulation.particles().size());
    simulation.expectUnchanged(previous);
}

TEST_F(ParticleGhostExchangeTest, WrongLocalityCountFailsWithoutMutation) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    ParticleStorage particles;
    CartesianGrid grid(1, 0, 0);
    seedOwned(particles, grid, [](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    auto const previous = storageState(particles, grid);
    EXPECT_ANY_THROW(cosmictiger::exchangeGhostParticles(particles, grid, {}, nextOperationId()));
    expectUnchanged(particles, grid, previous);
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

TEST_F(ParticleGhostExchangeTest, ConvenienceFunctionsUseLocalParticleStorage) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one HPX locality";
    }
    LocalStorageScope storageScope;
    auto& particles = cosmictiger::localParticleStorage();
    CartesianGrid grid(0, 0, 0);
    seedOwned(particles, grid, [](GridIndices) {
        PROFILE();
        return size_t(1);
    });
    std::vector<OwnedState> const expected{ownedState(particles, grid)};
    cosmictiger::exchangeGhostParticles(grid, nextOperationId());
    EXPECT_NO_THROW(verifyGhosts(particles, grid, expected));
    auto const capacity = particles.capacity();
    cosmictiger::clearGhostParticles(grid);
    EXPECT_NO_THROW(verifyCleared(particles, grid, expected[0], capacity));
}

}    // namespace

namespace cosmictiger::detail {

void runRemoteParticleGhostExchange(std::vector<hpx::id_type> localityIds, std::uint64_t operationId, size_t rank) {
    PROFILE();
    CartesianGrid grid(2, 1, rank);
    ParticleStorage particles;
    // A single corner-cell payload exceeds 64 KiB, exercising native transfers.
    auto const cellCount = [](GridIndices) {
        PROFILE();
        return size_t(3000);
    };
    seedOwned(particles, grid, cellCount);
    std::vector<OwnedState> expected;
    for (size_t sourceRank = 0; sourceRank < localityIds.size(); sourceRank++) {
        CartesianGrid sourceGrid(2, 1, sourceRank);
        ParticleStorage sourceParticles;
        seedOwned(sourceParticles, sourceGrid, cellCount);
        expected.push_back(ownedState(sourceParticles, sourceGrid));
    }
    exchangeGhostParticles(particles, grid, localityIds, operationId);
    verifyGhosts(particles, grid, expected);
    auto const capacity = particles.capacity();
    clearGhostParticles(particles, grid);
    verifyCleared(particles, grid, expected[rank], capacity);
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::runRemoteParticleGhostExchange, cosmictiger_run_remote_particle_ghost_exchange_action)

TEST_F(ParticleGhostExchangeTest, RemoteExchange) {
    PROFILE();
    auto const physicalLocalities = hpx::find_all_localities();
    if (physicalLocalities.size() != 2 && physicalLocalities.size() != 8) {
        GTEST_SKIP() << "requires two or eight HPX localities";
    }
    // Eight logical grid ranks can share two physical hosts while retaining
    // real remote transfers and all face, edge, and corner owner relations.
    std::vector<hpx::id_type> localityIds;
    localityIds.reserve(8);
    for (size_t rank = 0; rank < 8; rank++) {
        localityIds.push_back(physicalLocalities[rank % physicalLocalities.size()]);
    }
    auto const operationId = nextOperationId();
    std::vector<hpx::future<void>> futures;
    for (size_t rank = 0; rank < localityIds.size(); rank++) {
        futures.emplace_back(hpx::async<cosmictiger_run_remote_particle_ghost_exchange_action>(localityIds[rank], localityIds, operationId, rank));
    }
    for (auto& future : futures) {
        EXPECT_NO_THROW(future.get());
    }
}
