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
#include <cosmictiger/particle/particle_load_balance.hpp>

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
// Compare floating-point object representations, including the sign of zero.
using ParticleRecord = hpx::tuple<std::array<std::uint32_t, NDIM>, std::array<std::uint32_t, NDIM>, std::uint8_t, bool>;
using CellRecord = hpx::tuple<ParticleCount, ParticleCount, ParticleCount, ParticleCount, cosmictiger::CellBox>;

std::uint64_t nextOperationId() {
    PROFILE();
    static std::atomic<std::uint64_t> sequence{0x504c420000000000ULL};
    return sequence.fetch_add(1);
}

ParticleRecord makeParticle(std::array<std::uint32_t, NDIM> position, size_t identity, bool isGhost = false) {
    PROFILE();
    std::array<float, NDIM> const velocity{identity == 0 ? -0.0f : float(identity) + 0.25f, -float(identity) - 0.5f, float(identity) + 0.75f};
    std::array<std::uint32_t, NDIM> velocityBits{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        velocityBits[axis] = hpx::bit_cast<std::uint32_t>(velocity[axis]);
    }
    return ParticleRecord{position, velocityBits, static_cast<std::uint8_t>((identity * 37) % 64), isGhost};
}

void appendParticle(ParticleStorage& particles, ParticleRecord const& record) {
    PROFILE();
    std::array<Coordinate, NDIM> position{};
    std::array<float, NDIM> velocity{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        position[axis] = Coordinate::fromRaw(hpx::get<0>(record)[axis]);
        velocity[axis] = hpx::bit_cast<float>(hpx::get<1>(record)[axis]);
    }
    particles.append(position, velocity, hpx::get<2>(record), hpx::get<3>(record));
}

std::vector<ParticleRecord> particleRecords(ParticleStorage const& particles) {
    PROFILE();
    std::vector<ParticleRecord> records;
    records.reserve(static_cast<size_t>(particles.size()));
    for (ParticleCount index = 0; index < particles.size(); index++) {
        std::array<std::uint32_t, NDIM> position{};
        std::array<std::uint32_t, NDIM> velocity{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            position[axis] = particles.position(axis, index).raw();
            velocity[axis] = hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index));
        }
        records.emplace_back(position, velocity, particles.flags(index).rung(), particles.flags(index).isGhost());
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

// Independent integer reference: raw coordinates occupy the periodic unit cube.
GridIndices referenceGlobalCell(ParticleRecord const& record, size_t globalSide) {
    PROFILE();
    GridIndices cell{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        cell[axis] = static_cast<std::int64_t>((std::uint64_t(hpx::get<0>(record)[axis]) * globalSide) >> 32);
    }
    return cell;
}

size_t referenceOwner(ParticleRecord const& record, CartesianGrid const& grid) {
    PROFILE();
    auto const global = referenceGlobalCell(record, grid.layout().globalSideLength());
    auto const side = grid.layout().interiorSideLength();
    auto const q = grid.layout().localitiesPerAxis();
    return static_cast<size_t>(global[0]) / side + q * (static_cast<size_t>(global[1]) / side + q * (static_cast<size_t>(global[2]) / side));
}

void require(bool condition, std::string const& message) {
    PROFILE();
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void verifyRungRanges(ParticleStorage const& particles, cosmictiger::Cell const& cell) {
    static_assert(cosmictiger::Rung::maximumRung() == 63);
    static_assert(std::tuple_size_v<decltype(cosmictiger::Cell::rungParticles)> == 64);
    PROFILE();
    require(cell.rungRangesValid, "cell rung ranges are invalid after load balancing");
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

void verifyPartition(ParticleStorage const& particles, CartesianGrid const& grid, std::vector<ParticleRecord> expected) {
    PROFILE();
    auto actual = particleRecords(particles);
    auto sortedActual = actual;
    hpx::sort(sortedActual.begin(), sortedActual.end());
    hpx::sort(expected.begin(), expected.end());
    require(sortedActual == expected, "particle count, one of the seven fields, or ghost flag changed");
    for (ParticleCount index = 0; index < particles.size(); index++) {
        require(!particles.flags(index).isGhost(), "redistributed owned particle retains a ghost flag");
    }
    auto const& layout = grid.layout();
    auto const origin = layout.globalOrigin();
    auto const side = static_cast<std::int64_t>(layout.interiorSideLength());
    ParticleCount cursor = 0;
    for (std::int64_t z = 0; z < side; z++) {
        for (std::int64_t y = 0; y < side; y++) {
            for (std::int64_t x = 0; x < side; x++) {
                GridIndices const global{origin[0] + x, origin[1] + y, origin[2] + z};
                ParticleCount count = 0;
                for (auto const& record : expected) {
                    if (referenceGlobalCell(record, layout.globalSideLength()) == global) {
                        count++;
                    }
                }
                auto const& cell = grid.cell({x, y, z});
                require(cell.localParticles.begin == cursor && cell.localParticles.end == cursor + count,
                    "interior ranges do not partition the array in x-fastest order");
                require(cell.ownerParticles.begin == cursor && cell.ownerParticles.end == cursor + count, "interior local and owner ranges differ");
                for (ParticleCount index = cursor; index < cursor + count; index++) {
                    require(
                        referenceGlobalCell(actual[static_cast<size_t>(index)], layout.globalSideLength()) == global, "particle lies outside its cell range");
                }
                cursor += count;
            }
        }
    }
    require(cursor == particles.size(), "interior ranges omit particles");
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        auto const& cell = grid.cell(coordinates);
        verifyRungRanges(particles, cell);
        if (!layout.isInterior(coordinates)) {
            require(cell.localParticles.begin == 0 && cell.localParticles.end == 0 && cell.ownerParticles.begin == 0 && cell.ownerParticles.end == 0,
                "ghost ranges retain stale particle indices");
        }
        require(cell.physicalBox == layout.physicalBox(coordinates), "load balancing changed physical cell geometry");
    }
}

class LogicalLocalities {
public:
    LogicalLocalities(size_t cellExponent, size_t localityExponent)
      : localityIds_(size_t(1) << (NDIM * localityExponent), hpx::find_here()) {
        PROFILE();
        particles_.reserve(localityIds_.size());
        grids_.reserve(localityIds_.size());
        for (size_t rank = 0; rank < localityIds_.size(); rank++) {
            particles_.emplace_back();
            grids_.emplace_back(cellExponent, localityExponent, rank);
        }
    }

    // Await every participant even after failure so no operation escapes a test.
    std::vector<std::string> balance() {
        PROFILE();
        auto const operationId = nextOperationId();
        std::vector<hpx::future<void>> futures;
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            futures.emplace_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, rank, operationId] {
                        PROFILE();
                        cosmictiger::loadBalanceParticles(particles_[rank], grids_[rank], localityIds_, operationId);
                    },
                    "test.particle.balanceLogicalSite")));
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

    void verify(std::vector<ParticleRecord> const& expected) const {
        PROFILE();
        for (size_t rank = 0; rank < particles_.size(); rank++) {
            std::vector<ParticleRecord> localExpected;
            for (auto const& record : expected) {
                if (referenceOwner(record, grids_[rank]) == rank) {
                    localExpected.push_back(record);
                }
            }
            verifyPartition(particles_[rank], grids_[rank], std::move(localExpected));
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

class ParticleLoadBalanceTest : public ::testing::Test {
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

TEST_F(ParticleLoadBalanceTest, ExactFixedPointBoundariesAndCellPartition) {
    constexpr std::array<std::uint32_t, 8> kCoordinates{0, 0x3fffffffU, 0x40000000U, 0x40000001U, 0x7fffffffU, 0x80000000U, 0xc0000000U, 0xffffffffU};
    PROFILE();

    LogicalLocalities simulation(2, 0);
    std::vector<ParticleRecord> expected;
    for (auto const z : kCoordinates) {
        for (auto const y : kCoordinates) {
            for (auto const x : kCoordinates) {
                expected.push_back(makeParticle({x, y, z}, expected.size()));
            }
        }
    }
    for (auto iterator = expected.rbegin(); iterator != expected.rend(); iterator++) {
        appendParticle(simulation.particles()[0], *iterator);
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, SkewedSourceRedistributesAllSevenFieldsAcrossEightLogicalLocalities) {
    PROFILE();
    LogicalLocalities simulation(3, 1);
    std::vector<ParticleRecord> expected;
    for (std::uint32_t z = 0; z < 8; z++) {
        for (std::uint32_t y = 0; y < 8; y++) {
            for (std::uint32_t x = 0; x < 8; x++) {
                expected.push_back(makeParticle({x << 29, y << 29, z << 29}, expected.size()));
            }
        }
    }
    for (auto iterator = expected.rbegin(); iterator != expected.rend(); iterator++) {
        appendParticle(simulation.particles()[7], *iterator);
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    for (auto const& particles : simulation.particles()) {
        EXPECT_EQ(particles.size(), 64);
    }
}

TEST_F(ParticleLoadBalanceTest, GravityOutputFieldsSurviveMigrationAndRungSorting) {
    PROFILE();
    for (int configuration = 1; configuration < 4; configuration++) {
        bool const acceleration = (configuration & 1) != 0;
        bool const potential = (configuration & 2) != 0;
        LogicalLocalities simulation(2, 1);
        for (auto& particles : simulation.particles()) {
            particles.configureGravityFields(acceleration, potential);
        }
        auto& source = simulation.particles().back();
        std::vector<ParticleRecord> expected;
        for (std::uint32_t z = 0; z < 4; z++) {
            for (std::uint32_t y = 0; y < 4; y++) {
                for (std::uint32_t x = 0; x < 4; x++) {
                    expected.push_back(makeParticle({x << 30, y << 30, z << 30}, expected.size()));
                    appendParticle(source, expected.back());
                    ParticleCount const index = source.size() - 1;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        if (acceleration) {
                            source.acceleration(axis, index) = 2.0f * source.velocity(axis, index);
                        }
                    }
                    if (potential) {
                        source.potential(index) = source.velocity(0, index) - 128.0f;
                    }
                }
            }
        }
        ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
        EXPECT_NO_THROW(simulation.verify(expected));
        for (auto const& particles : simulation.particles()) {
            EXPECT_EQ(particles.storesAcceleration(), acceleration);
            EXPECT_EQ(particles.storesPotential(), potential);
            for (ParticleCount index = 0; index < particles.size(); index++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    if (acceleration) {
                        EXPECT_EQ(particles.acceleration(axis, index), 2.0f * particles.velocity(axis, index));
                    }
                }
                if (potential) {
                    EXPECT_EQ(particles.potential(index), particles.velocity(0, index) - 128.0f);
                }
            }
        }
    }
}

TEST_F(ParticleLoadBalanceTest, InconsistentGravityOutputFieldsFailEveryParticipant) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    simulation.particles().back().configureGravityFields(true, false);
    EXPECT_EQ(simulation.balance().size(), simulation.particles().size());
}

TEST_F(ParticleLoadBalanceTest, ChunkedSkewedSourcePreservesFieldsAndExcludesMixedGhosts) {
    constexpr size_t kSourceCount = 65537;
    constexpr ParticleCount kGhostBegin = 4093;
    constexpr ParticleCount kGhostMiddle = 8201;
    constexpr ParticleCount kGhostEnd = 9003;
    PROFILE();

    LogicalLocalities simulation(2, 1);
    auto& source = simulation.particles()[7];
    source.reserve(kSourceCount);
    std::vector<ParticleRecord> expected;
    expected.reserve(kSourceCount);
    std::uint32_t state = 0x504c4201U;
    for (size_t index = 0; index < kSourceCount; index++) {
        std::array<std::uint32_t, NDIM> position{};
        for (auto& coordinate : position) {
            state = state * 1664525U + 1013904223U;
            coordinate = state;
        }
        bool const flaggedGhost = index % 97 == 0;
        auto const record = makeParticle(position, index, flaggedGhost);
        appendParticle(source, record);
        if (!flaggedGhost && (index < kGhostBegin || index >= kGhostEnd)) {
            expected.push_back(record);
        }
    }
    // Overlapping periodic ghost copies cross coarse chunk boundaries. The
    // remaining owned intervals also contain explicitly flagged ghost particles.
    simulation.grids()[7].setGhostRanges({-1, 0, 0}, {kGhostBegin, kGhostMiddle}, {100, 100 + kGhostMiddle - kGhostBegin});
    simulation.grids()[7].setGhostRanges({-1, 1, 0}, {kGhostMiddle - 10, kGhostEnd}, {1000, 1000 + kGhostEnd - kGhostMiddle + 10});
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    std::vector<std::vector<ParticleRecord>> firstPass;
    for (auto const& particles : simulation.particles()) {
        EXPECT_GT(particles.size(), 4096U);
        firstPass.push_back(particleRecords(particles));
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    for (size_t rank = 0; rank < firstPass.size(); rank++) {
        EXPECT_EQ(particleRecords(simulation.particles()[rank]), firstPass[rank]);
    }
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, ChunkedManySourcesPopulateOneCellAndLeaveEmptyDestinations) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    std::vector<ParticleRecord> expected;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        size_t const count = 5003 + rank * 37;
        simulation.particles()[rank].reserve(count);
        for (size_t index = 0; index < count; index++) {
            auto const record = makeParticle({0x80000000U + static_cast<std::uint32_t>(index), 0x80000000U, 0}, expected.size());
            appendParticle(simulation.particles()[rank], record);
            expected.push_back(record);
        }
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(simulation.particles()[3].size(), expected.size());
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        if (rank != 3) {
            EXPECT_EQ(simulation.particles()[rank].size(), 0U);
        }
    }
    auto const firstPass = particleRecords(simulation.particles()[3]);
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_EQ(particleRecords(simulation.particles()[3]), firstPass);
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, EmptyDestinationsAndRepeatedRedistribution) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    std::vector<ParticleRecord> expected;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        auto const record = makeParticle({0x80000000U, 0x80000000U, 0}, rank);
        appendParticle(simulation.particles()[rank], record);
        expected.push_back(record);
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(simulation.particles()[3].size(), 8);
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        if (rank != 3) {
            EXPECT_EQ(simulation.particles()[rank].size(), 0);
        }
    }
    // Move particles to the periodic upper corner and run a fresh collective.
    expected.clear();
    auto& particles = simulation.particles()[3];
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, index) = Coordinate::fromRaw(0xffffffffU);
        }
    }
    expected = particleRecords(particles);
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(simulation.particles()[7].size(), 8);
    EXPECT_EQ(simulation.particles()[3].size(), 0);
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, AllEmptyLocalitiesResetInteriorAndGhostRanges) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    for (auto& grid : simulation.grids()) {
        grid.setInteriorRange({0, 0, 0}, {100, 200});
        grid.setGhostRanges({-1, 0, 0}, {0, 0}, {100, 200});
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify({}));
}

TEST_F(ParticleLoadBalanceTest, FlaggedGhostsOutsideGridRangesAreDiscarded) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    std::vector<ParticleRecord> expected;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        for (size_t index = 0; index < 4; index++) {
            bool const isGhost = (index % 2) != 0;
            auto const record = makeParticle({std::uint32_t(rank) << 29, std::uint32_t(index) << 30, 0xffffffffU}, rank * 4 + index, isGhost);
            appendParticle(simulation.particles()[rank], record);
            if (!isGhost) {
                expected.push_back(record);
            }
        }
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, FlagsAndPeriodicLocalGhostRangesAreExcludedAsAUnion) {
    PROFILE();
    LogicalLocalities simulation(2, 0);
    std::vector<ParticleRecord> expected;
    for (size_t index = 0; index < 6; index++) {
        bool const isGhost = index == 1 || index == 4;
        auto const record = makeParticle({std::uint32_t(index) << 29, 0, 0xffffffffU}, index, isGhost);
        appendParticle(simulation.particles()[0], record);
        if (index == 0 || index == 5) {
            expected.push_back(record);
        }
    }
    // Unflagged periodic copies still need range exclusion; one flagged copy
    // overlaps the ranges and must be discarded only once.
    simulation.grids()[0].setGhostRanges({-1, 0, 0}, {2, 4}, {20, 22});
    simulation.grids()[0].setGhostRanges({0, -1, 0}, {3, 5}, {30, 32});
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    EXPECT_EQ(simulation.particles()[0].size(), 2);
}

TEST_F(ParticleLoadBalanceTest, AllFlaggedGhostsLeaveEmptyOwnedStorage) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        appendParticle(simulation.particles()[rank], makeParticle({std::uint32_t(rank) << 29, 0, 0}, rank, true));
        simulation.grids()[rank].setInteriorRange({0, 0, 0}, {0, 1});
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify({}));
}

TEST_F(ParticleLoadBalanceTest, OverlappingGhostCopiesAreExcludedAsAUnion) {
    PROFILE();
    LogicalLocalities simulation(2, 1);
    std::vector<ParticleRecord> expected;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        for (size_t index = 0; index < 8; index++) {
            auto const record = makeParticle({std::uint32_t(index) << 29, std::uint32_t(rank) << 29, 0xffffffffU}, rank * 8 + index);
            appendParticle(simulation.particles()[rank], record);
            if (index == 0 || index >= 6) {
                expected.push_back(record);
            }
        }
        simulation.grids()[rank].setGhostRanges({-1, 0, 0}, {1, 4}, {90, 93});
        simulation.grids()[rank].setGhostRanges({0, -1, 0}, {3, 6}, {100, 103});
        simulation.grids()[rank].setGhostRanges({0, 0, -1}, {2, 3}, {200, 201});
    }
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
    size_t actualCount = 0;
    for (auto const& particles : simulation.particles()) {
        actualCount += static_cast<size_t>(particles.size());
    }
    EXPECT_EQ(actualCount, 24);
}

TEST_F(ParticleLoadBalanceTest, InvalidGhostRangeFailsEveryRankWithoutMutationAndCanRetry) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(2, 1);
    std::vector<ParticleRecord> expected;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        auto const record = makeParticle({std::uint32_t(rank) << 29, 0xffffffffU, 0}, rank);
        appendParticle(simulation.particles()[rank], record);
        expected.push_back(record);
        simulation.grids()[rank].setInteriorRange({1, 1, 1}, {15, 16});
        simulation.grids()[rank].setGhostRanges({-1, 0, 0}, {0, 0}, {23, 24});
    }
    simulation.grids()[5].setGhostRanges({0, -1, 0}, {0, 2}, {30, 32});
    std::vector<std::vector<ParticleRecord>> initialParticles;
    std::vector<std::vector<CellRecord>> initialCells;
    std::vector<ParticleCount> initialCapacities;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        initialParticles.push_back(particleRecords(simulation.particles()[rank]));
        initialCells.push_back(cellRecords(simulation.grids()[rank]));
        initialCapacities.push_back(simulation.particles()[rank].capacity());
    }
    EXPECT_EQ(simulation.balance().size(), simulation.particles().size());
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        EXPECT_EQ(particleRecords(simulation.particles()[rank]), initialParticles[rank]);
        EXPECT_EQ(cellRecords(simulation.grids()[rank]), initialCells[rank]);
        EXPECT_EQ(simulation.particles()[rank].capacity(), initialCapacities[rank]);
    }
    simulation.grids()[5].setGhostRanges({0, -1, 0}, {0, 0}, {30, 32});
    ASSERT_EQ(simulation.balance(), (std::vector<std::string>{}));
    EXPECT_NO_THROW(simulation.verify(expected));
}

TEST_F(ParticleLoadBalanceTest, RejectsWrongLocalityCountBeforeMutation) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    ParticleStorage particles;
    CartesianGrid grid(2, 0, 0);
    appendParticle(particles, makeParticle({0xffffffffU, 0, 0}, 0));
    grid.setInteriorRange({0, 0, 0}, {7, 8});
    auto const initialParticles = particleRecords(particles);
    auto const initialCells = cellRecords(grid);
    EXPECT_ANY_THROW(cosmictiger::loadBalanceParticles(particles, grid, {}, nextOperationId()));
    EXPECT_EQ(particleRecords(particles), initialParticles);
    EXPECT_EQ(cellRecords(grid), initialCells);
}

TEST_F(ParticleLoadBalanceTest, InconsistentGridTopologyFailsEveryRankWithoutMutation) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    LogicalLocalities simulation(2, 1);
    // Keep all eight participants and locality blocks, but change one cell grid.
    simulation.grids()[5] = CartesianGrid(3, 1, 5);
    std::vector<std::vector<ParticleRecord>> initialParticles;
    std::vector<std::vector<CellRecord>> initialCells;
    std::vector<ParticleCount> initialCapacities;
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        appendParticle(simulation.particles()[rank], makeParticle({std::uint32_t(rank) << 29, 0xffffffffU, 0}, rank));
        appendParticle(simulation.particles()[rank], makeParticle({0, 0, 0}, rank + 100));
        simulation.grids()[rank].setInteriorRange({1, 1, 1}, {15, 16});
        simulation.grids()[rank].setGhostRanges({-1, 0, 0}, {1, 2}, {23, 24});
        initialParticles.push_back(particleRecords(simulation.particles()[rank]));
        initialCells.push_back(cellRecords(simulation.grids()[rank]));
        initialCapacities.push_back(simulation.particles()[rank].capacity());
    }
    EXPECT_EQ(simulation.balance().size(), simulation.particles().size());
    for (size_t rank = 0; rank < simulation.particles().size(); rank++) {
        EXPECT_EQ(particleRecords(simulation.particles()[rank]), initialParticles[rank]);
        EXPECT_EQ(cellRecords(simulation.grids()[rank]), initialCells[rank]);
        EXPECT_EQ(simulation.particles()[rank].capacity(), initialCapacities[rank]);
    }
}

}    // namespace

namespace cosmictiger::detail {

void runRemoteParticleLoadBalance(std::vector<hpx::id_type> localityIds, std::uint64_t operationId, size_t rank) {
    PROFILE();
    CartesianGrid grid(2, 1, rank);
    ParticleStorage particles;
    size_t const destination = (rank + 1) % localityIds.size();
    auto makeDestinationRecords = [](size_t target, size_t source) {
        constexpr size_t kRecordCount = 4096;
        PROFILE();

        std::array<std::uint32_t, NDIM> origin{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            origin[axis] = static_cast<std::uint32_t>((target >> axis) & 1) << 31;
        }
        std::vector<ParticleRecord> records;
        records.reserve(kRecordCount);
        records.push_back(makeParticle(origin, source * kRecordCount));
        auto upper = origin;
        for (auto& coordinate : upper) {
            coordinate += 0x7fffffffU;
        }
        records.push_back(makeParticle(upper, source * kRecordCount + 1));
        auto middle = origin;
        middle[0] += 0x40000000U;
        records.push_back(makeParticle(middle, source * kRecordCount + 2));
        // Each transfer exceeds 64 KiB while sampling all local cells.
        for (size_t index = records.size(); index < kRecordCount; index++) {
            auto position = origin;
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto const offset = (std::uint32_t(index) * 0x1f123bb5U + std::uint32_t(axis) * 0x5f356495U) & 0x7fffffffU;
                position[axis] += offset;
            }
            records.push_back(makeParticle(position, source * kRecordCount + index));
        }
        return records;
    };
    auto const outgoing = makeDestinationRecords(destination, rank);
    for (auto const& record : outgoing) {
        appendParticle(particles, record);
    }
    auto const ghostBegin = particles.size();
    appendParticle(particles, makeParticle({0, 0, 0}, 100 + rank));
    grid.setGhostRanges({-1, 0, 0}, {ghostBegin, ghostBegin + 1}, {90, 91});
    loadBalanceParticles(particles, grid, localityIds, operationId);
    auto const expected = makeDestinationRecords(rank, (rank + localityIds.size() - 1) % localityIds.size());
    verifyPartition(particles, grid, expected);
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::runRemoteParticleLoadBalance, cosmictiger_run_remote_particle_load_balance_action)

TEST_F(ParticleLoadBalanceTest, RemoteRedistribution) {
    PROFILE();
    auto const localityIds = hpx::find_all_localities();
    if (localityIds.size() != 8) {
        GTEST_SKIP() << "requires eight HPX localities";
    }
    auto const operationId = nextOperationId();
    std::vector<hpx::future<void>> futures;
    for (size_t rank = 0; rank < localityIds.size(); rank++) {
        futures.emplace_back(hpx::async<cosmictiger_run_remote_particle_load_balance_action>(localityIds[rank], localityIds, operationId, rank));
    }
    for (auto& future : futures) {
        EXPECT_NO_THROW(future.get());
    }
}
