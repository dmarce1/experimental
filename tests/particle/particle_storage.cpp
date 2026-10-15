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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_storage.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <gtest/gtest.h>
#include <hpx/bit.hpp>

#include <limits>
#include <type_traits>
#include <utility>

namespace {

using Storage = cosmictiger::ParticleStorage;
using Rung = cosmictiger::Rung;
using Coordinate = Storage::Coordinate;
using Size = Storage::Size;

class ParticleStorageTest : public ::testing::Test {
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

TEST(ParticleStorageArithmetic, CountsAndCapacitiesExceed32Bits) {
    constexpr Size kBoundary = Size(1) << 32;

    static_assert(std::numeric_limits<Size>::digits == 64);
    PROFILE();
    EXPECT_EQ(Storage::capacityForSize(kBoundary), kBoundary);
    EXPECT_EQ(Storage::capacityForSize(kBoundary + 1), kBoundary * 2);
    if constexpr (sizeof(size_t) >= 8) {
        EXPECT_EQ(cosmictiger::detail::particleAllocationBytes<Coordinate>(kBoundary + 1), (kBoundary + 1) * 4);
    }
    EXPECT_EQ(Storage::capacityForSize(Size(1) << 63), Size(1) << 63);
    EXPECT_THROW(Storage::capacityForSize((Size(1) << 63) + 1), std::length_error);
    EXPECT_THROW(Storage::capacityForSize(std::numeric_limits<Size>::max()), std::length_error);
    EXPECT_EQ(Storage::capacityForSize(0), 0);
    EXPECT_EQ(Storage::capacityForSize(1), 1);
    EXPECT_EQ(Storage::capacityForSize(9), 16);
}

TEST(ParticleStorageArithmetic, AllocationBytesAreCheckedBeforeNarrowing) {
    PROFILE();
    Size const maximum = std::numeric_limits<size_t>::max() / sizeof(Coordinate);
    EXPECT_EQ(cosmictiger::detail::particleAllocationBytes<Coordinate>(maximum), static_cast<size_t>(maximum) * sizeof(Coordinate));
    EXPECT_THROW(cosmictiger::detail::particleAllocationBytes<Coordinate>(maximum + 1), std::length_error);
    Storage empty;
    EXPECT_THROW(empty.reserve(Size(1) << 63), std::length_error);
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(empty.capacity(), 0);
}

TEST(ParticleStorageArithmetic, BoundsModeMatchesValidationConfiguration) {
    static_assert(Storage::boundsChecking() == cosmictiger::validationChecksEnabled());
    PROFILE();
    EXPECT_EQ(sizeof(Coordinate), 4);
    EXPECT_EQ(Coordinate::fractionalBits(), 32);
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Storage&>().velocity(0, 0)), float&>) );
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Storage&>().flags(0)), Rung&>) );
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Storage const&>().flags(0)), Rung const&>) );
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Storage&>().rungData()), Rung*>) );
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Storage const&>().rungData()), Rung const*>) );
}

TEST(ParticleStorageArithmetic, RungAndGhostFlagOccupyOneByteAndMutateIndependently) {
    static_assert(sizeof(Rung) == 1);
    static_assert(std::is_trivially_copyable_v<Rung>);
    static_assert(Rung::maximumRung() == 63);
    PROFILE();
    Rung value{Rung::maximumRung(), true};
    EXPECT_EQ(value.rung(), Rung::maximumRung());
    EXPECT_TRUE(value.isGhost());
    value.setRung(0);
    EXPECT_EQ(value.rung(), 0);
    EXPECT_TRUE(value.isGhost());
    value.setRung(Rung::maximumRung());
    value.setGhost(false);
    EXPECT_EQ(value.rung(), Rung::maximumRung());
    EXPECT_FALSE(value.isGhost());
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(value.setRung(64), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(value.setRung(127), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(value.setRung(256), std::runtime_error);
    }
    EXPECT_EQ(value.rung(), Rung::maximumRung());
    EXPECT_FALSE(value.isGhost());
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW((Rung{64, false}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW((Rung{127, true}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW((Rung{256, false}), std::runtime_error);
    }
}

TEST(ParticleStorageArithmetic, RungSettersPreserveReservedBit) {
    PROFILE();
    Rung value = hpx::bit_cast<Rung>(std::uint8_t(0x80));
    value.setRung(63);
    value.setGhost(true);
    EXPECT_EQ(value.rung(), 63);
    EXPECT_TRUE(value.isGhost());
    EXPECT_EQ(hpx::bit_cast<std::uint8_t>(value) & 0x80, 0x80);
}

TEST_F(ParticleStorageTest, InitializesSevenContiguousFields) {
    PROFILE();
    Storage particles(5);
    EXPECT_EQ(particles.size(), 5);
    EXPECT_EQ(particles.capacity(), 8);
    Storage const& readOnly = particles;
    std::array<void const*, 2 * NDIM + 1> pointers{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        pointers[axis] = particles.positionData(axis);
        pointers[axis + NDIM] = particles.velocityData(axis);
        for (Size index = 0; index < particles.size(); index++) {
            EXPECT_EQ(&particles.position(axis, index), particles.positionData(axis) + index);
            EXPECT_EQ(&particles.velocity(axis, index), particles.velocityData(axis) + index);
            EXPECT_EQ(readOnly.position(axis, index).raw(), 0);
            EXPECT_EQ(readOnly.velocity(axis, index), 0.0f);
        }
    }
    pointers[2 * NDIM] = particles.rungData();
    for (size_t first = 0; first < pointers.size(); first++) {
        for (size_t second = first + 1; second < pointers.size(); second++) {
            EXPECT_NE(pointers[first], pointers[second]);
        }
    }
    for (Size index = 0; index < particles.size(); index++) {
        EXPECT_EQ(readOnly.flags(index).rung(), 0);
        EXPECT_FALSE(readOnly.flags(index).isGhost());
        EXPECT_EQ(&readOnly.flags(index), readOnly.rungData() + index);
    }
}

TEST_F(ParticleStorageTest, AppendDoublesCapacityAndPreservesValues) {
    PROFILE();
    Storage particles;
    for (Size index = 0; index < 17; index++) {
        auto const raw = static_cast<std::uint32_t>(index * 1234567);
        EXPECT_EQ(particles.append({Coordinate::fromRaw(raw), Coordinate::fromRaw(raw + 1), Coordinate::fromRaw(raw + 2)},
                      {float(index), -float(index), float(index) + 0.5f}, static_cast<std::uint8_t>(index), index % 2 != 0),
            index);
        EXPECT_EQ(particles.capacity(), Storage::capacityForSize(index + 1));
    }
    EXPECT_EQ(particles.capacity(), 32);
    for (Size index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.position(axis, index).raw(), index * 1234567 + axis);
        }
        EXPECT_EQ(particles.velocity(0, index), float(index));
        EXPECT_EQ(particles.velocity(1, index), -float(index));
        EXPECT_EQ(particles.velocity(2, index), float(index) + 0.5f);
        EXPECT_EQ(particles.flags(index).rung(), index);
        EXPECT_EQ(particles.flags(index).isGhost(), index % 2 != 0);
    }
}

TEST_F(ParticleStorageTest, GravityFieldsAllocateIndependentlyAndOnlyWhenEnabled) {
    PROFILE();
    Storage particles(3);
    EXPECT_FALSE(particles.storesAcceleration());
    EXPECT_FALSE(particles.storesPotential());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.accelerationData(axis), nullptr);
    }
    EXPECT_EQ(particles.potentialData(), nullptr);
    particles.configureGravityFields(true, false);
    EXPECT_TRUE(particles.storesAcceleration());
    EXPECT_FALSE(particles.storesPotential());
    EXPECT_EQ(particles.potentialData(), nullptr);
    Storage const& readOnly = particles;
    for (size_t axis = 0; axis < NDIM; axis++) {
        for (Size index = 0; index < particles.size(); index++) {
            EXPECT_EQ(readOnly.acceleration(axis, index), 0.0f);
            EXPECT_EQ(&readOnly.acceleration(axis, index), readOnly.accelerationData(axis) + index);
        }
    }
    particles.acceleration(0, 1) = 4.5f;
    particles.configureGravityFields(true, true);
    EXPECT_EQ(particles.acceleration(0, 1), 4.5f);
    for (Size index = 0; index < particles.size(); index++) {
        EXPECT_EQ(readOnly.potential(index), 0.0f);
        EXPECT_EQ(&readOnly.potential(index), readOnly.potentialData() + index);
    }
    particles.potential(1) = -9.0f;
    particles.configureGravityFields(false, true);
    EXPECT_FALSE(particles.storesAcceleration());
    EXPECT_TRUE(particles.storesPotential());
    EXPECT_EQ(particles.potential(1), -9.0f);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.accelerationData(axis), nullptr);
    }
    particles.configureGravityFields(false, false);
    EXPECT_EQ(particles.potentialData(), nullptr);
}

TEST_F(ParticleStorageTest, GravityFieldsFollowParticlesThroughGrowthPermutationRemovalAndMove) {
    constexpr std::array<Size, 4> kIdentities{3, 4, 2, 0};
    PROFILE();

    Storage particles;
    particles.configureGravityFields(true, true);
    for (Size index = 0; index < 5; index++) {
        particles.append({}, {}, index);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.acceleration(axis, index), 0.0f);
            particles.acceleration(axis, index) = float(index * NDIM + axis + 1);
        }
        EXPECT_EQ(particles.potential(index), 0.0f);
        particles.potential(index) = -float(index + 1);
    }
    particles.swapParticles(0, 3);
    particles.remove(1);
    particles.resize(9);
    for (Size index = 0; index < kIdentities.size(); index++) {
        EXPECT_EQ(particles.flags(index).rung(), kIdentities[index]);
        EXPECT_EQ(particles.potential(index), -float(kIdentities[index] + 1));
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.acceleration(axis, index), float(kIdentities[index] * NDIM + axis + 1));
        }
    }
    for (Size index = kIdentities.size(); index < particles.size(); index++) {
        EXPECT_EQ(particles.potential(index), 0.0f);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.acceleration(axis, index), 0.0f);
        }
    }
    Storage moved(std::move(particles));
    EXPECT_TRUE(moved.storesAcceleration());
    EXPECT_TRUE(moved.storesPotential());
    EXPECT_EQ(moved.potential(0), -4.0f);
    EXPECT_FALSE(particles.storesAcceleration());
    EXPECT_FALSE(particles.storesPotential());
    EXPECT_EQ(particles.potentialData(), nullptr);
}

TEST_F(ParticleStorageTest, AppendAcceptsMaximumRungAndDefaultsToLocalParticle) {
    PROFILE();
    Storage particles;
    particles.append({}, {}, Rung::maximumRung());
    particles.append({}, {}, Rung::maximumRung(), true);
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    EXPECT_FALSE(particles.flags(0).isGhost());
    EXPECT_EQ(particles.flags(1).rung(), Rung::maximumRung());
    EXPECT_TRUE(particles.flags(1).isGhost());
    particles.flags(0).setGhost(true);
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    particles.flags(1).setRung(1);
    EXPECT_TRUE(particles.flags(1).isGhost());
}

TEST_F(ParticleStorageTest, AppendRejectsRungsAboveMaximumBeforeMutation) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    Storage particles;
    for (std::uint64_t const invalid : {64, 65, 127, 256}) {
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.append({}, {}, invalid), std::runtime_error);
        }
    }
    EXPECT_EQ(particles.size(), 0);
    EXPECT_EQ(particles.capacity(), 0);
    EXPECT_EQ(particles.rungData(), nullptr);
    particles.append({Coordinate::fromRaw(1), Coordinate::fromRaw(2), Coordinate::fromRaw(3)}, {4.0f, 5.0f, 6.0f}, Rung::maximumRung(), true);
    auto* positionPointer = particles.positionData(0);
    auto* velocityPointer = particles.velocityData(0);
    auto* rungPointer = particles.rungData();
    for (std::uint64_t const invalid : {64, 65, 127, 256}) {
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.append({}, {}, invalid, true), std::runtime_error);
        }
    }
    EXPECT_EQ(particles.size(), 1);
    EXPECT_EQ(particles.capacity(), 1);
    EXPECT_EQ(particles.positionData(0), positionPointer);
    EXPECT_EQ(particles.velocityData(0), velocityPointer);
    EXPECT_EQ(particles.rungData(), rungPointer);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), axis + 1);
        EXPECT_EQ(particles.velocity(axis, 0), float(axis + 4));
    }
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    EXPECT_TRUE(particles.flags(0).isGhost());
}

TEST_F(ParticleStorageTest, RemovalMovesAllFieldsAndRetainsAllocations) {
    PROFILE();
    Storage particles;
    for (std::uint32_t index = 0; index < 4; index++) {
        particles.append({Coordinate::fromRaw(index), Coordinate::fromRaw(index + 1), Coordinate::fromRaw(index + 2)},
            {float(index), float(index + 1), float(index + 2)}, static_cast<std::uint8_t>(index), index == 3);
    }
    auto* positionPointer = particles.positionData(0);
    auto* velocityPointer = particles.velocityData(1);
    auto* rungPointer = particles.rungData();
    particles.remove(1);
    EXPECT_EQ(particles.size(), 3);
    EXPECT_EQ(particles.capacity(), 4);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 1).raw(), 3 + axis);
        EXPECT_EQ(particles.velocity(axis, 1), float(3 + axis));
    }
    EXPECT_EQ(particles.flags(1).rung(), 3);
    EXPECT_TRUE(particles.flags(1).isGhost());
    EXPECT_EQ(particles.position(0, 0).raw(), 0);
    EXPECT_EQ(particles.position(0, 2).raw(), 2);
    particles.remove(2);
    particles.resize(1);
    particles.clear();
    EXPECT_EQ(particles.size(), 0);
    EXPECT_EQ(particles.capacity(), 4);
    EXPECT_EQ(particles.positionData(0), positionPointer);
    EXPECT_EQ(particles.velocityData(1), velocityPointer);
    EXPECT_EQ(particles.rungData(), rungPointer);
    particles.resize(3);
    for (Size index = 0; index < particles.size(); index++) {
        EXPECT_EQ(particles.position(0, index).raw(), 0);
        EXPECT_EQ(particles.velocity(1, index), 0);
        EXPECT_EQ(particles.flags(index).rung(), 0);
        EXPECT_FALSE(particles.flags(index).isGhost());
    }
}

TEST_F(ParticleStorageTest, SwapParticlesMovesAllFieldsTogether) {
    PROFILE();
    Storage particles;
    particles.append({Coordinate::fromRaw(1), Coordinate::fromRaw(2), Coordinate::fromRaw(3)}, {4.0f, 5.0f, 6.0f}, 7, true);
    particles.append({Coordinate::fromRaw(11), Coordinate::fromRaw(12), Coordinate::fromRaw(13)}, {14.0f, 15.0f, 16.0f}, 17);
    auto* positionPointer = particles.positionData(0);
    auto* velocityPointer = particles.velocityData(0);
    auto* rungPointer = particles.rungData();

    particles.swapParticles(0, 1);
    particles.swapParticles(1, 1);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), 11 + axis);
        EXPECT_EQ(particles.position(axis, 1).raw(), 1 + axis);
        EXPECT_EQ(particles.velocity(axis, 0), float(14 + axis));
        EXPECT_EQ(particles.velocity(axis, 1), float(4 + axis));
    }
    EXPECT_EQ(particles.flags(0).rung(), 17);
    EXPECT_FALSE(particles.flags(0).isGhost());
    EXPECT_EQ(particles.flags(1).rung(), 7);
    EXPECT_TRUE(particles.flags(1).isGhost());
    EXPECT_EQ(particles.positionData(0), positionPointer);
    EXPECT_EQ(particles.velocityData(0), velocityPointer);
    EXPECT_EQ(particles.rungData(), rungPointer);
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(particles.swapParticles(0, particles.size()), std::runtime_error);
    }
}

TEST_F(ParticleStorageTest, ReserveAndShrinkPreserveSizeAndSurvivors) {
    PROFILE();
    Storage particles(3);
    particles.position(2, 0) = Coordinate::fromRaw(std::numeric_limits<std::uint32_t>::max());
    particles.velocity(1, 0) = -2.5f;
    particles.flags(0).setRung(Rung::maximumRung());
    particles.flags(0).setGhost(true);
    particles.reserve(20);
    EXPECT_EQ(particles.capacity(), 32);
    EXPECT_EQ(particles.size(), 3);
    auto* pointer = particles.positionData(2);
    particles.reserve(2);
    particles.resize(1);
    EXPECT_EQ(particles.capacity(), 32);
    EXPECT_EQ(particles.positionData(2), pointer);
    EXPECT_EQ(particles.position(2, 0).raw(), std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(particles.velocity(1, 0), -2.5f);
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    EXPECT_TRUE(particles.flags(0).isGhost());
    EXPECT_THROW(particles.resize(std::numeric_limits<Size>::max()), std::length_error);
    EXPECT_EQ(particles.size(), 1);
    EXPECT_EQ(particles.capacity(), 32);
    EXPECT_EQ(particles.positionData(2), pointer);
}

TEST_F(ParticleStorageTest, MovesTransferOwnershipAndLeaveEmptySource) {
    PROFILE();
    Storage source(3);
    source.flags(1).setRung(42);
    source.flags(1).setGhost(true);
    auto* pointer = source.rungData();
    Storage destination(std::move(source));
    EXPECT_EQ(destination.rungData(), pointer);
    EXPECT_EQ(destination.flags(1).rung(), 42);
    EXPECT_TRUE(destination.flags(1).isGhost());
    EXPECT_EQ(source.size(), 0);
    EXPECT_EQ(source.capacity(), 0);
    EXPECT_EQ(source.rungData(), nullptr);
    Storage assigned(9);
    assigned = std::move(destination);
    EXPECT_EQ(assigned.size(), 3);
    EXPECT_EQ(assigned.capacity(), 4);
    EXPECT_EQ(assigned.rungData(), pointer);
    EXPECT_EQ(assigned.flags(1).rung(), 42);
    EXPECT_TRUE(assigned.flags(1).isGhost());
    EXPECT_EQ(destination.size(), 0);
    EXPECT_EQ(destination.capacity(), 0);
    source.resize(2);
    EXPECT_EQ(source.flags(0).rung(), 0);
    EXPECT_FALSE(source.flags(0).isGhost());
}

TEST_F(ParticleStorageTest, ChecksLiveSizeAxisAndLargeIndicesWhenEnabled) {
    PROFILE();
    if constexpr (!Storage::boundsChecking()) {
        GTEST_SKIP() << "bounds checking disabled in this build";
    } else {
        Storage particles(3);
        Storage const& readOnly = particles;
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.position(0, 3), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(readOnly.position(0, Size(1) << 32), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.velocity(0, 3), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(readOnly.velocity(0, std::numeric_limits<Size>::max()), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.flags(3), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(readOnly.flags(Size(1) << 32), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.positionData(NDIM), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(readOnly.positionData(NDIM), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.velocityData(NDIM), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(readOnly.velocityData(NDIM), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.remove(3), std::runtime_error);
        }
        EXPECT_EQ(particles.size(), 3);
        particles.clear();
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.remove(0), std::runtime_error);
        }
        if constexpr (cosmictiger::validationChecksEnabled()) {
            EXPECT_THROW(particles.flags(0), std::runtime_error);
        }
        EXPECT_EQ(particles.size(), 0);
    }
}

TEST_F(ParticleStorageTest, LocalityAccessorRetainsIndependentOwner) {
    PROFILE();
    auto& local = cosmictiger::localParticleStorage();
    EXPECT_EQ(&local, &cosmictiger::localParticleStorage());
    local.resize(2);
    local.flags(0).setRung(17);
    local.flags(0).setGhost(true);
    Storage separate(1);
    separate.flags(0).setRung(63);
    EXPECT_EQ(cosmictiger::localParticleStorage().flags(0).rung(), 17);
    EXPECT_TRUE(cosmictiger::localParticleStorage().flags(0).isGhost());
    EXPECT_FALSE(separate.flags(0).isGhost());
    local.clear();
}

#ifdef COSMICTIGER_ENABLE_CUDA
TEST_F(ParticleStorageTest, EveryFieldUsesCudaManagedMemoryAfterGrowth) {
    PROFILE();
    Storage particles(3);
    for (Size const newSize : {Size(3), Size(9)}) {
        particles.resize(newSize);
        std::array<void const*, 2 * NDIM + 1> pointers{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            pointers[axis] = particles.positionData(axis);
            pointers[axis + NDIM] = particles.velocityData(axis);
        }
        pointers[2 * NDIM] = particles.rungData();
        for (auto const* pointer : pointers) {
            auto const attributes = cosmictiger::cuda::pointerAttributes(pointer).get();
            EXPECT_EQ(attributes.type, cudaMemoryTypeManaged);
        }
    }
}
#endif

}    // namespace
