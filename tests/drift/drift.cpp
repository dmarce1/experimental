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

#include <cosmictiger/drift/drift.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/bit.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <iostream>

#include <array>
#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::DriftRequest;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleRange;
using cosmictiger::ParticleStorage;
using cosmictiger::Rung;

std::uint64_t nextExchangeOperationId() {
    PROFILE();
    static std::uint64_t operationId = 0x6472696674000000ull;
    return operationId++;
}

// Exact signed-integer oracle for the dyadic test inputs. It wraps the full
// position plus displacement rather than reproducing the driver's reduction.
std::uint32_t periodicRaw(std::uint32_t raw, std::int64_t displacementRaw) {
    constexpr std::int64_t kPeriod = std::int64_t(1) << Coordinate::fractionalBits();
    PROFILE();

    std::int64_t wrapped = (std::int64_t(raw) + displacementRaw) % kPeriod;
    if (wrapped < 0) {
        wrapped += kPeriod;
    }
    return static_cast<std::uint32_t>(wrapped);
}

class DriftArrays {
public:
    explicit DriftArrays(size_t count)
      : flags_(count) {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions_[axis].resize(count);
            momenta_[axis].resize(count);
        }
    }

    DriftRequest request(ParticleRange range, double factor) {
        PROFILE();
        DriftRequest result{range, {}, factor};
        result.particles.flags = flags_.data();
        for (size_t axis = 0; axis < NDIM; axis++) {
            result.particles.positions[axis] = positions_[axis].data();
            result.particles.momenta[axis] = momenta_[axis].data();
        }
        return result;
    }

    Coordinate& position(size_t axis, size_t index) {
        PROFILE();
        return positions_[axis][index];
    }

    float& momentum(size_t axis, size_t index) {
        PROFILE();
        return momenta_[axis][index];
    }

    Rung& flags(size_t index) {
        PROFILE();
        return flags_[index];
    }

private:
    std::array<std::vector<Coordinate>, NDIM> positions_;
    std::array<std::vector<float>, NDIM> momenta_;
    std::vector<Rung> flags_;
};

class DriftStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable for managed particle storage";
            }
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        if (!cosmictiger::cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed particle storage";
        }
#endif
    }
};

TEST(DriftCpu, ExactPeriodicDisplacementsPreserveRawLowBits) {
    constexpr std::array<std::uint32_t, 6> kInitial{0, 1, 17, 0x80000001u, 0xfffffffdu, 0xffffffffu};
    constexpr std::array<std::int64_t, 7> kMomentumNumerators{-33, -8, -1, 0, 1, 8, 33};
    constexpr std::array<std::int64_t, 4> kFactorNumerators{-13, -1, 1, 13};
    PROFILE();

    for (auto const momentumNumerator : kMomentumNumerators) {
        for (auto const factorNumerator : kFactorNumerators) {
            DriftArrays particles(kInitial.size());
            for (size_t index = 0; index < kInitial.size(); index++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    particles.position(axis, index) = Coordinate::fromRaw(kInitial[index]);
                    particles.momentum(axis, index) = static_cast<float>(momentumNumerator) / 8.0f;
                }
            }
            cosmictiger::driftParticlesCpu(particles.request({0, kInitial.size()}, static_cast<double>(factorNumerator) / 4.0));
            std::int64_t const displacementRaw = momentumNumerator * factorNumerator * (std::int64_t(1) << 27);
            for (size_t index = 0; index < kInitial.size(); index++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    EXPECT_EQ(particles.position(axis, index).raw(), periodicRaw(kInitial[index], displacementRaw));
                }
            }
        }
    }
}

TEST(DriftCpu, SubLsbDisplacementsTruncateTowardZero) {
    constexpr std::array<std::int64_t, 5> kNumerators{-3, -1, 0, 1, 3};
    PROFILE();

    for (auto const numerator : kNumerators) {
        DriftArrays particles(2);
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, 0) = Coordinate::fromRaw(0);
            particles.position(axis, 1) = Coordinate::fromRaw(0xffffffffu);
            particles.momentum(axis, 0) = static_cast<float>(numerator);
            particles.momentum(axis, 1) = static_cast<float>(numerator);
        }
        cosmictiger::driftParticlesCpu(particles.request({0, 2}, 0x1p-33));
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.position(axis, 0).raw(), periodicRaw(0, numerator / 2));
            EXPECT_EQ(particles.position(axis, 1).raw(), periodicRaw(0xffffffffu, numerator / 2));
        }
    }
}

TEST(DriftCpu, HugeWholeBoxDisplacementsKeepEveryRawBit) {
    PROFILE();
    DriftArrays particles(2);
    for (size_t axis = 0; axis < NDIM; axis++) {
        particles.position(axis, 0) = Coordinate::fromRaw(0xfffffffdu);
        particles.position(axis, 1) = Coordinate::fromRaw(17);
        particles.momentum(axis, 0) = 0x1p100f;
        particles.momentum(axis, 1) = -0x1p100f;
    }
    cosmictiger::driftParticlesCpu(particles.request({0, 2}, -3.0));
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), 0xfffffffdu);
        EXPECT_EQ(particles.position(axis, 1).raw(), 17);
    }
}

TEST(DriftCpu, ParallelSubrangeDriftsAllRungsAndSkipsRemoteGhosts) {
    constexpr size_t kCount = 8197;
    PROFILE();

    DriftArrays particles(kCount);
    for (size_t index = 0; index < kCount; index++) {
        particles.flags(index) = Rung(index % (Rung::maximumRung() + 1), index % 67 == 0);
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const raw = static_cast<std::uint32_t>(std::uint64_t(index) * 104729 + axis + 1);
            particles.position(axis, index) = Coordinate::fromRaw(raw);
            particles.momentum(axis, index) = static_cast<float>(static_cast<int>(index % 13) - 6 + static_cast<int>(axis)) / 8.0f;
        }
    }
    cosmictiger::driftParticlesCpu(particles.request({1, kCount - 1}, -3.25));
    for (size_t index = 0; index < kCount; index++) {
        bool const remote = index % 67 == 0;
        EXPECT_EQ(particles.flags(index).rung(), index % (Rung::maximumRung() + 1));
        EXPECT_EQ(particles.flags(index).isGhost(), remote);
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const raw = static_cast<std::uint32_t>(std::uint64_t(index) * 104729 + axis + 1);
            std::int64_t const numerator = static_cast<int>(index % 13) - 6 + static_cast<int>(axis);
            bool const moved = index != 0 && index != kCount - 1 && !remote;
            EXPECT_EQ(particles.position(axis, index).raw(), periodicRaw(raw, moved ? numerator * -13 * (std::int64_t(1) << 27) : 0));
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(particles.momentum(axis, index)), hpx::bit_cast<std::uint32_t>(static_cast<float>(numerator) / 8.0f));
        }
    }
}

TEST(DriftCpu, EmptyAndZeroFactorRequestsNeedNoPointers) {
    PROFILE();
    EXPECT_NO_THROW(cosmictiger::driftParticlesCpu(DriftRequest{{7, 7}, {}, 1.0}));
    EXPECT_NO_THROW(cosmictiger::driftParticlesCpu(DriftRequest{{0, 8197}, {}, 0.0}));
}

TEST(DriftCpu, InvalidRequestsFailBeforeChangingPositions) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    DriftArrays particles(1);
    particles.position(0, 0) = Coordinate::fromRaw(17);
    EXPECT_THROW(cosmictiger::driftParticlesCpu(particles.request({1, 0}, 1.0)), std::runtime_error);
    auto request = particles.request({0, 1}, 1.0);
    request.particles.flags = nullptr;
    EXPECT_THROW(cosmictiger::driftParticlesCpu(request), std::runtime_error);
    for (size_t axis = 0; axis < NDIM; axis++) {
        request = particles.request({0, 1}, 1.0);
        request.particles.positions[axis] = nullptr;
        EXPECT_THROW(cosmictiger::driftParticlesCpu(request), std::runtime_error);
        request = particles.request({0, 1}, 1.0);
        request.particles.momenta[axis] = nullptr;
        EXPECT_THROW(cosmictiger::driftParticlesCpu(request), std::runtime_error);
    }
    EXPECT_EQ(particles.position(0, 0).raw(), 17);
}

TEST_F(DriftStorageTest, EveryOptionalGravityConfigurationPreservesNonPositionFields) {
    constexpr ParticleCount kCount = Rung::maximumRung() + 1;
    PROFILE();

    for (unsigned configuration = 0; configuration < 4; configuration++) {
        bool const acceleration = (configuration & 1) != 0;
        bool const potential = (configuration & 2) != 0;
        ParticleStorage particles(kCount);
        particles.configureGravityFields(acceleration, potential);
        for (ParticleCount index = 0; index < kCount; index++) {
            particles.flags(index) = Rung(index, false);
            for (size_t axis = 0; axis < NDIM; axis++) {
                particles.position(axis, index) = Coordinate::fromRaw(static_cast<std::uint32_t>(index * 4093 + axis));
                particles.velocity(axis, index) = index == 0 ? -0.0f : static_cast<float>(index + axis) / 8.0f;
                if (acceleration) {
                    particles.acceleration(axis, index) = -static_cast<float>(index + axis) - 0.25f;
                }
            }
            if (potential) {
                particles.potential(index) = -static_cast<float>(index) - 0.5f;
            }
        }
        cosmictiger::driftParticlesCpu(cosmictiger::makeDriftRequest(particles, {0, kCount}, 0.5));
        EXPECT_EQ(particles.size(), kCount);
        EXPECT_EQ(particles.storesAcceleration(), acceleration);
        EXPECT_EQ(particles.storesPotential(), potential);
        for (ParticleCount index = 0; index < kCount; index++) {
            EXPECT_EQ(particles.flags(index).rung(), index);
            EXPECT_FALSE(particles.flags(index).isGhost());
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto const raw = static_cast<std::uint32_t>(index * 4093 + axis);
                auto const displacementRaw = index == 0 ? 0 : static_cast<std::int64_t>(index + axis) * (std::int64_t(1) << 28);
                EXPECT_EQ(particles.position(axis, index).raw(), periodicRaw(raw, displacementRaw));
                float const momentum = index == 0 ? -0.0f : static_cast<float>(index + axis) / 8.0f;
                EXPECT_EQ(hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index)), hpx::bit_cast<std::uint32_t>(momentum));
                if (acceleration) {
                    EXPECT_EQ(particles.acceleration(axis, index), -static_cast<float>(index + axis) - 0.25f);
                }
            }
            if (potential) {
                EXPECT_EQ(particles.potential(index), -static_cast<float>(index) - 0.5f);
            }
        }
    }
}

TEST_F(DriftStorageTest, DriverRemovesSameOwnerPeriodicCopiesAndDriftsOwnedPrefixOnce) {
    PROFILE();
    ParticleStorage particles;
    CartesianGrid grid(0, 0, 0);
    std::array<Coordinate, NDIM> position{};
    std::array<float, NDIM> momentum{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        position[axis] = Coordinate::fromRaw(0x80000001u + static_cast<std::uint32_t>(axis));
        momentum[axis] = static_cast<float>(axis + 1) / 8.0f;
    }
    particles.append(position, momentum, Rung::maximumRung());
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    cosmictiger::exchangeGhostParticles(particles, grid, {hpx::find_here()}, nextExchangeOperationId());
    ASSERT_GT(particles.size(), 1);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        ASSERT_FALSE(particles.flags(index).isGhost());
    }
    auto const capacity = particles.capacity();
    cosmictiger::driftParticles(particles, grid, -3.25);
    ASSERT_EQ(particles.size(), 1);
    EXPECT_EQ(particles.capacity(), capacity);
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), periodicRaw(position[axis].raw(), -13 * static_cast<std::int64_t>(axis + 1) * (std::int64_t(1) << 27)));
        EXPECT_EQ(particles.velocity(axis, 0), momentum[axis]);
    }
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        auto const& cell = grid.cell(coordinates);
        if (grid.layout().isInterior(coordinates)) {
            EXPECT_EQ(cell.localParticles.begin, 0);
            EXPECT_EQ(cell.localParticles.end, 1);
        } else {
            EXPECT_EQ(cell.localParticles.begin, cell.localParticles.end);
            EXPECT_EQ(cell.ownerParticles.begin, cell.ownerParticles.end);
        }
    }
}

TEST_F(DriftStorageTest, StorageRequestRejectsOutOfBoundsRanges) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    ParticleStorage particles(1);
    EXPECT_THROW(cosmictiger::makeDriftRequest(particles, {0, 2}, 1.0), std::runtime_error);
    EXPECT_THROW(cosmictiger::makeDriftRequest(particles, {1, 0}, 1.0), std::runtime_error);
}

TEST(DriftCpu, PackedAndOwnedPathsMatchScalarOracleAtTailsAndWrapBoundaries) {
    constexpr std::array<size_t, 14> kCounts{0, 1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097};
    constexpr std::array<double, 6> kFactors{0x1p-33, 0.1, -3.25, 1.0, 0x1p50, -0x1p90};
    constexpr size_t kGuardCount = 2;
    constexpr std::uint32_t kInitial = 0xfffffffdu;
    PROFILE();
    for (bool const ownedOnly : {false, true}) {
        for (size_t const count : kCounts) {
            for (double const factor : kFactors) {
                DriftArrays particles(count + kGuardCount);
                for (size_t index = 0; index < count + kGuardCount; index++) {
                    particles.flags(index) = Rung(0, !ownedOnly && index % 19 == 0);
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        particles.position(axis, index) = Coordinate::fromRaw(kInitial);
                        particles.momentum(axis, index) = static_cast<float>(int(index % 23) - 11) / 7.0f + static_cast<float>(axis) * 0.5f;
                    }
                }
                auto request = particles.request({1, count + 1}, factor);
                request.ownedOnly = ownedOnly;
                cosmictiger::driftParticlesCpu(request);
                for (size_t index = 0; index < count + kGuardCount; index++) {
                    bool const moved = index > 0 && index <= count && !particles.flags(index).isGhost();
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        double const displacement = double(particles.momentum(axis, index)) * factor;
                        auto const shift = moved ? static_cast<std::int64_t>(std::ldexp(std::fmod(displacement, 1.0), Coordinate::fractionalBits())) : 0;
                        EXPECT_EQ(particles.position(axis, index).raw(), periodicRaw(kInitial, shift));
                    }
                }
            }
        }
    }
}

TEST(DriftCpu, DISABLED_PackedDriftBenchmark) {
    constexpr size_t kCount = 1048579;
    constexpr size_t kRepeats = 32;
    constexpr double kFactor = 0.00123456789;
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();
    DriftArrays particles(kCount);
    for (size_t index = 0; index < kCount; index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, index) = Coordinate::fromRaw(static_cast<std::uint32_t>(index));
            particles.momentum(axis, index) = static_cast<float>(int(index % 101) - 50) / 13.0f;
        }
    }
    auto request = particles.request({0, kCount}, kFactor);
    request.ownedOnly = true;
    auto const scalarStart = hpx::chrono::high_resolution_clock::now();
    for (size_t repeat = 0; repeat < kRepeats; repeat++) {
        cosmictiger::FpeGuard guard;
        for (size_t index = 0; index < kCount; index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const displacement = double(request.particles.momenta[axis][index]) * kFactor;
                auto const shift = static_cast<std::int64_t>(std::ldexp(std::fmod(displacement, 1.0), Coordinate::fractionalBits()));
                auto& position = request.particles.positions[axis][index];
                position = Coordinate::fromRaw(static_cast<std::uint32_t>(std::uint64_t(position.raw()) + static_cast<std::uint32_t>(shift)));
            }
        }
    }
    auto const packedStart = hpx::chrono::high_resolution_clock::now();
    for (size_t repeat = 0; repeat < kRepeats; repeat++) {
        cosmictiger::driftParticlesCpu(request);
    }
    auto const end = hpx::chrono::high_resolution_clock::now();
    std::cout << "Drift benchmark: particles=" << kCount << " repeats=" << kRepeats
              << " scalar_seconds=" << double(packedStart - scalarStart) * kSecondsPerNanosecond
              << " packed_seconds=" << double(end - packedStart) * kSecondsPerNanosecond << " checksum=" << particles.position(0, kCount / 2).raw() << '\n';
}

}    // namespace
