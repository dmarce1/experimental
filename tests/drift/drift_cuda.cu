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

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/drift/drift.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/bit.hpp>
#include <hpx/chrono.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <iostream>

#include <array>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::DriftExecution;
using cosmictiger::DriftRequest;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleStorage;
using cosmictiger::Rung;

std::uint64_t nextExchangeOperationId() {
    PROFILE();
    static std::uint64_t operationId = 0x6472696675000000ull;
    return operationId++;
}

std::uint32_t periodicRaw(std::uint32_t raw, std::int64_t displacementRaw) {
    constexpr std::int64_t kPeriod = std::int64_t(1) << Coordinate::fractionalBits();
    PROFILE();
    std::int64_t wrapped = (std::int64_t(raw) + displacementRaw) % kPeriod;
    if (wrapped < 0) {
        wrapped += kPeriod;
    }
    return static_cast<std::uint32_t>(wrapped);
}

void seedParticles(ParticleStorage& particles, ParticleCount count) {
    PROFILE();
    particles.resize(count);
    particles.configureGravityFields(true, true);
    for (ParticleCount index = 0; index < count; index++) {
        particles.flags(index) = Rung(index % (Rung::maximumRung() + 1), index % 19 == 0);
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const raw = static_cast<std::uint32_t>(std::uint64_t(index) * 104729 + axis + 1);
            particles.position(axis, index) = Coordinate::fromRaw(raw);
            particles.velocity(axis, index) = index == 0 ? -0.0f : static_cast<float>(static_cast<int>(index % 13) - 6 + static_cast<int>(axis)) / 8.0f;
            particles.acceleration(axis, index) = -static_cast<float>(index + axis) - 0.25f;
        }
        particles.potential(index) = -static_cast<float>(index) - 0.5f;
    }
}

class DriftCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable";
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
        stream_ = cosmictiger::cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cosmictiger::cuda::streamDestroy(stream_).get());
        }
    }

    cudaStream_t stream() const noexcept {
        PROFILE();
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

TEST_F(DriftCudaTest, ExternalStreamMatchesIntegerOracleAndCpuAcrossBlocksAndChunks) {
    std::array<ParticleCount, 8> constexpr kCounts{31, 32, 33, 255, 256, 257, 513, 8197};
    PROFILE();
    for (auto const count : kCounts) {
        ParticleStorage cpu;
        ParticleStorage gpu;
        seedParticles(cpu, count);
        seedParticles(gpu, count);
        cosmictiger::ParticleRange const range{3, count - 2};
        cosmictiger::driftParticlesCpu(cosmictiger::makeDriftRequest(cpu, range, -3.25));
        auto completed = cosmictiger::launchDriftParticlesCuda(cosmictiger::makeDriftRequest(gpu, range, -3.25), stream());
        ASSERT_TRUE(completed.valid());
        // GPU positions and caller-owned stream remain alive through get().
        // Every host read of the submitted positions follows this completion.
        ASSERT_NO_THROW(completed.get());
        for (ParticleCount index = 0; index < count; index++) {
            bool const remote = index % 19 == 0;
            EXPECT_EQ(gpu.flags(index).rung(), index % (Rung::maximumRung() + 1));
            EXPECT_EQ(gpu.flags(index).isGhost(), remote);
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto const raw = static_cast<std::uint32_t>(std::uint64_t(index) * 104729 + axis + 1);
                std::int64_t const numerator = static_cast<int>(index % 13) - 6 + static_cast<int>(axis);
                bool const moved = index >= range.begin && index < range.end && !remote;
                auto const expected = periodicRaw(raw, moved ? numerator * -13 * (std::int64_t(1) << 27) : 0);
                EXPECT_EQ(gpu.position(axis, index).raw(), expected);
                EXPECT_EQ(gpu.position(axis, index).raw(), cpu.position(axis, index).raw());
                EXPECT_EQ(hpx::bit_cast<std::uint32_t>(gpu.velocity(axis, index)), hpx::bit_cast<std::uint32_t>(cpu.velocity(axis, index)));
                EXPECT_EQ(gpu.acceleration(axis, index), -static_cast<float>(index + axis) - 0.25f);
            }
            EXPECT_EQ(gpu.potential(index), -static_cast<float>(index) - 0.5f);
        }
    }
}

TEST_F(DriftCudaTest, SmallAndHugeDisplacementsPreservePeriodicRawPrecision) {
    PROFILE();
    ParticleStorage particles(2);
    for (size_t axis = 0; axis < NDIM; axis++) {
        particles.position(axis, 0) = Coordinate::fromRaw(0xffffffffu);
        particles.position(axis, 1) = Coordinate::fromRaw(0);
        particles.velocity(axis, 0) = 3.0f;
        particles.velocity(axis, 1) = -3.0f;
    }
    ASSERT_NO_THROW(cosmictiger::launchDriftParticlesCuda(cosmictiger::makeDriftRequest(particles, {0, 2}, 0x1p-33), stream()).get());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), 0);
        EXPECT_EQ(particles.position(axis, 1).raw(), 0xffffffffu);
        particles.velocity(axis, 0) = 0x1p100f;
        particles.velocity(axis, 1) = -0x1p100f;
    }
    ASSERT_NO_THROW(cosmictiger::launchDriftParticlesCuda(cosmictiger::makeDriftRequest(particles, {0, 2}, -3.0), stream()).get());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), 0);
        EXPECT_EQ(particles.position(axis, 1).raw(), 0xffffffffu);
    }
}

TEST_F(DriftCudaTest, DriverChoosesCudaClearsSameOwnerCopiesAndWaitsForCompletion) {
    PROFILE();
    ParticleStorage particles;
    CartesianGrid grid(0, 0, 0);
    std::array<Coordinate, NDIM> position{};
    std::array<float, NDIM> momentum{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        position[axis] = Coordinate::fromRaw(0xfffffffdu - static_cast<std::uint32_t>(axis));
        momentum[axis] = static_cast<float>(axis + 1) / 8.0f;
    }
    particles.append(position, momentum, Rung::maximumRung());
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    cosmictiger::exchangeGhostParticles(particles, grid, {hpx::find_here()}, nextExchangeOperationId());
    ASSERT_GT(particles.size(), 1);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        ASSERT_FALSE(particles.flags(index).isGhost());
    }
    ASSERT_NO_THROW(cosmictiger::driftParticles(particles, grid, 3.25, DriftExecution::cuda));
    ASSERT_EQ(particles.size(), 1);
    EXPECT_EQ(particles.flags(0).rung(), Rung::maximumRung());
    EXPECT_FALSE(particles.flags(0).isGhost());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(particles.position(axis, 0).raw(), periodicRaw(position[axis].raw(), 13 * static_cast<std::int64_t>(axis + 1) * (std::int64_t(1) << 27)));
        EXPECT_EQ(particles.velocity(axis, 0), momentum[axis]);
    }
}

TEST(DriftCudaRequests, EmptyAndZeroFactorCompleteWithoutPointersOrDevice) {
    PROFILE();
    auto empty = cosmictiger::launchDriftParticlesCuda(DriftRequest{{7, 7}, {}, 1.0});
    ASSERT_TRUE(empty.is_ready());
    EXPECT_NO_THROW(empty.get());
    auto zero = cosmictiger::launchDriftParticlesCuda(DriftRequest{{0, 8197}, {}, 0.0});
    ASSERT_TRUE(zero.is_ready());
    EXPECT_NO_THROW(zero.get());
}

TEST(DriftCudaRequests, InvalidRequestsReturnExceptionalFuturesBeforeSubmission) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    hpx::future<void> invalidRange;
    EXPECT_NO_THROW(invalidRange = cosmictiger::launchDriftParticlesCuda(DriftRequest{{1, 0}, {}, 1.0}));
    ASSERT_TRUE(invalidRange.valid());
    EXPECT_THROW(invalidRange.get(), std::runtime_error);
    hpx::future<void> missingPointers;
    EXPECT_NO_THROW(missingPointers = cosmictiger::launchDriftParticlesCuda(DriftRequest{{0, 1}, {}, 1.0}));
    ASSERT_TRUE(missingPointers.valid());
    EXPECT_THROW(missingPointers.get(), std::runtime_error);
}

// Write momenta and read coordinates as the CUDA kick does, so the measured
// CPU drift starts with GPU-owned managed pages rather than warm host arrays.
struct HandoffParticles {
    Coordinate const* positions[NDIM];
    float* momenta[NDIM];
    ParticleCount count;
};

__global__ void handoffMomentumKernel(HandoffParticles request) {
    for (ParticleCount index = blockIdx.x * blockDim.x + threadIdx.x; index < request.count; index += gridDim.x * blockDim.x) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            request.momenta[axis][index] = (request.positions[axis][index].raw() & 1u) ? 0.125f : -0.125f;
        }
    }
}

TEST_F(DriftCudaTest, DISABLED_ManagedCpuHandoffBenchmark) {
    constexpr std::array<ParticleCount, 2> kCounts{32768, 1048576};
    constexpr size_t kRepeats = 24;
    constexpr unsigned kBlocks = 32;
    constexpr unsigned kThreads = 128;
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();
    for (auto const count : kCounts) {
        ParticleStorage particles(count);
        for (ParticleCount index = 0; index < count; index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                particles.position(axis, index) = Coordinate::fromRaw(static_cast<std::uint32_t>(index + axis));
            }
        }
        auto request = cosmictiger::makeDriftRequest(particles, {0, count}, 0.001);
        request.ownedOnly = true;
        HandoffParticles handoff{};
        handoff.count = count;
        std::vector<cosmictiger::cuda::ManagedMemoryRange> ranges;
        for (size_t axis = 0; axis < NDIM; axis++) {
            handoff.positions[axis] = particles.positionData(axis);
            handoff.momenta[axis] = particles.velocityData(axis);
            ranges.push_back({request.particles.positions[axis], size_t(count) * sizeof(Coordinate)});
            ranges.push_back({request.particles.momenta[axis], size_t(count) * sizeof(float)});
        }
        for (bool const prefetch : {false, true}) {
            double handoffSeconds = 0.0;
            double warmSeconds = 0.0;
            for (size_t repeat = 0; repeat < kRepeats; repeat++) {
                cosmictiger::cuda::launch(
                    hpx::annotated_function([handoff](cudaStream_t launchStream) { handoffMomentumKernel<<<kBlocks, kThreads, 0, launchStream>>>(handoff); },
                        "test.drift.handoffMomentum"),
                    stream())
                    .get();
                auto const begin = hpx::chrono::high_resolution_clock::now();
                if (prefetch) {
                    cosmictiger::cuda::prefetch(ranges, cudaCpuDeviceId, stream()).get();
                }
                cosmictiger::driftParticlesCpu(request);
                auto const warmBegin = hpx::chrono::high_resolution_clock::now();
                cosmictiger::driftParticlesCpu(request);
                auto const end = hpx::chrono::high_resolution_clock::now();
                handoffSeconds += double(warmBegin - begin) * kSecondsPerNanosecond;
                warmSeconds += double(end - warmBegin) * kSecondsPerNanosecond;
            }
            std::cout << "Managed drift: particles=" << count << " prefetch=" << prefetch << " handoff_seconds=" << handoffSeconds / kRepeats
                      << " warm_seconds=" << warmSeconds / kRepeats << '\n';
        }
    }
}

}    // namespace
