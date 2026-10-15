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

#include <cosmictiger/gravity/direct/direct_interactions_cuda.cuh>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/particle/particle_storage.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <new>
#include <stddef.h>
#include <vector>

namespace {

using cosmictiger::Acceleration;
using cosmictiger::Coordinate;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleStorage;
using cosmictiger::Rung;
using cosmictiger::gravity::DirectInteractionParticles;
using cosmictiger::gravity::DirectInteractionRequest;
using cosmictiger::gravity::directInteractionsCpu;
using cosmictiger::gravity::DirectKickBackend;
using cosmictiger::gravity::KickParameters;
using cosmictiger::gravity::KickPhase;
using cosmictiger::gravity::KickSums;
using cosmictiger::gravity::launchDirectInteractionsCuda;
using cosmictiger::gravity::makeFerrersSoftening;

struct InteractionRecord {
    Acceleration acceleration{};
    float potential = 0.0f;
    unsigned hits = 0;
    float rungZeroTimestep = 0.0f;
    std::uint8_t rung = 0;
};

class RecordingBackend {
public:
    __host__ __device__ explicit RecordingBackend(InteractionRecord* records) noexcept
      : records_(records) {
        PROFILE();
    }

    __host__ __device__ void operator()(
        ParticleCount sinkIndex, Acceleration acceleration, float potential, Rung flags, float rungZeroTimestep, DirectInteractionParticles) const noexcept {
        PROFILE();
        auto& record = records_[sinkIndex];
        record.acceleration = acceleration;
        record.potential = potential;
        record.rungZeroTimestep = rungZeroTimestep;
        record.rung = flags.rung();
        ++record.hits;
    }

private:
    InteractionRecord* records_;
};

class ManagedRecords {
public:
    ManagedRecords() = default;
    ManagedRecords(ManagedRecords const&) = delete;
    ManagedRecords& operator=(ManagedRecords const&) = delete;

    ~ManagedRecords() {
        PROFILE();
        if (values_ != nullptr) {
            try {
                cosmictiger::cuda::free(values_).get();
            } catch (std::exception const& error) {
                ADD_FAILURE() << "Managed record deallocation failed: " << error.what();
            }
        }
    }

    void allocate(size_t requestedCount) {
        PROFILE();
        values_ = static_cast<InteractionRecord*>(cosmictiger::cuda::mallocManaged(requestedCount * sizeof(InteractionRecord)).get());
        count_ = requestedCount;
        for (size_t index = 0; index < count_; index++) {
            ::new (static_cast<void*>(values_ + index)) InteractionRecord{};
        }
    }

    InteractionRecord* data() noexcept {
        PROFILE();
        return values_;
    }

    InteractionRecord const& operator[](size_t index) const noexcept {
        PROFILE();
        return values_[index];
    }

private:
    InteractionRecord* values_ = nullptr;
    size_t count_ = 0;
};

class ManagedKickSums {
public:
    ManagedKickSums() = default;
    ManagedKickSums(ManagedKickSums const&) = delete;
    ManagedKickSums& operator=(ManagedKickSums const&) = delete;

    ~ManagedKickSums() {
        PROFILE();
        if (value_ != nullptr) {
            try {
                cosmictiger::cuda::free(value_).get();
            } catch (std::exception const& error) {
                ADD_FAILURE() << "Managed kick sums deallocation failed: " << error.what();
            }
        }
    }

    void allocate() {
        PROFILE();
        value_ = static_cast<KickSums*>(cosmictiger::cuda::mallocManaged(sizeof(KickSums)).get());
        ::new (static_cast<void*>(value_)) KickSums{};
    }

    KickSums* data() noexcept {
        PROFILE();
        return value_;
    }

private:
    KickSums* value_ = nullptr;
};

DirectInteractionParticles particleView(ParticleStorage& particles) {
    PROFILE();
    DirectInteractionParticles view{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        view.positions[axis] = particles.positionData(axis);
        view.velocities[axis] = particles.velocityData(axis);
        view.accelerations[axis] = particles.accelerationData(axis);
    }
    view.flags = particles.rungData();
    view.potentials = particles.potentialData();
    return view;
}

__global__ void knownFieldKickKernel(DirectInteractionParticles particles, KickParameters parameters, KickSums* sums) {
    PROFILE();
    DirectKickBackend{parameters, sums}(0, Acceleration{8.0f, 0.0f, 0.0f}, -4.0f, particles.flags[0], 1.0, particles);
}

hpx::future<void> launchKnownFieldKick(ParticleStorage& particles, KickParameters parameters, KickSums* sums) {
    PROFILE();
    auto const view = particleView(particles);
    return cosmictiger::cuda::launch(hpx::annotated_function(
        [view, parameters, sums](cudaStream_t stream) {
            PROFILE();
            knownFieldKickKernel<<<1, 1, 0, stream>>>(view, parameters, sums);
        },
        "DirectInteractionsCudaTest::launchKnownFieldKick"));
}

void expectNear(float actual, float expected) {
    PROFILE();
    EXPECT_NEAR(actual, expected, 2.0e-3f * std::max(1.0f, std::fabs(expected)));
}

class DirectInteractionsCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
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
    }
};

TEST_F(DirectInteractionsCudaTest, RegularKickStoresEachHalfStepSeparately) {
    PROFILE();
    ParticleStorage particles(1);
    particles.velocity(0, 0) = 16777216.0f;
    particles.flags(0) = Rung{2, false};
    KickParameters parameters;
    parameters.eta = 1.0;
    parameters.epsilon = 0.25;
    parameters.phase = KickPhase::regular;
    parameters.timeTick = cosmictiger::gravity::rungTicks(2);

    ASSERT_NO_THROW(launchKnownFieldKick(particles, parameters, nullptr).get());

    // Each rung-two half kick adds exactly one. At 2^24, each stored float
    // rounds back to 2^24; combining them into one +2 update would differ.
    EXPECT_EQ(particles.velocity(0, 0), 16777216.0f);
    EXPECT_NE(particles.velocity(0, 0), 16777218.0f);
    EXPECT_EQ(particles.flags(0).rung(), 2);
}

TEST_F(DirectInteractionsCudaTest, KnownFieldKickUsesOldAndSelectedRungsAndHandlesEndpointPhases) {
    PROFILE();
    struct KickCase {
        char const* description;
        KickPhase phase;
        std::uint8_t oldRung;
        double epsilon;
        std::uint64_t timeTick;
        float initialMomentum;
        float expectedMomentum;
        std::uint8_t expectedRung;
    };
    std::array<KickCase, 5> const cases{{
        {"first kick opens only the selected half step", KickPhase::initial, 0, 0.25, 0, 3.0f, 4.0f, 2},
        {"refinement closes rung one and opens rung three", KickPhase::regular, 1, 0.0625, cosmictiger::gravity::rungTicks(1), 1.0f, 3.5f, 3},
        {"coarsening stops at the synchronized rung", KickPhase::regular, 3, 4.0, cosmictiger::gravity::rungTicks(1), 1.0f, 3.5f, 1},
        {"unsynchronized coarsening retains the old rung", KickPhase::regular, 3, 4.0, cosmictiger::gravity::rungTicks(3), 1.0f, 2.0f, 3},
        {"last kick closes only the old half step", KickPhase::final, 3, 4.0, cosmictiger::gravity::rungTicks(0), 3.0f, 3.5f, 3},
    }};
    for (auto const& kickCase : cases) {
        SCOPED_TRACE(kickCase.description);
        ParticleStorage particles(1);
        particles.configureGravityFields(true, true);
        particles.velocity(0, 0) = kickCase.initialMomentum;
        particles.flags(0) = Rung{kickCase.oldRung, false};
        ManagedKickSums sums;
        ASSERT_NO_THROW(sums.allocate());
        KickParameters parameters;
        parameters.eta = 1.0;
        parameters.epsilon = kickCase.epsilon;
        parameters.phase = kickCase.phase;
        parameters.timeTick = kickCase.timeTick;
        parameters.sumPotential = true;

        ASSERT_NO_THROW(launchKnownFieldKick(particles, parameters, sums.data()).get());

        EXPECT_EQ(particles.flags(0).rung(), kickCase.expectedRung);
        EXPECT_FALSE(particles.flags(0).isGhost());
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.velocity(axis, 0), axis == 0 ? kickCase.expectedMomentum : 0.0f);
            EXPECT_EQ(particles.acceleration(axis, 0), axis == 0 ? 8.0f : 0.0f);
            EXPECT_EQ(sums.data()->acceleration[axis], axis == 0 ? 8.0 : 0.0);
        }
        EXPECT_EQ(particles.potential(0), -4.0f);
        EXPECT_EQ(sums.data()->particleCount, 1);
        EXPECT_EQ(sums.data()->potential, -4.0);
        EXPECT_EQ(sums.data()->maximumRung, kickCase.expectedRung);
    }
}

TEST_F(DirectInteractionsCudaTest, CrossesSourceTilesAndSinkBlocksWithRungGhostAndMassFiltering) {
    ParticleCount constexpr kParticleCount = 259;
    std::array<std::uint32_t, NDIM> constexpr kCoordinateSteps{0x9e3779b9U, 0x7f4a7c15U, 0x85ebca6bU};
    unsigned constexpr kBlocks = static_cast<unsigned>((kParticleCount - 1) / cosmictiger::gravity::detail::kDirectInteractionTileSize + 1);
    PROFILE();
    ParticleStorage particles(kParticleCount);
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, index) = Coordinate::fromRaw(kCoordinateSteps[axis] * static_cast<std::uint32_t>(index + 1));
            particles.velocity(axis, index) = 0.25f * static_cast<float>(index + axis);
        }
        particles.flags(index) = Rung{index % 4, index % 29 == 0};
    }
    particles.flags(127) = Rung{3, false};
    particles.flags(128) = Rung{2, false};
    particles.flags(129) = Rung{1, false};
    particles.flags(130) = Rung{2, true};

    DirectInteractionRequest const request{{0, 131}, {0, kParticleCount}, particleView(particles), makeFerrersSoftening(0.1875f), 2.5f, 0.03125f, 2};
    ManagedRecords actual;
    ASSERT_NO_THROW(actual.allocate(static_cast<size_t>(kParticleCount)));

    auto* const records = actual.data();
    auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
        [request, kBlocks, records](cudaStream_t stream) {
            PROFILE();
            cosmictiger::gravity::detail::directInteractionsKernel<RecordingBackend, 2, true>
                <<<kBlocks, cosmictiger::gravity::detail::kDirectInteractionTileSize, 0, stream>>>(request, RecordingBackend{records});
        },
        "DirectInteractionsCudaTest::launchBatched"));
    ASSERT_NO_THROW(complete.get());

    std::vector<InteractionRecord> expected(static_cast<size_t>(kParticleCount));
    directInteractionsCpu<RecordingBackend, 2, true>(request, RecordingBackend{expected.data()});
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        SCOPED_TRACE(index);
        auto const& deviceRecord = actual[static_cast<size_t>(index)];
        auto const& hostRecord = expected[static_cast<size_t>(index)];
        EXPECT_EQ(deviceRecord.hits, hostRecord.hits);
        if (hostRecord.hits != 0) {
            EXPECT_EQ(deviceRecord.hits, 1U);
            expectNear(deviceRecord.acceleration.x, hostRecord.acceleration.x);
            expectNear(deviceRecord.acceleration.y, hostRecord.acceleration.y);
            expectNear(deviceRecord.acceleration.z, hostRecord.acceleration.z);
            expectNear(deviceRecord.potential, hostRecord.potential);
            EXPECT_EQ(deviceRecord.rung, particles.flags(index).rung());
            EXPECT_FLOAT_EQ(deviceRecord.rungZeroTimestep, request.rungZeroTimestep);
        }
    }
    EXPECT_EQ(actual[127].hits, 1U);
    EXPECT_EQ(actual[128].hits, 1U);
    EXPECT_EQ(actual[129].hits, 0U);
    EXPECT_EQ(actual[130].hits, 0U);
}

TEST_F(DirectInteractionsCudaTest, ExcludesSelfPotentialButKeepsDistinctCoincidentParticles) {
    float constexpr kParticleMass = 3.0f;
    PROFILE();
    ParticleStorage particles(2);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        particles.position(0, index) = Coordinate::fromRaw(0x10000000U);
        particles.position(1, index) = Coordinate::fromRaw(0x20000000U);
        particles.position(2, index) = Coordinate::fromRaw(0x30000000U);
        particles.flags(index) = Rung{index == 0 ? 0U : 3U, index == 0};
    }
    DirectInteractionRequest const request{{0, 2}, {0, 2}, particleView(particles), makeFerrersSoftening(0.25f), kParticleMass, 0.125f, 3};
    ManagedRecords actual;
    ASSERT_NO_THROW(actual.allocate(2));

    auto* const records = actual.data();
    auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
        [request, records](cudaStream_t stream) {
            PROFILE();
            cosmictiger::gravity::detail::directInteractionsKernel<RecordingBackend, 3, true>
                <<<1, cosmictiger::gravity::detail::kDirectInteractionTileSize, 0, stream>>>(request, RecordingBackend{records});
        },
        "DirectInteractionsCudaTest::launchSelfExclusion"));
    ASSERT_NO_THROW(complete.get());

    auto const distinctField =
        cosmictiger::gravity::ferrersAccelerationAndPotential<3>({particles.position(0, 0), particles.position(1, 0), particles.position(2, 0)},
            {particles.position(0, 1), particles.position(1, 1), particles.position(2, 1)}, request.softening);
    EXPECT_EQ(actual[0].hits, 0U);
    EXPECT_EQ(actual[1].hits, 1U);
    EXPECT_FLOAT_EQ(actual[1].acceleration.x, 0.0f);
    EXPECT_FLOAT_EQ(actual[1].acceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(actual[1].acceleration.z, 0.0f);
    expectNear(actual[1].potential, kParticleMass * distinctField.potential);
}

TEST_F(DirectInteractionsCudaTest, AccelerationOnlyFieldsAndPublicKickMatchCpu) {
    PROFILE();
    ParticleStorage particles(3);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.position(axis, index) = Coordinate::fromRaw(static_cast<std::uint32_t>((index + 1) * (axis + 2)) << 25);
            particles.velocity(axis, index) = -0.5f * static_cast<float>(index + axis + 1);
        }
        particles.flags(index) = Rung{1, false};
    }
    DirectInteractionRequest const request{{0, 2}, {1, 3}, particleView(particles), makeFerrersSoftening(0.25f), 1.75f, 0.0625f, 1};
    ManagedRecords actual;
    ASSERT_NO_THROW(actual.allocate(3));

    auto* const records = actual.data();
    auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
        [request, records](cudaStream_t stream) {
            PROFILE();
            cosmictiger::gravity::detail::directInteractionsKernel<RecordingBackend, 1, false>
                <<<1, cosmictiger::gravity::detail::kDirectInteractionTileSize, 0, stream>>>(request, RecordingBackend{records});
        },
        "DirectInteractionsCudaTest::launchAccelerationOnly"));
    ASSERT_NO_THROW(complete.get());

    std::array<InteractionRecord, 3> expected{};
    directInteractionsCpu<RecordingBackend, 1, false>(request, RecordingBackend{expected.data()});
    for (size_t index = 0; index < expected.size(); index++) {
        SCOPED_TRACE(index);
        EXPECT_EQ(actual[index].hits, expected[index].hits);
        EXPECT_FLOAT_EQ(actual[index].potential, 0.0f);
        if (expected[index].hits != 0) {
            expectNear(actual[index].acceleration.x, expected[index].acceleration.x);
            expectNear(actual[index].acceleration.y, expected[index].acceleration.y);
            expectNear(actual[index].acceleration.z, expected[index].acceleration.z);
        }
    }

    ParticleStorage reference(particles.size());
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            reference.position(axis, index) = particles.position(axis, index);
            reference.velocity(axis, index) = particles.velocity(axis, index);
        }
        reference.flags(index) = particles.flags(index);
    }
    auto cpuRequest = request;
    cpuRequest.particles = particleView(reference);
    directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 1, false>(cpuRequest);
    ASSERT_NO_THROW(launchDirectInteractionsCuda(request, 1, false).get());
    for (ParticleCount index = 0; index < particles.size(); index++) {
        EXPECT_EQ(particles.flags(index).rung(), reference.flags(index).rung());
        for (size_t axis = 0; axis < NDIM; axis++) {
            expectNear(particles.velocity(axis, index), reference.velocity(axis, index));
        }
    }
}

TEST_F(DirectInteractionsCudaTest, PublicKickUpdatesRungsMomentumPersistentFieldsAndManagedSums) {
    ParticleCount constexpr kParticleCount = 259;
    std::array<std::uint32_t, NDIM> constexpr kCoordinateSteps{0x9e3779b9U, 0x7f4a7c15U, 0x85ebca6bU};
    PROFILE();
    ParticleStorage actual(kParticleCount);
    ParticleStorage reference(kParticleCount);
    actual.configureGravityFields(true, true);
    reference.configureGravityFields(true, true);
    ParticleCount expectedCount = 0;
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const position = Coordinate::fromRaw(kCoordinateSteps[axis] * static_cast<std::uint32_t>(index + 1));
            float const momentum = 0.01f * static_cast<float>(index + axis + 1);
            float const acceleration = 100.0f + static_cast<float>(axis);
            actual.position(axis, index) = reference.position(axis, index) = position;
            actual.velocity(axis, index) = reference.velocity(axis, index) = momentum;
            actual.acceleration(axis, index) = reference.acceleration(axis, index) = acceleration;
        }
        actual.potential(index) = reference.potential(index) = 103.0f;
        actual.flags(index) = reference.flags(index) = Rung{1 + index % 3, index % 29 == 0};
        if (index >= 3 && actual.flags(index).rung() >= 2 && !actual.flags(index).isGhost()) {
            expectedCount++;
        }
    }

    ManagedKickSums deviceSums;
    ASSERT_NO_THROW(deviceSums.allocate());
    KickSums cpuSums;
    DirectInteractionRequest request{{0, 131}, {3, kParticleCount}, particleView(actual), makeFerrersSoftening(0.1875f), 0.015f, 0.5, 2};
    request.kick.phase = KickPhase::regular;
    request.kick.scaleFactor = 0.5;
    request.kick.timeTick = cosmictiger::gravity::rungTicks(1);
    request.kick.sumPotential = true;
    request.sums = deviceSums.data();
    auto cpuRequest = request;
    cpuRequest.particles = particleView(reference);
    cpuRequest.sums = &cpuSums;
    directInteractionsCpu<DirectKickBackend, 2, true>(cpuRequest);

    // Host and device have separate totals; host reads wait for GPU completion.
    ASSERT_NO_THROW(launchDirectInteractionsCuda(request, 2, true).get());
    EXPECT_EQ(deviceSums.data()->particleCount, expectedCount);
    EXPECT_EQ(deviceSums.data()->particleCount, cpuSums.particleCount);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(deviceSums.data()->acceleration[axis], cpuSums.acceleration[axis], 2.0e-5 * std::max(1.0, std::abs(cpuSums.acceleration[axis])));
    }
    EXPECT_NEAR(deviceSums.data()->potential, cpuSums.potential, 2.0e-5 * std::max(1.0, std::abs(cpuSums.potential)));

    bool changedRung = false;
    bool changedMomentum = false;
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        SCOPED_TRACE(index);
        EXPECT_EQ(actual.flags(index).rung(), reference.flags(index).rung());
        EXPECT_EQ(actual.flags(index).isGhost(), reference.flags(index).isGhost());
        if (actual.flags(index).rung() != 1 + index % 3) {
            changedRung = true;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            expectNear(actual.velocity(axis, index), reference.velocity(axis, index));
            expectNear(actual.acceleration(axis, index), reference.acceleration(axis, index));
            if (actual.velocity(axis, index) != 0.01f * static_cast<float>(index + axis + 1)) {
                changedMomentum = true;
            }
        }
        expectNear(actual.potential(index), reference.potential(index));
    }
    EXPECT_TRUE(changedRung);
    EXPECT_TRUE(changedMomentum);
}

TEST_F(DirectInteractionsCudaTest, PlummerFieldsMatchCpuBeyondTheSofteningLength) {
    constexpr ParticleCount kParticleCount = 2;
    constexpr float kEpsilon = 0.0625f;
    PROFILE();
    ParticleStorage particles(kParticleCount);
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        particles.position(0, index) = Coordinate::fromRaw(static_cast<std::uint32_t>(index) * 0x40000000U);
        particles.position(1, index) = Coordinate::fromRaw(0);
        particles.position(2, index) = Coordinate::fromRaw(0);
        particles.flags(index) = Rung{0, false};
    }
    DirectInteractionRequest const request{
        {0, kParticleCount}, {0, kParticleCount}, particleView(particles), cosmictiger::gravity::makePlummerSoftening(kEpsilon), 1.0f, 1.0, 0};
    ManagedRecords actual;
    actual.allocate(kParticleCount);
    auto* const records = actual.data();
    auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
        [request, records](cudaStream_t stream) {
            PROFILE();
            cosmictiger::gravity::detail::directInteractionsKernel<RecordingBackend, 2, true>
                <<<1, cosmictiger::gravity::detail::kDirectInteractionTileSize, 0, stream>>>(request, RecordingBackend{records});
        },
        "DirectInteractionsCudaTest::launchPlummer"));
    ASSERT_NO_THROW(complete.get());
    std::array<InteractionRecord, kParticleCount> expected{};
    directInteractionsCpu<RecordingBackend, 2, true>(request, RecordingBackend{expected.data()});
    for (ParticleCount index = 0; index < kParticleCount; index++) {
        EXPECT_EQ(actual[index].hits, 1U);
        expectNear(actual[index].acceleration.x, expected[index].acceleration.x);
        expectNear(actual[index].acceleration.y, expected[index].acceleration.y);
        expectNear(actual[index].acceleration.z, expected[index].acceleration.z);
        expectNear(actual[index].potential, expected[index].potential);
    }
}

TEST(DirectInteractionsCudaFuture, EmptySinkCompletesWithoutADevice) {
    PROFILE();
    DirectInteractionRequest const request{};
    auto complete = launchDirectInteractionsCuda(request, 1, false);
    EXPECT_TRUE(complete.is_ready());
    EXPECT_NO_THROW(complete.get());
}

TEST(DirectInteractionsCudaFuture, RejectsInvalidRequestsThroughTheFuture) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    DirectInteractionRequest request{};
    auto invalidOrder = launchDirectInteractionsCuda(request, 4, false);
    EXPECT_THROW(invalidOrder.get(), std::runtime_error);
    request.source = {2, 1};
    auto invalidRange = launchDirectInteractionsCuda(request, 1, false);
    EXPECT_THROW(invalidRange.get(), std::runtime_error);
    request.source = {0, 1};
    request.sink = {0, 1};
    auto missingParticles = launchDirectInteractionsCuda(request, 1, false);
    EXPECT_THROW(missingParticles.get(), std::runtime_error);
}

}    // namespace
