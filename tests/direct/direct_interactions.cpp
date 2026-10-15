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

#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <hpx/type_support/bit_cast.hpp>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using cosmictiger::Acceleration;
using cosmictiger::Coordinate;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleRange;
using cosmictiger::Position;
using cosmictiger::Rung;
using cosmictiger::gravity::DirectFieldBackend;
using cosmictiger::gravity::DirectInteractionParticles;
using cosmictiger::gravity::DirectInteractionRequest;
using cosmictiger::gravity::directInteractionsCpu;
using cosmictiger::gravity::ferrersAcceleration;
using cosmictiger::gravity::ferrersAccelerationAndPotential;
using cosmictiger::gravity::makeFerrersSoftening;

template <size_t count>
class TestParticles {
public:
    void setPosition(size_t index, std::array<std::uint32_t, NDIM> values) {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions_[axis][index] = Coordinate::fromRaw(values[axis]);
        }
    }

    Position position(size_t index) const {
        PROFILE();
        return {positions_[0][index], positions_[1][index], positions_[2][index]};
    }

    DirectInteractionParticles view() {
        PROFILE();
        DirectInteractionParticles result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result.positions[axis] = positions_[axis].data();
            result.velocities[axis] = velocities_[axis].data();
        }
        result.flags = flags_.data();
        return result;
    }
    std::array<Rung, count>& flags() noexcept {
        PROFILE();
        return flags_;
    }

private:
    std::array<Coordinate, count> positions_[NDIM]{};
    std::array<float, count> velocities_[NDIM]{};
    std::array<Rung, count> flags_{};
};

struct KickRecord {
    ParticleCount sinkIndex;
    Acceleration acceleration;
    float potential;
    Rung flags;
    float rungZeroTimestep;
    DirectInteractionParticles particles;
};

class RecordingBackend {
public:
    explicit RecordingBackend(std::vector<KickRecord>* records) noexcept
      : records_(records) {
        PROFILE();
    }

    void operator()(
        ParticleCount sinkIndex, Acceleration acceleration, float potential, Rung flags, float rungZeroTimestep, DirectInteractionParticles particles) const {
        PROFILE();
        records_->push_back({sinkIndex, acceleration, potential, flags, rungZeroTimestep, particles});
    }

private:
    std::vector<KickRecord>* records_;
};

class DisabledConcurrentRecordingBackend : public RecordingBackend {
public:
    using RecordingBackend::RecordingBackend;

    static constexpr bool supportsConcurrentSinks() noexcept {
        PROFILE();
        return false;
    }
};

// The same field writer with concurrency explicitly disabled is the scalar
// reference; only sink scheduling differs between the two runs.
class SerialFieldBackend {
public:
    explicit SerialFieldBackend(DirectFieldBackend backend)
      : backend_(backend) {
        PROFILE();
    }

    static constexpr bool supportsConcurrentSinks() noexcept {
        PROFILE();
        return false;
    }

    void operator()(
        ParticleCount sinkIndex, Acceleration acceleration, float potential, Rung flags, float timestep, DirectInteractionParticles particles) const {
        PROFILE();
        backend_(sinkIndex, acceleration, potential, flags, timestep, particles);
    }

private:
    DirectFieldBackend backend_;
};

class GatedSinkBackend {
public:
    GatedSinkBackend(std::atomic<size_t>& entered, std::atomic<size_t>& completed, hpx::promise<void>& ready, hpx::shared_future<void> release,
        size_t readyCount, bool failFirst = false)
      : entered_(&entered)
      , completed_(&completed)
      , ready_(&ready)
      , release_(std::move(release))
      , readyCount_(readyCount)
      , failFirst_(failFirst) {
        PROFILE();
    }

    static constexpr bool supportsConcurrentSinks() noexcept {
        PROFILE();
        return true;
    }

    void operator()(ParticleCount sinkIndex, Acceleration, float, Rung, float, DirectInteractionParticles) const {
        PROFILE();
        if (failFirst_ && sinkIndex == 0) {
            throw std::runtime_error("injected sink backend failure");
        }
        if (entered_->fetch_add(1) + 1 == readyCount_) {
            ready_->set_value();
        }
        // A supported callback may suspend. This also verifies that the CPU
        // arithmetic guard does not span a backend invocation.
        release_.get();
        completed_->fetch_add(1);
    }

private:
    std::atomic<size_t>* entered_;
    std::atomic<size_t>* completed_;
    hpx::promise<void>* ready_;
    hpx::shared_future<void> release_;
    size_t readyCount_;
    bool failFirst_;
};

template <bool computePotential>
void compareSinkFields() {
    constexpr size_t kCount = 263;
    constexpr ParticleRange kSinkRange{7, kCount - 3};
    constexpr size_t kSinkCount = kSinkRange.end - kSinkRange.begin;
    constexpr float kPotentialSentinel = 74.0f;
    constexpr Acceleration kSentinel{71.0f, -72.0f, 73.0f};
    PROFILE();

    TestParticles<kCount> particles;
    for (size_t index = 0; index < kCount; index++) {
        auto const word = static_cast<std::uint32_t>(index);
        particles.setPosition(index, {0x12345678U + word * 0x01112223U, 0x23456789U + word * 0x02334445U, 0x3456789aU + word * 0x03556667U});
        particles.flags()[index] = Rung(index % 5, index % 11 == 0);
    }
    // Both particles lie inside the overlapping source/sink ranges, exercising
    // self exclusion while preserving a distinct coincident source.
    particles.setPosition(7, {0x50000000U, 0x60000000U, 0x70000000U});
    particles.setPosition(8, {0x50000000U, 0x60000000U, 0x70000000U});
    std::vector<Acceleration> scalarAcceleration(kSinkCount + 2, kSentinel);
    std::vector<Acceleration> parallelAcceleration(kSinkCount + 2, kSentinel);
    std::vector<float> scalarPotential(kSinkCount + 2, kPotentialSentinel);
    std::vector<float> parallelPotential(kSinkCount + 2, kPotentialSentinel);
    DirectInteractionRequest const request{
        {5, 5 + 2 * cosmictiger::gravity::detail::kDirectFloatLanes + 3}, kSinkRange, particles.view(), makeFerrersSoftening(0.125f), 1.5f, 0.25f, 2};
    directInteractionsCpu<SerialFieldBackend, 2, computePotential>(
        request, SerialFieldBackend{DirectFieldBackend{kSinkRange, scalarAcceleration.data() + 1, scalarPotential.data() + 1}});
    directInteractionsCpu<DirectFieldBackend, 2, computePotential>(
        request, DirectFieldBackend{kSinkRange, parallelAcceleration.data() + 1, parallelPotential.data() + 1});
    for (size_t offset = 0; offset < kSinkCount + 2; offset++) {
        SCOPED_TRACE(::testing::Message() << "potential=" << computePotential << " output=" << offset);
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(parallelAcceleration[offset].x), hpx::bit_cast<std::uint32_t>(scalarAcceleration[offset].x));
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(parallelAcceleration[offset].y), hpx::bit_cast<std::uint32_t>(scalarAcceleration[offset].y));
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(parallelAcceleration[offset].z), hpx::bit_cast<std::uint32_t>(scalarAcceleration[offset].z));
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(parallelPotential[offset]), hpx::bit_cast<std::uint32_t>(scalarPotential[offset]));
        bool active = false;
        if (offset != 0 && offset != kSinkCount + 1) {
            Rung const flags = particles.flags()[kSinkRange.begin + offset - 1];
            active = !flags.isGhost() && flags.rung() >= request.rung;
        }
        if (!active) {
            EXPECT_EQ(parallelAcceleration[offset].x, kSentinel.x);
            EXPECT_EQ(parallelAcceleration[offset].y, kSentinel.y);
            EXPECT_EQ(parallelAcceleration[offset].z, kSentinel.z);
            EXPECT_EQ(parallelPotential[offset], kPotentialSentinel);
        } else {
            if constexpr (computePotential) {
                EXPECT_LT(parallelPotential[offset], 0.0f);
            } else {
                EXPECT_EQ(parallelPotential[offset], 0.0f);
            }
        }
    }
}

Acceleration add(Acceleration first, Acceleration second) {
    PROFILE();
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

void expectAccelerationNear(Acceleration actual, Acceleration expected) {
    PROFILE();
    EXPECT_NEAR(actual.x, expected.x, 2.0e-5f);
    EXPECT_NEAR(actual.y, expected.y, 2.0e-5f);
    EXPECT_NEAR(actual.z, expected.z, 2.0e-5f);
}

TEST(DirectInteractionsCpu, SourceAndSinkRangesKeepSlowAndGhostSourcesButSelectOwnedActiveSinks) {
    PROFILE();
    TestParticles<5> particles;
    particles.setPosition(0, {0x10000000U, 0x20000000U, 0x30000000U});
    particles.setPosition(1, {0x18000000U, 0x20000000U, 0x30000000U});
    particles.setPosition(2, {0x20000000U, 0x28000000U, 0x30000000U});
    particles.setPosition(3, {0x30000000U, 0x30000000U, 0x40000000U});
    particles.setPosition(4, {0x38000000U, 0x38000000U, 0x40000000U});
    particles.flags() = {Rung(2, false), Rung(1, true), Rung(2, false), Rung(3, false), Rung(2, true)};
    auto const softening = makeFerrersSoftening(0.125f);
    DirectInteractionRequest const request{{1, 4}, {0, 5}, particles.view(), softening, 1.0f, 0.25f, 2};
    std::vector<KickRecord> records;

    directInteractionsCpu<RecordingBackend, 2, false>(request, RecordingBackend{&records});

    ASSERT_EQ(records.size(), 3);
    EXPECT_EQ(records[0].sinkIndex, 0);
    EXPECT_EQ(records[1].sinkIndex, 2);
    EXPECT_EQ(records[2].sinkIndex, 3);
    auto const expected0 = add(add(ferrersAcceleration<2>(particles.position(0), particles.position(1), softening),
                                   ferrersAcceleration<2>(particles.position(0), particles.position(2), softening)),
        ferrersAcceleration<2>(particles.position(0), particles.position(3), softening));
    auto const expected2 = add(ferrersAcceleration<2>(particles.position(2), particles.position(1), softening),
        ferrersAcceleration<2>(particles.position(2), particles.position(3), softening));
    auto const expected3 = add(ferrersAcceleration<2>(particles.position(3), particles.position(1), softening),
        ferrersAcceleration<2>(particles.position(3), particles.position(2), softening));
    expectAccelerationNear(records[0].acceleration, expected0);
    expectAccelerationNear(records[1].acceleration, expected2);
    expectAccelerationNear(records[2].acceleration, expected3);
    for (auto const& record : records) {
        EXPECT_FLOAT_EQ(record.potential, 0.0f);
    }
}

TEST(DirectInteractionsCpu, SubrangeSinkSkipsSelfButKeepsCoincidentDistinctSource) {
    PROFILE();
    TestParticles<3> particles;
    particles.setPosition(0, {0x40000000U, 0x50000000U, 0x60000000U});
    particles.setPosition(1, {0x40000000U, 0x50000000U, 0x60000000U});
    particles.setPosition(2, {0x80000000U, 0x50000000U, 0x60000000U});
    particles.flags()[1] = Rung(0, false);
    auto const softening = makeFerrersSoftening(0.25f);
    DirectInteractionRequest const request{{0, 2}, {1, 2}, particles.view(), softening, 1.0f, 0.5f, 0};
    std::vector<KickRecord> records;

    directInteractionsCpu<RecordingBackend, 3, true>(request, RecordingBackend{&records});

    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].sinkIndex, 1);
    expectAccelerationNear(records[0].acceleration, {0.0f, 0.0f, 0.0f});
    auto const distinctParticle = ferrersAccelerationAndPotential<3>(particles.position(1), particles.position(0), softening);
    EXPECT_NEAR(records[0].potential, distinctParticle.potential, 2.0e-5f);
}

TEST(DirectInteractionsCpu, FullLaneChunksAndTailIncludeEveryDistinctSource) {
    constexpr size_t kCount = 2 * cosmictiger::gravity::detail::kDirectFloatLanes + 3;
    constexpr float kParticleMass = 1.5f;
    PROFILE();

    TestParticles<kCount> particles;
    auto const softening = makeFerrersSoftening(0.25f);
    DirectInteractionRequest const request{{0, kCount}, {0, 1}, particles.view(), softening, kParticleMass, 1.0f, 0};
    std::vector<KickRecord> records;

    directInteractionsCpu<RecordingBackend, 0, true>(request, RecordingBackend{&records});

    ASSERT_EQ(records.size(), 1);
    expectAccelerationNear(records[0].acceleration, {0.0f, 0.0f, 0.0f});
    auto const oneSource = ferrersAccelerationAndPotential<0>(particles.position(0), particles.position(1), softening);
    EXPECT_FLOAT_EQ(records[0].potential, kParticleMass * float(kCount - 1) * oneSource.potential);
}

TEST(DirectInteractionsCpu, ScalesAccelerationAndPotentialAndForwardsKickInputs) {
    constexpr float kParticleMass = 2.5f;
    constexpr float kRungZeroTimestep = 0.03125f;
    PROFILE();

    TestParticles<2> particles;
    particles.setPosition(0, {0x10000000U, 0x20000000U, 0x30000000U});
    particles.setPosition(1, {0x18000000U, 0x28000000U, 0x30000000U});
    particles.flags()[1] = Rung(7, false);
    auto const softening = makeFerrersSoftening(0.25f);
    auto const pointers = particles.view();
    DirectInteractionRequest const request{{0, 1}, {1, 2}, pointers, softening, kParticleMass, kRungZeroTimestep, 7};
    std::vector<KickRecord> records;

    directInteractionsCpu<RecordingBackend, 1, true>(request, RecordingBackend{&records});

    ASSERT_EQ(records.size(), 1);
    auto const& record = records[0];
    EXPECT_EQ(record.sinkIndex, 1);
    auto const unitField = ferrersAccelerationAndPotential<1>(particles.position(1), particles.position(0), softening);
    expectAccelerationNear(
        record.acceleration, {kParticleMass * unitField.acceleration.x, kParticleMass * unitField.acceleration.y, kParticleMass * unitField.acceleration.z});
    EXPECT_NEAR(record.potential, kParticleMass * unitField.potential, 2.0e-5f);
    EXPECT_EQ(record.flags.rung(), 7);
    EXPECT_FALSE(record.flags.isGhost());
    EXPECT_FLOAT_EQ(record.rungZeroTimestep, kRungZeroTimestep);
    EXPECT_EQ(record.particles.flags, pointers.flags);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(record.particles.positions[axis], pointers.positions[axis]);
        EXPECT_EQ(record.particles.velocities[axis], pointers.velocities[axis]);
    }
}

TEST(DirectInteractionsCpu, ConcurrentFieldBackendMatchesScalarBitsForPartialSinkChunks) {
    PROFILE();
    compareSinkFields<false>();
    compareSinkFields<true>();
}

TEST(DirectInteractionsCpu, LegacyAndExplicitlyDisabledBackendsRetainIncreasingSinkOrder) {
    constexpr size_t kCount = 263;
    PROFILE();

    TestParticles<kCount> particles;
    std::vector<ParticleCount> expected;
    for (size_t index = 0; index < kCount; index++) {
        particles.flags()[index] = Rung(index % 5, index % 11 == 0);
        if (index >= 7 && index < kCount - 3 && !particles.flags()[index].isGhost() && particles.flags()[index].rung() >= 2) {
            expected.push_back(index);
        }
    }
    DirectInteractionRequest const request{{0, 0}, {7, kCount - 3}, particles.view(), makeFerrersSoftening(0.25f), 1.0f, 1.0f, 2};
    std::vector<KickRecord> legacy;
    std::vector<KickRecord> disabled;
    directInteractionsCpu<RecordingBackend, 0, false>(request, RecordingBackend{&legacy});
    directInteractionsCpu<DisabledConcurrentRecordingBackend, 0, false>(request, DisabledConcurrentRecordingBackend{&disabled});
    ASSERT_EQ(legacy.size(), expected.size());
    ASSERT_EQ(disabled.size(), expected.size());
    for (size_t index = 0; index < expected.size(); index++) {
        EXPECT_EQ(legacy[index].sinkIndex, expected[index]);
        EXPECT_EQ(disabled[index].sinkIndex, expected[index]);
    }
}

TEST(DirectInteractionsCpu, FieldBackendSupportsAccelerationOnlyAndEmptySinkRanges) {
    PROFILE();
    TestParticles<2> particles;
    particles.setPosition(1, {0x20000000U, 0, 0});
    auto const softening = makeFerrersSoftening(0.25f);
    Acceleration acceleration{};
    DirectInteractionRequest request{{0, 1}, {1, 2}, particles.view(), softening, 1.0f, 0.5f, 0};
    directInteractionsCpu<DirectFieldBackend, 1, false>(request, DirectFieldBackend{{1, 2}, &acceleration});
    expectAccelerationNear(acceleration, ferrersAcceleration<1>(particles.position(1), particles.position(0), softening));
    request.sink = {1, 1};
    EXPECT_NO_THROW((directInteractionsCpu<DirectFieldBackend, 1, true>(request, DirectFieldBackend{{1, 1}, nullptr})));
}

TEST(DirectInteractionsCpu, SourceAwareChunksAllowConcurrentSuspendingBackendCalls) {
    PROFILE();
    if (hpx::get_os_thread_count() < 2) {
        GTEST_SKIP() << "requires at least two HPX workers";
    }
    TestParticles<513> particles;
    // 33 sinks still warrant several tasks with this source count; a fixed
    // 64-sink minimum would incorrectly serialize the callback rendezvous.
    DirectInteractionRequest const request{{0, 513}, {7, 40}, particles.view(), makeFerrersSoftening(0.25f), 1.0f, 1.0f, 0};
    std::atomic<size_t> entered{0};
    std::atomic<size_t> completed{0};
    hpx::promise<void> ready;
    auto readyFuture = ready.get_future();
    hpx::promise<void> release;
    auto releaseFuture = release.get_future().share();
    auto task = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                directInteractionsCpu<GatedSinkBackend, 0, false>(request, GatedSinkBackend{entered, completed, ready, releaseFuture, 2});
            },
            "test.direct.concurrentSinkChunks"));
    auto const status = readyFuture.wait_for(std::chrono::seconds(5));
    // Always release the first callback, including when concurrency regresses,
    // so a failed assertion cannot strand a task that captures these objects.
    release.set_value();
    EXPECT_EQ(status, hpx::future_status::ready);
    EXPECT_NO_THROW(task.get());
    EXPECT_EQ(entered.load(), 33U);
    EXPECT_EQ(completed.load(), 33U);
}

TEST(DirectInteractionsCpu, BackendFailureWaitsForOtherLaunchedSinkChunks) {
    constexpr size_t kCount = 193;
    PROFILE();

    if (hpx::get_os_thread_count() < 2) {
        GTEST_SKIP() << "requires at least two HPX workers";
    }
    TestParticles<kCount> particles;
    DirectInteractionRequest const request{{0, 0}, {0, kCount}, particles.view(), makeFerrersSoftening(0.25f), 1.0f, 1.0f, 0};
    std::atomic<size_t> entered{0};
    std::atomic<size_t> completed{0};
    hpx::promise<void> ready;
    auto readyFuture = ready.get_future();
    hpx::promise<void> release;
    auto releaseFuture = release.get_future().share();
    auto task = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                directInteractionsCpu<GatedSinkBackend, 0, false>(request, GatedSinkBackend{entered, completed, ready, releaseFuture, 1, true});
            },
            "test.direct.failedSinkChunks"));
    auto const status = readyFuture.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, hpx::future_status::ready);
    EXPECT_EQ(task.wait_for(std::chrono::milliseconds(20)), hpx::future_status::timeout);
    release.set_value();
    EXPECT_THROW(task.get(), std::runtime_error);
    // The failing chunk stops at its first sink. Both remaining chunks must
    // have completed all their callbacks before the exception escapes.
    EXPECT_EQ(entered.load(), 128U);
    EXPECT_EQ(completed.load(), 128U);
}

TEST(DirectInteractionsCpu, FieldBackendChecksDeclaredOutputRangeAsSinksAreConsumed) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    TestParticles<2> particles;
    Acceleration acceleration{};
    EXPECT_THROW((DirectFieldBackend{{2, 1}, &acceleration}), std::runtime_error);
    DirectInteractionRequest const request{{0, 0}, {0, 1}, particles.view(), makeFerrersSoftening(0.25f), 1.0f, 1.0f, 0};
    EXPECT_THROW((directInteractionsCpu<DirectFieldBackend, 0, false>(request, DirectFieldBackend{{1, 2}, &acceleration})), std::runtime_error);
    EXPECT_THROW((directInteractionsCpu<DirectFieldBackend, 0, false>(request, DirectFieldBackend{{0, 1}, nullptr})), std::runtime_error);
}

TEST(DirectInteractionsCpu, RejectsMalformedRangesRungsAndPointers) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    TestParticles<1> particles;
    DirectInteractionRequest request{{1, 0}, {0, 0}, particles.view(), makeFerrersSoftening(0.25f), 1.0f, 1.0f, 0};
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
    request.source = {0, 0};
    request.sink = {1, 0};
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
    request.sink = {0, 1};
    request.rung = Rung::maximumRung() + 1;
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
    request.rung = 0;
    request.particles.flags = nullptr;
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
    request.particles = particles.view();
    request.particles.positions[0] = nullptr;
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
    request.particles = particles.view();
    request.particles.velocities[0] = nullptr;
    EXPECT_THROW((directInteractionsCpu<cosmictiger::gravity::DirectKickBackend, 0, false>(request)), std::runtime_error);
}

}    // namespace

TEST(PlummerSoftening, AnalyticFieldRetainsTheTailAndCoincidentLimit) {
    constexpr float kEpsilon = 0.0625f;
    constexpr float kSeparation = 0.25f;
    constexpr float kRelativeTolerance = 2.0e-7f;
    constexpr int kFerrersOrder = 2;
    PROFILE();
    Position const source{Coordinate::fromRaw(0), Coordinate::fromRaw(0), Coordinate::fromRaw(0)};
    Position const sink{Coordinate::fromRaw(0x40000000U), Coordinate::fromRaw(0), Coordinate::fromRaw(0)};
    auto const softening = cosmictiger::gravity::makePlummerSoftening(kEpsilon);
    auto const field = cosmictiger::gravity::plummerAccelerationAndPotential(sink, source, softening);
    float const inverseRadius = 1.0f / std::sqrt(kSeparation * kSeparation + kEpsilon * kEpsilon);
    float const acceleration = -kSeparation * inverseRadius * inverseRadius * inverseRadius;
    EXPECT_NEAR(field.potential, -inverseRadius, kRelativeTolerance * inverseRadius);
    EXPECT_NEAR(field.acceleration.x, acceleration, kRelativeTolerance * std::abs(acceleration));
    EXPECT_FLOAT_EQ(field.acceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(field.acceleration.z, 0.0f);
    EXPECT_NE(field.potential, -1.0f / kSeparation);
    auto const coincident = cosmictiger::gravity::plummerAccelerationAndPotential(source, source, softening);
    EXPECT_FLOAT_EQ(coincident.potential, -1.0f / kEpsilon);
    EXPECT_FLOAT_EQ(coincident.acceleration.x, 0.0f);
    EXPECT_FLOAT_EQ(coincident.acceleration.y, 0.0f);
    EXPECT_FLOAT_EQ(coincident.acceleration.z, 0.0f);
    EXPECT_DOUBLE_EQ(cosmictiger::gravity::softeningEpsilon<kFerrersOrder>(softening), double(kEpsilon));
    auto const ferrers = makeFerrersSoftening(kEpsilon);
    EXPECT_LT(cosmictiger::gravity::softeningEpsilon<kFerrersOrder>(ferrers), double(kEpsilon));
}

TEST(DirectInteractionsCpu, PlummerDispatchExcludesSelfAndKeepsNoncompactTail) {
    constexpr size_t kCount = 2;
    constexpr float kEpsilon = 0.0625f;
    constexpr float kParticleMass = 0.5f;
    constexpr int kFerrersOrder = 2;
    PROFILE();
    TestParticles<kCount> particles;
    particles.setPosition(0, {0, 0, 0});
    particles.setPosition(1, {0x40000000U, 0, 0});
    std::array<Acceleration, kCount> accelerations{};
    std::array<float, kCount> potentials{};
    DirectInteractionRequest const request{
        {0, kCount}, {0, kCount}, particles.view(), cosmictiger::gravity::makePlummerSoftening(kEpsilon), kParticleMass, 1.0, 0};
    DirectFieldBackend backend({0, kCount}, accelerations.data(), potentials.data());
    directInteractionsCpu<DirectFieldBackend, kFerrersOrder, true>(request, backend);
    auto const reference = cosmictiger::gravity::plummerAccelerationAndPotential(particles.position(0), particles.position(1), request.softening);
    EXPECT_FLOAT_EQ(potentials[0], kParticleMass * reference.potential);
    EXPECT_FLOAT_EQ(potentials[1], potentials[0]);
    EXPECT_FLOAT_EQ(accelerations[0].x, kParticleMass * reference.acceleration.x);
    EXPECT_FLOAT_EQ(accelerations[1].x, -accelerations[0].x);
    EXPECT_FLOAT_EQ(accelerations[0].y, 0.0f);
    EXPECT_FLOAT_EQ(accelerations[0].z, 0.0f);
}
