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

#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <new>
#include <stddef.h>
#include <stdexcept>

namespace {

using cosmictiger::Coordinate;
using cosmictiger::Position;
using cosmictiger::gravity::DirectEwaldParameters;
using cosmictiger::gravity::directSolveCpu;
using cosmictiger::gravity::DirectSolveField;
using cosmictiger::gravity::DirectSolveRequest;
using cosmictiger::gravity::launchDirectSolveCuda;

template <typename T>
class ManagedDirectArray {
public:
    explicit ManagedDirectArray(size_t count) {
        PROFILE();
        values_ = static_cast<T*>(cosmictiger::cuda::mallocManaged(count * sizeof(T)).get());
        for (size_t index = 0; index < count; index++) {
            ::new (static_cast<void*>(values_ + index)) T{};
        }
    }

    ManagedDirectArray(ManagedDirectArray const&) = delete;
    ManagedDirectArray& operator=(ManagedDirectArray const&) = delete;

    ~ManagedDirectArray() {
        PROFILE();
        if (values_ != nullptr) {
            try {
                cosmictiger::cuda::free(values_).get();
            } catch (std::exception const& error) {
                ADD_FAILURE() << "managed direct-solve buffer release failed: " << error.what();
            }
        }
    }

    T* data() noexcept {
        PROFILE();
        return values_;
    }

    T& operator[](size_t index) noexcept {
        PROFILE();
        return values_[index];
    }

private:
    T* values_ = nullptr;
};

class DirectSolveCudaTest : public ::testing::Test {
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
            GTEST_SKIP() << "CUDA device does not support managed memory";
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

void expectMatchingField(DirectSolveField const& actual, DirectSolveField const& expected) {
    constexpr double kRelativeTolerance = 2.0e-11;
    PROFILE();
    EXPECT_NEAR(actual.potential, expected.potential, kRelativeTolerance * (1.0 + std::abs(expected.potential)));
    for (size_t axis = 0; axis < NDIM; axis++) {
        SCOPED_TRACE(axis);
        EXPECT_NEAR(actual.acceleration[axis], expected.acceleration[axis], kRelativeTolerance * (1.0 + std::abs(expected.acceleration[axis])));
    }
}

TEST_F(DirectSolveCudaTest, RetainsSelfImagesAndBackgroundWithoutSelfForce) {
    constexpr size_t kParticleCount = 1;
    constexpr double kRegularSelfPotential = 2.837297479480619;
    constexpr double kSelfTolerance = 2.0e-12;
    PROFILE();
    ManagedDirectArray<Position> sources(kParticleCount);
    ManagedDirectArray<size_t> sinks(kParticleCount);
    ManagedDirectArray<DirectSolveField> fields(kParticleCount);
    sources[0] = {Coordinate::fromRaw(0x80000000U), Coordinate::fromRaw(0x20000000U), Coordinate::fromRaw(0xe0000000U)};
    sinks[0] = 0;
    DirectSolveRequest request{};
    request.sources = sources.data();
    request.sourceCount = kParticleCount;
    request.sinks = sinks.data();
    request.sinkCount = kParticleCount;
    request.fields = fields.data();
    request.particleMass = 1.0;
    request.softeningLength = 0.03125;
    request.ferrersOrder = 1;
    DirectSolveField expected{};
    auto cpuRequest = request;
    cpuRequest.fields = &expected;
    directSolveCpu(cpuRequest);

    auto complete = launchDirectSolveCuda(request, {}, stream());
    ASSERT_TRUE(complete.valid());
    ASSERT_NO_THROW(complete.get());
    expectMatchingField(fields[0], expected);
    EXPECT_NEAR(fields[0].potential, kRegularSelfPotential, kSelfTolerance);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(fields[0].acceleration[axis], 0.0, kSelfTolerance);
    }
}

TEST_F(DirectSolveCudaTest, SoftenedPeriodicPairMatchesCpuAndIsAntisymmetric) {
    constexpr size_t kParticleCount = 2;
    constexpr double kSymmetryTolerance = 2.0e-12;
    PROFILE();
    ManagedDirectArray<Position> sources(kParticleCount);
    ManagedDirectArray<size_t> sinks(kParticleCount);
    ManagedDirectArray<DirectSolveField> fields(kParticleCount);
    sources[0] = {Coordinate::fromRaw(0x02000000U), Coordinate::fromRaw(0x50000000U), Coordinate::fromRaw(0x60000000U)};
    sources[1] = {Coordinate::fromRaw(0xfe000000U), Coordinate::fromRaw(0x50000000U), Coordinate::fromRaw(0x60000000U)};
    for (size_t index = 0; index < kParticleCount; index++) {
        sinks[index] = index;
    }
    DirectSolveRequest request{};
    request.sources = sources.data();
    request.sourceCount = kParticleCount;
    request.sinks = sinks.data();
    request.sinkCount = kParticleCount;
    request.fields = fields.data();
    request.particleMass = 0.5;
    request.softeningLength = 0.03125;
    request.ferrersOrder = 3;
    std::array<DirectSolveField, kParticleCount> expected{};
    auto cpuRequest = request;
    cpuRequest.fields = expected.data();
    directSolveCpu(cpuRequest);

    ASSERT_NO_THROW(launchDirectSolveCuda(request, {}, stream()).get());
    for (size_t index = 0; index < kParticleCount; index++) {
        SCOPED_TRACE(index);
        expectMatchingField(fields[index], expected[index]);
    }
    EXPECT_LT(fields[0].acceleration[0], 0.0);
    EXPECT_NEAR(fields[0].potential, fields[1].potential, kSymmetryTolerance);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(fields[0].acceleration[axis], -fields[1].acceleration[axis], kSymmetryTolerance);
    }
}

TEST_F(DirectSolveCudaTest, CooperatingSourcesMatchCpuForAllFerrersOrdersAndSelectedSinks) {
    constexpr size_t kSourceCount = 137;
    constexpr size_t kSinkCount = 6;
    constexpr std::array<size_t, kSinkCount> kSinks{0, 1, 2, 64, 128, 136};
    constexpr std::array<std::uint32_t, NDIM> kCoordinateSteps{0x9e3779b9U, 0x7f4a7c15U, 0x85ebca6bU};
    constexpr int kMaximumFerrersOrder = 3;
    PROFILE();
    ManagedDirectArray<Position> sources(kSourceCount);
    ManagedDirectArray<size_t> sinks(kSinkCount);
    ManagedDirectArray<DirectSolveField> fields(kSinkCount);
    for (size_t index = 0; index < kSourceCount; index++) {
        std::uint32_t const multiplier = static_cast<std::uint32_t>(index + 1);
        sources[index] = {Coordinate::fromRaw(kCoordinateSteps[0] * multiplier), Coordinate::fromRaw(kCoordinateSteps[1] * multiplier),
            Coordinate::fromRaw(kCoordinateSteps[2] * multiplier)};
    }
    // Distinct coincident particles retain softened interactions; a nearly
    // coincident periodic pair exercises cancellation in the Ewald correction.
    sources[0] = {Coordinate::fromRaw(0xfffffff0U), Coordinate::fromRaw(0x80000000U), Coordinate::fromRaw(0x20000000U)};
    sources[1] = sources[0];
    sources[2] = {Coordinate::fromRaw(0x00000010U), sources[0].y, sources[0].z};
    for (size_t index = 0; index < kSinkCount; index++) {
        sinks[index] = kSinks[index];
    }
    DirectSolveRequest request{};
    request.sources = sources.data();
    request.sourceCount = kSourceCount;
    request.sinks = sinks.data();
    request.sinkCount = kSinkCount;
    request.fields = fields.data();
    request.particleMass = 1.0 / static_cast<double>(kSourceCount);
    request.softeningLength = 0.0625;
    DirectEwaldParameters parameters{};
    parameters.alpha = 1.75;
    std::array<DirectSolveField, kSinkCount> expected{};
    for (int ferrersOrder = 0; ferrersOrder <= kMaximumFerrersOrder; ferrersOrder++) {
        request.ferrersOrder = ferrersOrder;
        auto cpuRequest = request;
        cpuRequest.fields = expected.data();
        directSolveCpu(cpuRequest, parameters);
        ASSERT_NO_THROW(launchDirectSolveCuda(request, parameters, stream()).get());
        for (size_t index = 0; index < kSinkCount; index++) {
            // GoogleTest traces belong to an OS thread; keep their scopes
            // after HPX work and future waits that can migrate this task.
            SCOPED_TRACE(::testing::Message() << "Ferrers order " << ferrersOrder << ", sink " << index);
            expectMatchingField(fields[index], expected[index]);
        }
    }
}

TEST_F(DirectSolveCudaTest, UniformPeriodicLatticeHasNoNetAcceleration) {
    constexpr size_t kAxisCount = 2;
    constexpr size_t kParticleCount = size_t{1} << NDIM;
    constexpr std::uint32_t kCenterOffset = 0x40000000U;
    constexpr std::uint32_t kCellWidth = 0x80000000U;
    constexpr double kLatticeTolerance = 2.0e-12;
    PROFILE();
    ManagedDirectArray<Position> sources(kParticleCount);
    ManagedDirectArray<size_t> sinks(kParticleCount);
    ManagedDirectArray<DirectSolveField> fields(kParticleCount);
    for (size_t index = 0; index < kParticleCount; index++) {
        std::array<Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const cell = static_cast<std::uint32_t>((index >> axis) % kAxisCount);
            position[axis] = Coordinate::fromRaw(kCenterOffset + kCellWidth * cell);
        }
        sources[index] = {position[0], position[1], position[2]};
        sinks[index] = index;
    }
    DirectSolveRequest request{};
    request.sources = sources.data();
    request.sourceCount = kParticleCount;
    request.sinks = sinks.data();
    request.sinkCount = kParticleCount;
    request.fields = fields.data();
    request.particleMass = 1.0 / static_cast<double>(kParticleCount);
    request.softeningLength = 0.03125;
    request.ferrersOrder = 2;
    std::array<DirectSolveField, kParticleCount> expected{};
    auto cpuRequest = request;
    cpuRequest.fields = expected.data();
    directSolveCpu(cpuRequest);

    ASSERT_NO_THROW(launchDirectSolveCuda(request, {}, stream()).get());
    for (size_t index = 0; index < kParticleCount; index++) {
        SCOPED_TRACE(index);
        expectMatchingField(fields[index], expected[index]);
        EXPECT_NEAR(fields[index].potential, fields[0].potential, kLatticeTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(fields[index].acceleration[axis], 0.0, kLatticeTolerance);
        }
    }
}

TEST(DirectSolveCudaRequests, EmptyBatchCompletesWithoutADevice) {
    PROFILE();
    auto complete = launchDirectSolveCuda({});
    ASSERT_TRUE(complete.is_ready());
    EXPECT_NO_THROW(complete.get());
}

TEST(DirectSolveCudaRequests, MissingBuffersFailThroughTheFuture) {
    PROFILE();
    DirectSolveRequest request{};
    request.sourceCount = 1;
    request.sinkCount = 1;
    auto complete = launchDirectSolveCuda(request);
    ASSERT_TRUE(complete.valid());
    EXPECT_THROW(complete.get(), std::invalid_argument);
}

TEST_F(DirectSolveCudaTest, PlummerSofteningIncludesPeriodicImagesAndMatchesCpu) {
    constexpr size_t kParticleCount = 2;
    constexpr double kEpsilon = 0.125;
    PROFILE();
    ManagedDirectArray<Position> sources(kParticleCount);
    ManagedDirectArray<size_t> sinks(kParticleCount);
    ManagedDirectArray<DirectSolveField> fields(kParticleCount);
    sources[0] = {Coordinate::fromRaw(0), Coordinate::fromRaw(0), Coordinate::fromRaw(0)};
    sources[1] = {Coordinate::fromRaw(0x40000000U), Coordinate::fromRaw(0x20000000U), Coordinate::fromRaw(0xe0000000U)};
    for (size_t index = 0; index < kParticleCount; index++) {
        sinks[index] = index;
    }
    DirectSolveRequest request{
        sources.data(), kParticleCount, sinks.data(), kParticleCount, fields.data(), 0.5, kEpsilon, 2, cosmictiger::gravity::SofteningModel::plummer};
    ManagedDirectArray<DirectSolveField> expected(kParticleCount);
    auto cpuRequest = request;
    cpuRequest.fields = expected.data();
    directSolveCpu(cpuRequest);
    ASSERT_NO_THROW(launchDirectSolveCuda(request, {}, stream()).get());
    for (size_t index = 0; index < kParticleCount; index++) {
        expectMatchingField(fields[index], expected[index]);
    }
}

}    // namespace
