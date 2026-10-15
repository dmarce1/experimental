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
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/gravity/ferrers.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stddef.h>

namespace {

using cosmictiger::Acceleration;
using cosmictiger::Coordinate;
using cosmictiger::Position;
using cosmictiger::gravity::ferrersAcceleration;
using cosmictiger::gravity::ferrersAccelerationAndPotential;
using cosmictiger::gravity::FerrersAccelerationPotential;
using cosmictiger::gravity::makeFerrersSoftening;
using cosmictiger::gravity::SofteningLength;

struct DeviceResult {
    Acceleration acceleration[4];
    FerrersAccelerationPotential both[4];
};

struct TestCase {
    Position target;
    Position source;
};

class DeviceResultStorage {
public:
    DeviceResultStorage()
      : result_(static_cast<DeviceResult*>(cosmictiger::cuda::malloc(sizeof(DeviceResult)).get())) {
        PROFILE();
    }

    DeviceResultStorage(DeviceResultStorage const&) = delete;
    DeviceResultStorage& operator=(DeviceResultStorage const&) = delete;

    ~DeviceResultStorage() {
        PROFILE();
        try {
            cosmictiger::cuda::free(result_).get();
        } catch (std::exception const& error) {
            ADD_FAILURE() << "Device result deallocation failed: " << error.what();
        }
    }

    DeviceResult* data() const noexcept {
        PROFILE();
        return result_;
    }

private:
    DeviceResult* result_;
};

__global__ void ferrersDeviceKernel(Position target, Position source, SofteningLength softening, DeviceResult* result) {
    PROFILE();
    result->acceleration[0] = ferrersAcceleration<0>(target, source, softening);
    result->acceleration[1] = ferrersAcceleration<1>(target, source, softening);
    result->acceleration[2] = ferrersAcceleration<2>(target, source, softening);
    result->acceleration[3] = ferrersAcceleration<3>(target, source, softening);

    result->both[0] = ferrersAccelerationAndPotential<0>(target, source, softening);
    result->both[1] = ferrersAccelerationAndPotential<1>(target, source, softening);
    result->both[2] = ferrersAccelerationAndPotential<2>(target, source, softening);
    result->both[3] = ferrersAccelerationAndPotential<3>(target, source, softening);
}

Position makePosition(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    PROFILE();
    return {Coordinate::fromRaw(x), Coordinate::fromRaw(y), Coordinate::fromRaw(z)};
}

DeviceResult evaluateCpu(TestCase const& testCase, SofteningLength softening) {
    PROFILE();
    DeviceResult result{};
    result.acceleration[0] = ferrersAcceleration<0>(testCase.target, testCase.source, softening);
    result.acceleration[1] = ferrersAcceleration<1>(testCase.target, testCase.source, softening);
    result.acceleration[2] = ferrersAcceleration<2>(testCase.target, testCase.source, softening);
    result.acceleration[3] = ferrersAcceleration<3>(testCase.target, testCase.source, softening);

    result.both[0] = ferrersAccelerationAndPotential<0>(testCase.target, testCase.source, softening);
    result.both[1] = ferrersAccelerationAndPotential<1>(testCase.target, testCase.source, softening);
    result.both[2] = ferrersAccelerationAndPotential<2>(testCase.target, testCase.source, softening);
    result.both[3] = ferrersAccelerationAndPotential<3>(testCase.target, testCase.source, softening);
    return result;
}

void expectNear(float actual, float expected) {
    PROFILE();
    EXPECT_NEAR(actual, expected, 2.0e-5f * std::max(1.0f, std::fabs(expected)));
}

TEST(FerrersCuda, DeviceMatchesCpuForEveryOrderAndOutputMode) {
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

    SofteningLength const softening = makeFerrersSoftening(0.25f);
    std::array<TestCase, 5> const cases{{
        {makePosition(0x20000000U, 0x20000000U, 0x30000000U), makePosition(0x10000000U, 0x20000000U, 0x30000000U)},
        {makePosition(0x08000000U, 0xff000000U, 0x80000000U), makePosition(0xf8000000U, 0x01000000U, 0x80000000U)},
        {makePosition(0x80000000U, 0x40000000U, 0U), makePosition(0U, 0U, 0U)},
        {makePosition(0x40000000U, 0U, 0U), makePosition(0U, 0U, 0U)},
        {makePosition(0x12345678U, 0xabcdef01U, 0xfedcba98U), makePosition(0x12345678U, 0xabcdef01U, 0xfedcba98U)},
    }};

    DeviceResultStorage storage;
    auto* const deviceResult = storage.data();
    for (size_t caseIndex = 0; caseIndex < cases.size(); caseIndex++) {
        TestCase const testCase = cases[caseIndex];
        auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
            [testCase, softening, deviceResult](cudaStream_t stream) {
                PROFILE();
                ferrersDeviceKernel<<<1, 1, 0, stream>>>(testCase.target, testCase.source, softening, deviceResult);
            },
            "FerrersCudaTest::launch"));
        ASSERT_NO_THROW(complete.get());
        DeviceResult actual{};
        ASSERT_NO_THROW(cosmictiger::cuda::memcpy(&actual, deviceResult, sizeof(actual), cudaMemcpyDeviceToHost).get());
        SCOPED_TRACE(caseIndex);
        DeviceResult const expected = evaluateCpu(testCase, softening);
        for (size_t order = 0; order < 4; order++) {
            SCOPED_TRACE(order);
            expectNear(actual.acceleration[order].x, expected.acceleration[order].x);
            expectNear(actual.acceleration[order].y, expected.acceleration[order].y);
            expectNear(actual.acceleration[order].z, expected.acceleration[order].z);
            expectNear(actual.both[order].acceleration.x, expected.both[order].acceleration.x);
            expectNear(actual.both[order].acceleration.y, expected.both[order].acceleration.y);
            expectNear(actual.both[order].acceleration.z, expected.both[order].acceleration.z);
            expectNear(actual.both[order].potential, expected.both[order].potential);
        }
    }
}

}    // namespace
