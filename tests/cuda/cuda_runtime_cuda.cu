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

#include <cosmictiger/debug.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

__global__ void delayedWriteKernel(int* result, unsigned long long delayClocks, int value) {
    PROFILE();
    unsigned long long const begin = clock64();
    while (clock64() - begin < delayClocks) {
    }
    *result = value;
}

__global__ void validationPrimitivesKernel(int* result, double value) {
    PROFILE();
    cosmictiger::FpeGuard guard(cosmictiger::toMask({cosmictiger::Fpe::invalid, cosmictiger::Fpe::divByZero, cosmictiger::Fpe::overflow}));
    cosmictiger::expectPositive(value);
    cosmictiger::expectNonNegative(value - 2.0);
    cosmictiger::expectNonPositive(2.0 - value);
    cosmictiger::expectNegative(-value);
    cosmictiger::expectNonZero(value);
    cosmictiger::expectRange(2.0, value, 3.0);
    cosmictiger::expectRange(1.0, value, 2.0);
    cosmictiger::expectIndex(threadIdx.x, blockDim.x);
    cosmictiger::expect(value == 2.0);
    cosmictiger::expectEqual(value, 2.0);
    cosmictiger::expectNonZero(result);
    // CUDA exposes no floating-point exception flags or hardware FPE traps.
    // This only verifies device compatibility and the explicit capability query.
    *result = (cosmictiger::validationChecksEnabled() ? 1 : 0) | (cosmictiger::FpeGuard::supportsHardwareTraps() ? 2 : 0);
}

class CudaRuntimeGpu : public ::testing::Test {
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
        properties_ = cosmictiger::cuda::deviceProperties(0).get();
        if (!properties_.managedMemory || !properties_.concurrentManagedAccess) {
            GTEST_SKIP() << "Concurrent managed memory unsupported";
        }
        result_ = static_cast<int*>(cosmictiger::cuda::mallocManaged(sizeof(int)).get());
        *result_ = -1;
    }

    void TearDown() override {
        PROFILE();
        EXPECT_NO_THROW(cosmictiger::cuda::free(result_).get());
    }

    int* result() const noexcept {
        PROFILE();
        return result_;
    }

    unsigned long long delayClocks() const noexcept {
        PROFILE();
        // The advertised clock rate is in kHz, or cycles per millisecond.
        return static_cast<unsigned long long>(properties_.clockRate) * 30;
    }

private:
    cudaDeviceProp properties_{};
    int* result_ = nullptr;
};

TEST_F(CudaRuntimeGpu, LaunchFutureWaitsForDelayedKernelCompletion) {
    constexpr int kExpected = 12345;
    PROFILE();
    auto* const output = result();
    unsigned long long const delay = delayClocks();
    // One CUDA thread waits approximately 30 ms before writing the managed result.
    auto completed = cosmictiger::cuda::launch(hpx::annotated_function(
        [output, delay](cudaStream_t stream) {
            PROFILE();
            delayedWriteKernel<<<1, 1, 0, stream>>>(output, delay, kExpected);
        },
        "CudaRuntimeTest::delayedWrite"));
    EXPECT_NO_THROW(completed.get());
    EXPECT_EQ(*output, kExpected);
}

TEST_F(CudaRuntimeGpu, ValidationPrimitivesAndFpeGuardAreDeviceCompatible) {
    PROFILE();
    auto* const output = result();
    auto completed = cosmictiger::cuda::launch(hpx::annotated_function(
        [output](cudaStream_t stream) {
            PROFILE();
            validationPrimitivesKernel<<<1, 1, 0, stream>>>(output, 2.0);
        },
        "CudaRuntimeTest::validationPrimitives"));
    ASSERT_NO_THROW(completed.get());
    EXPECT_EQ(*output, cosmictiger::validationChecksEnabled() ? 1 : 0);
}

TEST_F(CudaRuntimeGpu, InvalidLaunchFailsThroughFutureAndLaterLaunchRecovers) {
    constexpr int kExpected = 6789;
    PROFILE();
    auto* const output = result();
    hpx::future<void> failed;
    EXPECT_NO_THROW(failed = cosmictiger::cuda::launch(hpx::annotated_function(
                        [output](cudaStream_t stream) {
                            PROFILE();
                            delayedWriteKernel<<<0, 1, 0, stream>>>(output, 0, 0);
                        },
                        "CudaRuntimeTest::invalidLaunch")));
    ASSERT_TRUE(failed.valid());
    try {
        failed.get();
        FAIL() << "A kernel with zero grid blocks unexpectedly succeeded";
    } catch (cosmictiger::cuda::CudaError const& error) {
        EXPECT_EQ(error.code(), cudaErrorInvalidConfiguration);
    }
    EXPECT_EQ(*output, -1);

    unsigned long long const delay = delayClocks();
    auto recovered = cosmictiger::cuda::launch(hpx::annotated_function(
        [output, delay](cudaStream_t stream) {
            PROFILE();
            delayedWriteKernel<<<1, 1, 0, stream>>>(output, delay, kExpected);
        },
        "CudaRuntimeTest::recoveredLaunch"));
    EXPECT_NO_THROW(recovered.get());
    EXPECT_EQ(*output, kExpected);
}

}    // namespace
