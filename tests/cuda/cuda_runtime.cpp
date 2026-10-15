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

#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace {

class CudaRuntimeTest : public ::testing::Test {
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
    }

    void TearDown() override {
        PROFILE();
        for (void* pointer : allocations_) {
            EXPECT_NO_THROW(cosmictiger::cuda::free(pointer).get());
        }
    }

    void* track(void* pointer) {
        PROFILE();
        allocations_.push_back(pointer);
        return pointer;
    }

private:
    std::vector<void*> allocations_;
};

TEST_F(CudaRuntimeTest, DeviceQueriesUseFutures) {
    PROFILE();
    int const deviceId = cosmictiger::cuda::device().get();
    EXPECT_EQ(deviceId, 0);
    EXPECT_GT(cosmictiger::cuda::deviceCount().get(), deviceId);
    cudaDeviceProp const properties = cosmictiger::cuda::deviceProperties(deviceId).get();
    EXPECT_EQ(cosmictiger::cuda::deviceAttribute(cudaDevAttrMultiProcessorCount, deviceId).get(), properties.multiProcessorCount);
}

TEST_F(CudaRuntimeTest, CopiesComposeAsFutureDependencies) {
    constexpr size_t kSampleCount = 32;
    PROFILE();
    std::array<int, kSampleCount> source{};
    std::array<int, kSampleCount> destination{};
    for (size_t index = 0; index < kSampleCount; index++) {
        source[index] = static_cast<int>(index * index + 7);
    }
    void* pointer = nullptr;
    hpx::future<void> upload(cosmictiger::cuda::malloc(sizeof(source))
            .then(hpx::launch::async,
                hpx::annotated_function(
                    [&](hpx::future<void*> allocated) {
                        PROFILE();
                        pointer = track(allocated.get());
                        return cosmictiger::cuda::memcpy(pointer, source.data(), sizeof(source), cudaMemcpyHostToDevice);
                    },
                    "CudaRuntimeTest::upload")));
    hpx::future<void> download(upload.then(hpx::launch::async,
        hpx::annotated_function(
            [&](hpx::future<void> uploaded) {
                PROFILE();
                uploaded.get();
                return cosmictiger::cuda::memcpy(destination.data(), pointer, sizeof(destination), cudaMemcpyDeviceToHost);
            },
            "CudaRuntimeTest::download")));
    ASSERT_NO_THROW(download.get());
    EXPECT_EQ(destination, source);
    EXPECT_EQ(cosmictiger::cuda::pointerAttributes(pointer).get().type, cudaMemoryTypeDevice);
}

TEST_F(CudaRuntimeTest, RuntimeErrorsAreFutureExceptions) {
    PROFILE();
    hpx::future<cudaDeviceProp> failed;
    EXPECT_NO_THROW(failed = cosmictiger::cuda::deviceProperties(-1));
    ASSERT_TRUE(failed.valid());
    try {
        (void) failed.get();
        FAIL() << "Invalid device query unexpectedly succeeded";
    } catch (cosmictiger::cuda::CudaError const& error) {
        EXPECT_EQ(error.code(), cudaErrorInvalidDevice);
    }
    hpx::future<void> emptyLaunch;
    EXPECT_NO_THROW(emptyLaunch = cosmictiger::cuda::launch({}));
    EXPECT_THROW(emptyLaunch.get(), std::invalid_argument);
    EXPECT_THROW(cosmictiger::cuda::launch(hpx::annotated_function(
                                               [](cudaStream_t) {
                                                   PROFILE();
                                                   throw std::runtime_error("submission failure");
                                               },
                                               "CudaRuntimeTest::throwingLaunch"))
                     .get(),
        std::runtime_error);
}

TEST_F(CudaRuntimeTest, ExternalStreamsCanBeCreatedAndRetired) {
    int constexpr kSource = 123;
    PROFILE();
    cudaStream_t const stream = cosmictiger::cuda::streamCreate().get();
    ASSERT_NE(stream, nullptr);
    void* const pointer = track(cosmictiger::cuda::malloc(sizeof(int)).get());
    int destination = 0;
    EXPECT_NO_THROW(cosmictiger::cuda::memcpy(pointer, &kSource, sizeof(kSource), cudaMemcpyHostToDevice, stream).get());
    EXPECT_NO_THROW(cosmictiger::cuda::memcpy(&destination, pointer, sizeof(destination), cudaMemcpyDeviceToHost, stream).get());
    EXPECT_EQ(destination, kSource);
    EXPECT_NO_THROW(cosmictiger::cuda::synchronize(stream).get());
    EXPECT_NO_THROW(cosmictiger::cuda::streamDestroy(stream).get());
    EXPECT_THROW(cosmictiger::cuda::streamDestroy(nullptr).get(), std::invalid_argument);
}

TEST_F(CudaRuntimeTest, BatchedManagedPrefetchCompletesBeforeHostAccess) {
    constexpr size_t kCount = 4097;
    PROFILE();
    if (!cosmictiger::cuda::deviceAttribute(cudaDevAttrConcurrentManagedAccess, 0).get()) {
        GTEST_SKIP() << "device does not support managed prefetch";
    }
    auto* first = static_cast<int*>(track(cosmictiger::cuda::mallocManaged(kCount * sizeof(int)).get()));
    auto* second = static_cast<int*>(track(cosmictiger::cuda::mallocManaged(kCount * sizeof(int)).get()));
    std::vector<int> source(kCount);
    for (size_t index = 0; index < kCount; index++) {
        source[index] = static_cast<int>(index * 3 + 7);
    }
    auto const stream = cosmictiger::cuda::streamCreate().get();
    std::vector<cosmictiger::cuda::ManagedMemoryRange> const ranges{{first, kCount * sizeof(int)}, {nullptr, 0}, {second, kCount * sizeof(int)}};
    cosmictiger::cuda::prefetch(ranges, 0, stream).get();
    cosmictiger::cuda::memcpy(first, source.data(), kCount * sizeof(int), cudaMemcpyHostToDevice, stream).get();
    cosmictiger::cuda::memcpy(second, first, kCount * sizeof(int), cudaMemcpyDeviceToDevice, stream).get();
    auto ready = cosmictiger::cuda::prefetch(ranges, cudaCpuDeviceId, stream);
    ASSERT_TRUE(ready.valid());
    ready.get();
    for (size_t index = 0; index < kCount; index++) {
        EXPECT_EQ(first[index], source[index]);
        EXPECT_EQ(second[index], source[index]);
    }
    EXPECT_NO_THROW(cosmictiger::cuda::prefetch({}, cudaCpuDeviceId).get());
    // A late invalid range must propagate through the future after draining
    // the valid earlier submission, leaving the allocation safe to release.
    auto failed = cosmictiger::cuda::prefetch({{first, kCount * sizeof(int)}, {nullptr, sizeof(int)}}, 0, stream);
    EXPECT_THROW(failed.get(), cosmictiger::cuda::CudaError);
    cosmictiger::cuda::streamDestroy(stream).get();
}

}    // namespace
