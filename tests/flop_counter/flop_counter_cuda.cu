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

#include <cosmictiger/flop_counter.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <gtest/gtest.h>
#include <hpx/threading_base/annotated_function.hpp>

#include <cstdint>
#include <exception>
#include <stddef.h>
#include <type_traits>
#include <vector>

namespace {

using cosmictiger::FlopCounter;
using Counter = FlopCounter<>;

struct CounterRecord {
    std::uint64_t count;
    std::uint64_t constRead;
    std::uint64_t resetRead;
    std::uint64_t disabledRead;
    std::uint64_t defaultRead;
    bool countingEnabled;
    bool disabledCountingEnabled;
};

static_assert(std::is_trivially_copyable_v<Counter>);
static_assert(std::is_trivially_copyable_v<CounterRecord>);

__global__ void updateCountersKernel(Counter* counters, CounterRecord* records, size_t count) {
    PROFILE();
    size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    auto counter = counters[index];
    (counter = counters[index].read()) += index;
    if (threadIdx.x % 3 == 0) {
        counter += Counter{10};
        counter -= Counter{3};
    } else {
        counter += Counter{13};
        counter -= Counter{6};
    }
    Counter const constant = counter;
    std::uint64_t const converted = constant;
    auto reset = counter;
    reset.reset();

    using DisabledCounter = FlopCounter<std::uint64_t, false>;
    DisabledCounter disabled{counter.read()};
    (disabled = 11) += index;
    disabled += DisabledCounter{13};
    disabled -= DisabledCounter{19};
    DisabledCounter const disabledConstant = disabled;
    std::uint64_t const disabledConverted = disabledConstant;
    Counter const defaultCounter;
    counters[index] = counter;
    records[index] = {
        counter.read(), converted, reset.read(), disabledConverted, defaultCounter.read(), Counter::countingEnabled(), DisabledCounter::countingEnabled()};
}

template <typename Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count)
      : pointer_(static_cast<Value*>(cosmictiger::cuda::malloc(count * sizeof(Value)).get())) {
        PROFILE();
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    ~DeviceBuffer() {
        PROFILE();
        try {
            cosmictiger::cuda::free(pointer_).get();
        } catch (std::exception const& error) {
            ADD_FAILURE() << "Counter buffer deallocation failed: " << error.what();
        }
    }

    Value* data() const noexcept {
        PROFILE();
        return pointer_;
    }

private:
    Value* pointer_;
};

class FlopCounterCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "No CUDA device available";
            }
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA device unavailable: " << error.what();
            }
            throw;
        }
        stream_ = cosmictiger::cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cosmictiger::cuda::streamDestroy(stream_).get());
        }
    }

    void expectCounterRoundTrip(unsigned blockSize) {
        std::uint64_t constexpr kBase = (std::uint64_t(1) << 53) + 1;
        PROFILE();
        size_t const count = size_t(blockSize) * 2 + 3;
        std::vector<Counter> counters(count);
        std::vector<CounterRecord> records(count);
        for (size_t index = 0; index < count; index++) {
            counters[index] = kBase + index;
        }

        DeviceBuffer<Counter> deviceCounters(count);
        DeviceBuffer<CounterRecord> deviceRecords(count);
        Counter* const counterPointer = deviceCounters.data();
        CounterRecord* const recordPointer = deviceRecords.data();
        unsigned const blocks = static_cast<unsigned>((count + blockSize - 1) / blockSize);
        ASSERT_NO_THROW(cosmictiger::cuda::memcpy(counterPointer, counters.data(), count * sizeof(Counter), cudaMemcpyHostToDevice, stream_).get());

        auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
                                                      [counterPointer, recordPointer, count, blocks, blockSize](cudaStream_t stream) {
                                                          PROFILE();
                                                          updateCountersKernel<<<blocks, blockSize, 0, stream>>>(counterPointer, recordPointer, count);
                                                      },
                                                      "FlopCounterCudaTest::updateCounters"),
            stream_);
        ASSERT_TRUE(complete.valid());
        ASSERT_NO_THROW(complete.get());
        // Copies and host comparisons follow completion; buffers and the external
        // stream stay alive until every submitted operation has completed.
        ASSERT_NO_THROW(cosmictiger::cuda::memcpy(counters.data(), counterPointer, count * sizeof(Counter), cudaMemcpyDeviceToHost, stream_).get());
        ASSERT_NO_THROW(cosmictiger::cuda::memcpy(records.data(), recordPointer, count * sizeof(CounterRecord), cudaMemcpyDeviceToHost, stream_).get());
        for (size_t index = 0; index < count; index++) {
            SCOPED_TRACE(index);
            std::uint64_t const expected = kBase + 2 * index + 7;
            EXPECT_EQ(counters[index].read(), expected);
            EXPECT_EQ(records[index].count, expected);
            EXPECT_EQ(records[index].constRead, expected);
            EXPECT_EQ(records[index].resetRead, 0);
            EXPECT_EQ(records[index].disabledRead, 0);
            EXPECT_EQ(records[index].defaultRead, 0);
            EXPECT_TRUE(records[index].countingEnabled);
            EXPECT_FALSE(records[index].disabledCountingEnabled);
        }
    }

private:
    cudaStream_t stream_ = nullptr;
};

TEST_F(FlopCounterCudaTest, PreservesCountsInPartialWarpBlocksAndDivergentThreads) {
    PROFILE();
    expectCounterRoundTrip(65);
}

TEST_F(FlopCounterCudaTest, PreservesCountsInEveryWarpAndAcrossBlocks) {
    PROFILE();
    expectCounterRoundTrip(96);
}

}    // namespace
