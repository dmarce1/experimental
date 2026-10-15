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

#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stddef.h>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_device_vector.hpp>
#include <cosmictiger/cuda/cuda_memory_pool.hpp>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/profiler.hpp>

namespace {

namespace cuda = cosmictiger::cuda;
using cosmictiger::CudaMemoryPool;
using cosmictiger::CudaMemoryPoolView;

// The smallest allocation holds this many ints, so lists longer than it grow.
constexpr size_t kInitialIntCapacity = 256;
constexpr size_t kPushCount = 3000;
constexpr size_t kResizeExtra = 500;
constexpr size_t kPopCount = 100;
constexpr size_t kHeapBytes = size_t(64) << 20;

int pushedValue(size_t block, size_t index) {
    PROFILE();
    return static_cast<int>(block * 100003 + index * 7);
}

__host__ __device__ int filledValue(size_t block, size_t index) {
    PROFILE();
    return -static_cast<int>(block * 100003 + index);
}

// The results one block, or the host, records after the operation sequence.
struct SequenceResult {
    size_t pushedCapacity;
    size_t finalSize;
    size_t otherSize;
    bool pushesSucceeded;
    bool resizeSucceeded;
    bool aliasSucceeded;
};

// The same sequence runs on the host and in every block: append with growth,
// grow by resize and fill in parallel, pop, swap, and append an element of the
// vector itself at a capacity boundary so that the append reallocates.
__host__ __device__ void runSequence(cuda::DeviceVector<int>& vector, cuda::DeviceVector<int>& other, CudaMemoryPoolView pool, size_t block, int const* pushed,
    int* contents, SequenceResult* result) {
    PROFILE();
    vector.initialize(pool);
    other.initialize(pool);
    bool pushesSucceeded = true;
    for (size_t index = 0; index < kPushCount; index++) {
        pushesSucceeded = vector.pushBack(pushed[index]) && pushesSucceeded;
    }
    size_t const pushedCapacity = vector.capacity();
    bool const resizeSucceeded = vector.resize(kPushCount + kResizeExtra);
    for (size_t index = kPushCount + cuda::detail::blockThreadRank(); index < vector.size(); index += cuda::detail::blockThreadCount()) {
        vector[index] = filledValue(block, index);
    }
    for (size_t index = 0; index < kPopCount; index++) {
        vector.popBack();
    }
    vector.swap(other);
    bool aliasSucceeded = other.resize(other.capacity());
    aliasSucceeded = other.pushBack(other[0]) && aliasSucceeded;
    for (size_t index = cuda::detail::blockThreadRank(); index < other.size(); index += cuda::detail::blockThreadCount()) {
        contents[index] = other[index];
    }
    if (cuda::detail::blockThreadRank() == 0) {
        *result = {pushedCapacity, other.size(), vector.size(), pushesSucceeded, resizeSucceeded, aliasSucceeded};
    }
    other.release();
    vector.release();
}

__global__ void sequenceKernel(CudaMemoryPoolView pool, int const* pushed, int* contents, size_t contentsStride, SequenceResult* results) {
    PROFILE();
    __shared__ cuda::DeviceVector<int> vector;
    __shared__ cuda::DeviceVector<int> other;
    size_t const block = blockIdx.x;
    runSequence(vector, other, pool, block, pushed + block * kPushCount, contents + block * contentsStride, results + block);
}

// Every thread reports whether its exhausting resize failed and the vector
// was left as it was.
__global__ void exhaustionKernel(CudaMemoryPoolView pool, size_t count, bool* unchanged) {
    PROFILE();
    __shared__ cuda::DeviceVector<double> vector;
    vector.initialize(pool);
    bool const pushed = vector.pushBack(1.5);
    size_t const capacity = vector.capacity();
    bool const resized = vector.resize(count);
    unchanged[cuda::detail::blockThreadRank()] = pushed && !resized && vector.size() == 1 && vector.capacity() == capacity && vector[0] == 1.5;
    vector.release();
}

// Each block appends to its own vector, which lives in managed memory rather
// than shared memory, as the old drift kernel filled per-block lightcone lists.
__global__ void appendKernel(cuda::DeviceVector<int>* vectors, size_t count) {
    PROFILE();
    cuda::DeviceVector<int>& vector = vectors[blockIdx.x];
    for (size_t index = 0; index < count; index++) {
        bool const pushed = vector.pushBack(filledValue(blockIdx.x, index));
        if (!pushed) {
            return;
        }
    }
}

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(size_t count)
      : count_(count) {
        PROFILE();
        values_ = static_cast<T*>(cuda::mallocManaged((count_ == 0 ? 1 : count_) * sizeof(T)).get());
        for (size_t index = 0; index < count_; index++) {
            ::new (static_cast<void*>(values_ + index)) T{};
        }
    }

    ManagedArray(ManagedArray const&) = delete;
    ManagedArray& operator=(ManagedArray const&) = delete;

    ~ManagedArray() {
        PROFILE();
        try {
            cuda::free(values_).get();
        } catch (std::exception const& error) {
            ADD_FAILURE() << "managed device-vector buffer release failed: " << error.what();
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
    size_t count_ = 0;
};

// Host storage for a pool view; the view needs no CUDA runtime on the host.
class HostPool {
public:
    explicit HostPool(size_t heapBytes)
      : bytes_(CudaMemoryPoolView::metadataBytes() + heapBytes)
      , storage_(::operator new(bytes_, std::align_val_t(CudaMemoryPoolView::minimumAlignment())))
      , view_(CudaMemoryPoolView::initialize(storage_, bytes_)) {
        PROFILE();
    }

    HostPool(HostPool const&) = delete;
    HostPool& operator=(HostPool const&) = delete;

    ~HostPool() {
        PROFILE();
        ::operator delete(storage_, std::align_val_t(CudaMemoryPoolView::minimumAlignment()));
    }

    CudaMemoryPoolView view() const noexcept {
        PROFILE();
        return view_;
    }

private:
    size_t bytes_;
    void* storage_;
    CudaMemoryPoolView view_;
};

// Expected contents after runSequence: the pushed values, the parallel fill
// less the popped tail, padding up to the capacity, then the first element.
void checkSequence(SequenceResult const& result, size_t block, int const* contents) {
    PROFILE();
    size_t const keptSize = kPushCount + kResizeExtra - kPopCount;
    EXPECT_TRUE(result.pushesSucceeded) << "block " << block;
    EXPECT_TRUE(result.resizeSucceeded) << "block " << block;
    EXPECT_TRUE(result.aliasSucceeded) << "block " << block;
    EXPECT_GE(result.pushedCapacity, kPushCount) << "block " << block;
    EXPECT_GT(result.pushedCapacity, kInitialIntCapacity) << "block " << block;
    EXPECT_GT(result.finalSize, keptSize) << "block " << block;
    EXPECT_EQ(result.otherSize, 0u) << "block " << block;
    for (size_t index = 0; index < keptSize; index++) {
        int const expected = index < kPushCount ? pushedValue(block, index) : filledValue(block, index);
        ASSERT_EQ(contents[index], expected) << "block " << block << " index " << index;
    }
    EXPECT_EQ(contents[result.finalSize - 1], pushedValue(block, 0)) << "block " << block;
}

std::vector<int> pushedValues(size_t blockCount) {
    PROFILE();
    std::vector<int> values(blockCount * kPushCount);
    for (size_t block = 0; block < blockCount; block++) {
        for (size_t index = 0; index < kPushCount; index++) {
            values[block * kPushCount + index] = pushedValue(block, index);
        }
    }
    return values;
}

TEST(CudaDeviceVectorHostTest, SequenceMatchesExpectedContents) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    std::vector<int> const pushed = pushedValues(1);
    std::vector<int> contents(2 * (kPushCount + kResizeExtra));
    SequenceResult result{};
    cuda::DeviceVector<int> vector;
    cuda::DeviceVector<int> other;
    runSequence(vector, other, pool.view(), 0, pushed.data(), contents.data(), &result);
    checkSequence(result, 0, contents.data());
}

TEST(CudaDeviceVectorHostTest, CapacityUsesTheWholeSizeClass) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::DeviceVector<int> vector;
    vector.initialize(pool.view());
    EXPECT_EQ(vector.capacity(), 0u);
    EXPECT_TRUE(vector.reserve(1));
    EXPECT_EQ(vector.capacity(), kInitialIntCapacity);
    EXPECT_TRUE(vector.empty());
    EXPECT_TRUE(vector.reserve(kInitialIntCapacity + 1));
    EXPECT_EQ(vector.capacity(), 2 * kInitialIntCapacity);
    EXPECT_TRUE(vector.resize(3));
    vector.clear();
    EXPECT_TRUE(vector.empty());
    EXPECT_EQ(vector.capacity(), 2 * kInitialIntCapacity);
    vector.release();
    EXPECT_EQ(vector.capacity(), 0u);
}

// Released storage returns to the pool: rebuilding the same lists reserves no
// further heap.
TEST(CudaDeviceVectorHostTest, ReleaseReturnsStorageToThePool) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::DeviceVector<int> vector;
    size_t reservedAfterFirstPass = 0;
    for (int pass = 0; pass < 3; pass++) {
        vector.initialize(pool.view());
        for (size_t index = 0; index < kPushCount; index++) {
            ASSERT_TRUE(vector.pushBack(static_cast<int>(index)));
        }
        vector.release();
        if (pass == 0) {
            reservedAfterFirstPass = pool.view().bytesReserved();
        }
        EXPECT_EQ(pool.view().bytesReserved(), reservedAfterFirstPass) << "pass " << pass;
    }
}

TEST(CudaDeviceVectorHostTest, ExhaustionLeavesTheVectorUnchanged) {
    PROFILE();
    HostPool const pool(4096);
    cuda::DeviceVector<double> vector;
    vector.initialize(pool.view());
    ASSERT_TRUE(vector.pushBack(1.5));
    size_t const capacity = vector.capacity();
    EXPECT_FALSE(vector.resize(4096));
    EXPECT_FALSE(vector.reserve(4096));
    // Byte counts that overflow size_t are refused rather than wrapped.
    EXPECT_FALSE(vector.resize(~size_t{0} / 2));
    EXPECT_EQ(vector.size(), 1u);
    EXPECT_EQ(vector.capacity(), capacity);
    EXPECT_EQ(vector[0], 1.5);
    vector.release();
}

class CudaDeviceVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable";
            }
        } catch (cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        if (!cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
        stream_ = cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cuda::streamDestroy(stream_).get());
        }
    }

    // Runs the sequence in blockCount blocks of the given shape, which may be
    // multidimensional, and checks every block against the expected contents.
    void checkSequenceKernel(size_t blockCount, dim3 blockShape) {
        PROFILE();
        std::unique_ptr<CudaMemoryPool> pool = CudaMemoryPool::create(kHeapBytes).get();
        std::vector<int> const values = pushedValues(blockCount);
        ManagedArray<int> pushed(values.size());
        for (size_t index = 0; index < values.size(); index++) {
            pushed[index] = values[index];
        }
        size_t const contentsStride = 2 * (kPushCount + kResizeExtra);
        ManagedArray<int> contents(blockCount * contentsStride);
        ManagedArray<SequenceResult> results(blockCount);
        CudaMemoryPoolView const view = pool->view();
        int const* const pushedData = pushed.data();
        int* const contentsData = contents.data();
        SequenceResult* const resultsData = results.data();
        unsigned const blocks = static_cast<unsigned>(blockCount);
        cuda::launch(hpx::annotated_function(
                         [=](cudaStream_t launchStream) {
                             PROFILE();
                             sequenceKernel<<<blocks, blockShape, 0, launchStream>>>(view, pushedData, contentsData, contentsStride, resultsData);
                         },
                         "cudaDeviceVectorTest.sequenceKernel"),
            stream_)
            .get();
        for (size_t block = 0; block < blockCount; block++) {
            checkSequence(results[block], block, contents.data() + block * contentsStride);
        }
        pool->release().get();
    }

    cudaStream_t stream_ = nullptr;
};

TEST_F(CudaDeviceVectorTest, OneDimensionalBlocks) {
    PROFILE();
    checkSequenceKernel(37, dim3(96));
}

// The old container used threadIdx.x alone, so every row of a two-dimensional
// block acted as the first thread.
TEST_F(CudaDeviceVectorTest, MultidimensionalBlocks) {
    PROFILE();
    checkSequenceKernel(23, dim3(32, 4, 2));
}

// Vectors initialized on the host are filled by GPU blocks, then read and
// released on the host.
TEST_F(CudaDeviceVectorTest, ManagedVectorsFilledByBlocks) {
    PROFILE();
    constexpr unsigned kBlockCount = 19;
    constexpr unsigned kThreadCount = 64;
    std::unique_ptr<CudaMemoryPool> pool = CudaMemoryPool::create(kHeapBytes).get();
    ManagedArray<cuda::DeviceVector<int>> vectors(kBlockCount);
    for (unsigned block = 0; block < kBlockCount; block++) {
        vectors[block].initialize(pool->view());
    }
    cuda::DeviceVector<int>* const vectorsData = vectors.data();
    cuda::launch(hpx::annotated_function(
                     [=](cudaStream_t launchStream) {
                         PROFILE();
                         appendKernel<<<kBlockCount, kThreadCount, 0, launchStream>>>(vectorsData, kPushCount);
                     },
                     "cudaDeviceVectorTest.appendKernel"),
        stream_)
        .get();
    for (unsigned block = 0; block < kBlockCount; block++) {
        ASSERT_EQ(vectors[block].size(), kPushCount) << "block " << block;
        for (size_t index = 0; index < kPushCount; index++) {
            ASSERT_EQ(vectors[block][index], filledValue(block, index)) << "block " << block << " index " << index;
        }
        vectors[block].release();
    }
    pool->release().get();
}

TEST_F(CudaDeviceVectorTest, ExhaustionFailsOnEveryThread) {
    PROFILE();
    constexpr unsigned kThreadCount = 128;
    std::unique_ptr<CudaMemoryPool> pool = CudaMemoryPool::create(4096).get();
    ManagedArray<bool> unchanged(kThreadCount);
    CudaMemoryPoolView const view = pool->view();
    bool* const unchangedData = unchanged.data();
    cuda::launch(hpx::annotated_function(
                     [=](cudaStream_t launchStream) {
                         PROFILE();
                         exhaustionKernel<<<1, kThreadCount, 0, launchStream>>>(view, 4096, unchangedData);
                     },
                     "cudaDeviceVectorTest.exhaustionKernel"),
        stream_)
        .get();
    for (unsigned thread = 0; thread < kThreadCount; thread++) {
        EXPECT_TRUE(unchanged[thread]) << "thread " << thread;
    }
    pool->release().get();
}

}    // namespace
