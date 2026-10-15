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

#include <exception>
#include <memory>
#include <new>
#include <stddef.h>

#include <gtest/gtest.h>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_managed_vector.hpp>
#include <cosmictiger/cuda/cuda_memory_pool.hpp>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/profiler.hpp>

namespace {

namespace cuda = cosmictiger::cuda;
using cosmictiger::CudaMemoryPool;
using cosmictiger::CudaMemoryPoolView;

// The smallest allocation holds this many ints, so arrays longer than it grow.
constexpr size_t kInitialIntCapacity = 256;
constexpr size_t kPushCount = 3000;
constexpr size_t kHeapBytes = size_t(64) << 20;

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

TEST(CudaManagedVectorHostTest, ConstructorsInitializeElements) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::ManagedVector<int> const zeros(pool.view(), 10);
    cuda::ManagedVector<int> const sevens(pool.view(), 10, 7);
    ASSERT_EQ(zeros.size(), 10u);
    ASSERT_EQ(sevens.size(), 10u);
    for (size_t index = 0; index < 10; index++) {
        EXPECT_EQ(zeros[index], 0);
        EXPECT_EQ(sevens[index], 7);
    }
    cuda::ManagedVector<int> const empty(pool.view());
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.capacity(), 0u);
}

TEST(CudaManagedVectorHostTest, AppendsGrowGeometrically) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::ManagedVector<int> vector(pool.view());
    size_t capacityChanges = 0;
    size_t capacity = 0;
    for (size_t index = 0; index < kPushCount; index++) {
        vector.pushBack(static_cast<int>(index * 7));
        if (vector.capacity() != capacity) {
            capacity = vector.capacity();
            capacityChanges++;
        }
    }
    // 256, 512, ..., 4096 ints.
    EXPECT_EQ(capacityChanges, 5u);
    EXPECT_EQ(vector.capacity(), 16 * kInitialIntCapacity);
    for (size_t index = 0; index < kPushCount; index++) {
        ASSERT_EQ(vector[index], static_cast<int>(index * 7));
    }
    vector.popBack();
    EXPECT_EQ(vector.size(), kPushCount - 1);
    EXPECT_EQ(vector.back(), static_cast<int>((kPushCount - 2) * 7));
    vector.clear();
    EXPECT_TRUE(vector.empty());
    EXPECT_EQ(vector.capacity(), 16 * kInitialIntCapacity);
}

// Appending an element of the vector at a full capacity reallocates, which
// frees the storage the argument refers to.
TEST(CudaManagedVectorHostTest, AppendingOwnElementAcrossGrowth) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::ManagedVector<int> vector(pool.view(), 1, 41);
    vector.resize(vector.capacity(), 5);
    size_t const capacity = vector.capacity();
    vector.pushBack(vector[0]);
    EXPECT_GT(vector.capacity(), capacity);
    EXPECT_EQ(vector.back(), 41);
    vector.resize(vector.capacity() + 1, vector[1]);
    EXPECT_EQ(vector.back(), 5);
}

TEST(CudaManagedVectorHostTest, CopiesAreDeepAndMovesTransferStorage) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::ManagedVector<int> original(pool.view(), kPushCount, 3);
    cuda::ManagedVector<int> copy(original);
    copy[0] = 4;
    EXPECT_EQ(original[0], 3);
    EXPECT_NE(copy.data(), original.data());
    ASSERT_EQ(copy.size(), kPushCount);

    cuda::ManagedVector<int> assigned(pool.view(), 2, 9);
    assigned = original;
    EXPECT_EQ(assigned.size(), kPushCount);
    EXPECT_EQ(assigned[kPushCount - 1], 3);
    auto const& self = assigned;
    assigned = self;
    EXPECT_EQ(assigned.size(), kPushCount);
    EXPECT_EQ(assigned[0], 3);

    int const* const storage = original.data();
    cuda::ManagedVector<int> moved(static_cast<cuda::ManagedVector<int>&&>(original));
    EXPECT_EQ(moved.data(), storage);
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.data(), nullptr);
    original.pushBack(1);
    EXPECT_EQ(original[0], 1);

    copy = static_cast<cuda::ManagedVector<int>&&>(moved);
    EXPECT_EQ(copy.data(), storage);
    EXPECT_EQ(copy[0], 3);
}

// Destroyed vectors return their storage: rebuilding reserves no more heap.
TEST(CudaManagedVectorHostTest, DestructionReturnsStorageToThePool) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    size_t reservedAfterFirstPass = 0;
    for (int pass = 0; pass < 3; pass++) {
        {
            cuda::ManagedVector<int> vector(pool.view());
            for (size_t index = 0; index < kPushCount; index++) {
                vector.pushBack(static_cast<int>(index));
            }
            cuda::ManagedVector<int> const copy(vector);
        }
        if (pass == 0) {
            reservedAfterFirstPass = pool.view().bytesReserved();
        }
        EXPECT_EQ(pool.view().bytesReserved(), reservedAfterFirstPass) << "pass " << pass;
    }
}

TEST(CudaManagedVectorHostTest, AllocationFailureThrowsAndLeavesTheVectorUnchanged) {
    PROFILE();
    HostPool const pool(4096);
    cuda::ManagedVector<double> vector(pool.view(), 1, 1.5);
    size_t const capacity = vector.capacity();
    EXPECT_THROW(vector.resize(4096), std::bad_alloc);
    EXPECT_THROW(vector.reserve(4096), std::bad_alloc);
    // Byte counts that overflow size_t are refused rather than wrapped.
    EXPECT_THROW(vector.resize(~size_t{0} / 2), std::bad_alloc);
    EXPECT_EQ(vector.size(), 1u);
    EXPECT_EQ(vector.capacity(), capacity);
    EXPECT_EQ(vector[0], 1.5);
    cuda::ManagedVector<double> withoutPool;
    EXPECT_THROW(withoutPool.pushBack(1.0), std::bad_alloc);
}

TEST(CudaManagedVectorHostTest, ViewsSeeTheElements) {
    PROFILE();
    HostPool const pool(kHeapBytes);
    cuda::ManagedVector<int> vector(pool.view(), 5, 2);
    cuda::ManagedVectorView<int> const view = vector.view();
    view[3] = 8;
    cuda::ManagedVectorView<int const> const constView = view;
    EXPECT_EQ(constView.size(), 5u);
    EXPECT_EQ(constView[3], 8);
    EXPECT_EQ(vector[3], 8);
    int sum = 0;
    for (int const value : constView) {
        sum += value;
    }
    EXPECT_EQ(sum, 16);
}

__global__ void transformKernel(cuda::ManagedVectorView<int const> input, cuda::ManagedVectorView<int> output) {
    PROFILE();
    size_t const index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < output.size()) {
        output[index] = 3 * input[index] + 1;
    }
}

class CudaManagedVectorTest : public ::testing::Test {
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
        if (cuda::deviceAttribute(cudaDevAttrConcurrentManagedAccess, 0).get() == 0) {
            GTEST_SKIP() << "CUDA device does not support concurrent managed access";
        }
        stream_ = cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cuda::streamDestroy(stream_).get());
        }
    }

    cudaStream_t stream_ = nullptr;
};

// Built on the host, passed to a kernel through views, read back on the host.
TEST_F(CudaManagedVectorTest, HostFilledVectorsReachTheKernel) {
    PROFILE();
    constexpr unsigned kBlockSize = 128;
    constexpr size_t kCount = 100003;
    std::unique_ptr<CudaMemoryPool> pool = CudaMemoryPool::create(kHeapBytes).get();
    {
        cuda::ManagedVector<int> input(pool->view());
        for (size_t index = 0; index < kCount; index++) {
            input.pushBack(static_cast<int>(index) - 5000);
        }
        cuda::ManagedVector<int> output(pool->view(), kCount);
        cuda::ManagedVectorView<int const> const inputView = input.view();
        cuda::ManagedVectorView<int> const outputView = output.view();
        unsigned const blocks = static_cast<unsigned>((kCount + kBlockSize - 1) / kBlockSize);
        cuda::launch(hpx::annotated_function(
                         [=](cudaStream_t launchStream) {
                             PROFILE();
                             transformKernel<<<blocks, kBlockSize, 0, launchStream>>>(inputView, outputView);
                         },
                         "cudaManagedVectorTest.transformKernel"),
            stream_)
            .get();
        for (size_t index = 0; index < kCount; index++) {
            ASSERT_EQ(output[index], 3 * (static_cast<int>(index) - 5000) + 1) << "index " << index;
        }
    }
    pool->release().get();
}

}    // namespace
