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

#include <array>
#include <cstdint>
#include <exception>
#include <new>
#include <random>
#include <stddef.h>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/cuda/cuda_warp_partition.cuh>
#include <cosmictiger/profiler.hpp>

namespace {

namespace cuda = cosmictiger::cuda;

constexpr unsigned kBlockSize = 128;
constexpr unsigned kWarpsPerBlock = kBlockSize / cuda::kWarpSize;
constexpr std::uint32_t kTestSeed = 42;

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
            ADD_FAILURE() << "managed warp-partition buffer release failed: " << error.what();
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

// Each warp appends the indices of its entries to its own listCount lists,
// each list a slice of capacity entries. Lanes past the end take part with
// kNoWarpList. A failed append stops the warp and records the failing batch.
template <int listCount>
struct AppendRequest {
    int const* categories;
    size_t entriesPerWarp;
    size_t warpCount;
    size_t capacity;
    std::uint32_t* lists;
    size_t* sizes;
    long long* failedBatch;
};

template <int listCount>
__global__ void appendKernel(AppendRequest<listCount> request) {
    PROFILE();
    size_t const warp = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / cuda::kWarpSize;
    unsigned const lane = threadIdx.x % cuda::kWarpSize;
    if (warp >= request.warpCount) {
        return;
    }
    std::uint32_t* lists[listCount];
    size_t sizes[listCount];
    size_t capacities[listCount];
#pragma unroll
    for (int list = 0; list < listCount; list++) {
        lists[list] = request.lists + (warp * listCount + list) * request.capacity;
        sizes[list] = 0;
        capacities[list] = request.capacity;
    }
    long long failedBatch = -1;
    int const* categories = request.categories + warp * request.entriesPerWarp;
    for (size_t batch = 0; batch * cuda::kWarpSize < request.entriesPerWarp; batch++) {
        size_t const entry = batch * cuda::kWarpSize + lane;
        int const list = entry < request.entriesPerWarp ? categories[entry] : cuda::kNoWarpList;
        if (!cuda::appendWarp<std::uint32_t, listCount>(list, static_cast<std::uint32_t>(entry), lists, sizes, capacities)) {
            failedBatch = static_cast<long long>(batch);
            break;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int list = 0; list < listCount; list++) {
            request.sizes[warp * listCount + list] = sizes[list];
        }
        request.failedBatch[warp] = failedBatch;
    }
}

class CudaWarpPartitionTest : public ::testing::Test {
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

    // Runs the kernel and checks it against a host stable partition that
    // applies the same all-or-nothing rule to each 32-entry batch.
    template <int listCount>
    size_t checkAppend(size_t warpCount, size_t entriesPerWarp, size_t capacity, std::uint32_t seed) {
        PROFILE();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> category(cuda::kNoWarpList, listCount - 1);
        size_t const entryCount = warpCount * entriesPerWarp;
        ManagedArray<int> categories(entryCount);
        for (size_t entry = 0; entry < entryCount; entry++) {
            categories[entry] = category(generator);
        }
        ManagedArray<std::uint32_t> lists(warpCount * listCount * capacity);
        ManagedArray<size_t> sizes(warpCount * listCount);
        ManagedArray<long long> failedBatches(warpCount);
        AppendRequest<listCount> const request{categories.data(), entriesPerWarp, warpCount, capacity, lists.data(), sizes.data(), failedBatches.data()};
        unsigned const blocks = static_cast<unsigned>((warpCount + kWarpsPerBlock - 1) / kWarpsPerBlock);
        cuda::launch(hpx::annotated_function(
                         [request, blocks](cudaStream_t launchStream) {
                             PROFILE();
                             appendKernel<listCount><<<blocks, kBlockSize, 0, launchStream>>>(request);
                         },
                         "cudaWarpPartitionTest.appendKernel"),
            stream_)
            .get();

        size_t failedWarps = 0;
        for (size_t warp = 0; warp < warpCount; warp++) {
            std::array<std::vector<std::uint32_t>, listCount> expected;
            long long expectedFailure = -1;
            for (size_t batch = 0; batch * cuda::kWarpSize < entriesPerWarp && expectedFailure < 0; batch++) {
                std::array<std::vector<std::uint32_t>, listCount> pending;
                for (size_t lane = 0; lane < cuda::kWarpSize; lane++) {
                    size_t const entry = batch * cuda::kWarpSize + lane;
                    if (entry < entriesPerWarp && categories[warp * entriesPerWarp + entry] != cuda::kNoWarpList) {
                        pending[categories[warp * entriesPerWarp + entry]].push_back(static_cast<std::uint32_t>(entry));
                    }
                }
                bool fits = true;
                for (int list = 0; list < listCount; list++) {
                    fits = fits && expected[list].size() + pending[list].size() <= capacity;
                }
                if (!fits) {
                    expectedFailure = static_cast<long long>(batch);
                    break;
                }
                for (int list = 0; list < listCount; list++) {
                    expected[list].insert(expected[list].end(), pending[list].begin(), pending[list].end());
                }
            }
            EXPECT_EQ(failedBatches[warp], expectedFailure) << "warp " << warp;
            failedWarps += expectedFailure >= 0 ? 1 : 0;
            for (int list = 0; list < listCount; list++) {
                size_t const listIndex = warp * listCount + list;
                EXPECT_EQ(sizes[listIndex], expected[list].size()) << "warp " << warp << " list " << list;
                for (size_t position = 0; position < expected[list].size(); position++) {
                    EXPECT_EQ(lists[listIndex * capacity + position], expected[list][position]) << "warp " << warp << " list " << list << " " << position;
                }
            }
        }
        return failedWarps;
    }

private:
    cudaStream_t stream_ = nullptr;
};

TEST_F(CudaWarpPartitionTest, SingleListCompactionIsStable) {
    PROFILE();
    EXPECT_EQ(checkAppend<1>(97, 1000, 1000, kTestSeed), 0u);
}

// The old tree walk split its checklist among five lists.
TEST_F(CudaWarpPartitionTest, FiveChecklistListsMatchHostStablePartition) {
    PROFILE();
    EXPECT_EQ(checkAppend<5>(211, 777, 777, kTestSeed + 1), 0u);
}

TEST_F(CudaWarpPartitionTest, ManyListsAndPartialFinalBatch) {
    PROFILE();
    EXPECT_EQ(checkAppend<9>(64, 45, 45, kTestSeed + 2), 0u);
}

// Small capacities make some warps fail part way; a failed batch must write
// nothing and leave every size at its previous value.
TEST_F(CudaWarpPartitionTest, OverflowWritesNothingAndReportsTheBatch) {
    PROFILE();
    EXPECT_EQ(checkAppend<5>(128, 512, 70, kTestSeed + 3), 128u);
}

}    // namespace
