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

// GPU consumers use the future-returning runtime interface around the
// independently reusable allocator view.
#include <cosmictiger/cuda/cuda_memory_pool.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

void yieldForGpuSubmission();

namespace {

using cosmictiger::CudaMemoryPool;
using cosmictiger::CudaMemoryPoolView;

void require(bool condition, char const* message) {
    PROFILE();
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(size_t count)
      : values_(static_cast<T*>(cosmictiger::cuda::mallocManaged(count * sizeof(T)).get())) {
        PROFILE();
    }

    ManagedArray(ManagedArray const&) = delete;
    ManagedArray& operator=(ManagedArray const&) = delete;

    ~ManagedArray() {
        PROFILE();
        try {
            cosmictiger::cuda::free(values_).get();
        } catch (std::exception const& error) {
            std::cerr << "CUDA memory pool result deallocation failed: " << error.what() << '\n';
            std::terminate();
        }
    }

    T* data() const noexcept {
        PROFILE();
        return values_;
    }

    T& operator[](size_t index) const noexcept {
        PROFILE();
        return values_[index];
    }

private:
    T* values_ = nullptr;
};

struct AllocationRecord {
    void* pointer;
    size_t requestedBytes;
    bool succeeded;
};

__host__ __device__ unsigned char payloadValue(size_t owner, size_t round, size_t index) {
    PROFILE();
    return static_cast<unsigned char>((owner * 131 + round * 29 + index * 17) & 255);
}

__device__ void writePayload(void* pointer, size_t bytes, size_t owner, size_t round) {
    PROFILE();
    auto* payload = static_cast<unsigned char*>(pointer);
    for (size_t index = 0; index < bytes; index++) {
        payload[index] = payloadValue(owner, round, index);
    }
}

__host__ __device__ bool checkPayload(void* pointer, size_t bytes, size_t owner, size_t round) {
    PROFILE();
    auto const* payload = static_cast<unsigned char const*>(pointer);
    for (size_t index = 0; index < bytes; index++) {
        if (payload[index] != payloadValue(owner, round, index)) {
            return false;
        }
    }
    return true;
}

__global__ void allocateKernel(CudaMemoryPoolView view, AllocationRecord* records, size_t count, size_t round) {
    PROFILE();
    size_t const owner = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (owner >= count) {
        return;
    }
    size_t const requestedBytes = size_t(16) << (owner % 6);
    void* const pointer = view.allocate(requestedBytes);
    records[owner] = {pointer, requestedBytes, pointer != nullptr};
    if (pointer != nullptr) {
        writePayload(pointer, requestedBytes, owner, round);
    }
}

__global__ void releaseKernel(CudaMemoryPoolView view, AllocationRecord* records, size_t count) {
    PROFILE();
    size_t const owner = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (owner < count) {
        records[owner].succeeded = view.deallocate(records[owner].pointer);
    }
}

__global__ void mixedStressKernel(CudaMemoryPoolView view, AllocationRecord* records, size_t count) {
    PROFILE();
    size_t const owner = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (owner >= count) {
        return;
    }
    bool success = true;
    for (size_t round = 0; round < 128; round++) {
        size_t const requestedBytes = size_t(16) << ((owner + round) % 6);
        // Varying alignments and sizes make simultaneous requests from one warp
        // follow different allocation paths, including empty and reused bins.
        size_t const alignment = size_t(16) << ((owner + round) % 4);
        void* const pointer = view.allocate(requestedBytes, alignment);
        if (pointer == nullptr) {
            success = false;
            break;
        }
        success = reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0 && success;
        writePayload(pointer, requestedBytes, owner, round);
        __threadfence();
        success = checkPayload(pointer, requestedBytes, owner, round) && success;
        success = view.deallocate(pointer) && success;
    }
    records[owner].succeeded = success;
}

__global__ void exactFitKernel(CudaMemoryPoolView view, AllocationRecord* record) {
    PROFILE();
    bool success = view.allocate(1024) == nullptr && view.bytesReserved() == 0;
    success = view.allocate(static_cast<size_t>(-1)) == nullptr && success;
    success = view.bytesReserved() == 0 && success;
    void* const pointer = view.allocate(16);
    success = pointer != nullptr && success;
    success = view.bytesReserved() == view.capacity() && success;
    success = view.allocate(1) == nullptr && success;
    success = view.bytesReserved() == view.capacity() && success;
    if (pointer != nullptr) {
        writePayload(pointer, 16, 3, 4);
        success = checkPayload(pointer, 16, 3, 4) && success;
        success = view.deallocate(pointer) && success;
        void* const reused = view.allocate(16);
        success = reused == pointer && success;
        success = view.deallocate(reused) && success;
    }
    *record = {pointer, 16, success};
}

void checkExactFit() {
    PROFILE();
    CudaMemoryPool pool(CudaMemoryPoolView::blockOverhead() + 16);
    ManagedArray<AllocationRecord> result(1);
    auto const view = pool.view();
    auto* const records = result.data();
    cosmictiger::cuda::launch(hpx::annotated_function(
                                  [view, records](cudaStream_t stream) {
                                      PROFILE();
                                      exactFitKernel<<<1, 1, 0, stream>>>(view, records);
                                  },
                                  "CudaMemoryPoolTest::exactFit"))
        .get();
    require(result[0].succeeded, "device exact-fit, overflow, or exhaustion regression");
    pool.reset();
    require(pool.view().bytesReserved() == 0, "managed owner reset left reservations");
}

void checkCrossKernelReuse() {
    size_t constexpr kAllocationCount = 2048;
    unsigned constexpr kThreadsPerBlock = 128;
    unsigned constexpr kBlocks = static_cast<unsigned>((kAllocationCount + kThreadsPerBlock - 1) / kThreadsPerBlock);
    PROFILE();
    // All size classes fit simultaneously. Repeated rounds may only reuse the
    // first round's blocks, so a lost free pointer cannot hide in fresh space.
    size_t heapCapacity = 0;
    for (size_t owner = 0; owner < kAllocationCount; owner++) {
        heapCapacity += CudaMemoryPoolView::blockOverhead() + (size_t(16) << (owner % 6));
    }
    CudaMemoryPool pool(heapCapacity);
    ManagedArray<AllocationRecord> records(kAllocationCount);
    auto const view = pool.view();
    auto* const allocationRecords = records.data();
    std::vector<void*> firstPointers(kAllocationCount);
    for (size_t round = 0; round < 12; round++) {
        cosmictiger::cuda::launch(hpx::annotated_function(
                                      [view, allocationRecords, round](cudaStream_t stream) {
                                          PROFILE();
                                          allocateKernel<<<kBlocks, kThreadsPerBlock, 0, stream>>>(view, allocationRecords, kAllocationCount, round);
                                      },
                                      "CudaMemoryPoolTest::allocate"))
            .get();
        for (size_t owner = 0; owner < kAllocationCount; owner++) {
            auto const& record = records[owner];
            require(record.succeeded, "multi-block allocation lost a free pointer");
            require(reinterpret_cast<std::uintptr_t>(record.pointer) % 16 == 0, "device allocation is not 16-byte aligned");
            require(checkPayload(record.pointer, record.requestedBytes, owner, round), "device live allocation payload overwritten");
            for (size_t previous = 0; previous < owner; previous++) {
                require(records[previous].pointer != record.pointer, "two live device allocations share a pointer");
            }
            if (round == 0) {
                firstPointers[owner] = record.pointer;
            } else {
                bool found = false;
                for (void* previous : firstPointers) {
                    found = previous == record.pointer || found;
                }
                require(found, "cross-kernel reuse unexpectedly reserved a new pointer");
            }
        }
        require(pool.view().bytesReserved() == heapCapacity, "device exact-fit reservation total incorrect");
        cosmictiger::cuda::launch(hpx::annotated_function(
                                      [view, allocationRecords](cudaStream_t stream) {
                                          PROFILE();
                                          releaseKernel<<<kBlocks, kThreadsPerBlock, 0, stream>>>(view, allocationRecords, kAllocationCount);
                                      },
                                      "CudaMemoryPoolTest::release"))
            .get();
        for (size_t owner = 0; owner < kAllocationCount; owner++) {
            require(records[owner].succeeded, "multi-block free failed");
        }
    }
}

void checkMixedStress() {
    size_t constexpr kAllocationCount = 2048;
    unsigned constexpr kThreadsPerBlock = 128;
    unsigned constexpr kBlocks = static_cast<unsigned>((kAllocationCount + kThreadsPerBlock - 1) / kThreadsPerBlock);
    PROFILE();
    CudaMemoryPool pool(16 << 20);
    ManagedArray<AllocationRecord> records(kAllocationCount);
    auto const view = pool.view();
    auto* const allocationRecords = records.data();
    std::atomic<bool> submitted{false};
    auto complete = cosmictiger::cuda::launch(hpx::annotated_function(
        [view, allocationRecords, &submitted](cudaStream_t stream) {
            PROFILE();
            mixedStressKernel<<<kBlocks, kThreadsPerBlock, 0, stream>>>(view, allocationRecords, kAllocationCount);
            submitted.store(true, std::memory_order_release);
        },
        "CudaMemoryPoolTest::mixedStress"));
    // Device selection can fail before the launch lambda runs. Wait for either
    // submission or completion so such failures cannot strand this test.
    while (!submitted.load(std::memory_order_acquire) && !complete.is_ready()) {
        yieldForGpuSubmission();
    }
    if (!submitted.load(std::memory_order_acquire)) {
        complete.get();
    }
    bool hostSuccess = true;
    // The calling host thread also uses the same managed pool before joining
    // the GPU. This exercises system-scope ownership transfers without adding
    // a threading library or another allocator dependency to this consumer.
    for (size_t round = 0; round < 4096; round++) {
        size_t const requestedBytes = size_t(16) << (round % 6);
        size_t const alignment = size_t(16) << (round % 4);
        void* const pointer = view.allocate(requestedBytes, alignment);
        if (pointer == nullptr) {
            hostSuccess = false;
            break;
        }
        auto* payload = static_cast<unsigned char*>(pointer);
        for (size_t index = 0; index < requestedBytes; index++) {
            payload[index] = payloadValue(kAllocationCount, round, index);
        }
        hostSuccess = checkPayload(pointer, requestedBytes, kAllocationCount, round) && hostSuccess;
        hostSuccess = view.deallocate(pointer) && hostSuccess;
    }
    complete.get();
    require(hostSuccess, "simultaneous host/device allocation stress failed on host");
    for (size_t owner = 0; owner < kAllocationCount; owner++) {
        require(records[owner].succeeded, "mixed device allocate/free stress failed");
    }
}

}    // namespace

int runGpuTests() {
    PROFILE();
    try {
        int deviceCount = 0;
        try {
            deviceCount = cosmictiger::cuda::deviceCount().get();
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                std::cout << "CUDA memory pool GPU tests skipped: " << error.what() << '\n';
                return 77;
            }
            throw;
        }
        if (deviceCount == 0) {
            std::cout << "CUDA memory pool GPU tests skipped: no CUDA device available\n";
            return 77;
        }
        int const device = cosmictiger::cuda::device().get();
        auto const properties = cosmictiger::cuda::deviceProperties(device).get();
        if (!properties.managedMemory || !properties.concurrentManagedAccess) {
            std::cout << "CUDA memory pool GPU tests skipped: concurrent managed memory unsupported\n";
            return 77;
        }
        checkExactFit();
        checkCrossKernelReuse();
        checkMixedStress();
        std::cout << "CUDA memory pool GPU tests passed\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "CUDA memory pool GPU test failed: " << error.what() << '\n';
        return 1;
    }
}
