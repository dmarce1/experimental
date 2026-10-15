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

#include <cosmictiger/cuda/cuda_memory_pool_view.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/future.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace {

using cosmictiger::CudaMemoryPoolView;

void require(bool condition, char const* message) {
    PROFILE();
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Host tests supply ordinary aligned storage, proving the view has no CUDA
// runtime dependency and checking its algorithms before a GPU is available.
class Buffer {
public:
    explicit Buffer(size_t heapCapacity)
      : storage_(::operator new(CudaMemoryPoolView::metadataBytes() + heapCapacity, std::align_val_t(CudaMemoryPoolView::minimumAlignment())))
      , view_(CudaMemoryPoolView::initialize(storage_, CudaMemoryPoolView::metadataBytes() + heapCapacity)) {
        PROFILE();
        require(view_.valid(), "aligned pool storage rejected");
    }

    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;

    ~Buffer() {
        PROFILE();
        ::operator delete(storage_, std::align_val_t(CudaMemoryPoolView::minimumAlignment()));
    }

    CudaMemoryPoolView view() const noexcept {
        PROFILE();
        return view_;
    }

private:
    void* storage_;
    CudaMemoryPoolView view_;
};

unsigned char payloadValue(size_t owner, size_t round, size_t index) {
    PROFILE();
    return static_cast<unsigned char>((owner * 131 + round * 29 + index * 17) & 255);
}

void writePayload(void* pointer, size_t bytes, size_t owner, size_t round) {
    PROFILE();
    auto* payload = static_cast<unsigned char*>(pointer);
    for (size_t index = 0; index < bytes; index++) {
        payload[index] = payloadValue(owner, round, index);
    }
}

bool checkPayload(void* pointer, size_t bytes, size_t owner, size_t round) {
    PROFILE();
    auto const* payload = static_cast<unsigned char const*>(pointer);
    for (size_t index = 0; index < bytes; index++) {
        if (payload[index] != payloadValue(owner, round, index)) {
            return false;
        }
    }
    return true;
}

void checkSizingAndInitialization() {
    size_t constexpr kLargeRequest = (size_t(1) << 30) + 1;
    PROFILE();
    require(CudaMemoryPoolView::minimumAlignment() == 16, "minimum alignment changed");
    require(CudaMemoryPoolView::sizeClassCapacity(0) == 0, "zero size has a size class");
    require(CudaMemoryPoolView::sizeClassCapacity(1) == 16, "minimum size class changed");
    require(CudaMemoryPoolView::sizeClassCapacity(16) == 16, "exact size class rounded upward");
    require(CudaMemoryPoolView::sizeClassCapacity(17) == 32, "size class rounded downward");
    require(CudaMemoryPoolView::sizeClassCapacity(std::numeric_limits<size_t>::max()) == 0, "size class overflow not rejected");
    if constexpr (sizeof(size_t) > 4) {
        require(CudaMemoryPoolView::sizeClassCapacity(kLargeRequest) == (size_t(1) << 31), "size class above 1 GiB overflowed");
    }

    require(!CudaMemoryPoolView::initialize(nullptr, 1024).valid(), "null storage accepted");
    alignas(16) unsigned char undersized[16]{};
    require(!CudaMemoryPoolView::initialize(undersized, sizeof(undersized)).valid(), "undersized metadata storage accepted");
    Buffer buffer(1024);
    auto const view = buffer.view();
    require(view.capacity() == 1024, "heap capacity includes metadata");
    require(view.bytesReserved() == 0, "new pool has reserved blocks");
    require(view.allocate(0) == nullptr, "zero-byte allocation unexpectedly succeeded");
    require(view.allocate(16, 0) == nullptr, "zero alignment accepted");
    require(view.allocate(16, 3) == nullptr, "non-power-of-two alignment accepted");
    require(view.deallocate(nullptr), "freeing nullptr failed");
    require(view.bytesReserved() == 0, "invalid allocation changed heap position");
}

void checkExhaustionAndExactFit() {
    size_t constexpr kPayloadBytes = 16;
    size_t constexpr kBlockBytes = CudaMemoryPoolView::blockOverhead() + kPayloadBytes;
    PROFILE();
    Buffer buffer(kBlockBytes);
    auto const view = buffer.view();

    // A failed large reservation must not move the bump pointer. The following
    // small allocation then consumes the heap exactly, including its header.
    require(view.allocate(1024) == nullptr, "oversized allocation unexpectedly succeeded");
    require(view.allocate(std::numeric_limits<size_t>::max()) == nullptr, "overflowing allocation unexpectedly succeeded");
    require(view.bytesReserved() == 0, "failed allocation consumed heap space");
    void* const pointer = view.allocate(kPayloadBytes);
    require(pointer != nullptr, "exact-fit allocation failed");
    require(view.bytesReserved() == kBlockBytes, "exact fit reserved incorrect byte count");
    require(view.allocate(1) == nullptr, "exhausted pool returned another allocation");
    require(view.bytesReserved() == kBlockBytes, "heap exhaustion advanced bump pointer");
    writePayload(pointer, kPayloadBytes, 1, 2);
    require(checkPayload(pointer, kPayloadBytes, 1, 2), "allocation payload damaged metadata");
    require(view.deallocate(pointer), "exact-fit block could not be freed");
    require(view.allocate(kPayloadBytes) == pointer, "freed exact-fit block was lost");
    require(view.deallocate(pointer), "reused exact-fit block could not be freed");
    view.reset();
    require(view.bytesReserved() == 0, "reset left heap reservations");
    require(view.allocate(kPayloadBytes) == pointer, "reset failed to restore initial heap position");
    require(view.deallocate(pointer), "reset block could not be freed");
}

void checkAlignment() {
    size_t constexpr kAlignments[]{16, 32, 64, 128, 256, 1024};
    size_t constexpr kRequests[]{1, 15, 16, 17, 127};
    PROFILE();
    Buffer buffer(1 << 16);
    auto const view = buffer.view();
    for (size_t alignment : kAlignments) {
        for (size_t requestedBytes : kRequests) {
            void* const pointer = view.allocate(requestedBytes, alignment);
            require(pointer != nullptr, "aligned allocation failed");
            require(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0, "returned pointer has incorrect alignment");
            writePayload(pointer, requestedBytes, alignment, requestedBytes);
            require(checkPayload(pointer, requestedBytes, alignment, requestedBytes), "aligned payload corrupted");
            require(view.deallocate(pointer), "aligned allocation could not be freed");
            void* const reused = view.allocate(requestedBytes, alignment);
            require(reused == pointer, "aligned allocation could not be reused");
            require(view.deallocate(reused), "aligned reused block could not be freed");
        }
    }
}

void checkConcurrentReuse() {
    size_t constexpr kTaskCount = 128;
    size_t constexpr kPayloadBytes = 64;
    size_t constexpr kRounds = 16;
    size_t constexpr kHeapBytes = kTaskCount * (CudaMemoryPoolView::blockOverhead() + kPayloadBytes);
    PROFILE();
    Buffer buffer(kHeapBytes);
    auto const view = buffer.view();
    std::vector<void*> pointers(kTaskCount);

    for (size_t round = 0; round < kRounds; round++) {
        std::vector<hpx::future<void*>> allocations;
        allocations.reserve(kTaskCount);
        for (size_t owner = 0; owner < kTaskCount; owner++) {
            allocations.emplace_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [view, owner, round] {
                        PROFILE();
                        void* const pointer = view.allocate(kPayloadBytes);
                        if (pointer != nullptr) {
                            writePayload(pointer, kPayloadBytes, owner, round);
                        }
                        return pointer;
                    },
                    "CudaMemoryPoolHostTest::allocate")));
        }
        // Finish every task before testing results or destroying its backing
        // storage, including on a failed allocation.
        for (size_t owner = 0; owner < kTaskCount; owner++) {
            pointers[owner] = allocations[owner].get();
        }
        for (size_t owner = 0; owner < kTaskCount; owner++) {
            require(pointers[owner] != nullptr, "concurrent allocation lost a free block");
            require(checkPayload(pointers[owner], kPayloadBytes, owner, round), "live concurrent allocation payload overwritten");
            for (size_t previous = 0; previous < owner; previous++) {
                require(pointers[previous] != pointers[owner], "two live allocations share a pointer");
            }
        }
        require(view.bytesReserved() == kHeapBytes, "concurrent exact-fit reservation has wrong size");
        std::vector<hpx::future<bool>> releases;
        releases.reserve(kTaskCount);
        for (void* pointer : pointers) {
            releases.emplace_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [view, pointer] {
                        PROFILE();
                        return view.deallocate(pointer);
                    },
                    "CudaMemoryPoolHostTest::release")));
        }
        bool released = true;
        for (auto& release : releases) {
            released = release.get() && released;
        }
        require(released, "concurrent free failed");
    }

    // Mixed allocation and release interleavings exercise both ends of the
    // same free list. Yield while holding live data so other HPX tasks can
    // expose duplicate ownership through the payload sentinels.
    std::vector<hpx::future<bool>> workers;
    workers.reserve(kTaskCount);
    for (size_t owner = 0; owner < kTaskCount; owner++) {
        workers.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [view, owner] {
                    PROFILE();
                    bool success = true;
                    for (size_t round = 0; round < 256; round++) {
                        void* const pointer = view.allocate(kPayloadBytes);
                        if (pointer == nullptr) {
                            return false;
                        }
                        writePayload(pointer, kPayloadBytes, owner, round);
                        hpx::this_thread::yield();
                        success = checkPayload(pointer, kPayloadBytes, owner, round) && success;
                        success = view.deallocate(pointer) && success;
                    }
                    return success;
                },
                "CudaMemoryPoolHostTest::mixedStress")));
    }
    bool success = true;
    for (auto& worker : workers) {
        success = worker.get() && success;
    }
    require(success, "mixed concurrent allocation/free corrupted or lost a block");

    // Recover the entire pool after stress: a missing node is observable even
    // if contention happened not to overwrite another participant's payload.
    for (void*& pointer : pointers) {
        pointer = view.allocate(kPayloadBytes);
        require(pointer != nullptr, "mixed stress leaked a free block");
    }
    require(view.allocate(kPayloadBytes) == nullptr, "mixed stress duplicated a free block");
    for (void* pointer : pointers) {
        require(view.deallocate(pointer), "post-stress recovery free failed");
    }
}

}    // namespace

int hpx_main() {
    PROFILE();
    int result = 0;
    try {
        checkSizingAndInitialization();
        checkExhaustionAndExactFit();
        checkAlignment();
        checkConcurrentReuse();
        std::cout << "CUDA memory pool host tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "CUDA memory pool host test failed: " << error.what() << '\n';
        result = 1;
    }
    hpx::finalize();
    return result;
}

int main(int argc, char* argv[]) {
    PROFILE();
    hpx::init_params params;
    params.cfg = {"hpx.os_threads=2"};
    return hpx::init(argc, argv, params);
}
