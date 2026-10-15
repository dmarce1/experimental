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

#pragma once

// This allocator view is self-contained and usable in CUDA C++20 kernels.
// The HPX host owner is declared separately in cuda_memory_pool.hpp.
#include <cosmictiger/profiler.hpp>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <string>

namespace cosmictiger {

class CudaMemoryPoolView;

namespace detail {

    inline constexpr size_t kPoolAlignment = 16;
    inline constexpr size_t kPoolSizeClassCount = sizeof(size_t) * CHAR_BIT;
    inline constexpr unsigned long long kPoolNullOffset = ~0ULL;

    // Links never change after publication. A block stays in its size class for
    // the entire pool lifetime; only its ownership state changes on reuse.
    class alignas(kPoolAlignment) CudaMemoryPoolBlock {
        friend class cosmictiger::CudaMemoryPoolView;

        unsigned long long nextOffset_;
        unsigned long long state_;
        size_t sizeClass_;
        size_t payloadBytes_;

        __host__ __device__ CudaMemoryPoolBlock(size_t sizeClass, size_t payloadBytes) noexcept
          : nextOffset_(kPoolNullOffset)
          , state_(1)
          , sizeClass_(sizeClass)
          , payloadBytes_(payloadBytes) {
            PROFILE();
        }
    };

    class alignas(kPoolAlignment) CudaMemoryPoolState {
        friend class cosmictiger::CudaMemoryPoolView;

        unsigned long long bytesReserved_;
        unsigned long long heads_[kPoolSizeClassCount];
        unsigned long long freeHints_[kPoolSizeClassCount];
        unsigned long long freeCounts_[kPoolSizeClassCount];

        __host__ __device__ CudaMemoryPoolState() noexcept {
            PROFILE();
            reset();
        }

        // Initialization/reset is exclusively owned by the caller. During
        // normal operation all accesses to these words use atomic_ref.
        __host__ __device__ void reset() noexcept {
            PROFILE();
            bytesReserved_ = 0;
            for (size_t index = 0; index < kPoolSizeClassCount; index++) {
                heads_[index] = kPoolNullOffset;
                freeHints_[index] = kPoolNullOffset;
                freeCounts_[index] = 0;
            }
        }
    };

}    // namespace detail

// A copyable, non-owning handle that can be passed to kernels by value. Multiple
// CPU threads and CUDA threads may allocate/free concurrently with system-scope
// atomics, provided the backing memory supports those atomics. Managed memory
// requires concurrentManagedAccess == 1. Ordinary CPU memory is valid for CPU
// use; a device allocation is valid for use by threads on that GPU alone.
//
// There are no spinlocks, collective operations, or block barriers here: one
// lane may call the allocator independently of the other lanes in its warp.
// CUDA device use requires compute capability 6.0 or newer. Compile consumers
// with C++20 and an appropriate architecture (for example, -arch=sm_60).
class CudaMemoryPoolView {
public:
    __host__ __device__ CudaMemoryPoolView() noexcept {
        PROFILE();
    }

    __host__ __device__ static constexpr size_t minimumAlignment() noexcept {
        PROFILE();
        return detail::kPoolAlignment;
    }

    __host__ __device__ static constexpr size_t metadataBytes() noexcept {
        PROFILE();
        return sizeof(detail::CudaMemoryPoolState);
    }

    __host__ __device__ static constexpr size_t blockOverhead() noexcept {
        PROFILE();
        return sizeof(detail::CudaMemoryPoolBlock);
    }

    // Zero and requests whose rounded size cannot be represented return zero.
    // No signed arithmetic or shifts are used, including for requests > 1 GiB.
    __host__ __device__ static constexpr size_t sizeClassCapacity(size_t bytes) noexcept {
        PROFILE();
        if (bytes == 0) {
            return 0;
        }
        size_t roundedBytes = minimumAlignment();
        while (roundedBytes < bytes) {
            if (roundedBytes > ~size_t{0} / 2) {
                return 0;
            }
            roundedBytes *= 2;
        }
        return roundedBytes;
    }

    // Place bookkeeping at the beginning of caller-owned, 16-byte-aligned
    // storage. The remaining bytes form the heap. Initialization must finish
    // before sharing the view. No CUDA allocation or runtime call is performed.
    __host__ __device__ static CudaMemoryPoolView initialize(void* buffer, size_t bufferBytes) noexcept {
        PROFILE();
        CudaMemoryPoolView view;
        if (buffer == nullptr || reinterpret_cast<std::uintptr_t>(buffer) % minimumAlignment() != 0 || bufferBytes < metadataBytes()) {
            return view;
        }
        view.state_ = ::new (buffer) detail::CudaMemoryPoolState;
        view.heap_ = static_cast<unsigned char*>(buffer) + metadataBytes();
        view.capacity_ = bufferBytes - metadataBytes();
        return view;
    }

    __host__ __device__ bool valid() const noexcept {
        PROFILE();
        return state_ != nullptr;
    }

    __host__ __device__ size_t capacity() const noexcept {
        PROFILE();
        return capacity_;
    }

    // Includes block headers and alignment padding. Freed blocks remain
    // reserved for reuse by their original size class; they are not coalesced.
    __host__ __device__ size_t bytesReserved() const noexcept {
        PROFILE();
        return valid() ? static_cast<size_t>(AtomicReference(state_->bytesReserved_).load(::cuda::std::memory_order_relaxed)) : 0;
    }

    // Return nullptr for zero, overflow, invalid alignment, or exhaustion.
    // Alignment must be a power of two; alignments below 16 are promoted to 16.
    // Each successful call owns a distinct block until deallocate is called.
    __host__ __device__ void* allocate(size_t bytes, size_t alignment = minimumAlignment()) const noexcept {
        PROFILE();
        size_t const payloadBytes = sizeClassCapacity(bytes);
        if (!valid() || payloadBytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
            return nullptr;
        }
        if (alignment < minimumAlignment()) {
            alignment = minimumAlignment();
        }
        size_t sizeClass = 0;
        for (size_t value = payloadBytes; value > 1; value /= 2) {
            sizeClass++;
        }

        if (void* pointer = findFreeBlock(sizeClass, alignment)) {
            return pointer;
        }

        size_t headerOffset = 0;
        if (!reserveBlock(payloadBytes, alignment, headerOffset)) {
            // A concurrent free may have completed while the reservation was
            // attempted. Recheck once before reporting exhaustion.
            return findFreeBlock(sizeClass, alignment);
        }
        auto* block = ::new (static_cast<void*>(heap_ + headerOffset)) detail::CudaMemoryPoolBlock(sizeClass, payloadBytes);
        AtomicReference head(state_->heads_[sizeClass]);
        auto nextOffset = head.load(::cuda::std::memory_order_acquire);
        do {
            block->nextOffset_ = nextOffset;
            // Publish the complete header with release ordering. The creator
            // owns this block already; allocation never exposes a partial header.
        } while (!head.compare_exchange_weak(
            nextOffset, static_cast<unsigned long long>(headerOffset), ::cuda::std::memory_order_acq_rel, ::cuda::std::memory_order_acquire));
        return heap_ + headerOffset + blockOverhead();
    }

    // Null is a successful no-op. Other pointers must be live allocations from
    // this pool. Immediate duplicate frees return false; stale pointers after
    // reuse/reset and arbitrary interior pointers are caller errors.
    // Finish every payload access before releasing ownership of its block.
    __host__ __device__ bool deallocate(void* pointer) const noexcept {
        PROFILE();
        if (pointer == nullptr) {
            return true;
        }
        auto const address = reinterpret_cast<std::uintptr_t>(pointer);
        auto const heapAddress = reinterpret_cast<std::uintptr_t>(heap_);
        if (!valid() || address < heapAddress || address - heapAddress < blockOverhead() || address - heapAddress >= capacity_ ||
            address % minimumAlignment() != 0) {
            return false;
        }
        size_t const headerOffset = address - heapAddress - blockOverhead();
        auto* block = blockAt(headerOffset);
        if (block->sizeClass_ >= detail::kPoolSizeClassCount) {
            return false;
        }
        AtomicReference state(block->state_);
        unsigned long long expected = kAllocatedState_;
        if (!state.compare_exchange_strong(expected, kReleasingState_, ::cuda::std::memory_order_acq_rel, ::cuda::std::memory_order_relaxed)) {
            return false;
        }
        // Count before publishing free ownership. The intermediate state keeps
        // claimers out until the count exists, preventing a count underflow.
        AtomicReference(state_->freeCounts_[block->sizeClass_]).fetch_add(1, ::cuda::std::memory_order_relaxed);
        state.store(kFreeState_, ::cuda::std::memory_order_release);
        AtomicReference(state_->freeHints_[block->sizeClass_]).store(headerOffset, ::cuda::std::memory_order_release);
        return true;
    }

    // Invalidate all previous allocations. Every CPU/GPU operation using this
    // pool must have finished before reset, reinitialization, or destruction.
    __host__ __device__ void reset() const noexcept {
        PROFILE();
        if (valid()) {
            state_->reset();
        }
    }

private:
    using AtomicReference = ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>;
    static constexpr unsigned long long kFreeState_ = 0;
    static constexpr unsigned long long kAllocatedState_ = 1;
    static constexpr unsigned long long kReleasingState_ = 2;

    static_assert(sizeof(size_t) == sizeof(unsigned long long), "CudaMemoryPool requires 64-bit size_t");
    static_assert(AtomicReference::is_always_lock_free, "CudaMemoryPool requires lock-free 64-bit atomics");
    static_assert(alignof(detail::CudaMemoryPoolState) >= AtomicReference::required_alignment);
    static_assert(alignof(detail::CudaMemoryPoolBlock) >= AtomicReference::required_alignment);
    static_assert(sizeof(detail::CudaMemoryPoolState) % detail::kPoolAlignment == 0 && sizeof(detail::CudaMemoryPoolBlock) % detail::kPoolAlignment == 0);

    detail::CudaMemoryPoolState* state_ = nullptr;
    unsigned char* heap_ = nullptr;
    size_t capacity_ = 0;

    __host__ __device__ detail::CudaMemoryPoolBlock* blockAt(size_t offset) const noexcept {
        PROFILE();
        return reinterpret_cast<detail::CudaMemoryPoolBlock*>(heap_ + offset);
    }

    __host__ __device__ void* tryClaimBlock(unsigned long long offset, size_t sizeClass, size_t alignment) const noexcept {
        PROFILE();
        if (offset == detail::kPoolNullOffset) {
            return nullptr;
        }
        auto* block = blockAt(static_cast<size_t>(offset));
        auto* pointer = heap_ + static_cast<size_t>(offset) + blockOverhead();
        if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
            return nullptr;
        }
        unsigned long long expected = kFreeState_;
        if (!AtomicReference(block->state_)
                .compare_exchange_strong(expected, kAllocatedState_, ::cuda::std::memory_order_acquire, ::cuda::std::memory_order_relaxed)) {
            return nullptr;
        }
        AtomicReference(state_->freeCounts_[sizeClass]).fetch_sub(1, ::cuda::std::memory_order_relaxed);
        return pointer;
    }

    __host__ __device__ void* findFreeBlock(size_t sizeClass, size_t alignment) const noexcept {
        PROFILE();
        if (AtomicReference(state_->freeCounts_[sizeClass]).load(::cuda::std::memory_order_relaxed) == 0) {
            return nullptr;
        }
        auto const hint = AtomicReference(state_->freeHints_[sizeClass]).load(::cuda::std::memory_order_acquire);
        if (void* pointer = tryClaimBlock(hint, sizeClass, alignment)) {
            return pointer;
        }
        // A hint is only an optimization: overwriting it can never lose a block.
        // Immutable list membership avoids ABA and bounded queue capacity. A
        // contended/fragmented bin can require an O(number of blocks) scan.
        auto offset = AtomicReference(state_->heads_[sizeClass]).load(::cuda::std::memory_order_acquire);
        while (offset != detail::kPoolNullOffset) {
            if (void* pointer = tryClaimBlock(offset, sizeClass, alignment)) {
                return pointer;
            }
            offset = blockAt(static_cast<size_t>(offset))->nextOffset_;
        }
        return nullptr;
    }

    __host__ __device__ bool reserveBlock(size_t payloadBytes, size_t alignment, size_t& headerOffset) const noexcept {
        PROFILE();
        AtomicReference reserved(state_->bytesReserved_);
        auto offset = reserved.load(::cuda::std::memory_order_relaxed);
        auto const heapAddress = reinterpret_cast<std::uintptr_t>(heap_);
        size_t const alignmentMask = alignment - 1;
        for (;;) {
            // Check every addition before performing it, including alignment
            // of the absolute address. A failed attempt never advances the bump.
            if (offset > capacity_ || blockOverhead() > capacity_ - offset || heapAddress > ~std::uintptr_t{0} - offset - blockOverhead()) {
                return false;
            }
            auto const unalignedAddress = heapAddress + offset + blockOverhead();
            if (unalignedAddress > ~std::uintptr_t{0} - alignmentMask) {
                return false;
            }
            size_t const payloadOffset = ((unalignedAddress + alignmentMask) & ~static_cast<std::uintptr_t>(alignmentMask)) - heapAddress;
            if (payloadOffset > capacity_ || payloadBytes > capacity_ - payloadOffset) {
                return false;
            }
            auto const nextOffset = static_cast<unsigned long long>(payloadOffset + payloadBytes);
            if (reserved.compare_exchange_weak(offset, nextOffset, ::cuda::std::memory_order_relaxed, ::cuda::std::memory_order_relaxed)) {
                headerOffset = payloadOffset - blockOverhead();
                return true;
            }
        }
    }
};

}    // namespace cosmictiger
