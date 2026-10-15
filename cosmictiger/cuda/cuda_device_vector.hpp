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

/** @file
 * @brief Block-cooperative growable array in CudaMemoryPool storage.
 *
 * Ported from the old tree code's device_vector, which held the per-block
 * checklists and interaction lists of the CUDA tree walk. The object holds only
 * a pointer, size, and capacity, so in a kernel it must live where every thread
 * of the block sees the same members: shared memory for scratch lists, or
 * managed memory for a vector that the host initializes, one block fills, and
 * the host reads after the kernel. Host-built arrays passed to kernels are
 * ManagedVector instead (cuda_managed_vector.hpp). Every mutating method
 * is a collective: all threads of the block call it convergently with the same
 * arguments, and the block's first thread does the bookkeeping between
 * barriers. Host callers act as a block of one thread.
 *
 * The type has trivial construction and destruction so that it can be declared
 * __shared__. Call initialize() before use and release() before the storage
 * goes away; there is no destructor to return the allocation to the pool.
 * Growth that exhausts the pool returns false on every thread and leaves the
 * vector unchanged, so the caller decides how to recover.
 */
#pragma once

#include <cstring>
#include <new>
#include <stddef.h>
#include <type_traits>

#include <cosmictiger/cuda/cuda_memory_pool_view.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::cuda {

namespace detail {

    // The thread's rank within its block, flattened over every block dimension.
    __host__ __device__ inline unsigned blockThreadRank() {
        PROFILE();
#ifdef __CUDA_ARCH__
        return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
        return 0;
#endif
    }

    __host__ __device__ inline unsigned blockThreadCount() {
        PROFILE();
#ifdef __CUDA_ARCH__
        return blockDim.x * blockDim.y * blockDim.z;
#else
        return 1;
#endif
    }

    __host__ __device__ inline void synchronizeBlock() {
        PROFILE();
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }

}    // namespace detail

// Elements are relocated by copying their bytes when the storage grows and are
// never destroyed, so they must be trivially copyable and destructible.
// Elements added by resize are default-initialized: trivial types are left
// indeterminate. Reads (size, operator[], and so on) are not collectives;
// a barrier must separate them from any later collective that changes the
// vector, and every collective ends with one for the reads that follow it.
template <typename Value>
class DeviceVector {
    static_assert(std::is_trivially_copyable_v<Value>, "DeviceVector relocates elements by copying their bytes");
    static_assert(std::is_trivially_destructible_v<Value>, "DeviceVector never runs element destructors");

public:
    DeviceVector() = default;
    DeviceVector(DeviceVector const&) = delete;
    DeviceVector& operator=(DeviceVector const&) = delete;

    // Collective. Begins empty with no storage; nothing is allocated until the
    // first growth. Do not call it again before release().
    __host__ __device__ void initialize(CudaMemoryPoolView pool) {
        PROFILE();
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            storePool(pool);
            data_ = nullptr;
            staging_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
        detail::synchronizeBlock();
    }

    // Collective. Returns the storage to the pool and leaves the vector empty,
    // ready for another initialize().
    __host__ __device__ void release() {
        PROFILE();
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            [[maybe_unused]] bool const released = loadPool().deallocate(data_);
            expect(released);
            data_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
        detail::synchronizeBlock();
    }

    __host__ __device__ size_t size() const {
        PROFILE();
        return size_;
    }

    __host__ __device__ size_t capacity() const {
        PROFILE();
        return capacity_;
    }

    __host__ __device__ bool empty() const {
        PROFILE();
        return size_ == 0;
    }

    __host__ __device__ Value* data() {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value const* data() const {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value* begin() {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value const* begin() const {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value* end() {
        PROFILE();
        return data_ + size_;
    }

    __host__ __device__ Value const* end() const {
        PROFILE();
        return data_ + size_;
    }

    __host__ __device__ Value& operator[](size_t index) {
        PROFILE();
        expectIndex(index, size_);
        return data_[index];
    }

    __host__ __device__ Value const& operator[](size_t index) const {
        PROFILE();
        expectIndex(index, size_);
        return data_[index];
    }

    __host__ __device__ Value& back() {
        PROFILE();
        expectPositive(size_);
        return data_[size_ - 1];
    }

    __host__ __device__ Value const& back() const {
        PROFILE();
        expectPositive(size_);
        return data_[size_ - 1];
    }

    // Collective. Ensures room for count elements without changing the size.
    // Returns false, leaving the vector unchanged, if the pool cannot supply it.
    [[nodiscard]] __host__ __device__ bool reserve(size_t count) {
        PROFILE();
        detail::synchronizeBlock();
        bool const grown = count <= capacity_ || grow(count);
        detail::synchronizeBlock();
        return grown;
    }

    // Collective. Shrinking keeps the capacity. Growth that the pool cannot
    // supply returns false and leaves the vector unchanged.
    [[nodiscard]] __host__ __device__ bool resize(size_t count) {
        PROFILE();
        detail::synchronizeBlock();
        size_t const oldSize = size_;
        if (count > capacity_ && !grow(count)) {
            detail::synchronizeBlock();
            return false;
        }
        unsigned const rank = detail::blockThreadRank();
        unsigned const threadCount = detail::blockThreadCount();
        for (size_t index = oldSize + rank; index < count; index += threadCount) {
            ::new (static_cast<void*>(data_ + index)) Value;
        }
        // Every thread has read the old size before the first thread changes it.
        detail::synchronizeBlock();
        if (rank == 0) {
            size_ = count;
        }
        detail::synchronizeBlock();
        return true;
    }

    // Collective. Appends the first thread's value; the other threads' values
    // are ignored. value may refer to an element of this vector.
    [[nodiscard]] __host__ __device__ bool pushBack(Value const& value) {
        PROFILE();
        // Growth frees the storage that value may refer to.
        Value const copy = value;
        detail::synchronizeBlock();
        if (size_ == capacity_ && !grow(size_ + 1)) {
            detail::synchronizeBlock();
            return false;
        }
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            data_[size_] = copy;
            size_++;
        }
        detail::synchronizeBlock();
        return true;
    }

    // Collective. The vector must not be empty.
    __host__ __device__ void popBack() {
        PROFILE();
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            expectPositive(size_);
            size_--;
        }
        detail::synchronizeBlock();
    }

    // Collective. Keeps the capacity.
    __host__ __device__ void clear() {
        PROFILE();
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            size_ = 0;
        }
        detail::synchronizeBlock();
    }

    // Collective over both vectors, which must belong to the same block. Each
    // vector keeps its storage's pool.
    __host__ __device__ void swap(DeviceVector& other) {
        PROFILE();
        detail::synchronizeBlock();
        if (detail::blockThreadRank() == 0) {
            CudaMemoryPoolView const pool = loadPool();
            storePool(other.loadPool());
            other.storePool(pool);
            exchange(data_, other.data_);
            exchange(size_, other.size_);
            exchange(capacity_, other.capacity_);
        }
        detail::synchronizeBlock();
    }

private:
    static constexpr size_t kMinimumAllocationBytes_ = 1024;

    // CudaMemoryPoolView's default constructor is not empty, and a member of
    // that type would keep this class from being declared __shared__. The view
    // is trivially copyable, so its bytes are kept instead.
    static_assert(std::is_trivially_copyable_v<CudaMemoryPoolView>);
    alignas(CudaMemoryPoolView) unsigned char poolBytes_[sizeof(CudaMemoryPoolView)];
    Value* data_;
    // Publishes the first thread's new allocation, or null on failure, to the block.
    Value* staging_;
    size_t size_;
    size_t capacity_;

    __host__ __device__ CudaMemoryPoolView loadPool() const {
        PROFILE();
        CudaMemoryPoolView pool;
        ::memcpy(static_cast<void*>(&pool), poolBytes_, sizeof(pool));
        return pool;
    }

    __host__ __device__ void storePool(CudaMemoryPoolView const& pool) {
        PROFILE();
        ::memcpy(poolBytes_, static_cast<void const*>(&pool), sizeof(pool));
    }

    // The smallest request is large enough that short lists do not regrow on
    // every append.
    __host__ __device__ static size_t allocationBytes(size_t count) {
        PROFILE();
        size_t const bytes = count * sizeof(Value);
        return bytes < kMinimumAllocationBytes_ ? kMinimumAllocationBytes_ : bytes;
    }

    template <typename Member>
    __host__ __device__ static void exchange(Member& left, Member& right) {
        PROFILE();
        Member const saved = left;
        left = right;
        right = saved;
    }

    // Called collectively between barriers with count > capacity_. Moves the
    // elements to storage for at least count elements and returns true on every
    // thread, or returns false on every thread with the vector unchanged.
    __host__ __device__ bool grow(size_t count) {
        PROFILE();
        unsigned const rank = detail::blockThreadRank();
        unsigned const threadCount = detail::blockThreadCount();
        if (rank == 0) {
            staging_ = nullptr;
            // Refuse counts whose byte size cannot be represented.
            if (count <= ~size_t{0} / sizeof(Value)) {
                staging_ = static_cast<Value*>(loadPool().allocate(allocationBytes(count), alignof(Value)));
            }
        }
        detail::synchronizeBlock();
        Value* const storage = staging_;
        if (storage == nullptr) {
            return false;
        }
        for (size_t index = rank; index < size_; index += threadCount) {
            storage[index] = data_[index];
        }
        // Every thread has finished copying and has read staging_ and the old
        // members before the first thread replaces them.
        detail::synchronizeBlock();
        if (rank == 0) {
            [[maybe_unused]] bool const released = loadPool().deallocate(data_);
            expect(released);
            data_ = storage;
            // The pool rounds each request up to a power-of-two size class.
            capacity_ = CudaMemoryPoolView::sizeClassCapacity(allocationBytes(count)) / sizeof(Value);
        }
        detail::synchronizeBlock();
        return true;
    }
};

}    // namespace cosmictiger::cuda
