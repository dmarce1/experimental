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
 * @brief Host-owned growable array in CudaMemoryPool managed storage.
 *
 * Ported from the host use of the old tree code's device_vector: arrays such
 * as Barnes-Hut nodes and particles were built on the CPU in managed memory
 * and their pointers passed straight to kernel launches without a copy.
 * ManagedVector owns its storage like a std::vector and is used on the host.
 * Kernels receive a non-owning ManagedVectorView by value. The block-shared
 * scratch lists of the old CUDA tree walk are DeviceVector instead.
 *
 * The pool must be backed by managed memory (CudaMemoryPool) for the GPU to
 * reach the elements, and must outlive every vector allocated from it. Do not
 * touch a vector on the host, or change its size or storage, while a kernel
 * that uses its view is running.
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

// A pointer and size passed to kernels by value. It does not own the elements
// and is invalidated by any change to the size or storage of its vector.
template <typename Value>
class ManagedVectorView {
public:
    __host__ __device__ ManagedVectorView() noexcept {
        PROFILE();
    }

    __host__ __device__ ManagedVectorView(Value* data, size_t size) noexcept
      : data_(data)
      , size_(size) {
        PROFILE();
    }

    // A view of mutable elements converts to a view of const elements.
    template <typename Other>
        requires std::is_same_v<Value, Other const>
    __host__ __device__ ManagedVectorView(ManagedVectorView<Other> other) noexcept
      : data_(other.data())
      , size_(other.size()) {
        PROFILE();
    }

    __host__ __device__ size_t size() const noexcept {
        PROFILE();
        return size_;
    }

    __host__ __device__ bool empty() const noexcept {
        PROFILE();
        return size_ == 0;
    }

    __host__ __device__ Value* data() const noexcept {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value* begin() const noexcept {
        PROFILE();
        return data_;
    }

    __host__ __device__ Value* end() const noexcept {
        PROFILE();
        return data_ + size_;
    }

    __host__ __device__ Value& operator[](size_t index) const {
        PROFILE();
        expectIndex(index, size_);
        return data_[index];
    }

private:
    Value* data_ = nullptr;
    size_t size_ = 0;
};

// Copying allocates new storage from the same pool. Moving transfers the
// storage and leaves the source empty with the same pool. Elements are
// relocated by copying their bytes, so they must be trivially copyable.
// Elements added by the constructor or resize are value-initialized, as in
// std::vector. Allocation failure, including a vector without a pool, throws
// std::bad_alloc and leaves the vector unchanged.
template <typename Value>
class ManagedVector {
    static_assert(std::is_trivially_copyable_v<Value>, "ManagedVector relocates elements by copying their bytes");
    static_assert(std::is_trivially_destructible_v<Value>, "ManagedVector never runs element destructors");

public:
    ManagedVector() noexcept {
        PROFILE();
    }

    explicit ManagedVector(CudaMemoryPoolView pool) noexcept
      : pool_(pool) {
        PROFILE();
    }

    ManagedVector(CudaMemoryPoolView pool, size_t count)
      : pool_(pool) {
        PROFILE();
        resize(count);
    }

    ManagedVector(CudaMemoryPoolView pool, size_t count, Value const& value)
      : pool_(pool) {
        PROFILE();
        resize(count, value);
    }

    ManagedVector(ManagedVector const& other)
      : pool_(other.pool_) {
        PROFILE();
        assignFrom(other);
    }

    ManagedVector(ManagedVector&& other) noexcept
      : pool_(other.pool_)
      , data_(other.data_)
      , size_(other.size_)
      , capacity_(other.capacity_) {
        PROFILE();
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Keeps this vector's pool.
    ManagedVector& operator=(ManagedVector const& other) {
        PROFILE();
        if (this != &other) {
            assignFrom(other);
        }
        return *this;
    }

    // Takes the other vector's storage and pool.
    ManagedVector& operator=(ManagedVector&& other) noexcept {
        PROFILE();
        ManagedVector moved(static_cast<ManagedVector&&>(other));
        swap(moved);
        return *this;
    }

    ~ManagedVector() {
        PROFILE();
        deallocate(data_);
    }

    CudaMemoryPoolView pool() const noexcept {
        PROFILE();
        return pool_;
    }

    ManagedVectorView<Value> view() noexcept {
        PROFILE();
        return ManagedVectorView<Value>(data_, size_);
    }

    ManagedVectorView<Value const> view() const noexcept {
        PROFILE();
        return ManagedVectorView<Value const>(data_, size_);
    }

    size_t size() const noexcept {
        PROFILE();
        return size_;
    }

    size_t capacity() const noexcept {
        PROFILE();
        return capacity_;
    }

    bool empty() const noexcept {
        PROFILE();
        return size_ == 0;
    }

    Value* data() noexcept {
        PROFILE();
        return data_;
    }

    Value const* data() const noexcept {
        PROFILE();
        return data_;
    }

    Value* begin() noexcept {
        PROFILE();
        return data_;
    }

    Value const* begin() const noexcept {
        PROFILE();
        return data_;
    }

    Value* end() noexcept {
        PROFILE();
        return data_ + size_;
    }

    Value const* end() const noexcept {
        PROFILE();
        return data_ + size_;
    }

    Value& operator[](size_t index) {
        PROFILE();
        expectIndex(index, size_);
        return data_[index];
    }

    Value const& operator[](size_t index) const {
        PROFILE();
        expectIndex(index, size_);
        return data_[index];
    }

    Value& back() {
        PROFILE();
        expectPositive(size_);
        return data_[size_ - 1];
    }

    Value const& back() const {
        PROFILE();
        expectPositive(size_);
        return data_[size_ - 1];
    }

    // Ensures room for count elements without changing the size.
    void reserve(size_t count) {
        PROFILE();
        if (count > capacity_) {
            reallocate(count);
        }
    }

    // Shrinking keeps the capacity.
    void resize(size_t count) {
        PROFILE();
        resize(count, Value{});
    }

    void resize(size_t count, Value const& value) {
        PROFILE();
        // Growth frees the storage that value may refer to.
        Value const copy = value;
        if (count > capacity_) {
            reallocate(growthCount(count));
        }
        for (size_t index = size_; index < count; index++) {
            ::new (static_cast<void*>(data_ + index)) Value(copy);
        }
        size_ = count;
    }

    // value may refer to an element of this vector.
    void pushBack(Value const& value) {
        PROFILE();
        resize(size_ + 1, value);
    }

    // The vector must not be empty.
    void popBack() {
        PROFILE();
        expectPositive(size_);
        size_--;
    }

    // Keeps the capacity.
    void clear() noexcept {
        PROFILE();
        size_ = 0;
    }

    // Each vector keeps its storage's pool.
    void swap(ManagedVector& other) noexcept {
        PROFILE();
        exchange(pool_, other.pool_);
        exchange(data_, other.data_);
        exchange(size_, other.size_);
        exchange(capacity_, other.capacity_);
    }

private:
    static constexpr size_t kMinimumAllocationBytes_ = 1024;

    CudaMemoryPoolView pool_;
    Value* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

    template <typename Member>
    static void exchange(Member& left, Member& right) noexcept {
        PROFILE();
        Member const saved = left;
        left = right;
        right = saved;
    }

    // The smallest request is large enough that short arrays do not regrow on
    // every append.
    static size_t allocationBytes(size_t count) noexcept {
        PROFILE();
        size_t const bytes = count * sizeof(Value);
        return bytes < kMinimumAllocationBytes_ ? kMinimumAllocationBytes_ : bytes;
    }

    // Repeated appends grow the capacity geometrically.
    size_t growthCount(size_t count) const noexcept {
        PROFILE();
        return count < 2 * capacity_ ? 2 * capacity_ : count;
    }

    void deallocate(Value* storage) noexcept {
        PROFILE();
        [[maybe_unused]] bool const released = pool_.deallocate(storage);
        expect(released);
    }

    // Moves the elements to storage for at least count elements, or throws
    // with the vector unchanged.
    void reallocate(size_t count) {
        PROFILE();
        if (count > ~size_t{0} / sizeof(Value)) {
            throw std::bad_alloc();
        }
        auto* const storage = static_cast<Value*>(pool_.allocate(allocationBytes(count), alignof(Value)));
        if (storage == nullptr) {
            throw std::bad_alloc();
        }
        if (size_ != 0) {
            ::memcpy(static_cast<void*>(storage), static_cast<void const*>(data_), size_ * sizeof(Value));
        }
        deallocate(data_);
        data_ = storage;
        // The pool rounds each request up to a power-of-two size class.
        capacity_ = CudaMemoryPoolView::sizeClassCapacity(allocationBytes(count)) / sizeof(Value);
    }

    void assignFrom(ManagedVector const& other) {
        PROFILE();
        if (other.size_ > capacity_) {
            reallocate(other.size_);
        }
        if (other.size_ != 0) {
            ::memcpy(static_cast<void*>(data_), static_cast<void const*>(other.data_), other.size_ * sizeof(Value));
        }
        size_ = other.size_;
    }
};

}    // namespace cosmictiger::cuda
