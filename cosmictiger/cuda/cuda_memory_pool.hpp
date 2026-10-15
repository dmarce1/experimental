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

#include <cosmictiger/cuda/cuda_memory_pool_view.hpp>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/profiler.hpp>

#include <memory>
#include <stddef.h>

namespace cosmictiger {

// Managed backing storage owned by an active HPX runtime. All CPU and GPU users
// must finish before resetting or releasing the owner. The allocator view itself
// has no HPX dependency and may be passed to CUDA kernels by value.
class CudaMemoryPool {
public:
    // Synchronous convenience; use create() to obtain allocation as a future.
    explicit CudaMemoryPool(size_t heapCapacity);
    CudaMemoryPool(CudaMemoryPool const&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool const&) = delete;
    ~CudaMemoryPool();

    static hpx::future<std::unique_ptr<CudaMemoryPool>> create(size_t heapCapacity);

    CudaMemoryPoolView view() const noexcept {
        PROFILE();
        return view_;
    }

    void reset() noexcept {
        PROFILE();
        view_.reset();
    }

    // Exclusive access is required. Invalidates the view immediately and returns
    // a future reporting completion and any CUDA error. May be called repeatedly.
    hpx::future<void> release();

private:
    void* buffer_ = nullptr;
    CudaMemoryPoolView view_;
};

}    // namespace cosmictiger
