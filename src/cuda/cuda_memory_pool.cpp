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

#include <cosmictiger/cuda/cuda_memory_pool.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/include/async.hpp>

#include <stdexcept>

namespace cosmictiger {

CudaMemoryPool::CudaMemoryPool(size_t heapCapacity) {
    PROFILE();
    if (heapCapacity == 0 || heapCapacity > ~size_t{0} - CudaMemoryPoolView::metadataBytes()) {
        throw std::invalid_argument("CudaMemoryPool: invalid heap capacity");
    }
    int const deviceId = cuda::device().get();
    if (cuda::deviceAttribute(cudaDevAttrConcurrentManagedAccess, deviceId).get() == 0) {
        throw std::runtime_error("CudaMemoryPool requires concurrent managed access for system-scope atomics");
    }
    size_t const bufferBytes = CudaMemoryPoolView::metadataBytes() + heapCapacity;
    buffer_ = cuda::mallocManaged(bufferBytes).get();
    view_ = CudaMemoryPoolView::initialize(buffer_, bufferBytes);
}

CudaMemoryPool::~CudaMemoryPool() {
    PROFILE();
    // The active HPX runtime and exclusive ownership must outlive cleanup.
    // Explicit release() reports errors; destruction cannot throw.
    try {
        release().get();
    } catch (...) {
    }
}

hpx::future<std::unique_ptr<CudaMemoryPool>> CudaMemoryPool::create(size_t heapCapacity) {
    PROFILE();
    return hpx::async(hpx::annotated_function(
        [heapCapacity] {
            PROFILE();
            return std::make_unique<CudaMemoryPool>(heapCapacity);
        },
        "CudaMemoryPool::create"));
}

hpx::future<void> CudaMemoryPool::release() {
    PROFILE();
    void* const buffer = buffer_;
    buffer_ = nullptr;
    view_ = CudaMemoryPoolView{};
    return cuda::free(buffer);
}

}    // namespace cosmictiger
