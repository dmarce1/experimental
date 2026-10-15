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

#ifdef COSMICTIGER_ENABLE_CUDA

#include <cuda_runtime_api.h>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/future.hpp>

#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger::cuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t status, char const* operation);

    cudaError_t code() const noexcept;

private:
    cudaError_t code_;
};

// Call these interfaces while the HPX runtime is active. Runtime operations run
// on HPX's I/O pool; errors become future exceptions, with allocation failures
// represented by std::bad_alloc. Device IDs are explicit and default to zero,
// so CUDA's OS-thread-local device selection cannot follow a migrating HPX task.
hpx::future<int> deviceCount();
hpx::future<int> device();
hpx::future<int> deviceAttribute(cudaDeviceAttr attribute, int deviceId);
hpx::future<cudaDeviceProp> deviceProperties(int deviceId);
hpx::future<cudaPointerAttributes> pointerAttributes(void const* pointer);

hpx::future<void*> malloc(size_t bytes, int deviceId = 0);
hpx::future<void*> mallocManaged(size_t bytes, unsigned flags = cudaMemAttachGlobal, int deviceId = 0);
hpx::future<void> free(void* pointer, int deviceId = 0);
hpx::future<cudaStream_t> streamCreate(unsigned flags = cudaStreamNonBlocking, int deviceId = 0);
hpx::future<void> streamDestroy(cudaStream_t stream, int deviceId = 0);

// A null stream selects a shared explicit stream for the requested device,
// independent of CUDA's legacy and per-thread default-stream modes. Chain
// futures to order dependent submissions; independent I/O tasks can submit in
// either order. Keep both buffers and any external stream alive until the future is
// ready. Completion means the copy has finished, including the destination's
// visibility to the CPU for device-to-host copies.
hpx::future<void> memcpy(void* destination, void const* source, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream = nullptr, int deviceId = 0);

struct ManagedMemoryRange {
    void const* pointer;
    size_t bytes;
};

// Batch managed-memory migration through one stream and completion future.
// destination is a device ordinal or cudaCpuDeviceId. Keep the allocations
// alive and do not access them until completion. Empty ranges are ignored.
hpx::future<void> prefetch(std::vector<ManagedMemoryRange> ranges, int destination, cudaStream_t stream = nullptr, int deviceId = 0);

// The operation must only enqueue kernels on the supplied stream and must not
// yield or call HPX. Its captures remain alive through GPU completion. Launch
// status and stream completion both become exceptions on the returned future.
// Keep an external stream alive until completion. Application continuations are
// completed from an HPX task, outside CUDA's restricted callback context.
hpx::future<void> launch(hpx::move_only_function<void(cudaStream_t)> operation, cudaStream_t stream = nullptr, int deviceId = 0);
hpx::future<void> synchronize(cudaStream_t stream = nullptr, int deviceId = 0);

}    // namespace cosmictiger::cuda

#endif
