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

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/dimensions.hpp>

#include <stddef.h>

namespace cosmictiger::gravity::stf {

// One source-target evaluation. The source points to at least
// (COSMICTIGER_MULTIPOLE_ORDER + 1)^2 contiguous STF coefficients in the
// layout of Multipole<double, COSMICTIGER_MULTIPOLE_ORDER>.
// The source pointers and the pointer table itself must be accessible from the
// selected CUDA device. Pointer accessibility is a caller precondition.
struct M2pCudaSample {
    double const* source = nullptr;
    double separation[NDIM]{};
    double sourceScale = 1.0;
};

struct M2pCudaField {
    double potential = 0.0;
    double acceleration[NDIM]{};
};

struct M2pCudaRequest {
    M2pCudaSample const* samples = nullptr;
    M2pCudaField* fields = nullptr;
    size_t count = 0;
};

// Evaluate one independent field per sample at the configured multipole order,
// using the same generated STF M2P polynomial as evaluateM2P. Outputs are
// overwritten, not accumulated. Output buffers must not alias sample or source
// buffers. All buffers must remain GPU-accessible and alive through completion.
// Each sourceScale and squared separation must be positive. The returned future is
// ready only after GPU work completes and output is visible to the host; keep
// any caller-owned stream alive until then. Pointer accessibility is not
// checked. An empty batch completes without touching pointers or the device.
hpx::future<void> launchM2pCuda(M2pCudaRequest request, bool centered = false, cudaStream_t stream = nullptr, int deviceId = 0);

}    // namespace cosmictiger::gravity::stf

#endif
