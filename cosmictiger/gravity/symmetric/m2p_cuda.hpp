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

namespace cosmictiger::gravity::symmetric {

/// One full Cartesian source-target evaluation at COSMICTIGER_MULTIPOLE_ORDER.
/// source addresses coefficientCount(order) coefficients in symmetric::index
/// layout. All pointers must be accessible from the selected CUDA device.
struct M2pCudaSample {
    double const* source = nullptr;
    double separation[NDIM]{};
    double sourceScale = 1.0;
    double softening = 0.0;
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

/// Batched full-moment M2P using the same host/device arithmetic as evaluateM2P.
/// softening=0 selects Newtonian gravity; softening>0 selects Plummer gravity.
/// The future becomes ready only after GPU completion. As with the STF API,
/// callers retain source, sample, output, and caller-owned stream lifetimes
/// until that future is ready. Output must not alias samples or sources.
/// sourceScale must be positive; r^2+softening^2 must be positive. An empty
/// request completes without inspecting pointers or submitting device work.
hpx::future<void> launchM2pCuda(M2pCudaRequest request, cudaStream_t stream = nullptr, int deviceId = 0);

}    // namespace cosmictiger::gravity::symmetric

#endif
