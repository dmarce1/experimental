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
 * @brief Batched cell-level FMM interactions: M2L, P2L, and M2P.
 *
 * These are the CosmicTiger ports of the old tree-code cc, cp, and pc kernels,
 * rebuilt on the generated STF operators so that they share the coefficient
 * layout, normalization, and sign conventions of the cell multipoles and
 * expansions. They evaluate the exterior Newtonian kernel with G=1 and no
 * softening; callers decide which cell pairs are sufficiently separated.
 *
 * The runtime FMM order p selects source rank p-1 and local rank p, as in the
 * PMMM path. Multipole sources use the CellMultipole layout (raw STF moments
 * divided by spacing^rank about the source cell center). Expansion targets use
 * the CellExpansion layout (potential derivatives multiplied by spacing^rank
 * about the target cell center). Only the ranks selected by p are read or
 * accumulated; higher coefficients and the appended Laplacian are untouched.
 *
 * Every separation is target cell center minus source cell center in
 * unit-box coordinates, for the periodic image chosen by the caller. Particle
 * offsets are taken from a cell center with the 32-bit fixed-point minimum
 * image, so each particle must lie within half a box of its own cell center;
 * the cell-to-cell separation is never wrapped.
 *
 * The CPU and CUDA entry points accept the same requests. Targets of one
 * request must not alias: distinct expansion pointers, and disjoint sink
 * ranges for M2P. Sources are read-only and may be shared among targets.
 */
#pragma once

#include <stddef.h>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/spatial_types.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

namespace cosmictiger::gravity::fmm {

// A cell multipole seen from one target. The pointer addresses the first
// coefficient of a CellMultipole; ranks above p-1 are ignored.
struct FmmMultipoleSource {
    float const* multipole = nullptr;
    double separation[NDIM]{};
};

// The particles of one source cell. Offsets are measured from center, which
// is also the reference of separation (target center minus this center).
struct FmmParticleSource {
    ParticleRange particles;
    Position center{};
    double separation[NDIM]{};
};

// One cell expansion receiving sources [sourceBegin, sourceEnd). The pointer
// addresses the first coefficient of a CellExpansion.
struct FmmExpansionTarget {
    float* expansion = nullptr;
    size_t sourceBegin = 0;
    size_t sourceEnd = 0;
};

// The sink particles of one cell receiving sources [sourceBegin, sourceEnd).
// Offsets are measured from center, the reference of the source separations.
struct FmmParticleTarget {
    ParticleRange sinks;
    Position center{};
    size_t sourceBegin = 0;
    size_t sourceEnd = 0;
};

// cc: add source cell multipoles to target cell expansions.
struct FmmM2lRequest {
    FmmExpansionTarget const* targets = nullptr;
    size_t targetCount = 0;
    FmmMultipoleSource const* sources = nullptr;
    size_t sourceCount = 0;
    double spacing = 0.0;
    int fmmOrder = kDefaultFmmOrder;
};

// cp: add source cell particles, each of mass particleMass, to target cell
// expansions. Every particle in a source range contributes.
struct FmmP2lRequest {
    FmmExpansionTarget const* targets = nullptr;
    size_t targetCount = 0;
    FmmParticleSource const* sources = nullptr;
    size_t sourceCount = 0;
    Coordinate const* positions[NDIM]{};
    float particleMass = 0.0f;
    double spacing = 0.0;
    int fmmOrder = kDefaultFmmOrder;
};

// pc: add the field of source cell multipoles at target cell particles. Output
// arrays are indexed by particle index, like DirectInteractionParticles, and
// are accumulated rather than overwritten. Potentials are optional. Every sink
// in a target range is evaluated; select owned active sinks in the request.
struct FmmM2pRequest {
    FmmParticleTarget const* targets = nullptr;
    size_t targetCount = 0;
    FmmMultipoleSource const* sources = nullptr;
    size_t sourceCount = 0;
    Coordinate const* positions[NDIM]{};
    float* accelerations[NDIM]{};
    float* potentials = nullptr;
    double spacing = 0.0;
    int fmmOrder = kDefaultFmmOrder;
};

// The CPU entry points evaluate in double precision and add each target's
// complete sum to its float outputs once, in source order. Targets are split
// among HPX tasks; all tasks finish before return or rethrow, and a failure can
// leave other targets partially updated. Requests with an unsupported order,
// a nonpositive spacing, or missing buffers for a nonempty batch throw
// std::invalid_argument. Zero separations violate a numerical contract.
void addM2lCpu(FmmM2lRequest const& request);
void addP2lCpu(FmmP2lRequest const& request);
void addM2pCpu(FmmM2pRequest const& request);

#ifdef COSMICTIGER_ENABLE_CUDA
// The CUDA entry points evaluate in single precision with the same generated
// operators. Every block is one warp and owns one target. M2L and P2L reduce
// the lanes in a fixed order. M2P stages source multipoles in shared memory and
// gives each lane every 32nd sink; targets of more than kBucketSize sinks take
// additional passes. Every buffer and table must be CUDA-accessible and remain alive
// until the returned future is ready, which happens only after the GPU work
// completes. Invalid requests and CUDA failures are reported through the
// future. An empty batch completes without touching pointers or the device.
hpx::future<void> launchM2lCuda(FmmM2lRequest request, cudaStream_t stream = nullptr, int deviceId = 0);
hpx::future<void> launchP2lCuda(FmmP2lRequest request, cudaStream_t stream = nullptr, int deviceId = 0);
hpx::future<void> launchM2pCuda(FmmM2pRequest request, cudaStream_t stream = nullptr, int deviceId = 0);
#endif

}    // namespace cosmictiger::gravity::fmm
