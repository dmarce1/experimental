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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <cstdint>
#include <stddef.h>

namespace cosmictiger {

enum class DriftExecution { cpu, cuda };

struct DriftParticles {
    Coordinate* positions[NDIM]{};
    float const* momenta[NDIM]{};
    Rung const* flags = nullptr;
};

// tau = H0 integral(dt/a), q = x/L, stored momentum p = a dq/dtau.
// factor = integral(d tau/a); drift holds p constant and advances q by p*factor.
// The half-open range must exclude same-owner periodic ghost copies, whose
// flags do not identify them as ghosts. Remote ghost flags are also skipped.
struct DriftRequest {
    ParticleRange range;
    DriftParticles particles;
    double factor = 0.0;
    // Set only after ownership ranges establish a ghost-free particle prefix.
    bool ownedOnly = false;
};

namespace detail {

    inline void validateDriftRequest(DriftRequest const& request) {
        PROFILE();
        expect(request.range.begin <= request.range.end);
        if (request.range.begin == request.range.end || request.factor == 0.0) {
            return;
        }
        expectNonZero(request.particles.flags);
        for (size_t axis = 0; axis < NDIM; axis++) {
            expectNonZero(request.particles.positions[axis]);
            expectNonZero(request.particles.momenta[axis]);
        }
    }

}    // namespace detail

// Positions alone change. All rungs drift over the same interval; momentum,
// rung flags and retained gravity fields remain unchanged. Requires exclusive
// access and completed prior GPU work. Raw arrays cover at least range.end.
DriftRequest makeDriftRequest(ParticleStorage& particles, ParticleRange range, double factor);
void driftParticlesCpu(DriftRequest const& request);

#ifdef COSMICTIGER_ENABLE_CUDA
// Arrays must be device-accessible. Keep them and any external stream alive
// until the future is ready; do not resize storage or access positions before
// completion. Dependent submissions must be ordered through their futures.
hpx::future<void> launchDriftParticlesCuda(DriftRequest request, cudaStream_t stream = nullptr, int deviceId = 0);
#endif

// Each call selects one processor for the entire update; CPU and CUDA have
// separate update implementations.
// Local driver: discard the appended ghost suffix, then drift every owned
// particle on the selected processor. Returns after CPU/GPU completion.
// Requires exclusive access and completed prior GPU work. Interior ranges
// must partition the owned prefix. After movement, cell membership, moments,
// expansions and retained gravity fields are stale; call loadBalanceParticles
// and refresh ghosts/fields before using them. No migration occurs here.
void driftParticles(ParticleStorage& particles, CartesianGrid& grid, double factor, DriftExecution execution = DriftExecution::cpu, int deviceId = 0);

}    // namespace cosmictiger
