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

#include <cosmictiger/cartesian_grid.hpp>

#include <stddef.h>
#include <vector>

namespace cosmictiger::gravity::detail {

struct P2mCudaCell {
    ParticleRange localParticles;
    double center[NDIM];
};

struct P2mCudaPositions {
    Coordinate const* coordinates[NDIM];
};

struct P2mCudaResult {
    float multipole[stf::coefficientCount(kMultipoleOrder)];
    float fullMultipole[symmetric::coefficientCount(kMultipoleOrder)];
    double secondMomentTrace;
};

// Inputs and returned results are host-owned. The implementation holds managed
// staging buffers until its HPX CUDA launch future reports GPU completion.
std::vector<P2mCudaResult> computeP2mCuda(std::vector<P2mCudaCell> const& cells, P2mCudaPositions positions, ParticleCount particleCount, double spacing,
    float particleMass, int deviceId, bool fullMoments = false);

}    // namespace cosmictiger::gravity::detail

#endif
