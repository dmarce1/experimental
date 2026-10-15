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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/spatial_types.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <cstdint>
#include <stddef.h>
#include <vector>

namespace cosmictiger::gravity {

struct DirectSolveField {
    double acceleration[NDIM]{};
    double potential = 0.0;
};

// Positions and softening use the unit periodic box. Each source is one owned
// particle; do not include ghost copies. Sink indices refer to this complete
// source array, and fields follow the order of those indices. Only a particle's
// primary self interaction is excluded; its periodic images remain.
struct DirectSolveRequest {
    Position const* sources = nullptr;
    size_t sourceCount = 0;
    size_t const* sinks = nullptr;
    size_t sinkCount = 0;
    DirectSolveField* fields = nullptr;
    double particleMass = 1.0;
    double softeningLength = 0.01;
    int ferrersOrder = 2;
    SofteningModel softeningModel = SofteningModel::ferrers;
};

// The default Ewald split and cutoffs converge below double precision for a
// nearest-image unit-box displacement. Custom values support convergence tests.
// Ferrers uses the PMMM point-mass potential gauge and replaces the primary
// interaction within its support. Plummer uses its own zero-mean periodic
// Green function, including softened images, reciprocal waves, and background.
struct DirectEwaldParameters {
    double alpha = 2.0;
    int imageCutoff = 4;
    int waveCutoff = 4;
};

void directSolveCpu(DirectSolveRequest const& request, DirectEwaldParameters const& parameters = {});

// Deterministically select ceil(fraction*sourceCount) distinct sink indices,
// sorted in source order. A zero fraction selects no sinks. This has a separate
// random stream from the initial-condition realization even for equal seeds.
std::vector<size_t> sampleDirectSinks(size_t sourceCount, double fraction, std::uint64_t seed = 42);

namespace detail {
    // Host-side argument validation does not dereference request pointers, so
    // the CUDA entry point can use device-only source, sink, and field arrays.
    void validateDirectSolveRequest(DirectSolveRequest const& request, DirectEwaldParameters const& parameters);
}    // namespace detail

#ifdef COSMICTIGER_ENABLE_CUDA
// Every request array must be CUDA-accessible and remain alive until the
// returned completion future is ready. No positions or particle state change.
hpx::future<void> launchDirectSolveCuda(DirectSolveRequest request, DirectEwaldParameters parameters = {}, cudaStream_t stream = nullptr, int deviceId = 0);
#endif

}    // namespace cosmictiger::gravity
