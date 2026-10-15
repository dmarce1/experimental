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

#include <cosmictiger/evolution/evolution_state.hpp>
#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/particle/particle_silo.hpp>

#include <string>

namespace cosmictiger {

struct EvolutionCheckpointMetadata {
    GlobalConstantsData constants;
    EvolutionState state;
    InitialConditionsSiloMetadata initial;
};

// Commit owned particles, their unchanged integration momenta, the complete
// restart state, and original initial-condition metadata in one atomic file
// replacement. A distributed checkpoint manifest must be committed only after
// every locality's file has completed. Existing particle Silo readers remain
// able to read the particle data; readParticleSilo restores the saved fields.
void writeEvolutionCheckpoint(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid, GlobalConstantsData const& constants,
    EvolutionState const& state, InitialConditionsSiloMetadata const& initial, ParticleSiloOutput const* output = nullptr);

// Read and validate the complete metadata before publishing run constants or
// modifying particle storage. Current scale factors can exceed one or lie on
// the contracting branch. The IC metadata continues to describe the original
// starting epoch and lattice; the gravity mesh and ghost width are retained
// independently. Operational format/state checks remain active in Release.
EvolutionCheckpointMetadata readEvolutionCheckpointMetadata(std::string const& filename);

}    // namespace cosmictiger
