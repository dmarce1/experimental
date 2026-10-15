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

#include <hpx/naming_base/id_type.hpp>

#include <cstdint>
#include <vector>

namespace cosmictiger {

// Collective: invoke once on every locality from an HPX task, using the same
// ordered locality list, grid topology, and globally unique operationId.
// The grid's locality index identifies the caller in that list. Logical
// localities may share a process if each has its own storage and grid.
// Requires exclusive access to both objects and completed GPU work.
// Native transfer buffers require identical particle representation/ABI on
// all participating processes.
//
// Every live particle with isGhost == false outside ghost localParticles ranges is
// owned, even before interior ranges have first been populated. Ghost copies
// are excluded from redistribution and discarded; all ghost ranges are
// invalidated on success for a later ghost refresh. Ghosts are never sorted
// or exchanged. Positions, velocities, and rungs travel together.
//
// Direct transfers place owned particles on their cell's locality. A local
// counting partition then sets both interior ranges, including empty cells,
// in x-fastest cell order. Bounded HPX chunks count/pack outgoing particles and
// count/scatter received particles using private histograms and disjoint output
// intervals. All chunk tasks finish before the next collective or storage swap.
// Input order is retained within each source's cell contribution before rung
// sorting; order within a rung is otherwise unspecified.
// Validation/allocation errors before commit preserve storage and ranges on
// every participant. A post-commit sort allocation or scheduling failure is
// reported to every participant; replacement remains committed and rung
// metadata may be incomplete. Runtime/process/network failures are not
// recoverable.
// Successful return follows a global completion barrier.
void loadBalanceParticles(ParticleStorage& particles, CartesianGrid& grid, std::vector<hpx::id_type> const& localities, std::uint64_t operationId);

// Uses localParticleStorage() and HPX's ordered list of all localities.
void loadBalanceParticles(CartesianGrid& grid, std::uint64_t operationId);

}    // namespace cosmictiger
