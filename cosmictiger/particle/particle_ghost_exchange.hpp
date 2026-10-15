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

// Collective from HPX tasks: use the same ordered locality list, grid topology,
// and fresh operationId on every participant. Requires exclusive access and
// completed GPU work. Interior ranges must partition the owned prefix in
// x-fastest order, as produced by loadBalanceParticles; particles must still
// belong to those cells. Existing ghosts form a trailing suffix; refresh and
// clear discard that suffix without consuming its previous range metadata.
//
// Refresh all ghost cells, including periodic faces, edges, and corners.
// Append separate copies per cell in stored x-fastest order, preserving the
// owned prefix and interior ranges. ownerParticles indexes the owner's
// interior range; localParticles indexes the appended copies. Only copies
// owned by another locality set isGhost. Empty cells receive empty ranges.
// Packing and suffix population use bounded HPX particle chunks, retaining
// peer request order and splitting heavily populated cells across tasks.
// Storage grows before copying; all tasks finish before range publication,
// rung sorting, and collective error agreement. Small/one-worker work is inline.
//
// Allocation errors before commit preserve live particle values, size, and
// grid ranges. Validation checks occur as data is consumed; a failed check
// during ghost population may leave partial changes. reserve may change
// capacity and field pointers even on failure; reacquire pointers afterward.
// Native transfer buffers require identical particle ABI. Runtime/network failures need runtime-level
// recovery. Successful return follows a global completion barrier.
void exchangeGhostParticles(ParticleStorage& particles, CartesianGrid& grid, std::vector<hpx::id_type> const& localities, std::uint64_t operationId);
void exchangeGhostParticles(CartesianGrid& grid, std::uint64_t operationId);

// Local operation: truncate the ghost suffix, then reset every
// ghost range. Preserve owned particles, interior ranges, and capacity. Safe
// to repeat. Requires exclusive access and completed GPU work; positions may
// have moved since the exchange. Same-owner periodic copies are also removed.
void clearGhostParticles(ParticleStorage& particles, CartesianGrid& grid);
void clearGhostParticles(CartesianGrid& grid);

}    // namespace cosmictiger
