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
 * @brief Local FMM solve of the PMMM near region over per-cell kd-trees.
 *
 * The near region of an interior sink cell is the (2w+1)^NDIM cells excluded
 * from the PMMM far field, chosen exactly as the direct near field chooses
 * them. Its kd-tree is walked against the trees of those cells, descending
 * sink and source trees together:
 *
 * - cc (M2L) when R > max((rs+ro)/theta, rs+ro+h);
 * - for two leaves, pc (M2P) when R > max(rs+ro/theta, rs+ro+h) or every sink
 *   particle lies farther than max(ro/theta, ro+h) from the source center;
 *   cp (P2L) is the mirror test. When both pass, the one looping over fewer
 *   particles is used, and equal counts fall back to pp;
 * - pp (direct softened P2P) for the remaining leaf pairs.
 *
 * h is the Ferrers support radius, so every multipole interaction is between
 * particles farther apart than h, where the softened kernel is Newtonian.
 * Separations and box distances are exact integer fixed-point differences,
 * and every test is symmetric in the two nodes, so the pair (A, B) receives
 * mirrored interactions (cc/cc, pc/cp, pp/pp) from both sides. With the
 * generated operators this makes the forces antisymmetric up to rounding when
 * every particle is a sink.
 *
 * Fields are those of unit-mass sources, like the raw direct near lanes; the
 * caller applies the particle mass. The trees must therefore be built with
 * KdTreeParameters::particleMass = 1.
 */
#pragma once

#include <array>
#include <cstdint>
#include <stddef.h>
#include <vector>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/gravity/tree/kd_tree.hpp>
#include <cosmictiger/particle/particle_storage.hpp>

namespace cosmictiger::gravity::tree {

enum class TreeInteractionType : std::uint8_t { m2l, p2l, m2p, p2p };

// One accepted node pair: sink node, source node, and the source cell's
// offset from the sink cell in cells (the periodic image actually used).
struct TreeInteraction {
    size_t sinkNode;
    size_t sourceNode;
    std::array<std::int64_t, NDIM> cellOffset;
    TreeInteractionType type;
};

struct TreeNearFieldRequest {
    CartesianGrid const* grid = nullptr;
    ParticleStorage const* particles = nullptr;
    // Built over this grid and storage with unit particle mass. Rung-threshold
    // trees use their at-or-above tree as the sink tree; position trees use
    // their single tree. Every tree of a near cell is a source.
    CellKdTrees const* trees = nullptr;
    SofteningLength softening{};
    int ferrersOrder = 2;
    int fmmOrder = kDefaultFmmOrder;
    int ghostWidth = 1;
    double openingAngle = kDefaultOpeningAngle;
    // Sinks are owned particles of interior cells with rung >= activeRung.
    std::uint8_t activeRung = 0;
    bool computePotential = false;
    // Outputs indexed by particle, accumulated for every evaluated sink and
    // untouched elsewhere. The potential may be null without computePotential.
    float* accelerations[NDIM]{};
    float* potentials = nullptr;
    // Optional diagnostics, filled in sink-cell order.
    std::vector<TreeInteraction>* interactions = nullptr;
};

// Evaluate the near field of every sink. Cells run as HPX tasks; all tasks
// finish before return or rethrow. Invalid requests throw
// std::invalid_argument. No FpeGuard may span this call.
void evaluateTreeNearField(TreeNearFieldRequest const& request);

}    // namespace cosmictiger::gravity::tree
