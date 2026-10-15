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
 * @brief Per-cell kd-trees for local FMM solves.
 *
 * Every stored mesh cell, interior or ghost, is the root of a binary tree.
 * Each split partitions a node's particle range in place, so every node owns
 * one contiguous ParticleRange and its two children tile that range and the
 * node's box. All particle fields move together, as in sortParticlesByRung.
 * The splitting criterion is a KdSplitter; the default splits at the geometric
 * center of the node box along x, y, z in turn (axis = depth mod NDIM).
 *
 * Node boxes, particle bounding boxes, and split planes are Box<Coordinate>
 * values holding 32-bit fixed-point offsets from the cell's lower corner, not
 * absolute positions. A cell is narrower than the periodic box, so no offset
 * box reaches the periodic endpoint. Wrapping subtraction makes ghost copies
 * across the periodic seam work unchanged, and dyadic midpoints keep geometric
 * splits exact.
 *
 * Two particle orderings are provided. Position ordering builds one tree per
 * cell. Rung-threshold ordering first separates each cell into particles with
 * rung below a threshold and those at or above it, then builds one tree per
 * part, both rooted at the full cell box. Either ordering replaces the
 * PM-FMM per-cell rung ordering, so building marks the grid's rung ranges
 * invalid; CartesianGrid::sortParticlesByRung restores them.
 *
 * Node multipoles use the CellMultipole layout and the cell spacing as their
 * scale, matching the cell-level FMM interactions. By default they are taken
 * about the node center of mass, where the dipole vanishes and is stored as
 * zero; MultipoleCenter::geometricCenter and minimumRadius keep a computed
 * dipole instead. Every geometric quantity (boxes, centers, radii) depends
 * only on the set of particles in a node, never on their order, so a ghost
 * copy of a cell builds the same tree geometry as its owner.
 */
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <vector>

#include <cosmictiger/box.hpp>
#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/spatial_types.hpp>

namespace cosmictiger::gravity::tree {

inline constexpr size_t kNoChild = std::numeric_limits<size_t>::max();

// Half-open box [min, max) on each axis, holding fixed-point offsets from the
// lower corner of the tree's cell rather than absolute positions.
using KdBox = Box<Coordinate>;

// One tree node. Child indices address the owning CellKdTrees node array, in
// depth-first order with the left child immediately after its parent; leaves
// have kNoChild for both. A leaf is either small enough or not splittable.
struct KdNode {
    ParticleRange particles;
    KdBox box;
    // Tight bounds of the node's particle offsets, half-open: the largest
    // offset on each axis is one raw unit below max. Empty for empty nodes.
    KdBox particleBox;
    size_t leftChild = kNoChild;
    size_t rightChild = kNoChild;
    size_t depth = 0;
    // Left particles have offset[splitAxis] < splitPlane. Unused for leaves.
    int splitAxis = -1;
    Coordinate splitPlane{};
    // Multipole expansion center, an absolute unit-box position. Empty nodes
    // use their geometric center.
    Position center{};
    // Upper bound on |particle - center| for every particle of the node, in
    // unit-box coordinates. Zero for empty nodes.
    double radius = 0.0;
    double mass = 0.0;
};

// What a splitter sees: the node to split and read access to its particles.
// cellLower is the raw position of the cell's lower corner; a particle's box
// offset on an axis is position.raw() - cellLower (modulo 2^32).
struct KdSplitContext {
    KdBox box;
    size_t depth = 0;
    ParticleRange particles;
    ParticleStorage const* storage = nullptr;
    std::array<std::uint32_t, NDIM> cellLower{};
};

// A split decision. When split is true, plane must lie strictly inside the
// node box on axis, so both child boxes are nonempty and every split shrinks
// the box; a child's particle range may still be empty.
struct KdSplit {
    bool split = false;
    int axis = 0;
    Coordinate plane{};
};

// Splitting criterion. Implementations must be safe to call concurrently for
// different nodes; they must not modify particles.
class KdSplitter {
public:
    virtual ~KdSplitter() = default;
    virtual KdSplit chooseSplit(KdSplitContext const& node) const = 0;
};

// Split at the geometric center of the box along axis depth mod NDIM. A box
// one raw unit wide on that axis is at fixed-point resolution and stays a leaf.
class GeometricMidpointSplitter final : public KdSplitter {
public:
    KdSplit chooseSplit(KdSplitContext const& node) const override;
};

enum class MultipoleCenter {
    // Center of mass, rounded to the fixed-point grid; dipole stored as zero.
    centerOfMass,
    // Geometric center of the node box; the dipole is retained.
    geometricCenter,
    // The center giving the smallest radius bound among the candidates of the
    // old CosmicTiger tree: for a leaf, the center of its particle bounding
    // box; for an internal node, the smallest of that center (with an exact
    // particle scan for small nodes or the box half-diagonal otherwise) and
    // the sphere enclosing both children's spheres. The dipole is retained.
    minimumRadius,
};

enum class KdOrdering {
    position,
    rungThreshold,
};

struct KdTreeParameters {
    MultipoleCenter multipoleCenter = MultipoleCenter::centerOfMass;
    // Common particle mass, as in CartesianGrid::computeMultipoles.
    float particleMass = 1.0f;
};

// The trees of every stored cell of one grid, in flat arrays suitable for a
// later device copy. Builds reorder particles within each cell's localParticles
// range, leaving every cell range unchanged, and run cells as HPX tasks.
// Optional companion arrays follow the particle permutation exactly, as in
// CartesianGrid::sortParticlesByRung. Building requires exclusive access to
// the grid and particles and completed GPU work on them.
class CellKdTrees {
public:
    CellKdTrees() = default;

    // One tree per cell, ordered by position only.
    static CellKdTrees buildByPosition(CartesianGrid& grid, ParticleStorage& particles, KdTreeParameters const& parameters = {},
        KdSplitter const& splitter = GeometricMidpointSplitter{}, double* const* companionFields = nullptr);

    // Each cell holds particles with rung < threshold, then rung >= threshold,
    // each ordered by position under its own tree.
    static CellKdTrees buildByRungThreshold(CartesianGrid& grid, ParticleStorage& particles, std::uint8_t rungThreshold,
        KdTreeParameters const& parameters = {}, KdSplitter const& splitter = GeometricMidpointSplitter{}, double* const* companionFields = nullptr);

    KdOrdering ordering() const noexcept;
    std::uint8_t rungThreshold() const;
    size_t cellCount() const noexcept;
    size_t nodeCount() const noexcept;

    // Cell indices are CartesianGridLayout::cellIndex values.
    size_t positionRoot(size_t cellIndex) const;
    size_t belowThresholdRoot(size_t cellIndex) const;
    size_t atOrAboveThresholdRoot(size_t cellIndex) const;

    KdNode const& node(size_t index) const;
    CellMultipole const& multipole(size_t index) const;
    KdNode const* nodeData() const noexcept;
    CellMultipole const* multipoleData() const noexcept;

private:
    KdOrdering ordering_ = KdOrdering::position;
    std::uint8_t rungThreshold_ = 0;
    size_t treesPerCell_ = 1;
    std::vector<KdNode> nodes_;
    std::vector<CellMultipole> multipoles_;
    std::vector<size_t> roots_;

    static CellKdTrees build(CartesianGrid& grid, ParticleStorage& particles, KdOrdering ordering, std::uint8_t rungThreshold,
        KdTreeParameters const& parameters, KdSplitter const& splitter, double* const* companionFields);
};

}    // namespace cosmictiger::gravity::tree
