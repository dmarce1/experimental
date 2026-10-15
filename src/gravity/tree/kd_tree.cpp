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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/gravity/tree/kd_tree.hpp>
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::gravity::tree {

namespace detail {

    using DoubleMultipole = stf::Multipole<double, kMultipoleOrder>;

    // The box and scale shared by every tree of one cell.
    struct CellFrame {
        std::array<std::uint32_t, NDIM> lower{};
        std::uint32_t width = 0;
        double spacing = 0.0;
    };

    // One cell's trees with cell-local node indices, before assembly.
    struct CellTrees {
        std::vector<KdNode> nodes;
        std::vector<CellMultipole> multipoles;
        std::array<size_t, 2> roots{};
    };

    std::uint32_t boxOffset(ParticleStorage const& particles, size_t axis, ParticleCount particle, std::array<std::uint32_t, NDIM> const& cellLower) {
        PROFILE();
        return particles.position(axis, particle).raw() - cellLower[axis];
    }

    void setAxis(Position& position, size_t axis, Coordinate value) {
        PROFILE();
        Coordinate* const axes[NDIM] = {&position.x, &position.y, &position.z};
        *axes[expectIndex(axis, size_t(NDIM))] = value;
    }

    void swapCompanions(double* const* companionFields, ParticleCount first, ParticleCount second) {
        PROFILE();
        if (companionFields == nullptr) {
            return;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (companionFields[axis] != nullptr) {
                std::swap(companionFields[axis][first], companionFields[axis][second]);
            }
        }
    }

    // Move particles for which goesRight is false before the others, with all
    // fields and companions. Returns the first index of the right part.
    template <typename GoesRight>
    ParticleCount partitionRange(ParticleStorage& particles, ParticleRange range, double* const* companionFields, GoesRight const& goesRight) {
        PROFILE();
        ParticleCount left = range.begin;
        ParticleCount right = range.end;
        while (true) {
            while (left < right && !goesRight(left)) {
                left++;
            }
            while (left < right && goesRight(right - 1)) {
                right--;
            }
            if (left >= right) {
                return left;
            }
            particles.swapParticles(left, right - 1);
            swapCompanions(companionFields, left, right - 1);
            left++;
            right--;
        }
    }

    // Append one tree over range, rooted at the cell box, in depth-first order
    // with each left child immediately after its parent. Iterative, because
    // HPX thread stacks are small and depth can reach NDIM * fractional bits.
    void buildTopology(
        CellTrees& trees, ParticleRange range, CellFrame const& frame, ParticleStorage& particles, KdSplitter const& splitter, double* const* companionFields) {
        PROFILE();
        struct Pending {
            ParticleRange particles;
            KdBox box;
            size_t depth;
            size_t parent;
            bool right;
        };
        KdBox cellBox{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            cellBox.upper[axis] = frame.width;
        }
        std::vector<Pending> pending{{range, cellBox, 0, kNoChild, false}};
        while (!pending.empty()) {
            Pending const item = pending.back();
            pending.pop_back();
            size_t const index = trees.nodes.size();
            if (item.parent != kNoChild) {
                auto& parent = trees.nodes[item.parent];
                (item.right ? parent.rightChild : parent.leftChild) = index;
            }
            KdNode node{};
            node.particles = item.particles;
            node.box = item.box;
            node.depth = item.depth;
            ParticleCount const count = item.particles.end - item.particles.begin;
            if (count > kBucketSize) {
                KdSplit const split = splitter.chooseSplit({item.box, item.depth, item.particles, &particles, frame.lower});
                if (split.split) {
                    if (split.axis < 0 || split.axis >= NDIM) {
                        throw std::logic_error("kd-tree split axis is out of range");
                    }
                    size_t const axis = static_cast<size_t>(split.axis);
                    if (split.plane <= item.box.lower[axis] || split.plane >= item.box.upper[axis]) {
                        throw std::logic_error("kd-tree split plane must lie strictly inside the node box");
                    }
                    ParticleCount const middle = partitionRange(particles, item.particles, companionFields,
                        [&](ParticleCount particle) { return boxOffset(particles, axis, particle, frame.lower) >= split.plane; });
                    node.splitAxis = split.axis;
                    node.splitPlane = split.plane;
                    Pending left{{item.particles.begin, middle}, item.box, item.depth + 1, index, false};
                    left.box.upper[axis] = split.plane;
                    Pending right{{middle, item.particles.end}, item.box, item.depth + 1, index, true};
                    right.box.lower[axis] = split.plane;
                    pending.push_back(right);
                    pending.push_back(left);
                }
            }
            trees.nodes.push_back(node);
        }
    }

    // Fill centers, masses, radii, and multipoles. Children follow their
    // parents, so a reverse pass finishes both children before each parent.
    // Moments are accumulated in double and rounded once for storage.
    void computeMoments(CellTrees& trees, CellFrame const& frame, ParticleStorage const& particles, KdTreeParameters const& parameters) {
        constexpr double kRawScale = 0x1p-32;
        PROFILE();
        size_t const count = trees.nodes.size();
        std::vector<DoubleMultipole> moments(count);
        std::vector<std::array<std::uint64_t, NDIM>> offsetSums(count);
        std::vector<std::array<std::uint32_t, NDIM>> centers(count);
        trees.multipoles.assign(count, CellMultipole{});
        double const particleMass = static_cast<double>(parameters.particleMass);
        FpeGuard guard;
        for (size_t index = count; index-- > 0;) {
            KdNode& node = trees.nodes[index];
            ParticleRange const range = node.particles;
            ParticleCount const particleCount = range.end - range.begin;
            bool const leaf = node.leftChild == kNoChild;
            auto& sum = offsetSums[index];
            if (leaf) {
                for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        std::uint32_t const offset = boxOffset(particles, axis, particle, frame.lower);
                        expectIndex(offset, frame.width);
                        sum[axis] += offset;
                    }
                }
            } else {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    sum[axis] = offsetSums[node.leftChild][axis] + offsetSums[node.rightChild][axis];
                }
            }
            // The center is a raw offset, so it is exactly representable both as
            // a fixed-point position and in double.
            auto& center = centers[index];
            for (size_t axis = 0; axis < NDIM; axis++) {
                std::uint32_t const lower = node.box.lower[axis];
                std::uint32_t const upper = node.box.upper[axis];
                if (particleCount == 0 || parameters.multipoleCenter == MultipoleCenter::geometricCenter) {
                    center[axis] = lower + (upper - lower) / 2;
                } else {
                    center[axis] = static_cast<std::uint32_t>((sum[axis] + particleCount / 2) / particleCount);
                }
                setAxis(node.center, axis, Coordinate::fromRaw(frame.lower[axis] + center[axis]));
            }
            node.mass = particleMass * static_cast<double>(particleCount);
            auto& moment = moments[index];
            double radius = 0.0;
            if (leaf) {
                for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                    std::array<double, NDIM> offset{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        offset[axis] = (static_cast<double>(boxOffset(particles, axis, particle, frame.lower)) - static_cast<double>(center[axis])) * kRawScale;
                    }
                    double const normalizedRadiusSquared = stf::addPoint<double, kMultipoleOrder>(moment, particleMass, offset, frame.spacing);
                    radius = std::max(radius, std::sqrt(normalizedRadiusSquared) * frame.spacing);
                }
            } else {
                for (size_t const child : {node.leftChild, node.rightChild}) {
                    KdNode const& childNode = trees.nodes[child];
                    if (childNode.particles.begin == childNode.particles.end) {
                        continue;
                    }
                    std::array<double, NDIM> displacement{};
                    double distanceSquared = 0.0;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        displacement[axis] = (static_cast<double>(centers[child][axis]) - static_cast<double>(center[axis])) * kRawScale;
                        distanceSquared += displacement[axis] * displacement[axis];
                    }
                    stf::addM2M<double, kMultipoleOrder>(moment, moments[child], displacement, frame.spacing, frame.spacing);
                    radius = std::max(radius, childNode.radius + std::sqrt(distanceSquared));
                }
                // The farthest box corner is also a bound; keep the tighter one.
                double cornerSquared = 0.0;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    double const below = static_cast<double>(center[axis] - node.box.lower[axis]);
                    double const above = static_cast<double>(node.box.upper[axis] - center[axis]);
                    double const extent = std::max(below, above) * kRawScale;
                    cornerSquared += extent * extent;
                }
                if (particleCount > 0) {
                    radius = std::min(radius, std::sqrt(cornerSquared));
                }
            }
            node.radius = radius;
            if (parameters.multipoleCenter == MultipoleCenter::centerOfMass) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    moment[stf::index(axis == 0, axis == 1, axis == 2)] = 0.0;
                }
            }
            for (size_t coefficient = 0; coefficient < moment.size(); coefficient++) {
                trees.multipoles[index][coefficient] = static_cast<float>(moment[coefficient]);
            }
        }
    }

}    // namespace detail

KdSplit GeometricMidpointSplitter::chooseSplit(KdSplitContext const& node) const {
    PROFILE();
    size_t const axis = node.depth % NDIM;
    std::uint32_t const lower = node.box.lower[axis];
    std::uint32_t const upper = node.box.upper[axis];
    expectRange(lower, lower, upper);
    if (upper - lower < 2) {
        return {};
    }
    return {true, static_cast<int>(axis), lower + (upper - lower) / 2};
}

CellKdTrees CellKdTrees::buildByPosition(
    CartesianGrid& grid, ParticleStorage& particles, KdTreeParameters const& parameters, KdSplitter const& splitter, double* const* companionFields) {
    PROFILE();
    return build(grid, particles, KdOrdering::position, 0, parameters, splitter, companionFields);
}

CellKdTrees CellKdTrees::buildByRungThreshold(CartesianGrid& grid, ParticleStorage& particles, std::uint8_t rungThreshold, KdTreeParameters const& parameters,
    KdSplitter const& splitter, double* const* companionFields) {
    PROFILE();
    return build(grid, particles, KdOrdering::rungThreshold, rungThreshold, parameters, splitter, companionFields);
}

CellKdTrees CellKdTrees::build(CartesianGrid& grid, ParticleStorage& particles, KdOrdering ordering, std::uint8_t rungThreshold,
    KdTreeParameters const& parameters, KdSplitter const& splitter, double* const* companionFields) {
    constexpr size_t kMinimumCellsPerChunk = 8;
    constexpr size_t kMaximumTreesPerCell = 2;
    PROFILE();
    if (!(parameters.particleMass > 0.0f)) {
        throw std::invalid_argument("kd-tree particle mass must be positive");
    }
    if (rungThreshold > Rung::maximumRung() + 1) {
        throw std::invalid_argument("kd-tree rung threshold exceeds the maximum rung");
    }
    auto const& layout = grid.layout();
    int const cellExponent = std::countr_zero(layout.globalSideLength());
    // The cell width, 2^(fractional bits - exponent), must fit the raw offsets.
    if (cellExponent < 1) {
        throw std::invalid_argument("kd-tree cells must be narrower than the periodic box");
    }
    int const positionShift = Coordinate::fractionalBits() - cellExponent;
    size_t const cellCount = layout.storedCellCount();
    size_t const treesPerCell = ordering == KdOrdering::position ? 1 : kMaximumTreesPerCell;
    // Partitioning destroys the PM-FMM rung order even if the build fails.
    grid.invalidateRungRanges();

    std::vector<detail::CellTrees> cells(cellCount);
    cosmictiger::detail::ParallelChunkPlan const chunks(cellCount, kMinimumCellsPerChunk);
    cosmictiger::detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    auto const coordinates = layout.cellCoordinates(index);
                    auto const global = layout.globalCoordinates(coordinates);
                    detail::CellFrame frame{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        frame.lower[axis] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(global[axis]) << positionShift);
                    }
                    frame.width = std::uint32_t(1) << positionShift;
                    frame.spacing = layout.spacing();
                    ParticleRange const range = grid.cell(coordinates).localParticles;
                    expectRange(0, range.begin, range.end);
                    expectRange(0, range.end, particles.size());
                    auto& trees = cells[index];
                    if (ordering == KdOrdering::position) {
                        trees.roots[0] = 0;
                        detail::buildTopology(trees, range, frame, particles, splitter, companionFields);
                    } else {
                        ParticleCount const middle = detail::partitionRange(
                            particles, range, companionFields, [&](ParticleCount particle) { return particles.flags(particle).rung() >= rungThreshold; });
                        trees.roots[0] = 0;
                        detail::buildTopology(trees, {range.begin, middle}, frame, particles, splitter, companionFields);
                        trees.roots[1] = trees.nodes.size();
                        detail::buildTopology(trees, {middle, range.end}, frame, particles, splitter, companionFields);
                    }
                    detail::computeMoments(trees, frame, particles, parameters);
                }
            },
            "kdTree.buildCells"));

    CellKdTrees result;
    result.ordering_ = ordering;
    result.rungThreshold_ = rungThreshold;
    result.treesPerCell_ = treesPerCell;
    std::vector<size_t> offsets(cellCount + 1, 0);
    for (size_t index = 0; index < cellCount; index++) {
        offsets[index + 1] = offsets[index] + cells[index].nodes.size();
    }
    result.nodes_.resize(offsets.back());
    result.multipoles_.resize(offsets.back());
    result.roots_.resize(cellCount * treesPerCell);
    cosmictiger::detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    size_t const offset = offsets[index];
                    auto& trees = cells[index];
                    for (size_t local = 0; local < trees.nodes.size(); local++) {
                        KdNode node = trees.nodes[local];
                        if (node.leftChild != kNoChild) {
                            node.leftChild += offset;
                            node.rightChild += offset;
                        }
                        result.nodes_[offset + local] = node;
                        result.multipoles_[offset + local] = trees.multipoles[local];
                    }
                    for (size_t tree = 0; tree < treesPerCell; tree++) {
                        result.roots_[index * treesPerCell + tree] = offset + trees.roots[tree];
                    }
                    trees = {};
                }
            },
            "kdTree.assembleCells"));
    return result;
}

KdOrdering CellKdTrees::ordering() const noexcept {
    PROFILE();
    return ordering_;
}

std::uint8_t CellKdTrees::rungThreshold() const {
    PROFILE();
    expect(ordering_ == KdOrdering::rungThreshold);
    return rungThreshold_;
}

size_t CellKdTrees::cellCount() const noexcept {
    PROFILE();
    return roots_.size() / treesPerCell_;
}

size_t CellKdTrees::nodeCount() const noexcept {
    PROFILE();
    return nodes_.size();
}

size_t CellKdTrees::positionRoot(size_t cellIndex) const {
    PROFILE();
    expect(ordering_ == KdOrdering::position);
    return roots_[expectIndex(cellIndex, cellCount())];
}

size_t CellKdTrees::belowThresholdRoot(size_t cellIndex) const {
    PROFILE();
    expect(ordering_ == KdOrdering::rungThreshold);
    return roots_[expectIndex(cellIndex, cellCount()) * treesPerCell_];
}

size_t CellKdTrees::atOrAboveThresholdRoot(size_t cellIndex) const {
    PROFILE();
    expect(ordering_ == KdOrdering::rungThreshold);
    return roots_[expectIndex(cellIndex, cellCount()) * treesPerCell_ + 1];
}

KdNode const& CellKdTrees::node(size_t index) const {
    PROFILE();
    return nodes_[expectIndex(index, nodes_.size())];
}

CellMultipole const& CellKdTrees::multipole(size_t index) const {
    PROFILE();
    return multipoles_[expectIndex(index, multipoles_.size())];
}

KdNode const* CellKdTrees::nodeData() const noexcept {
    PROFILE();
    return nodes_.data();
}

CellMultipole const* CellKdTrees::multipoleData() const noexcept {
    PROFILE();
    return multipoles_.data();
}

}    // namespace cosmictiger::gravity::tree
