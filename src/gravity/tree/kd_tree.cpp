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
#include <limits>
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

    std::uint32_t lowerRaw(KdBox const& box, size_t axis) {
        PROFILE();
        return box.min(axis).raw();
    }

    std::uint32_t upperRaw(KdBox const& box, size_t axis) {
        PROFILE();
        return box.max(axis).raw();
    }

    KdBox makeBox(std::array<std::uint32_t, NDIM> const& lower, std::array<std::uint32_t, NDIM> const& upper) {
        PROFILE();
        KdBox::Point lowerPoint{};
        KdBox::Point upperPoint{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            lowerPoint[axis] = Coordinate::fromRaw(lower[axis]);
            upperPoint[axis] = Coordinate::fromRaw(upper[axis]);
        }
        return KdBox(lowerPoint, upperPoint);
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
        std::array<std::uint32_t, NDIM> cellUpper{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            cellUpper[axis] = frame.width;
        }
        std::vector<Pending> pending{{range, makeBox({}, cellUpper), 0, kNoChild, false}};
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
                    std::uint32_t const plane = split.plane.raw();
                    if (plane <= lowerRaw(item.box, axis) || plane >= upperRaw(item.box, axis)) {
                        throw std::logic_error("kd-tree split plane must lie strictly inside the node box");
                    }
                    ParticleCount const middle = partitionRange(particles, item.particles, companionFields,
                        [&](ParticleCount particle) { return boxOffset(particles, axis, particle, frame.lower) >= plane; });
                    node.splitAxis = split.axis;
                    node.splitPlane = split.plane;
                    auto const [leftBox, rightBox] = item.box.split(axis, split.plane);
                    pending.push_back({{middle, item.particles.end}, rightBox, item.depth + 1, index, true});
                    pending.push_back({{item.particles.begin, middle}, leftBox, item.depth + 1, index, false});
                }
            }
            trees.nodes.push_back(node);
        }
    }

    using RawPoint = std::array<std::uint32_t, NDIM>;

    // Distance in unit-box coordinates between two raw offsets of one cell.
    double rawDistance(RawPoint const& first, RawPoint const& second) {
        constexpr double kRawScale = 0x1p-32;
        PROFILE();
        double squared = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const delta = (static_cast<double>(first[axis]) - static_cast<double>(second[axis])) * kRawScale;
            squared += delta * delta;
        }
        return std::sqrt(squared);
    }

    // The largest distance from center to a particle of range.
    double scanRadius(ParticleStorage const& particles, ParticleRange range, CellFrame const& frame, RawPoint const& center) {
        PROFILE();
        double radius = 0.0;
        for (ParticleCount particle = range.begin; particle < range.end; particle++) {
            RawPoint offset{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                offset[axis] = boxOffset(particles, axis, particle, frame.lower);
            }
            radius = std::max(radius, rawDistance(offset, center));
        }
        return radius;
    }

    // The sphere enclosing two nonempty child spheres. The exact center lies
    // on the line between the child centers; rounding it to a raw offset can
    // move it, so the radius is recomputed from the rounded center and stays
    // an upper bound of both children's bounds.
    std::pair<RawPoint, double> enclosingSphere(RawPoint const& leftCenter, double leftRadius, RawPoint const& rightCenter, double rightRadius) {
        constexpr double kHalf = 0.5;
        PROFILE();
        double const separation = rawDistance(leftCenter, rightCenter);
        if (separation + rightRadius <= leftRadius) {
            return {leftCenter, leftRadius};
        }
        if (separation + leftRadius <= rightRadius) {
            return {rightCenter, rightRadius};
        }
        double const exactRadius = kHalf * (separation + leftRadius + rightRadius);
        double const fraction = (exactRadius - leftRadius) / separation;
        RawPoint center{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const lower = static_cast<double>(leftCenter[axis]);
            double const delta = static_cast<double>(rightCenter[axis]) - lower;
            center[axis] = static_cast<std::uint32_t>(std::llround(lower + fraction * delta));
        }
        return {center, std::max(rawDistance(center, leftCenter) + leftRadius, rawDistance(center, rightCenter) + rightRadius)};
    }

    // Fill centers, masses, radii, particle boxes, and multipoles. Children
    // follow their parents, so a reverse pass finishes both children before
    // each parent. Moments are accumulated in double and rounded once.
    void computeMoments(CellTrees& trees, CellFrame const& frame, ParticleStorage const& particles, KdTreeParameters const& parameters) {
        constexpr double kRawScale = 0x1p-32;
        // Internal nodes up to this many buckets are scanned exactly, as in
        // the old CosmicTiger tree; larger ones use the box half-diagonal.
        constexpr ParticleCount kExactScanBuckets = 64;
        constexpr ParticleCount kExactScanCount = kExactScanBuckets * kBucketSize;
        PROFILE();
        size_t const count = trees.nodes.size();
        std::vector<DoubleMultipole> moments(count);
        std::vector<std::array<std::uint64_t, NDIM>> offsetSums(count);
        std::vector<RawPoint> centers(count);
        // Inclusive particle offset bounds of every nonempty node.
        std::vector<RawPoint> minimumOffsets(count);
        std::vector<RawPoint> maximumOffsets(count);
        trees.multipoles.assign(count, CellMultipole{});
        double const particleMass = static_cast<double>(parameters.particleMass);
        FpeGuard guard;
        for (size_t index = count; index-- > 0;) {
            KdNode& node = trees.nodes[index];
            ParticleRange const range = node.particles;
            ParticleCount const particleCount = range.end - range.begin;
            bool const leaf = node.leftChild == kNoChild;
            auto& sum = offsetSums[index];
            auto& minimum = minimumOffsets[index];
            auto& maximum = maximumOffsets[index];
            minimum.fill(std::numeric_limits<std::uint32_t>::max());
            maximum.fill(0);
            if (leaf) {
                for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        std::uint32_t const offset = boxOffset(particles, axis, particle, frame.lower);
                        expectIndex(offset, frame.width);
                        sum[axis] += offset;
                        minimum[axis] = std::min(minimum[axis], offset);
                        maximum[axis] = std::max(maximum[axis], offset);
                    }
                }
            } else {
                for (size_t const child : {node.leftChild, node.rightChild}) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        sum[axis] += offsetSums[child][axis];
                        if (trees.nodes[child].particles.begin != trees.nodes[child].particles.end) {
                            minimum[axis] = std::min(minimum[axis], minimumOffsets[child][axis]);
                            maximum[axis] = std::max(maximum[axis], maximumOffsets[child][axis]);
                        }
                    }
                }
            }
            RawPoint boxCenter{};
            RawPoint particleBoxCenter{};
            RawPoint upper{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                std::uint32_t const lower = lowerRaw(node.box, axis);
                boxCenter[axis] = lower + (upperRaw(node.box, axis) - lower) / 2;
                particleBoxCenter[axis] = minimum[axis] + (maximum[axis] - minimum[axis]) / 2;
                upper[axis] = maximum[axis] + 1;
            }
            if (particleCount > 0) {
                node.particleBox = makeBox(minimum, upper);
            }
            // Every center is a raw offset, so it is exactly representable both
            // as a fixed-point position and in double.
            auto& center = centers[index];
            double radius = 0.0;
            if (particleCount == 0) {
                center = boxCenter;
            } else if (parameters.multipoleCenter == MultipoleCenter::minimumRadius) {
                if (leaf) {
                    center = particleBoxCenter;
                    radius = scanRadius(particles, range, frame, center);
                } else {
                    center = particleBoxCenter;
                    if (particleCount <= kExactScanCount) {
                        radius = scanRadius(particles, range, frame, center);
                    } else {
                        double cornerSquared = 0.0;
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            double const extent = static_cast<double>(std::max(center[axis] - minimum[axis], maximum[axis] - center[axis])) * kRawScale;
                            cornerSquared += extent * extent;
                        }
                        radius = std::sqrt(cornerSquared);
                    }
                    KdNode const& left = trees.nodes[node.leftChild];
                    KdNode const& right = trees.nodes[node.rightChild];
                    bool const leftEmpty = left.particles.begin == left.particles.end;
                    bool const rightEmpty = right.particles.begin == right.particles.end;
                    auto const sphere = leftEmpty ? std::pair{centers[node.rightChild], right.radius} :
                        rightEmpty                ? std::pair{centers[node.leftChild], left.radius} :
                                                    enclosingSphere(centers[node.leftChild], left.radius, centers[node.rightChild], right.radius);
                    if (sphere.second < radius) {
                        center = sphere.first;
                        radius = sphere.second;
                    }
                }
            } else if (parameters.multipoleCenter == MultipoleCenter::geometricCenter) {
                center = boxCenter;
            } else {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    center[axis] = static_cast<std::uint32_t>((sum[axis] + particleCount / 2) / particleCount);
                }
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                setAxis(node.center, axis, Coordinate::fromRaw(frame.lower[axis] + center[axis]));
            }
            node.mass = particleMass * static_cast<double>(particleCount);
            auto& moment = moments[index];
            if (leaf) {
                for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                    std::array<double, NDIM> offset{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        offset[axis] = (static_cast<double>(boxOffset(particles, axis, particle, frame.lower)) - static_cast<double>(center[axis])) * kRawScale;
                    }
                    double const normalizedRadiusSquared = stf::addPoint<double, kMultipoleOrder>(moment, particleMass, offset, frame.spacing);
                    if (parameters.multipoleCenter != MultipoleCenter::minimumRadius) {
                        radius = std::max(radius, std::sqrt(normalizedRadiusSquared) * frame.spacing);
                    }
                }
            } else {
                double childBound = 0.0;
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
                    childBound = std::max(childBound, childNode.radius + std::sqrt(distanceSquared));
                }
                if (parameters.multipoleCenter != MultipoleCenter::minimumRadius && particleCount > 0) {
                    // The farthest box corner is also a bound; keep the tighter one.
                    double cornerSquared = 0.0;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        double const below = static_cast<double>(center[axis] - lowerRaw(node.box, axis));
                        double const above = static_cast<double>(upperRaw(node.box, axis) - center[axis]);
                        double const extent = std::max(below, above) * kRawScale;
                        cornerSquared += extent * extent;
                    }
                    radius = std::min(childBound, std::sqrt(cornerSquared));
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
    std::uint32_t const lower = node.box.min(axis).raw();
    std::uint32_t const upper = node.box.max(axis).raw();
    expectRange(lower, lower, upper);
    if (upper - lower < 2) {
        return {};
    }
    return {true, static_cast<int>(axis), Coordinate::fromRaw(lower + (upper - lower) / 2)};
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
