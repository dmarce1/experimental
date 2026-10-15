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
#include <cmath>
#include <cstdint>
#include <random>
#include <stddef.h>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/gravity/tree/kd_tree.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/profiler.hpp>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleRange;
using cosmictiger::ParticleStorage;
using cosmictiger::Position;
namespace gravity = cosmictiger::gravity;
namespace stf = cosmictiger::gravity::stf;
namespace tree = cosmictiger::gravity::tree;

constexpr size_t kCellExponent = 2;
constexpr int kCoordinateBits = 32;
constexpr std::uint32_t kTestSeed = 42;
constexpr std::uint64_t kMaximumTestRung = 5;
constexpr float kParticleMass = 0.25f;
constexpr double kRawScale = 0x1p-32;

static_assert(cosmictiger::gravity::kBucketSize == 128);

// One locality with a one-cell ghost shell; every ghost cell is a periodic
// image, so seam wrapping is exercised in every direction.
class KdTreeFixture {
public:
    KdTreeFixture()
      : grid_(kCellExponent, 0, 0, 1) {
        PROFILE();
        std::mt19937 generator(kTestSeed);
        auto const& layout = grid_.layout();
        std::vector<size_t> order;
        // Interior cells first, so interior ranges partition the owned prefix.
        for (int interior = 1; interior >= 0; interior--) {
            for (size_t index = 0; index < layout.storedCellCount(); index++) {
                if (layout.isInterior(layout.cellCoordinates(index)) == (interior == 1)) {
                    order.push_back(index);
                }
            }
        }
        for (size_t slot = 0; slot < order.size(); slot++) {
            size_t const index = order[slot];
            auto const coordinates = layout.cellCoordinates(index);
            ParticleRange range{particles_.size(), particles_.size()};
            fillCell(index, slot, generator);
            range.end = particles_.size();
            if (layout.isInterior(coordinates)) {
                grid_.setInteriorRange(coordinates, range);
            } else {
                grid_.setGhostRanges(coordinates, range, range);
            }
        }
        grid_.sortParticlesByRung(particles_);
        for (size_t axis = 0; axis < NDIM; axis++) {
            companion_[axis].resize(particles_.size());
        }
        records_.resize(particles_.size());
        for (ParticleCount particle = 0; particle < particles_.size(); particle++) {
            auto const id = static_cast<size_t>(particles_.velocity(0, particle));
            for (size_t axis = 0; axis < NDIM; axis++) {
                companion_[axis][particle] = companionValue(particles_.velocity(0, particle), axis);
            }
            Record record{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                record.raw[axis] = particles_.position(axis, particle).raw();
            }
            record.rung = particles_.flags(particle).rung();
            records_[id] = record;
        }
    }

    CartesianGrid& grid() {
        PROFILE();
        return grid_;
    }

    ParticleStorage& particles() {
        PROFILE();
        return particles_;
    }

    double* const* companions() {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            companionPointers_[axis] = companion_[axis].data();
        }
        return companionPointers_.data();
    }

    std::array<std::uint32_t, NDIM> cellLower(size_t index) const {
        PROFILE();
        auto const& layout = grid_.layout();
        auto const global = layout.globalCoordinates(layout.cellCoordinates(index));
        std::array<std::uint32_t, NDIM> lower{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            lower[axis] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(global[axis]) << (kCoordinateBits - kCellExponent));
        }
        return lower;
    }

    // Every particle keeps its fields and companions and stays in its cell.
    void expectPermutationPreserved() const {
        PROFILE();
        auto const& layout = grid_.layout();
        std::vector<bool> seen(records_.size(), false);
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            ParticleRange const range = grid_.cell(layout.cellCoordinates(index)).localParticles;
            for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                auto const id = static_cast<size_t>(particles_.velocity(0, particle));
                ASSERT_LT(id, records_.size());
                EXPECT_FALSE(seen[id]);
                seen[id] = true;
                EXPECT_EQ(cellOfId_[id], index);
                for (size_t axis = 0; axis < NDIM; axis++) {
                    EXPECT_EQ(particles_.position(axis, particle).raw(), records_[id].raw[axis]);
                    EXPECT_EQ(companion_[axis][particle], companionValue(static_cast<float>(id), axis));
                }
                EXPECT_EQ(particles_.flags(particle).rung(), records_[id].rung);
            }
        }
        EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
    }

private:
    struct Record {
        std::array<std::uint32_t, NDIM> raw;
        std::uint8_t rung;
    };

    CartesianGrid grid_;
    ParticleStorage particles_;
    std::vector<Record> records_;
    std::vector<size_t> cellOfId_;
    std::array<std::vector<double>, NDIM> companion_;
    std::array<double*, NDIM> companionPointers_{};

    static double companionValue(float id, size_t axis) {
        PROFILE();
        return static_cast<double>(id) * 3.0 + static_cast<double>(axis);
    }

    void append(size_t cellIndex, std::array<std::uint32_t, NDIM> const& raw, std::mt19937& generator) {
        PROFILE();
        std::uniform_int_distribution<std::uint64_t> rung(0, kMaximumTestRung);
        float const id = static_cast<float>(particles_.size());
        particles_.append({Coordinate::fromRaw(raw[0]), Coordinate::fromRaw(raw[1]), Coordinate::fromRaw(raw[2])}, {id, 0.0f, 0.0f}, rung(generator));
        cellOfId_.push_back(cellIndex);
    }

    // Cells cycle through uniform, clustered, coincident, single, and empty.
    void fillCell(size_t index, size_t slot, std::mt19937& generator) {
        constexpr std::uint32_t kWidth = std::uint32_t(1) << (kCoordinateBits - kCellExponent);
        constexpr std::uint32_t kClusterWidth = 1u << 12;
        constexpr size_t kPatternCount = 5;
        PROFILE();
        auto const lower = cellLower(index);
        std::uniform_int_distribution<std::uint32_t> uniform(0, kWidth - 1);
        std::uniform_int_distribution<std::uint32_t> cluster(0, kClusterWidth - 1);
        std::uniform_int_distribution<size_t> count(150, 450);
        std::array<std::uint32_t, NDIM> raw{};
        switch (slot % kPatternCount) {
        case 0:
        case 1: {
            size_t const particleCount = count(generator);
            for (size_t particle = 0; particle < particleCount; particle++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    raw[axis] = lower[axis] + uniform(generator);
                }
                append(index, raw, generator);
            }
            break;
        }
        case 2: {
            // Half uniform, half packed near one corner of the cell.
            size_t const particleCount = count(generator);
            for (size_t particle = 0; particle < particleCount; particle++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    raw[axis] = lower[axis] + (particle % 2 == 0 ? uniform(generator) : kWidth - kClusterWidth + cluster(generator));
                }
                append(index, raw, generator);
            }
            break;
        }
        case 3: {
            // Coincident particles reach the fixed-point depth limit.
            for (size_t axis = 0; axis < NDIM; axis++) {
                raw[axis] = lower[axis] + uniform(generator);
            }
            for (size_t particle = 0; particle < 2 * cosmictiger::gravity::kBucketSize; particle++) {
                append(index, raw, generator);
            }
            break;
        }
        default: {
            if (slot % 2 == 0) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    raw[axis] = lower[axis] + uniform(generator);
                }
                append(index, raw, generator);
            }
            break;
        }
        }
    }
};

std::uint32_t offsetOf(ParticleStorage const& particles, size_t axis, ParticleCount particle, std::array<std::uint32_t, NDIM> const& lower) {
    PROFILE();
    return particles.position(axis, particle).raw() - lower[axis];
}

std::uint32_t rawAxis(Position const& position, size_t axis) {
    PROFILE();
    Coordinate const axes[NDIM] = {position.x, position.y, position.z};
    return axes[axis].raw();
}

struct CheckOptions {
    tree::MultipoleCenter center = tree::MultipoleCenter::centerOfMass;
    // -1 checks geometric x, y, z alternation; otherwise every split is on this axis.
    int fixedAxis = -1;
};

// Check every invariant of one tree against its particles. Returns the number
// of nodes visited so the caller can check that trees are disjoint and complete.
size_t checkTree(tree::CellKdTrees const& trees, size_t root, ParticleRange range, std::array<std::uint32_t, NDIM> const& lower,
    ParticleStorage const& particles, double spacing, CheckOptions const& options) {
    constexpr std::uint32_t kWidth = std::uint32_t(1) << (kCoordinateBits - kCellExponent);
    constexpr double kMomentTolerance = 1e-10;
    constexpr double kFloatTolerance = 1e-6;
    PROFILE();
    auto const& rootNode = trees.node(root);
    EXPECT_EQ(rootNode.particles.begin, range.begin);
    EXPECT_EQ(rootNode.particles.end, range.end);
    EXPECT_EQ(rootNode.depth, 0u);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(rootNode.box.lower[axis], 0u);
        EXPECT_EQ(rootNode.box.upper[axis], kWidth);
    }
    size_t visited = 0;
    std::vector<size_t> pending{root};
    while (!pending.empty()) {
        size_t const index = pending.back();
        pending.pop_back();
        visited++;
        auto const& node = trees.node(index);
        ParticleCount const count = node.particles.end - node.particles.begin;
        int const expectedAxis = options.fixedAxis >= 0 ? options.fixedAxis : static_cast<int>(node.depth % NDIM);

        // Particles lie in the node box.
        for (ParticleCount particle = node.particles.begin; particle < node.particles.end; particle++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                std::uint32_t const offset = offsetOf(particles, axis, particle, lower);
                EXPECT_GE(offset, node.box.lower[axis]);
                EXPECT_LT(offset, node.box.upper[axis]);
            }
        }

        if (node.leftChild == tree::kNoChild) {
            EXPECT_EQ(node.rightChild, tree::kNoChild);
            std::uint32_t const width = node.box.upper[expectedAxis] - node.box.lower[expectedAxis];
            EXPECT_TRUE(count <= cosmictiger::gravity::kBucketSize || width < 2) << "unsplit node of " << count << " particles";
        } else {
            EXPECT_GT(count, cosmictiger::gravity::kBucketSize);
            EXPECT_EQ(node.leftChild, index + 1);
            EXPECT_EQ(node.splitAxis, expectedAxis);
            auto const axis = static_cast<size_t>(node.splitAxis);
            EXPECT_EQ(node.splitPlane, node.box.lower[axis] + (node.box.upper[axis] - node.box.lower[axis]) / 2);
            auto const& left = trees.node(node.leftChild);
            auto const& right = trees.node(node.rightChild);
            EXPECT_EQ(left.particles.begin, node.particles.begin);
            EXPECT_EQ(left.particles.end, right.particles.begin);
            EXPECT_EQ(right.particles.end, node.particles.end);
            EXPECT_EQ(left.depth, node.depth + 1);
            EXPECT_EQ(right.depth, node.depth + 1);
            for (size_t other = 0; other < NDIM; other++) {
                std::uint32_t const leftUpper = other == axis ? node.splitPlane : node.box.upper[other];
                std::uint32_t const rightLower = other == axis ? node.splitPlane : node.box.lower[other];
                EXPECT_EQ(left.box.lower[other], node.box.lower[other]);
                EXPECT_EQ(left.box.upper[other], leftUpper);
                EXPECT_EQ(right.box.lower[other], rightLower);
                EXPECT_EQ(right.box.upper[other], node.box.upper[other]);
            }
            pending.push_back(node.rightChild);
            pending.push_back(node.leftChild);
        }

        // Center, mass, radius, and multipole against a direct reference.
        EXPECT_DOUBLE_EQ(node.mass, static_cast<double>(kParticleMass) * static_cast<double>(count));
        std::array<double, NDIM> centerOffset{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            std::uint32_t const center = rawAxis(node.center, axis) - lower[axis];
            centerOffset[axis] = static_cast<double>(center);
            if (count == 0 || options.center == tree::MultipoleCenter::geometricCenter) {
                EXPECT_EQ(center, node.box.lower[axis] + (node.box.upper[axis] - node.box.lower[axis]) / 2);
            } else {
                double mean = 0.0;
                for (ParticleCount particle = node.particles.begin; particle < node.particles.end; particle++) {
                    mean += static_cast<double>(offsetOf(particles, axis, particle, lower));
                }
                mean /= static_cast<double>(count);
                EXPECT_LE(std::abs(centerOffset[axis] - mean), 0.5 + 1e-6);
            }
        }
        stf::Multipole<double, gravity::kMultipoleOrder> reference{};
        double farthest = 0.0;
        for (ParticleCount particle = node.particles.begin; particle < node.particles.end; particle++) {
            std::array<double, NDIM> offset{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                offset[axis] = (static_cast<double>(offsetOf(particles, axis, particle, lower)) - centerOffset[axis]) * kRawScale;
            }
            double const normalized = stf::addPoint<double, gravity::kMultipoleOrder>(reference, static_cast<double>(kParticleMass), offset, spacing);
            farthest = std::max(farthest, std::sqrt(normalized) * spacing);
        }
        EXPECT_GE(node.radius * (1.0 + 1e-12), farthest);
        if (count == 0) {
            EXPECT_EQ(node.radius, 0.0);
        }
        auto const& multipole = trees.multipole(index);
        for (size_t axis = 0; axis < NDIM; axis++) {
            size_t const dipole = stf::index(axis == 0, axis == 1, axis == 2);
            if (options.center == tree::MultipoleCenter::centerOfMass) {
                EXPECT_EQ(multipole[dipole], 0.0f);
                reference[dipole] = 0.0;
            }
        }
        double const tolerance = kMomentTolerance * node.mass;
        for (size_t coefficient = 0; coefficient < multipole.size(); coefficient++) {
            EXPECT_NEAR(multipole[coefficient], reference[coefficient], tolerance + kFloatTolerance * std::abs(reference[coefficient]))
                << "node " << index << " coefficient " << coefficient;
        }
    }
    return visited;
}

TEST(KdTree, PositionOrderingBuildsGeometricTreesInEveryCell) {
    PROFILE();
    KdTreeFixture fixture;
    auto& grid = fixture.grid();
    auto const& layout = grid.layout();
    tree::KdTreeParameters parameters{};
    parameters.particleMass = kParticleMass;
    auto const trees = tree::CellKdTrees::buildByPosition(grid, fixture.particles(), parameters, tree::GeometricMidpointSplitter{}, fixture.companions());
    EXPECT_EQ(trees.ordering(), tree::KdOrdering::position);
    ASSERT_EQ(trees.cellCount(), layout.storedCellCount());
    fixture.expectPermutationPreserved();
    size_t visited = 0;
    size_t deepest = 0;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const& cell = grid.cell(layout.cellCoordinates(index));
        if (cell.localParticles.begin != cell.localParticles.end) {
            EXPECT_FALSE(cell.rungRangesValid);
        }
        visited +=
            checkTree(trees, trees.positionRoot(index), cell.localParticles, fixture.cellLower(index), fixture.particles(), layout.spacing(), CheckOptions{});
    }
    for (size_t index = 0; index < trees.nodeCount(); index++) {
        deepest = std::max(deepest, trees.node(index).depth);
    }
    EXPECT_EQ(visited, trees.nodeCount());
    // Coincident particles split until the box is one raw unit on every axis.
    EXPECT_EQ(deepest, NDIM * (kCoordinateBits - kCellExponent));

    // The PM-FMM rung ordering is restored independently of the trees.
    grid.sortParticlesByRung(fixture.particles());
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        EXPECT_TRUE(grid.cell(layout.cellCoordinates(index)).rungRangesValid);
    }
}

TEST(KdTree, RungThresholdOrderingSeparatesThenSortsByPosition) {
    constexpr std::uint8_t kThreshold = 3;
    PROFILE();
    KdTreeFixture fixture;
    auto& grid = fixture.grid();
    auto const& layout = grid.layout();
    auto const& particles = fixture.particles();
    tree::KdTreeParameters parameters{};
    parameters.particleMass = kParticleMass;
    auto const trees =
        tree::CellKdTrees::buildByRungThreshold(grid, fixture.particles(), kThreshold, parameters, tree::GeometricMidpointSplitter{}, fixture.companions());
    EXPECT_EQ(trees.ordering(), tree::KdOrdering::rungThreshold);
    EXPECT_EQ(trees.rungThreshold(), kThreshold);
    fixture.expectPermutationPreserved();
    CheckOptions options{};
    size_t visited = 0;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        ParticleRange const range = grid.cell(layout.cellCoordinates(index)).localParticles;
        auto const& below = trees.node(trees.belowThresholdRoot(index));
        auto const& above = trees.node(trees.atOrAboveThresholdRoot(index));
        EXPECT_EQ(below.particles.begin, range.begin);
        EXPECT_EQ(below.particles.end, above.particles.begin);
        EXPECT_EQ(above.particles.end, range.end);
        for (ParticleCount particle = below.particles.begin; particle < below.particles.end; particle++) {
            EXPECT_LT(particles.flags(particle).rung(), kThreshold);
        }
        for (ParticleCount particle = above.particles.begin; particle < above.particles.end; particle++) {
            EXPECT_GE(particles.flags(particle).rung(), kThreshold);
        }
        visited += checkTree(trees, trees.belowThresholdRoot(index), below.particles, fixture.cellLower(index), particles, layout.spacing(), options);
        visited += checkTree(trees, trees.atOrAboveThresholdRoot(index), above.particles, fixture.cellLower(index), particles, layout.spacing(), options);
    }
    EXPECT_EQ(visited, trees.nodeCount());
}

TEST(KdTree, GeometricCenterRetainsDipole) {
    PROFILE();
    KdTreeFixture fixture;
    auto& grid = fixture.grid();
    auto const& layout = grid.layout();
    tree::KdTreeParameters parameters{};
    parameters.particleMass = kParticleMass;
    parameters.multipoleCenter = tree::MultipoleCenter::geometricCenter;
    auto const trees = tree::CellKdTrees::buildByPosition(grid, fixture.particles(), parameters);
    CheckOptions options{};
    options.center = tree::MultipoleCenter::geometricCenter;
    bool sawDipole = false;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        size_t const root = trees.positionRoot(index);
        checkTree(
            trees, root, grid.cell(layout.cellCoordinates(index)).localParticles, fixture.cellLower(index), fixture.particles(), layout.spacing(), options);
        sawDipole = sawDipole || trees.multipole(root)[stf::index(1, 0, 0)] != 0.0f;
    }
    EXPECT_TRUE(sawDipole);
}

// Splitting criteria are replaceable: this one always halves the x extent.
class XOnlySplitter final : public tree::KdSplitter {
public:
    tree::KdSplit chooseSplit(tree::KdSplitContext const& node) const override {
        PROFILE();
        std::uint32_t const lower = node.box.lower[0];
        std::uint32_t const upper = node.box.upper[0];
        if (upper - lower < 2) {
            return {};
        }
        return {true, 0, lower + (upper - lower) / 2};
    }
};

class BoundarySplitter final : public tree::KdSplitter {
public:
    tree::KdSplit chooseSplit(tree::KdSplitContext const& node) const override {
        PROFILE();
        return {true, 1, node.box.lower[1]};
    }
};

TEST(KdTree, CustomSplitterReplacesGeometricAlternation) {
    PROFILE();
    KdTreeFixture fixture;
    auto& grid = fixture.grid();
    auto const& layout = grid.layout();
    tree::KdTreeParameters parameters{};
    parameters.particleMass = kParticleMass;
    auto const trees = tree::CellKdTrees::buildByPosition(grid, fixture.particles(), parameters, XOnlySplitter{}, fixture.companions());
    fixture.expectPermutationPreserved();
    CheckOptions options{};
    options.fixedAxis = 0;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        checkTree(trees, trees.positionRoot(index), grid.cell(layout.cellCoordinates(index)).localParticles, fixture.cellLower(index), fixture.particles(),
            layout.spacing(), options);
    }
}

TEST(KdTree, RejectsInvalidRequests) {
    PROFILE();
    KdTreeFixture fixture;
    tree::KdTreeParameters parameters{};
    EXPECT_THROW(tree::CellKdTrees::buildByPosition(fixture.grid(), fixture.particles(), parameters, BoundarySplitter{}), std::logic_error);
    parameters.particleMass = 0.0f;
    EXPECT_THROW(tree::CellKdTrees::buildByPosition(fixture.grid(), fixture.particles(), parameters), std::invalid_argument);
    parameters.particleMass = 1.0f;
    EXPECT_THROW(tree::CellKdTrees::buildByRungThreshold(fixture.grid(), fixture.particles(), 65, parameters), std::invalid_argument);
}

}    // namespace
