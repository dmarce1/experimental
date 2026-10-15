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
#include <cstdio>
#include <map>
#include <random>
#include <stddef.h>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/gravity/tree/kd_tree.hpp>
#include <cosmictiger/gravity/tree/tree_near_field.hpp>
#include <cosmictiger/profiler.hpp>

namespace {

using namespace cosmictiger;
using namespace cosmictiger::gravity;

constexpr size_t kMeshExponent = 3;
constexpr size_t kUniformCount = 2000;
constexpr size_t kClusterCount = 3000;
constexpr double kClusterWidth = 0.02;
constexpr double kSofteningCellFraction = 0.05;
constexpr int kFerrersOrder = 2;
constexpr std::uint64_t kSeed = 42;

// Interior particles in cell order with their ranges assigned, half of them
// uniform and half in one dense cluster, so some cells need deep trees.
void seedParticles(
    CartesianGrid& grid, ParticleStorage& particles, bool storeFields, size_t uniformCount = kUniformCount, size_t clusterCount = kClusterCount) {
    PROFILE();
    auto const& layout = grid.layout();
    std::mt19937_64 generator(kSeed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, kClusterWidth);
    std::vector<std::array<Coordinate, NDIM>> positions;
    for (size_t index = 0; index < uniformCount + clusterCount; index++) {
        std::array<Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            // The cluster straddles a cell corner and the periodic seam.
            double value = index < uniformCount ? uniform(generator) : normal(generator) + (axis == 0 ? 0.0 : 0.25);
            value -= std::floor(value);
            position[axis] = Coordinate(value);
        }
        positions.push_back(position);
    }
    std::stable_sort(positions.begin(), positions.end(), [&layout](auto const& first, auto const& second) {
        return layout.cellIndex(layout.particleOwner(first).coordinates) < layout.cellIndex(layout.particleOwner(second).coordinates);
    });
    particles.configureGravityFields(storeFields, storeFields);
    for (auto const& position : positions) {
        particles.append(position, {0.0f, 0.0f, 0.0f}, 0);
    }
    ParticleCount cursor = 0;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        if (!layout.isInterior(coordinates)) {
            continue;
        }
        ParticleCount const begin = cursor;
        while (cursor < particles.size()) {
            std::array<Coordinate, NDIM> position{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                position[axis] = particles.position(axis, cursor);
            }
            if (layout.particleOwner(position).coordinates != coordinates) {
                break;
            }
            cursor++;
        }
        grid.setInteriorRange(coordinates, {begin, cursor});
    }
    ASSERT_EQ(cursor, particles.size());
}

struct Walk {
    std::vector<tree::TreeInteraction> interactions;
    tree::CellKdTrees trees;
};

Walk walkNearField(CartesianGrid& grid, ParticleStorage& particles, int ghostWidth, double openingAngle) {
    PROFILE();
    tree::KdTreeParameters parameters;
    parameters.multipoleCenter = tree::MultipoleCenter::minimumRadius;
    parameters.particleMass = 1.0f;
    Walk result;
    result.trees = tree::CellKdTrees::buildByRungThreshold(grid, particles, 0, parameters);
    std::array<std::vector<float>, NDIM> accelerations;
    for (auto& values : accelerations) {
        values.assign(particles.size(), 0.0f);
    }
    tree::TreeNearFieldRequest request;
    request.grid = &grid;
    request.particles = &particles;
    request.trees = &result.trees;
    request.softening = makeFerrersSoftening(static_cast<float>(kSofteningCellFraction * grid.layout().spacing()));
    request.ferrersOrder = kFerrersOrder;
    request.fmmOrder = 6;
    request.ghostWidth = ghostWidth;
    request.openingAngle = openingAngle;
    for (size_t axis = 0; axis < NDIM; axis++) {
        request.accelerations[axis] = accelerations[axis].data();
    }
    request.interactions = &result.interactions;
    tree::evaluateTreeNearField(request);
    return result;
}

ParticleCount nodeCount(tree::CellKdTrees const& trees, size_t node) {
    PROFILE();
    auto const range = trees.node(node).particles;
    return range.end - range.begin;
}

class TreeNearFieldTest : public ::testing::TestWithParam<std::tuple<int, double>> {};

// Every sink sees every other particle of its near region exactly once,
// through exactly one accepted node pair.
TEST_P(TreeNearFieldTest, EveryNearPairIsCoveredExactlyOnce) {
    PROFILE();
    auto const [ghostWidth, openingAngle] = GetParam();
    CartesianGrid grid(kMeshExponent, 0, 0, static_cast<size_t>(ghostWidth));
    ParticleStorage particles;
    seedParticles(grid, particles, false);
    auto const walk = walkNearField(grid, particles, ghostWidth, openingAngle);
    auto const& layout = grid.layout();
    std::vector<ParticleCount> covered(particles.size(), 0);
    std::array<size_t, 4> typeCounts{};
    for (auto const& interaction : walk.interactions) {
        typeCounts[static_cast<size_t>(interaction.type)]++;
        auto const sinkRange = walk.trees.node(interaction.sinkNode).particles;
        ParticleCount sources = nodeCount(walk.trees, interaction.sourceNode);
        bool const self = interaction.sinkNode == interaction.sourceNode &&
            std::all_of(interaction.cellOffset.begin(), interaction.cellOffset.end(), [](std::int64_t value) { return value == 0; });
        if (self) {
            EXPECT_EQ(interaction.type, tree::TreeInteractionType::p2p);
            sources--;
        }
        for (ParticleCount sink = sinkRange.begin; sink < sinkRange.end; sink++) {
            covered[sink] += sources;
        }
    }
    auto const side = static_cast<std::int64_t>(layout.globalSideLength());
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        if (!layout.isInterior(coordinates)) {
            continue;
        }
        ParticleCount nearCount = 0;
        for (std::int64_t dz = -ghostWidth; dz <= ghostWidth; dz++) {
            for (std::int64_t dy = -ghostWidth; dy <= ghostWidth; dy++) {
                for (std::int64_t dx = -ghostWidth; dx <= ghostWidth; dx++) {
                    GridIndices neighbor{(coordinates[0] + dx + side) % side, (coordinates[1] + dy + side) % side, (coordinates[2] + dz + side) % side};
                    auto const range = grid.cell(neighbor).localParticles;
                    nearCount += range.end - range.begin;
                }
            }
        }
        auto const range = grid.cell(coordinates).localParticles;
        for (ParticleCount sink = range.begin; sink < range.end; sink++) {
            ASSERT_EQ(covered[sink], nearCount - 1) << "sink " << sink;
        }
    }
    std::printf("tree interactions: width=%d theta=%.2f m2l=%zu p2l=%zu m2p=%zu p2p=%zu\n", ghostWidth, openingAngle, typeCounts[0], typeCounts[1],
        typeCounts[2], typeCounts[3]);
    std::fflush(stdout);
    EXPECT_GT(typeCounts[static_cast<size_t>(tree::TreeInteractionType::m2l)], 0u);
}

// The pair (A, B) receives mirrored interactions from both sides, so the
// forces cancel when every particle is a sink.
TEST_P(TreeNearFieldTest, InteractionListsAreMirrorSymmetric) {
    PROFILE();
    auto const [ghostWidth, openingAngle] = GetParam();
    CartesianGrid grid(kMeshExponent, 0, 0, static_cast<size_t>(ghostWidth));
    ParticleStorage particles;
    seedParticles(grid, particles, false);
    auto const walk = walkNearField(grid, particles, ghostWidth, openingAngle);
    using Key = std::tuple<size_t, size_t, std::int64_t, std::int64_t, std::int64_t>;
    std::map<Key, tree::TreeInteractionType> types;
    for (auto const& interaction : walk.interactions) {
        Key const key{interaction.sinkNode, interaction.sourceNode, interaction.cellOffset[0], interaction.cellOffset[1], interaction.cellOffset[2]};
        EXPECT_TRUE(types.emplace(key, interaction.type).second) << "duplicate interaction";
    }
    auto mirror = [](tree::TreeInteractionType type) {
        switch (type) {
        case tree::TreeInteractionType::m2p:
            return tree::TreeInteractionType::p2l;
        case tree::TreeInteractionType::p2l:
            return tree::TreeInteractionType::m2p;
        default:
            return type;
        }
    };
    size_t mismatches = 0;
    for (auto const& [key, type] : types) {
        auto const [sink, source, x, y, z] = key;
        auto const found = types.find(Key{source, sink, -x, -y, -z});
        if (found == types.end() || found->second != mirror(type)) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

INSTANTIATE_TEST_SUITE_P(WidthsAndAngles, TreeNearFieldTest,
    ::testing::Values(std::tuple{1, 0.3}, std::tuple{1, 0.55}, std::tuple{1, 0.8}, std::tuple{2, 0.55}, std::tuple{2, 0.8}));

struct FieldSample {
    std::vector<std::array<double, NDIM>> accelerations;
    std::vector<double> potentials;
};

// Near-only diagnostic fields: every cell expansion stays zero.
FieldSample nearFields(KickParameters parameters) {
    PROFILE();
    CartesianGrid grid(kMeshExponent, 0, 0, static_cast<size_t>(parameters.ghostWidth));
    ParticleStorage particles;
    seedParticles(grid, particles, true);
    float const particleMass = 1.0f / static_cast<float>(particles.size());
    auto const softening = makeFerrersSoftening(static_cast<float>(kSofteningCellFraction * grid.layout().spacing()));
    parameters.phase = KickPhase::diagnostics;
    parameters.sumPotential = true;
    kickPmmmParticlesCpu(grid, particles, softening, kFerrersOrder, particleMass, 1.0, 0, parameters);
    // Index fields by position so the two particle orders can be compared.
    std::vector<std::pair<std::array<std::uint32_t, NDIM>, ParticleCount>> order;
    for (ParticleCount index = 0; index < particles.size(); index++) {
        order.push_back({{particles.position(0, index).raw(), particles.position(1, index).raw(), particles.position(2, index).raw()}, index});
    }
    std::sort(order.begin(), order.end());
    FieldSample sample;
    for (auto const& [position, index] : order) {
        sample.accelerations.push_back({particles.acceleration(0, index), particles.acceleration(1, index), particles.acceleration(2, index)});
        sample.potentials.push_back(particles.potential(index));
    }
    return sample;
}

std::pair<double, double> relativeErrors(FieldSample const& actual, FieldSample const& expected) {
    PROFILE();
    double accelerationError = 0.0;
    double accelerationNorm = 0.0;
    double potentialError = 0.0;
    double potentialNorm = 0.0;
    for (size_t index = 0; index < expected.accelerations.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const delta = actual.accelerations[index][axis] - expected.accelerations[index][axis];
            accelerationError += delta * delta;
            accelerationNorm += expected.accelerations[index][axis] * expected.accelerations[index][axis];
        }
        double const delta = actual.potentials[index] - expected.potentials[index];
        potentialError += delta * delta;
        potentialNorm += expected.potentials[index] * expected.potentials[index];
    }
    return {std::sqrt(accelerationError / accelerationNorm), std::sqrt(potentialError / potentialNorm)};
}

// The tree near field converges to the direct near field as theta shrinks
// and as the expansion order rises.
TEST(TreeNearFieldAccuracy, MatchesDirectNearField) {
    PROFILE();
    for (int const ghostWidth : {1, 2}) {
        KickParameters direct;
        direct.ghostWidth = ghostWidth;
        auto const expected = nearFields(direct);
        double previous = 0.0;
        for (double const openingAngle : {0.8, 0.55, 0.3}) {
            for (int const fmmOrder : {4, 6}) {
                KickParameters treeParameters = direct;
                treeParameters.nearFieldMethod = NearFieldMethod::tree;
                treeParameters.openingAngle = openingAngle;
                treeParameters.fmmOrder = fmmOrder;
                auto const [acceleration, potential] = relativeErrors(nearFields(treeParameters), expected);
                std::printf("tree accuracy: width=%d theta=%.2f order=%d acceleration=%.3e potential=%.3e\n", ghostWidth, openingAngle, fmmOrder, acceleration,
                    potential);
                std::fflush(stdout);
                SCOPED_TRACE(::testing::Message() << "width=" << ghostWidth << " theta=" << openingAngle << " order=" << fmmOrder);
                EXPECT_LT(acceleration, 1.0e-2);
                EXPECT_LT(potential, 1.0e-2);
                if (openingAngle < 0.5 && fmmOrder == 6) {
                    EXPECT_LT(acceleration, 1.0e-5);
                    EXPECT_LT(potential, 1.0e-5);
                }
                if (fmmOrder == 6) {
                    if (previous != 0.0) {
                        EXPECT_LT(acceleration, previous);
                    }
                    previous = acceleration;
                }
            }
        }
    }
}

// Opt-in timing of one complete near-only diagnostic kick per solver:
// --gtest_also_run_disabled_tests --gtest_filter=*Benchmark*
TEST(TreeNearFieldBenchmark, DISABLED_DirectAndTreeKickTimes) {
    constexpr size_t kBenchmarkUniform = 131072;
    constexpr size_t kBenchmarkCluster = 131072;
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();
    for (size_t const meshExponent : {size_t(3), size_t(4)}) {
        for (auto const method : {NearFieldMethod::direct, NearFieldMethod::tree}) {
            CartesianGrid grid(meshExponent, 0, 0, 1);
            ParticleStorage particles;
            seedParticles(grid, particles, true, kBenchmarkUniform, kBenchmarkCluster);
            float const particleMass = 1.0f / static_cast<float>(particles.size());
            auto const softening = makeFerrersSoftening(static_cast<float>(kSofteningCellFraction * grid.layout().spacing()));
            KickParameters parameters;
            parameters.ghostWidth = 1;
            parameters.phase = KickPhase::diagnostics;
            parameters.nearFieldMethod = method;
            auto const start = hpx::chrono::high_resolution_clock::now();
            kickPmmmParticlesCpu(grid, particles, softening, kFerrersOrder, particleMass, 1.0, 0, parameters);
            double const seconds = double(hpx::chrono::high_resolution_clock::now() - start) * kSecondsPerNanosecond;
            std::printf("near-field benchmark: particles=%zu cells=%zu^3 method=%s seconds=%.3f\n", particles.size(), size_t(1) << meshExponent,
                method == NearFieldMethod::tree ? "tree" : "direct", seconds);
            std::fflush(stdout);
        }
    }
}

}    // namespace
