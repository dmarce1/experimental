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
#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include "fmm_test_cells.hpp"

namespace {

namespace fmm = cosmictiger::gravity::fmm;
namespace stf = cosmictiger::gravity::stf;
namespace test = cosmictiger::gravity::fmm::test;

using cosmictiger::CellExpansion;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleRange;
using cosmictiger::gravity::kMaximumFmmOrder;
using cosmictiger::gravity::kMinimumFmmOrder;
using Vector = std::array<double, NDIM>;

constexpr Vector kTargetCell{5.5, 5.5, 5.5};
constexpr std::array<Vector, 3> kSourceOffsets{{{5.0, -2.0, 6.0}, {-6.0, 1.0, 4.0}, {1.0, 7.0, -3.0}}};
constexpr size_t kParticlesPerCell = 37;

template <typename Function, int... orders>
void forEachOrder(Function&& function, std::integer_sequence<int, orders...>) {
    PROFILE();
    (function(std::integral_constant<int, kMinimumFmmOrder + orders>{}), ...);
}

template <typename Function>
void forEachOrder(Function&& function) {
    PROFILE();
    forEachOrder(std::forward<Function>(function), std::make_integer_sequence<int, kMaximumFmmOrder - kMinimumFmmOrder + 1>{});
}

Vector scaled(Vector const& cells) {
    PROFILE();
    return {cells[0] * test::kSpacing, cells[1] * test::kSpacing, cells[2] * test::kSpacing};
}

// A coefficient must match to float rounding relative to its own rank.
void expectCoefficientsNear(float const* actual, double const* expected, int order, double relativeTolerance, char const* label) {
    PROFILE();
    for (int rank = 0; rank <= order; rank++) {
        double rankScale = 0.0;
        for (int coefficient = rank * rank; coefficient < (rank + 1) * (rank + 1); coefficient++) {
            rankScale = std::max(rankScale, std::abs(expected[coefficient]));
        }
        for (int coefficient = rank * rank; coefficient < (rank + 1) * (rank + 1); coefficient++) {
            EXPECT_NEAR(actual[coefficient], expected[coefficient], relativeTolerance * rankScale + 1e-30)
                << label << " order " << order << " rank " << rank << " coefficient " << coefficient;
        }
    }
}

class Scene {
public:
    Scene() {
        PROFILE();
        targetCenter_ = scaled(kTargetCell);
        cells_.addCell(targetCenter_, kParticlesPerCell);
        for (auto const& offset : kSourceOffsets) {
            Vector center{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                center[axis] = targetCenter_[axis] + offset[axis] * test::kSpacing;
            }
            cells_.addCell(center, kParticlesPerCell);
        }
        for (size_t source = 0; source < kSourceOffsets.size(); source++) {
            fmm::FmmMultipoleSource multipoleSource{cells_.multipole(source + 1).data(), {}};
            fmm::FmmParticleSource particleSource{cells_.range(source + 1), cells_.center(source + 1), {}};
            for (size_t axis = 0; axis < NDIM; axis++) {
                multipoleSource.separation[axis] = -kSourceOffsets[source][axis] * test::kSpacing;
                particleSource.separation[axis] = multipoleSource.separation[axis];
            }
            multipoleSources_.push_back(multipoleSource);
            particleSources_.push_back(particleSource);
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions_[axis] = cells_.positions(axis).data();
        }
    }

    test::TestCells const& cells() const {
        PROFILE();
        return cells_;
    }

    std::vector<fmm::FmmMultipoleSource> const& multipoleSources() const {
        PROFILE();
        return multipoleSources_;
    }

    std::vector<fmm::FmmParticleSource> const& particleSources() const {
        PROFILE();
        return particleSources_;
    }

    cosmictiger::Coordinate const* const* positions() const {
        PROFILE();
        return positions_.data();
    }

    // Offset of a particle from the target cell center.
    Vector targetOffset(ParticleCount particle) const {
        PROFILE();
        auto const center = test::axesOf(cells_.center(0));
        Vector result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = test::wrappedOffset(cells_.positions(axis)[particle], center[axis]);
        }
        return result;
    }

    // Position of a source particle relative to the target cell center.
    Vector sourceRelativeToTarget(size_t source, ParticleCount particle) const {
        PROFILE();
        auto const center = test::axesOf(cells_.center(source + 1));
        Vector result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = test::wrappedOffset(cells_.positions(axis)[particle], center[axis]) + kSourceOffsets[source][axis] * test::kSpacing;
        }
        return result;
    }

    // Newtonian potential and acceleration at a point relative to the target center.
    std::array<double, NDIM + 1> directField(Vector const& point) const {
        PROFILE();
        std::array<double, NDIM + 1> field{};
        for (size_t source = 0; source < kSourceOffsets.size(); source++) {
            auto const range = cells_.range(source + 1);
            for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                auto const position = sourceRelativeToTarget(source, particle);
                Vector difference{};
                double radiusSquared = 0.0;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    difference[axis] = point[axis] - position[axis];
                    radiusSquared += difference[axis] * difference[axis];
                }
                double const inverseRadius = 1.0 / std::sqrt(radiusSquared);
                double const inverseCube = inverseRadius * inverseRadius * inverseRadius;
                field[0] -= test::kParticleMass * inverseRadius;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    field[axis + 1] -= test::kParticleMass * difference[axis] * inverseCube;
                }
            }
        }
        return field;
    }

private:
    Vector targetCenter_{};
    test::TestCells cells_;
    std::vector<fmm::FmmMultipoleSource> multipoleSources_;
    std::vector<fmm::FmmParticleSource> particleSources_;
    std::array<cosmictiger::Coordinate const*, NDIM> positions_{};
};

// Evaluate a cell expansion at displacement from its center, as the PMMM L2P does.
std::array<double, NDIM + 1> evaluateLocal(CellExpansion const& expansion, int order, Vector const& displacement) {
    constexpr int kOrder = kMaximumFmmOrder;
    PROFILE();
    stf::Expansion<double, kOrder> local{};
    for (size_t coefficient = 0; coefficient < stf::coefficientCount(order); coefficient++) {
        local[coefficient] = expansion[coefficient];
    }
    stf::Expansion<double, kOrder> point{};
    stf::addL2L<double, kOrder>(point, local, displacement, test::kSpacing, 1.0);
    return {point[0], -point[stf::index(1, 0, 0)], -point[stf::index(0, 1, 0)], -point[stf::index(0, 0, 1)]};
}

void expectFieldNear(std::array<double, NDIM + 1> const& actual, std::array<double, NDIM + 1> const& expected, double relativeTolerance, char const* label) {
    PROFILE();
    double const accelerationScale = std::sqrt(expected[1] * expected[1] + expected[2] * expected[2] + expected[3] * expected[3]);
    EXPECT_NEAR(actual[0], expected[0], relativeTolerance * std::abs(expected[0])) << label << " potential";
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(actual[axis + 1], expected[axis + 1], relativeTolerance * accelerationScale) << label << " acceleration " << axis;
    }
}

fmm::FmmM2lRequest m2lRequest(Scene const& scene, fmm::FmmExpansionTarget const* targets, size_t targetCount, int order) {
    PROFILE();
    return {targets, targetCount, scene.multipoleSources().data(), scene.multipoleSources().size(), test::kSpacing, order};
}

fmm::FmmP2lRequest p2lRequest(Scene const& scene, fmm::FmmExpansionTarget const* targets, size_t targetCount, int order) {
    PROFILE();
    fmm::FmmP2lRequest request{
        targets, targetCount, scene.particleSources().data(), scene.particleSources().size(), {}, test::kParticleMass, test::kSpacing, order};
    for (size_t axis = 0; axis < NDIM; axis++) {
        request.positions[axis] = scene.positions()[axis];
    }
    return request;
}

TEST(FmmInteractionsCpu, M2lMatchesGeneratedReferenceAtEveryOrder) {
    PROFILE();
    Scene const scene;
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        CellExpansion expansion{};
        fmm::FmmExpansionTarget const target{expansion.data(), 0, scene.multipoleSources().size()};
        fmm::addM2lCpu(m2lRequest(scene, &target, 1, kOrder));

        stf::Expansion<double, kOrder> reference{};
        for (size_t source = 0; source < scene.multipoleSources().size(); source++) {
            stf::Multipole<double, kOrder - 1> moments{};
            for (size_t coefficient = 0; coefficient < moments.size(); coefficient++) {
                moments[coefficient] = scene.cells().multipole(source + 1)[coefficient];
            }
            auto const& separation = scene.multipoleSources()[source].separation;
            stf::addM2L<double, kOrder - 1, kOrder>(reference, moments, {separation[0], separation[1], separation[2]}, test::kSpacing, test::kSpacing);
        }
        expectCoefficientsNear(expansion.data(), reference.data(), kOrder, 1e-6, "M2L");
    });
}

TEST(FmmInteractionsCpu, P2lMatchesGeneratedReferenceAtEveryOrder) {
    PROFILE();
    Scene const scene;
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        CellExpansion expansion{};
        fmm::FmmExpansionTarget const target{expansion.data(), 0, scene.particleSources().size()};
        fmm::addP2lCpu(p2lRequest(scene, &target, 1, kOrder));

        stf::Expansion<double, kOrder> reference{};
        for (size_t source = 0; source < scene.particleSources().size(); source++) {
            auto const range = scene.cells().range(source + 1);
            for (ParticleCount particle = range.begin; particle < range.end; particle++) {
                auto const position = scene.sourceRelativeToTarget(source, particle);
                stf::addP2L<double, kOrder>(reference, test::kParticleMass, {-position[0], -position[1], -position[2]}, test::kSpacing);
            }
        }
        expectCoefficientsNear(expansion.data(), reference.data(), kOrder, 1e-6, "P2L");
    });
}

TEST(FmmInteractionsCpu, M2pMatchesGeneratedReferenceAtEveryOrder) {
    PROFILE();
    Scene const scene;
    auto const sinks = scene.cells().range(0);
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        size_t const count = scene.cells().particleCount();
        std::array<std::vector<float>, NDIM> accelerations;
        for (auto& values : accelerations) {
            values.assign(count, 0.0f);
        }
        std::vector<float> potentials(count, 0.0f);
        fmm::FmmParticleTarget const target{sinks, scene.cells().center(0), 0, scene.multipoleSources().size()};
        fmm::FmmM2pRequest request{
            &target, 1, scene.multipoleSources().data(), scene.multipoleSources().size(), {}, {}, potentials.data(), test::kSpacing, kOrder};
        for (size_t axis = 0; axis < NDIM; axis++) {
            request.positions[axis] = scene.positions()[axis];
            request.accelerations[axis] = accelerations[axis].data();
        }
        fmm::addM2pCpu(request);

        for (ParticleCount sink = sinks.begin; sink < sinks.end; sink++) {
            auto const offset = scene.targetOffset(sink);
            std::array<double, NDIM + 1> reference{};
            for (size_t source = 0; source < scene.multipoleSources().size(); source++) {
                stf::Multipole<double, kOrder - 1> moments{};
                for (size_t coefficient = 0; coefficient < moments.size(); coefficient++) {
                    moments[coefficient] = scene.cells().multipole(source + 1)[coefficient];
                }
                auto const& separation = scene.multipoleSources()[source].separation;
                auto const field = stf::evaluateM2P<double, kOrder - 1>(
                    moments, {offset[0] + separation[0], offset[1] + separation[1], offset[2] + separation[2]}, test::kSpacing);
                reference[0] += field.potential;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    reference[axis + 1] += field.acceleration[axis];
                }
            }
            expectFieldNear({potentials[sink], accelerations[0][sink], accelerations[1][sink], accelerations[2][sink]}, reference, 1e-6, "M2P");
        }
        // Particles outside the sink range are never written.
        EXPECT_EQ(potentials[sinks.end], 0.0f);
    });
}

TEST(FmmInteractionsCpu, NinthOrderFieldsMatchDirectNewtonianSum) {
    constexpr int kOrder = kMaximumFmmOrder;
    constexpr double kTolerance = 2e-5;
    PROFILE();
    Scene const scene;
    auto const sinks = scene.cells().range(0);
    CellExpansion fromMultipoles{};
    CellExpansion fromParticles{};
    fmm::FmmExpansionTarget const m2lTarget{fromMultipoles.data(), 0, scene.multipoleSources().size()};
    fmm::FmmExpansionTarget const p2lTarget{fromParticles.data(), 0, scene.particleSources().size()};
    fmm::addM2lCpu(m2lRequest(scene, &m2lTarget, 1, kOrder));
    fmm::addP2lCpu(p2lRequest(scene, &p2lTarget, 1, kOrder));

    size_t const count = scene.cells().particleCount();
    std::array<std::vector<float>, NDIM> accelerations;
    for (auto& values : accelerations) {
        values.assign(count, 0.0f);
    }
    std::vector<float> potentials(count, 0.0f);
    fmm::FmmParticleTarget const m2pTarget{sinks, scene.cells().center(0), 0, scene.multipoleSources().size()};
    fmm::FmmM2pRequest request{
        &m2pTarget, 1, scene.multipoleSources().data(), scene.multipoleSources().size(), {}, {}, potentials.data(), test::kSpacing, kOrder};
    for (size_t axis = 0; axis < NDIM; axis++) {
        request.positions[axis] = scene.positions()[axis];
        request.accelerations[axis] = accelerations[axis].data();
    }
    fmm::addM2pCpu(request);

    for (ParticleCount sink = sinks.begin; sink < sinks.end; sink++) {
        auto const offset = scene.targetOffset(sink);
        auto const expected = scene.directField(offset);
        expectFieldNear(evaluateLocal(fromMultipoles, kOrder, offset), expected, kTolerance, "M2L+L2P");
        expectFieldNear(evaluateLocal(fromParticles, kOrder, offset), expected, kTolerance, "P2L+L2P");
        expectFieldNear({potentials[sink], accelerations[0][sink], accelerations[1][sink], accelerations[2][sink]}, expected, kTolerance, "M2P");
    }
}

// The source cell straddles the periodic boundary at the origin, and the
// target cell lies on the far side of the z boundary.
TEST(FmmInteractionsCpu, MinimumImageOffsetsAcrossThePeriodicBoundary) {
    constexpr int kOrder = kMaximumFmmOrder;
    constexpr double kTolerance = 2e-5;
    PROFILE();
    test::TestCells cells;
    Vector const sourceCenter{0.0, 0.0, 0.0};
    Vector const shift{5.0 * test::kSpacing, 2.0 * test::kSpacing, -4.0 * test::kSpacing};
    auto const sources = cells.addCell(sourceCenter, kParticlesPerCell);
    auto const sinks = cells.addCell(shift, kParticlesPerCell);
    std::array<cosmictiger::Coordinate const*, NDIM> positions{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        positions[axis] = cells.positions(axis).data();
    }
    auto const sourceAxes = test::axesOf(cells.center(0));
    auto const sinkAxes = test::axesOf(cells.center(1));
    EXPECT_GT(cells.center(1).z.raw(), 0x80000000u);

    auto directAt = [&](Vector const& point) {
        PROFILE();
        std::array<double, NDIM + 1> field{};
        for (ParticleCount particle = sources.begin; particle < sources.end; particle++) {
            Vector difference{};
            double radiusSquared = 0.0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                difference[axis] = point[axis] + shift[axis] - test::wrappedOffset(cells.positions(axis)[particle], sourceAxes[axis]);
                radiusSquared += difference[axis] * difference[axis];
            }
            double const inverseRadius = 1.0 / std::sqrt(radiusSquared);
            field[0] -= test::kParticleMass * inverseRadius;
            for (size_t axis = 0; axis < NDIM; axis++) {
                field[axis + 1] -= test::kParticleMass * difference[axis] * inverseRadius * inverseRadius * inverseRadius;
            }
        }
        return field;
    };

    CellExpansion expansion{};
    fmm::FmmParticleSource const particleSource{sources, cells.center(0), {shift[0], shift[1], shift[2]}};
    fmm::FmmExpansionTarget const expansionTarget{expansion.data(), 0, 1};
    fmm::FmmP2lRequest p2l{&expansionTarget, 1, &particleSource, 1, {}, test::kParticleMass, test::kSpacing, kOrder};
    for (size_t axis = 0; axis < NDIM; axis++) {
        p2l.positions[axis] = positions[axis];
    }
    fmm::addP2lCpu(p2l);

    size_t const count = cells.particleCount();
    std::array<std::vector<float>, NDIM> accelerations;
    for (auto& values : accelerations) {
        values.assign(count, 0.0f);
    }
    std::vector<float> potentials(count, 0.0f);
    fmm::FmmMultipoleSource const multipoleSource{cells.multipole(0).data(), {shift[0], shift[1], shift[2]}};
    fmm::FmmParticleTarget const particleTarget{sinks, cells.center(1), 0, 1};
    fmm::FmmM2pRequest m2p{&particleTarget, 1, &multipoleSource, 1, {}, {}, potentials.data(), test::kSpacing, kOrder};
    for (size_t axis = 0; axis < NDIM; axis++) {
        m2p.positions[axis] = positions[axis];
        m2p.accelerations[axis] = accelerations[axis].data();
    }
    fmm::addM2pCpu(m2p);

    for (ParticleCount sink = sinks.begin; sink < sinks.end; sink++) {
        Vector offset{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            offset[axis] = test::wrappedOffset(cells.positions(axis)[sink], sinkAxes[axis]);
        }
        auto const expected = directAt(offset);
        expectFieldNear(evaluateLocal(expansion, kOrder, offset), expected, kTolerance, "periodic P2L");
        expectFieldNear({potentials[sink], accelerations[0][sink], accelerations[1][sink], accelerations[2][sink]}, expected, kTolerance, "periodic M2P");
    }
}

TEST(FmmInteractionsCpu, AccumulatesActiveRanksAndPreservesTheRest) {
    constexpr int kOrder = 4;
    constexpr float kSentinel = 0.625f;
    constexpr size_t kActiveCount = stf::coefficientCount(kOrder);
    PROFILE();
    Scene const scene;
    CellExpansion fresh{};
    CellExpansion preset{};
    preset.fill(kSentinel);
    fmm::FmmExpansionTarget const freshTarget{fresh.data(), 0, scene.multipoleSources().size()};
    fmm::FmmExpansionTarget const presetTarget{preset.data(), 0, scene.multipoleSources().size()};
    fmm::addM2lCpu(m2lRequest(scene, &freshTarget, 1, kOrder));
    fmm::addM2lCpu(m2lRequest(scene, &presetTarget, 1, kOrder));
    for (size_t coefficient = 0; coefficient < preset.size(); coefficient++) {
        if (coefficient < kActiveCount) {
            EXPECT_FLOAT_EQ(preset[coefficient], kSentinel + fresh[coefficient]) << coefficient;
        } else {
            EXPECT_EQ(preset[coefficient], kSentinel) << coefficient;
            EXPECT_EQ(fresh[coefficient], 0.0f) << coefficient;
        }
    }
    // An empty source list leaves its target unchanged.
    CellExpansion untouched{};
    untouched.fill(kSentinel);
    fmm::FmmExpansionTarget const emptyTarget{untouched.data(), 1, 1};
    fmm::addM2lCpu(m2lRequest(scene, &emptyTarget, 1, kOrder));
    for (float const value : untouched) {
        EXPECT_EQ(value, kSentinel);
    }
}

TEST(FmmInteractionsCpu, ParallelTargetsMatchSingleTargetEvaluation) {
    constexpr int kOrder = 7;
    constexpr size_t kTargetCount = 67;
    PROFILE();
    Scene const scene;
    CellExpansion single{};
    fmm::FmmExpansionTarget const singleTarget{single.data(), 0, scene.particleSources().size()};
    fmm::addP2lCpu(p2lRequest(scene, &singleTarget, 1, kOrder));

    std::vector<CellExpansion> expansions(kTargetCount);
    std::vector<fmm::FmmExpansionTarget> targets;
    for (auto& expansion : expansions) {
        targets.push_back({expansion.data(), 0, scene.particleSources().size()});
    }
    fmm::addP2lCpu(p2lRequest(scene, targets.data(), targets.size(), kOrder));
    for (auto const& expansion : expansions) {
        EXPECT_EQ(expansion, single);
    }
}

TEST(FmmInteractionsCpu, RejectsInvalidRequests) {
    PROFILE();
    Scene const scene;
    CellExpansion expansion{};
    fmm::FmmExpansionTarget const target{expansion.data(), 0, scene.multipoleSources().size()};
    EXPECT_THROW(fmm::addM2lCpu(m2lRequest(scene, &target, 1, kMinimumFmmOrder - 1)), std::invalid_argument);
    EXPECT_THROW(fmm::addM2lCpu(m2lRequest(scene, &target, 1, kMaximumFmmOrder + 1)), std::invalid_argument);
    auto zeroSpacing = m2lRequest(scene, &target, 1, kMinimumFmmOrder);
    zeroSpacing.spacing = 0.0;
    EXPECT_THROW(fmm::addM2lCpu(zeroSpacing), std::invalid_argument);
    EXPECT_THROW(fmm::addM2lCpu(m2lRequest(scene, nullptr, 1, kMinimumFmmOrder)), std::invalid_argument);
    auto missingPositions = p2lRequest(scene, &target, 1, kMinimumFmmOrder);
    missingPositions.positions[1] = nullptr;
    EXPECT_THROW(fmm::addP2lCpu(missingPositions), std::invalid_argument);
    fmm::FmmParticleTarget const sinkTarget{scene.cells().range(0), scene.cells().center(0), 0, scene.multipoleSources().size()};
    fmm::FmmM2pRequest missingOutputs{
        &sinkTarget, 1, scene.multipoleSources().data(), scene.multipoleSources().size(), {}, {}, nullptr, test::kSpacing, kMinimumFmmOrder};
    for (size_t axis = 0; axis < NDIM; axis++) {
        missingOutputs.positions[axis] = scene.positions()[axis];
    }
    EXPECT_THROW(fmm::addM2pCpu(missingOutputs), std::invalid_argument);

    // A valid empty batch touches no pointers.
    EXPECT_NO_THROW(fmm::addM2lCpu(fmm::FmmM2lRequest{nullptr, 0, nullptr, 0, test::kSpacing, kMinimumFmmOrder}));
    EXPECT_NO_THROW(fmm::addP2lCpu(fmm::FmmP2lRequest{nullptr, 0, nullptr, 0, {}, test::kParticleMass, test::kSpacing, kMinimumFmmOrder}));
    EXPECT_NO_THROW(fmm::addM2pCpu(fmm::FmmM2pRequest{nullptr, 0, nullptr, 0, {}, {}, nullptr, test::kSpacing, kMinimumFmmOrder}));
}

}    // namespace
