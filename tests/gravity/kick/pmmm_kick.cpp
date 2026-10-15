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

#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <gtest/gtest.h>
#include <hpx/bit.hpp>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace cosmictiger;
using namespace cosmictiger::gravity;

class PmmmKickTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        try {
            if (cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires a CUDA device";
            }
            auto const properties = cuda::deviceProperties(cuda::device().get()).get();
            if (!properties.managedMemory) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires managed memory";
            }
        } catch (cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires a device and driver";
            }
            throw;
        }
#endif
    }
};

void assignRanges(CartesianGrid& grid, ParticleStorage const& particles) {
    PROFILE();
    ParticleCount cursor = 0;
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        if (!grid.layout().isInterior(coordinates)) {
            continue;
        }
        ParticleCount const begin = cursor;
        while (cursor < particles.size()) {
            std::array<Coordinate, NDIM> position{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                position[axis] = particles.position(axis, cursor);
            }
            auto const owner = grid.layout().particleOwner(position);
            if (owner.coordinates != coordinates || owner.localityIndex != grid.layout().localityIndex()) {
                break;
            }
            cursor++;
        }
        grid.setInteriorRange(coordinates, {begin, cursor});
    }
    ASSERT_EQ(cursor, particles.size());
}

Position position(ParticleStorage const& particles, ParticleCount index) {
    PROFILE();
    return {particles.position(0, index), particles.position(1, index), particles.position(2, index)};
}

std::vector<std::uint32_t> particleSnapshot(ParticleStorage const& particles) {
    PROFILE();
    std::vector<std::uint32_t> result;
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            result.push_back(particles.position(axis, index).raw());
            result.push_back(hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index)));
            if (particles.storesAcceleration()) {
                result.push_back(hpx::bit_cast<std::uint32_t>(particles.acceleration(axis, index)));
            }
        }
        if (particles.storesPotential()) {
            result.push_back(hpx::bit_cast<std::uint32_t>(particles.potential(index)));
        }
        result.push_back(particles.flags(index).rung());
        result.push_back(particles.flags(index).isGhost());
    }
    return result;
}

std::vector<ParticleCount> cellSnapshot(CartesianGrid const& grid) {
    PROFILE();
    std::vector<ParticleCount> result;
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const& cell = grid.cell(grid.layout().cellCoordinates(index));
        result.push_back(cell.localParticles.begin);
        result.push_back(cell.localParticles.end);
        result.push_back(cell.ownerParticles.begin);
        result.push_back(cell.ownerParticles.end);
        result.push_back(cell.rungRangesValid);
        for (auto const range : cell.rungParticles) {
            result.push_back(range.begin);
            result.push_back(range.end);
        }
    }
    return result;
}

void expectMatchingKick(CartesianGrid const& actualGrid, ParticleStorage const& actual, CartesianGrid const& expectedGrid, ParticleStorage const& expected,
    KickSums const& actualSums, KickSums const& expectedSums) {
    PROFILE();
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual.storesAcceleration(), expected.storesAcceleration());
    EXPECT_EQ(actual.storesPotential(), expected.storesPotential());
    EXPECT_EQ(particleSnapshot(actual), particleSnapshot(expected));
    EXPECT_EQ(cellSnapshot(actualGrid), cellSnapshot(expectedGrid));
    EXPECT_EQ(actualSums.particleCount, expectedSums.particleCount);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(actualSums.acceleration[axis], expectedSums.acceleration[axis], 1e-12 * (1.0 + std::abs(expectedSums.acceleration[axis])));
    }
    EXPECT_NEAR(actualSums.potential, expectedSums.potential, 1e-12 * (1.0 + std::abs(expectedSums.potential)));
}

void seedOverlapParticles(CartesianGrid& grid, ParticleStorage& particles, bool acceleration, bool potential, bool sorted) {
    PROFILE();
    particles.configureGravityFields(acceleration, potential);
    for (size_t x = 0; x < 8; x++) {
        if (x != 0 && x != 1 && x != 2 && x != 7) {
            continue;
        }
        for (size_t index = 0; index < 3; index++) {
            particles.append(
                {Coordinate((double(x) + 0.375 + 0.125 * double(index % 2)) / 8.0), Coordinate((0.5 + 0.0625 * double(index)) / 8.0), Coordinate(0.0625)},
                {float(x * 16 + index) + 0.25f, -float(index) - 0.5f, float(x) + 0.75f}, index == 1 ? 0 : index + 2);
        }
    }
    assignRanges(grid, particles);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (acceleration) {
                particles.acceleration(axis, index) = float(1024 + index * NDIM + axis);
            }
        }
        if (potential) {
            particles.potential(index) = -float(2048 + index);
        }
    }
    if (sorted) {
        grid.sortParticlesByRung(particles);
    }
}

void installOverlapPolynomial(CartesianGrid& grid) {
    PROFILE();
    double const spacing = grid.layout().spacing();
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        if (!grid.layout().isInterior(coordinates)) {
            continue;
        }
        auto& expansion = grid.expansion(coordinates);
        expansion = {};
        expansion[0] = float(index + 1) * 0.0625f;
        expansion[stf::index(1, 0, 0)] = float(2.0 * spacing);
        expansion[stf::index(0, 1, 0)] = float(-3.0 * spacing);
        expansion[stf::index(0, 0, 1)] = float(4.0 * spacing);
        expansion[stf::index(2, 0, 0)] = float(2.0 * spacing * spacing);
        expansion[stf::index(0, 2, 0)] = float(2.0 * spacing * spacing);
        stf::setLaplacian<float, kExpansionOrder>(expansion, float(6.0 * spacing * spacing));
    }
}

KickParameters overlapParameters(KickPhase phase = KickPhase::initial) {
    PROFILE();
    KickParameters result;
    result.phase = phase;
    result.scaleFactor = 0.25;
    result.sumPotential = true;
    return result;
}

// Declare every referenced future before this guard and launch afterward.
// Failed launches or observations release suspended work, then drain it while
// the test's grid, storage, promises, and future references are still alive.
template <typename Drain>
class GatedTasks {
public:
    GatedTasks(hpx::promise<void>& gate, Drain drain)
      : gate_(gate)
      , drain_(std::move(drain)) {
        PROFILE();
    }

    GatedTasks(GatedTasks const&) = delete;
    GatedTasks& operator=(GatedTasks const&) = delete;

    ~GatedTasks() {
        PROFILE();
        if (!released_) {
            try {
                release();
            } catch (...) {
                // If completion itself failed, still unblock the waiting
                // branch without replacing the exception already unwinding.
                try {
                    gate_.set_exception(std::current_exception());
                } catch (...) {
                    // An already-completed gate is also safe to drain.
                }
            }
        }
        drain_();
    }

    void release() {
        PROFILE();
        if (!released_) {
            gate_.set_value();
            released_ = true;
        }
    }

private:
    hpx::promise<void>& gate_;
    Drain drain_;
    bool released_ = false;
};

TEST_F(PmmmKickTest, CombinesFarPolynomialAndAllNearCellsBeforeOneDiagnosticPerSink) {
    float constexpr kMass = 0.125f;
    double constexpr kDx = 0.03125;
    PROFILE();

    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    particles.append({Coordinate(0.09375), Coordinate(0.0625), Coordinate(0.0625)}, {1.0f, 2.0f, 3.0f}, 0);
    particles.append({Coordinate(0.09375), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    particles.append({Coordinate((double(kPmmmNeighborWidth) + 0.5) / 8.0), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    particles.append({Coordinate(0.9375), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    assignRanges(grid, particles);
    double const spacing = grid.layout().spacing();
    auto& expansion = grid.expansion({0, 0, 0});
    expansion[0] = 5.0f;
    expansion[stf::index(1, 0, 0)] = float(2.0 * spacing);
    expansion[stf::index(0, 1, 0)] = float(3.0 * spacing);
    expansion[stf::index(0, 0, 1)] = float(-4.0 * spacing);
    expansion[stf::index(2, 0, 0)] = float(2.0 * spacing * spacing);
    expansion[stf::index(0, 2, 0)] = float(2.0 * spacing * spacing);
    stf::setLaplacian<float, kExpansionOrder>(expansion, float(6.0 * spacing * spacing));
    auto const softening = makeFerrersSoftening(0.04f);
    std::array<double, NDIM> expected{-2.0 - 2.0 * kDx, -3.0, 4.0};
    double expectedPotential = 5.0 + 2.0 * kDx + kDx * kDx;
    for (ParticleCount source = 1; source < particles.size(); source++) {
        auto const direct = ferrersAccelerationAndPotential<2>(position(particles, 0), position(particles, source), softening);
        expected[0] += kMass * direct.acceleration.x;
        expected[1] += kMass * direct.acceleration.y;
        expected[2] += kMass * direct.acceleration.z;
        expectedPotential += kMass * direct.potential;
    }
    KickParameters parameters;
    parameters.phase = KickPhase::diagnostics;
    parameters.sumPotential = true;
    KickSums sums;
    kickPmmmParticlesCpu(grid, particles, softening, 2, kMass, 1.0, 0, parameters, &sums);
    EXPECT_EQ(sums.particleCount, particles.size());
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(particles.acceleration(axis, 0), expected[axis], 2e-5);
        EXPECT_EQ(particles.velocity(axis, 0), float(axis + 1));
    }
    EXPECT_NEAR(particles.potential(0), expectedPotential, 2e-5);
    double potentialSum = 0.0;
    for (ParticleCount index = 0; index < particles.size(); index++) {
        EXPECT_EQ(particles.flags(index).rung(), 0);
        potentialSum += particles.potential(index);
    }
    EXPECT_DOUBLE_EQ(sums.potential, potentialSum);
}

TEST_F(PmmmKickTest, RuntimeGhostWidthSelectsTheDirectMaskOnEveryBackend) {
    constexpr size_t kMeshExponent = 3;
    constexpr int kSide = 1 << kMeshExponent;
    constexpr float kMass = 0.125f;
    constexpr std::array<int, 4> kSourceCells{0, 1, 2, kSide - 1};
    constexpr double kCellCenterOffset = 0.5;
    constexpr double kFirstCenter = kCellCenterOffset / kSide;
    constexpr float kSoftening = 0.02f;
    constexpr double kTolerance = 2.0e-5;
    PROFILE();

    for (int ghostWidth : {1, 2}) {
        CartesianGrid grid(kMeshExponent, 0, 0, ghostWidth);
        ParticleStorage particles;
        particles.configureGravityFields(true, true);
        for (int cell : kSourceCells) {
            particles.append({Coordinate((cell + kCellCenterOffset) / kSide), Coordinate(kFirstCenter), Coordinate(kFirstCenter)}, {}, 0);
        }
        assignRanges(grid, particles);
        auto const softening = makeFerrersSoftening(kSoftening);
        std::array<double, NDIM> expected{};
        double expectedPotential = 0.0;
        for (size_t source = 1; source < kSourceCells.size(); source++) {
            int const distance = std::min(kSourceCells[source], kSide - kSourceCells[source]);
            if (distance <= ghostWidth) {
                auto const pair = ferrersAccelerationAndPotential<2>(position(particles, 0), position(particles, source), softening);
                expected[0] += kMass * pair.acceleration.x;
                expected[1] += kMass * pair.acceleration.y;
                expected[2] += kMass * pair.acceleration.z;
                expectedPotential += kMass * pair.potential;
            }
        }
        KickParameters parameters;
        parameters.phase = KickPhase::diagnostics;
        parameters.sumPotential = true;
        parameters.ghostWidth = ghostWidth;
        auto check = [&] {
            PROFILE();
            SCOPED_TRACE(ghostWidth);
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(particles.acceleration(axis, 0), expected[axis], kTolerance);
                EXPECT_EQ(particles.velocity(axis, 0), 0.0f);
            }
            EXPECT_NEAR(particles.potential(0), expectedPotential, kTolerance);
        };
        kickPmmmParticlesCpu(grid, particles, softening, 2, kMass, 1.0, 0, parameters);
        check();
        PmmmNearField cpuField(grid, particles, softening, 2, kMass, 1.0, 0, true, false, {}, ghostWidth);
        cpuField.kick(grid, particles, parameters);
        check();
#ifdef COSMICTIGER_ENABLE_CUDA
        PmmmNearField cudaField(grid, particles, softening, 2, kMass, 1.0, 0, true, true, {}, ghostWidth);
        cudaField.kickCuda(grid, particles, parameters);
        check();
#endif
    }
}

TEST_F(PmmmKickTest, RuntimeGhostWidthRejectsInvalidMasksAndMismatchedPreparedFields) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    auto const softening = makeFerrersSoftening(0.01f);
    KickParameters parameters;
    parameters.phase = KickPhase::diagnostics;
    for (int ghostWidth : {0, 3}) {
        parameters.ghostWidth = ghostWidth;
        EXPECT_THROW(kickPmmmParticlesCpu(grid, particles, softening, 1, 1.0f, 1.0, 0, parameters), std::invalid_argument);
    }
    parameters.ghostWidth = 1;
    CartesianGrid narrow(3, 1, 0, 1);
    EXPECT_NO_THROW(kickPmmmParticlesCpu(narrow, particles, softening, 1, 1.0f, 1.0, 0, parameters));
    EXPECT_THROW(kickPmmmParticlesCpu(grid, particles, makeFerrersSoftening(0.2f), 1, 1.0f, 1.0, 0, parameters), std::invalid_argument);
    CartesianGrid small(2, 0, 0);
    EXPECT_NO_THROW(kickPmmmParticlesCpu(small, particles, softening, 1, 1.0f, 1.0, 0, parameters));
    PmmmNearField field(grid, particles, softening, 1, 1.0f, 1.0, 0, false, false, {}, 1);
    parameters.ghostWidth = 2;
    EXPECT_THROW(kickPmmmParticlesCpu(small, particles, softening, 1, 1.0f, 1.0, 0, parameters), std::invalid_argument);
    EXPECT_THROW(field.kick(grid, particles, parameters), std::invalid_argument);
}

TEST_F(PmmmKickTest, CancellingNearAndFarAccelerationKeepsRungZeroAndDoesNotKickTwice) {
    float constexpr kMass = 0.125f;
    PROFILE();

    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, false);
    particles.append({Coordinate(0.0625), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    particles.append({Coordinate(0.1875), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    assignRanges(grid, particles);
    // Near acceleration at particle zero is +8, cancelled by the local gradient.
    grid.expansion({0, 0, 0})[stf::index(1, 0, 0)] = 1.0f;
    KickParameters parameters;
    parameters.phase = KickPhase::initial;
    parameters.scaleFactor = 0.25;
    KickSums sums;
    kickPmmmParticlesCpu(grid, particles, makeFerrersSoftening(0.01f), 1, kMass, 1.0, 0, parameters, &sums);
    EXPECT_EQ(sums.particleCount, 2);
    EXPECT_EQ(particles.acceleration(0, 0), 0.0f);
    EXPECT_EQ(particles.velocity(0, 0), 0.0f);
    EXPECT_EQ(particles.flags(0).rung(), 0);
    EXPECT_GT(particles.flags(1).rung(), 0);
    EXPECT_TRUE(grid.cell({0, 0, 0}).rungRangesValid);
    EXPECT_TRUE(grid.cell({1, 0, 0}).rungRangesValid);
}

TEST_F(PmmmKickTest, InactiveSinksRetainFieldsWhileStillContributingAsSources) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    particles.append({Coordinate(0.0625), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    particles.append({Coordinate(0.1875), Coordinate(0.0625), Coordinate(0.0625)}, {}, 2);
    assignRanges(grid, particles);
    particles.acceleration(0, 0) = 123.0f;
    particles.potential(0) = 456.0f;
    KickParameters parameters;
    parameters.phase = KickPhase::diagnostics;
    KickSums sums;
    kickPmmmParticlesCpu(grid, particles, makeFerrersSoftening(0.01f), 0, 1.0f, 1.0, 1, parameters, &sums);
    EXPECT_EQ(sums.particleCount, 1);
    EXPECT_EQ(sums.potential, 0.0);
    EXPECT_EQ(particles.acceleration(0, 0), 123.0f);
    EXPECT_EQ(particles.potential(0), 456.0f);
    EXPECT_EQ(particles.acceleration(0, 1), -64.0f);
    EXPECT_EQ(particles.potential(1), -8.0f);
    EXPECT_EQ(particles.flags(1).rung(), 2);
}

TEST_F(PmmmKickTest, RejectsMissingRemoteHaloAndSofteningBeyondTheNearMask) {
    PROFILE();
    ParticleStorage particles;
    KickParameters parameters;
    parameters.phase = KickPhase::diagnostics;
    if constexpr (kPmmmNeighborWidth > 1) {
        CartesianGrid narrow(3, 1, 0);
        EXPECT_THROW(kickPmmmParticlesCpu(narrow, particles, makeFerrersSoftening(0.01f), 1, 1.0f, 1.0, 0, parameters), std::invalid_argument);
    }
    CartesianGrid grid(3, 0, 0);
    EXPECT_THROW(kickPmmmParticlesCpu(grid, particles, makeFerrersSoftening(0.5f), 1, 1.0f, 1.0, 0, parameters), std::invalid_argument);
}

TEST_F(PmmmKickTest, OverlappedFieldsAndKickMatchFusedBitsAcrossOrdersPhasesAndOutputFields) {
    constexpr std::array<KickPhase, 4> kPhases{KickPhase::initial, KickPhase::regular, KickPhase::final, KickPhase::diagnostics};
    PROFILE();

    for (int order = 0; order <= 3; order++) {
        for (auto const phase : kPhases) {
            for (size_t fields = 0; fields < 4; fields++) {
                for (bool const sorted : {false, true}) {
                    for (bool const sumPotential : {false, true}) {
                        CartesianGrid expectedGrid(3, 0, 0);
                        CartesianGrid actualGrid(3, 0, 0);
                        ParticleStorage expected;
                        ParticleStorage actual;
                        seedOverlapParticles(expectedGrid, expected, (fields & 1) != 0, (fields & 2) != 0, sorted);
                        seedOverlapParticles(actualGrid, actual, (fields & 1) != 0, (fields & 2) != 0, sorted);
                        auto parameters = overlapParameters(phase);
                        parameters.sumPotential = sumPotential;
                        auto const softening = makeFerrersSoftening(0.02f);
                        KickSums expectedSums;
                        KickSums actualSums;
                        installOverlapPolynomial(expectedGrid);
                        kickPmmmParticlesCpu(expectedGrid, expected, softening, order, 0.125f, 1.0, 2, parameters, &expectedSums);
                        size_t farCalls = 0;
                        kickPmmmParticlesOverlappedCpu(actualGrid, actual, softening, order, 0.125f, 1.0, 2, parameters,
                            hpx::annotated_function(
                                [&] {
                                    PROFILE();
                                    farCalls++;
                                    installOverlapPolynomial(actualGrid);
                                },
                                "test.pmmm.installCountedFarField"),
                            &actualSums);
                        SCOPED_TRACE(::testing::Message() << "order=" << order << " phase=" << int(phase) << " fields=" << fields << " sorted=" << sorted
                                                          << " sumPotential=" << sumPotential);
                        EXPECT_EQ(farCalls, 1);
                        EXPECT_EQ(actualSums.particleCount, 8);
                        expectMatchingKick(actualGrid, actual, expectedGrid, expected, actualSums, expectedSums);
                    }
                }
            }
        }
    }
}

#ifdef COSMICTIGER_ENABLE_CUDA
TEST_F(PmmmKickTest, CudaEvaluatesNearAndFarFieldsBeforeOneKickPerOwnedSink) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    for (int order = 0; order <= 3; order++) {
        for (auto const phase : {KickPhase::initial, KickPhase::diagnostics}) {
            CartesianGrid cpuGrid(3, 0, 0);
            CartesianGrid cudaGrid(3, 0, 0);
            ParticleStorage cpuParticles;
            ParticleStorage cudaParticles;
            seedOverlapParticles(cpuGrid, cpuParticles, true, true, true);
            seedOverlapParticles(cudaGrid, cudaParticles, true, true, true);
            auto parameters = overlapParameters(phase);
            parameters.sumKinetic = true;
            parameters.momentaStaggered = phase == KickPhase::diagnostics;
            auto const softening = makeFerrersSoftening(0.02f);
            KickSums cpuSums;
            kickPmmmParticlesOverlappedCpu(cpuGrid, cpuParticles, softening, order, 0.125f, 1.0, 0, parameters,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        installOverlapPolynomial(cpuGrid);
                    },
                    "test.pmmm.installCpuFarField"),
                &cpuSums);
            static std::atomic<std::uint64_t> sequence{0x43554441504d0000ULL};
            auto const cudaSums = kickPmmmParticlesOverlappedCuda(cudaGrid, cudaParticles, softening, order, 0.125f, 1.0, 0, parameters,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        installOverlapPolynomial(cudaGrid);
                    },
                    "test.pmmm.installCudaFarField"),
                sequence.fetch_add(1));
            SCOPED_TRACE(order);
            SCOPED_TRACE(int(phase));
            EXPECT_EQ(cudaSums.particleCount, cpuSums.particleCount);
            EXPECT_EQ(cudaSums.maximumRung, cpuSums.maximumRung);
            EXPECT_EQ(cellSnapshot(cudaGrid), cellSnapshot(cpuGrid));
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(cudaSums.acceleration[axis], cpuSums.acceleration[axis], 3.0e-4 * (1.0 + std::abs(cpuSums.acceleration[axis])));
            }
            EXPECT_NEAR(cudaSums.potential, cpuSums.potential, 3.0e-4 * (1.0 + std::abs(cpuSums.potential)));
            EXPECT_NEAR(cudaSums.momentumSquared, cpuSums.momentumSquared, 3.0e-4 * (1.0 + std::abs(cpuSums.momentumSquared)));
            for (ParticleCount sink = 0; sink < cpuParticles.size(); sink++) {
                EXPECT_EQ(cudaParticles.flags(sink).rung(), cpuParticles.flags(sink).rung());
                EXPECT_EQ(cudaParticles.flags(sink).isGhost(), cpuParticles.flags(sink).isGhost());
                for (size_t axis = 0; axis < NDIM; axis++) {
                    EXPECT_NEAR(
                        cudaParticles.velocity(axis, sink), cpuParticles.velocity(axis, sink), 3.0e-4f * (1.0f + std::abs(cpuParticles.velocity(axis, sink))));
                    EXPECT_NEAR(cudaParticles.acceleration(axis, sink), cpuParticles.acceleration(axis, sink),
                        3.0e-4f * (1.0f + std::abs(cpuParticles.acceleration(axis, sink))));
                }
                EXPECT_NEAR(cudaParticles.potential(sink), cpuParticles.potential(sink), 3.0e-4f * (1.0f + std::abs(cpuParticles.potential(sink))));
            }
        }
    }
}

TEST_F(PmmmKickTest, CudaL2pIncludesBackgroundTraceInPotentialAndZAcceleration) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    CartesianGrid cpuGrid(3, 0, 0);
    CartesianGrid cudaGrid(3, 0, 0);
    ParticleStorage cpuParticles;
    ParticleStorage cudaParticles;
    auto seed = [](CartesianGrid& grid, ParticleStorage& particles) {
        PROFILE();
        particles.configureGravityFields(true, true);
        particles.append({Coordinate(0.5 / 8.0), Coordinate(0.5 / 8.0), Coordinate(0.75 / 8.0)}, {}, 0);
        grid.setInteriorRange({0, 0, 0}, {0, 1});
    };
    seed(cpuGrid, cpuParticles);
    seed(cudaGrid, cudaParticles);
    auto const installTrace = [](CartesianGrid& grid) {
        PROFILE();
        auto& expansion = grid.expansion({0, 0, 0});
        expansion = {};
        stf::setLaplacian<float, kExpansionOrder>(expansion, 2.0f);
    };
    auto parameters = overlapParameters(KickPhase::diagnostics);
    KickSums cpuSums;
    kickPmmmParticlesOverlappedCpu(cpuGrid, cpuParticles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 0, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installTrace(cpuGrid);
            },
            "test.pmmm.installCpuTrace"),
        &cpuSums);
    static std::atomic<std::uint64_t> sequence{0x43554441504d1000ULL};
    auto const cudaSums = kickPmmmParticlesOverlappedCuda(cudaGrid, cudaParticles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 0, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installTrace(cudaGrid);
            },
            "test.pmmm.installCudaTrace"),
        sequence.fetch_add(1));
    EXPECT_EQ(cudaSums.particleCount, 1);
    EXPECT_NEAR(cudaParticles.potential(0), cpuParticles.potential(0), 1.0e-6f);
    EXPECT_NEAR(cudaParticles.acceleration(2, 0), cpuParticles.acceleration(2, 0), 1.0e-6f);
    EXPECT_NEAR(cudaSums.potential, cpuSums.potential, 1.0e-6);
    EXPECT_NEAR(cudaSums.acceleration[2], cpuSums.acceleration[2], 1.0e-6);
    EXPECT_NEAR(cudaParticles.potential(0), 0.0625f, 1.0e-6f);
    EXPECT_NEAR(cudaParticles.acceleration(2, 0), -4.0f, 1.0e-6f);
}

TEST_F(PmmmKickTest, CudaUsesRemoteGhostSourcesAcrossEightLogicalLocalities) {
    size_t constexpr kSiteCount = size_t(1) << NDIM;
    float constexpr kMass = 0.125f;
    PROFILE();

    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    for (int ghostWidth : {1, 2}) {
        std::vector<CartesianGrid> grids;
        std::vector<ParticleStorage> storage;
        std::array<Position, kSiteCount> positions{};
        std::vector<hpx::id_type> localities(kSiteCount, hpx::find_here());
        for (size_t site = 0; site < kSiteCount; site++) {
            grids.emplace_back(3, 1, site, ghostWidth);
            storage.emplace_back();
            auto& particles = storage.back();
            particles.configureGravityFields(true, true);
            std::array<Coordinate, NDIM> coordinates{};
            auto const block = grids.back().layout().localityCoordinates();
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const cell = block[axis] == 0 ? 3.0 : 3.0 + ghostWidth;
                coordinates[axis] = Coordinate((cell + 0.5) / 8.0);
            }
            particles.append(coordinates, {}, 0);
            positions[site] = position(particles, 0);
            assignRanges(grids.back(), particles);
        }

        static std::atomic<std::uint64_t> sequence{0x4355444147480000ULL};
        auto const operationId = sequence.fetch_add(1);
        std::vector<hpx::future<void>> exchanges;
        for (size_t site = 0; site < kSiteCount; site++) {
            exchanges.push_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, site] {
                        PROFILE();
                        exchangeGhostParticles(storage[site], grids[site], localities, operationId);
                    },
                    "test.pmmm.exchangeCudaLogicalGhosts")));
        }
        for (auto& exchange : exchanges) {
            EXPECT_NO_THROW(exchange.get());
        }

        auto const softening = makeFerrersSoftening(0.01f);
        KickParameters parameters;
        parameters.phase = KickPhase::diagnostics;
        parameters.sumPotential = true;
        parameters.ghostWidth = ghostWidth;
        for (size_t site = 0; site < kSiteCount; site++) {
            bool hasRemoteGhost = false;
            for (ParticleCount index = 1; index < storage[site].size(); index++) {
                hasRemoteGhost |= storage[site].flags(index).isGhost();
            }
            EXPECT_TRUE(hasRemoteGhost);
            KickSums cpuSums;
            kickPmmmParticlesCpu(grids[site], storage[site], softening, 3, kMass, 1.0, 0, parameters, &cpuSums);
            std::array<float, NDIM> cpuAcceleration{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                cpuAcceleration[axis] = storage[site].acceleration(axis, 0);
                storage[site].acceleration(axis, 0) = 123.0f;
            }
            float const cpuPotential = storage[site].potential(0);
            storage[site].potential(0) = 456.0f;

            PmmmNearField cudaField(grids[site], storage[site], softening, 3, kMass, 1.0, 0, true, true, {}, ghostWidth);
            KickSums cudaSums;
            ASSERT_NO_THROW(cudaField.kickCuda(grids[site], storage[site], parameters, &cudaSums));
            // Google Test keeps this trace in OS-thread-local storage; establish it
            // after HPX futures may have suspended and resumed this test elsewhere.
            SCOPED_TRACE(::testing::Message() << "ghostWidth=" << ghostWidth << " site=" << site);
            EXPECT_EQ(cudaSums.particleCount, 1);
            EXPECT_EQ(cudaSums.particleCount, cpuSums.particleCount);
            std::array<double, NDIM> pairwiseAcceleration{};
            double pairwisePotential = 0.0;
            for (size_t source = 0; source < kSiteCount; source++) {
                if (source == site) {
                    continue;
                }
                auto const direct = ferrersAccelerationAndPotential<3>(positions[site], positions[source], softening);
                std::array<float, NDIM> const component{direct.acceleration.x, direct.acceleration.y, direct.acceleration.z};
                for (size_t axis = 0; axis < NDIM; axis++) {
                    pairwiseAcceleration[axis] += kMass * component[axis];
                }
                pairwisePotential += kMass * direct.potential;
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const tolerance = 2.0e-4 * (1.0 + std::abs(pairwiseAcceleration[axis]));
                EXPECT_NEAR(storage[site].acceleration(axis, 0), cpuAcceleration[axis], tolerance);
                EXPECT_NEAR(storage[site].acceleration(axis, 0), pairwiseAcceleration[axis], tolerance);
                EXPECT_NEAR(cudaSums.acceleration[axis], cpuSums.acceleration[axis], tolerance);
            }
            double const potentialTolerance = 2.0e-4 * (1.0 + std::abs(pairwisePotential));
            EXPECT_NEAR(storage[site].potential(0), cpuPotential, potentialTolerance);
            EXPECT_NEAR(storage[site].potential(0), pairwisePotential, potentialTolerance);
            EXPECT_NEAR(cudaSums.potential, cpuSums.potential, potentialTolerance);
        }
    }
}

TEST_F(PmmmKickTest, CudaRegularKickPreservesFilteredAndGhostParticlesAndSeededSums) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    for (size_t fields = 0; fields < 4; fields++) {
        bool const storeAcceleration = (fields & 1) != 0;
        bool const storePotential = (fields & 2) != 0;
        CartesianGrid cpuGrid(3, 1, 0, kPmmmNeighborWidth);
        CartesianGrid cudaGrid(3, 1, 0, kPmmmNeighborWidth);
        ParticleStorage cpuParticles;
        ParticleStorage cudaParticles;
        auto seed = [&](CartesianGrid& grid, ParticleStorage& particles) {
            PROFILE();
            particles.configureGravityFields(storeAcceleration, storePotential);
            for (size_t index = 0; index < 3; index++) {
                particles.append({Coordinate((double(index) + 1.5) / 8.0), Coordinate(0.5 / 8.0), Coordinate(0.5 / 8.0)},
                    {float(10 * index + 1), float(10 * index + 2), float(10 * index + 3)}, index == 0 ? 0 : index + 1);
            }
            assignRanges(grid, particles);
            particles.append({Coordinate(4.5 / 8.0), Coordinate(0.5 / 8.0), Coordinate(0.5 / 8.0)}, {31.0f, 32.0f, 33.0f}, 3, true);
            grid.setGhostRanges({4, 0, 0}, {3, 4}, {0, 1});
            for (ParticleCount index = 0; index < particles.size(); index++) {
                if (storeAcceleration) {
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        particles.acceleration(axis, index) = float(100 + index * NDIM + axis);
                    }
                }
                if (storePotential) {
                    particles.potential(index) = float(200 + index);
                }
            }
            installOverlapPolynomial(grid);
        };
        seed(cpuGrid, cpuParticles);
        seed(cudaGrid, cudaParticles);

        KickParameters parameters;
        parameters.phase = KickPhase::regular;
        parameters.scaleFactor = 0.5;
        parameters.minimumKickRung = 3;
        parameters.sumPotential = storePotential;
        parameters.sumKinetic = true;
        KickSums seeded;
        seeded.acceleration = {1.25, -2.5, 3.75};
        seeded.potential = 4.5;
        seeded.particleCount = 7;
        seeded.momentumSquared = 6.5;
        seeded.maximumRung = 1;
        KickSums cpuSums = seeded;
        KickSums cudaSums = seeded;
        auto const softening = makeFerrersSoftening(0.02f);
        kickPmmmParticlesCpu(cpuGrid, cpuParticles, softening, 2, 0.125f, 1.0, 2, parameters, &cpuSums);
        PmmmNearField cudaField(cudaGrid, cudaParticles, softening, 2, 0.125f, 1.0, 2, parameters.sumPotential, true);
        ASSERT_NO_THROW(cudaField.kickCuda(cudaGrid, cudaParticles, parameters, &cudaSums));

        SCOPED_TRACE(fields);
        EXPECT_EQ(cpuSums.particleCount, seeded.particleCount + 2);
        EXPECT_EQ(cudaSums.particleCount, cpuSums.particleCount);
        EXPECT_EQ(cudaSums.maximumRung, cpuSums.maximumRung);
        EXPECT_EQ(cellSnapshot(cudaGrid), cellSnapshot(cpuGrid));
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(cudaSums.acceleration[axis], cpuSums.acceleration[axis], 3.0e-4 * (1.0 + std::abs(cpuSums.acceleration[axis])));
        }
        EXPECT_NEAR(cudaSums.potential, cpuSums.potential, 3.0e-4 * (1.0 + std::abs(cpuSums.potential)));
        EXPECT_NEAR(cudaSums.momentumSquared, cpuSums.momentumSquared, 3.0e-4 * (1.0 + std::abs(cpuSums.momentumSquared)));
        if (!parameters.sumPotential) {
            EXPECT_DOUBLE_EQ(cudaSums.potential, seeded.potential);
        }
        EXPECT_EQ(cpuParticles.flags(1).rung(), 2);
        EXPECT_GT(cpuParticles.flags(2).rung(), 3);
        for (ParticleCount sink = 0; sink < cpuParticles.size(); sink++) {
            EXPECT_EQ(cudaParticles.flags(sink).rung(), cpuParticles.flags(sink).rung());
            EXPECT_EQ(cudaParticles.flags(sink).isGhost(), cpuParticles.flags(sink).isGhost());
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(
                    cudaParticles.velocity(axis, sink), cpuParticles.velocity(axis, sink), 3.0e-4f * (1.0f + std::abs(cpuParticles.velocity(axis, sink))));
                if (storeAcceleration) {
                    EXPECT_NEAR(cudaParticles.acceleration(axis, sink), cpuParticles.acceleration(axis, sink),
                        3.0e-4f * (1.0f + std::abs(cpuParticles.acceleration(axis, sink))));
                }
                if (sink == 0 || sink == 1 || sink == 3) {
                    EXPECT_EQ(cudaParticles.velocity(axis, sink), float(10 * sink + axis + 1));
                }
                if (storeAcceleration && (sink == 0 || sink == 3)) {
                    EXPECT_EQ(cudaParticles.acceleration(axis, sink), float(100 + sink * NDIM + axis));
                }
            }
            if (storePotential) {
                EXPECT_NEAR(cudaParticles.potential(sink), cpuParticles.potential(sink), 3.0e-4f * (1.0f + std::abs(cpuParticles.potential(sink))));
                if (sink == 0 || sink == 3) {
                    EXPECT_EQ(cudaParticles.potential(sink), float(200 + sink));
                }
            }
        }
    }
}
#endif

TEST_F(PmmmKickTest, OverlappedCancellationSelectsRungFromTheCompleteFieldAndKicksOnce) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    particles.append({Coordinate(0.0625), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    particles.append({Coordinate(0.1875), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
    assignRanges(grid, particles);
    auto const parameters = overlapParameters();
    KickSums sums;
    kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.01f), 1, 0.125f, 1.0, 0, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                grid.expansion({0, 0, 0})[stf::index(1, 0, 0)] = 1.0f;
            },
            "test.pmmm.installLinearFarField"),
        &sums);
    EXPECT_EQ(sums.particleCount, 2);
    EXPECT_EQ(particles.acceleration(0, 0), 0.0f);
    EXPECT_EQ(particles.velocity(0, 0), 0.0f);
    EXPECT_EQ(particles.flags(0).rung(), 0);
    EXPECT_GT(particles.flags(1).rung(), 0);
    EXPECT_EQ(particles.velocity(0, 1), float(0.5 * rungTimestep(1.0, particles.flags(1).rung()) * particles.acceleration(0, 1)));
    EXPECT_DOUBLE_EQ(sums.potential, double(particles.potential(0)) + particles.potential(1));
}

TEST_F(PmmmKickTest, CrowdedSingleCellSplitsNearScratchAndKickChunksWithoutChangingBits) {
    PROFILE();
    for (bool const sorted : {false, true}) {
        CartesianGrid grid(3, 0, 0);
        CartesianGrid expectedGrid(3, 0, 0);
        ParticleStorage particles;
        ParticleStorage expected;
        auto seed = [&](CartesianGrid& targetGrid, ParticleStorage& target) {
            PROFILE();
            target.configureGravityFields(true, true);
            // Sixty-five active sinks share one cell. The inactive prefix and
            // empty surrounding cell spans exercise both compact range modes.
            for (size_t index = 0; index < 70; index++) {
                target.append({Coordinate(0.1875 + double(index) / 65536.0), Coordinate(0.0625 + double(index % 5) / 65536.0), Coordinate(0.0625)},
                    {float(index) + 0.25f, -float(index) - 0.5f, float(index) + 0.75f}, index < 5 ? 0 : 2 + index % 2);
            }
            target.append({Coordinate(0.9375), Coordinate(0.0625), Coordinate(0.0625)}, {}, 0);
            assignRanges(targetGrid, target);
            if (sorted) {
                targetGrid.sortParticlesByRung(target);
            }
        };
        seed(grid, particles);
        seed(expectedGrid, expected);
        auto const parameters = overlapParameters(KickPhase::diagnostics);
        auto const softening = makeFerrersSoftening(0.02f);
        KickSums expectedSums;
        KickSums sums;
        installOverlapPolynomial(expectedGrid);
        kickPmmmParticlesCpu(expectedGrid, expected, softening, 2, 0.125f, 1.0, 2, parameters, &expectedSums);
        kickPmmmParticlesOverlappedCpu(grid, particles, softening, 2, 0.125f, 1.0, 2, parameters,
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    installOverlapPolynomial(grid);
                },
                "test.pmmm.installPhasedFarField"),
            &sums);
        SCOPED_TRACE(::testing::Message() << "sorted=" << sorted);
        EXPECT_EQ(sums.particleCount, 65);
        expectMatchingKick(grid, particles, expectedGrid, expected, sums, expectedSums);
    }
}

TEST_F(PmmmKickTest, NearPreparationCompletesWhileTheFarTaskIsSuspended) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    CartesianGrid expectedGrid(3, 0, 0);
    ParticleStorage particles;
    ParticleStorage expected;
    seedOverlapParticles(grid, particles, true, true, true);
    seedOverlapParticles(expectedGrid, expected, true, true, true);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto const parameters = overlapParameters();
    auto const softening = makeFerrersSoftening(0.02f);
    KickSums expectedSums;
    installOverlapPolynomial(expectedGrid);
    kickPmmmParticlesCpu(expectedGrid, expected, softening, 2, 0.125f, 1.0, 2, parameters, &expectedSums);
    hpx::promise<void> farStarted;
    auto started = farStarted.get_future();
    hpx::promise<void> releaseFar;
    auto release = releaseFar.get_future();
    hpx::future<PmmmNearField> near;
    hpx::future<void> far;
    GatedTasks tasks(releaseFar,
        hpx::annotated_function(
            [&] {
                PROFILE();
                if (near.valid()) {
                    near.wait();
                }
                if (far.valid()) {
                    far.wait();
                }
            },
            "test.pmmm.drainNearAndFarTasks"));
    near = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                return PmmmNearField(grid, particles, softening, 2, 0.125f, 1.0, 2, true);
            },
            "test.pmmm.prepareGatedNearField"));
    far = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                farStarted.set_value();
                release.get();
                installOverlapPolynomial(grid);
            },
            "test.pmmm.prepareGatedFarField"));
    started.get();
    auto field = near.get();
    EXPECT_FALSE(far.is_ready());
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    tasks.release();
    far.get();
    KickSums sums;
    field.kick(grid, particles, parameters, &sums);
    expectMatchingKick(grid, particles, expectedGrid, expected, sums, expectedSums);
    auto const afterKick = particleSnapshot(particles);
    EXPECT_THROW(field.kick(grid, particles, parameters, &sums), std::logic_error);
    EXPECT_EQ(particleSnapshot(particles), afterKick);
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, OverlappedCallbackCannotCommitParticleFieldsBeforeFarCompletion) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, true, true, true);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto const parameters = overlapParameters();
    hpx::promise<void> farStarted;
    auto started = farStarted.get_future();
    hpx::promise<void> releaseFar;
    auto release = releaseFar.get_future();
    KickSums sums;
    hpx::future<void> task;
    GatedTasks tasks(releaseFar,
        hpx::annotated_function(
            [&] {
                PROFILE();
                if (task.valid()) {
                    task.wait();
                }
            },
            "test.pmmm.drainOverlappedTask"));
    task = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
                    hpx::annotated_function(
                        [&] {
                            PROFILE();
                            farStarted.set_value();
                            release.get();
                            installOverlapPolynomial(grid);
                        },
                        "test.pmmm.waitForFarFieldRelease"),
                    &sums);
            },
            "test.pmmm.runGatedOverlap"));
    started.get();
    EXPECT_FALSE(task.is_ready());
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
    EXPECT_DOUBLE_EQ(sums.potential, 0.0);
    for (double const component : sums.acceleration) {
        EXPECT_DOUBLE_EQ(component, 0.0);
    }
    tasks.release();
    EXPECT_NO_THROW(task.get());
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, NearFailureStillEntersAndDrainsFarWorkBeforeReportingAndCanRetry) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, true, true, true);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto const parameters = overlapParameters();
    hpx::promise<void> farStarted;
    auto started = farStarted.get_future();
    hpx::promise<void> releaseFar;
    auto release = releaseFar.get_future();
    KickSums sums;
    hpx::future<void> task;
    GatedTasks tasks(releaseFar,
        hpx::annotated_function(
            [&] {
                PROFILE();
                if (task.valid()) {
                    task.wait();
                }
            },
            "test.pmmm.drainFailedNearTask"));
    task = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 4, 0.125f, 1.0, 2, parameters,
                    hpx::annotated_function(
                        [&] {
                            PROFILE();
                            farStarted.set_value();
                            release.get();
                            installOverlapPolynomial(grid);
                        },
                        "test.pmmm.waitForFailedNearRelease"),
                    &sums);
            },
            "test.pmmm.runFailedNearOverlap"));
    started.get();
    EXPECT_FALSE(task.is_ready());
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    tasks.release();
    EXPECT_THROW(task.get(), std::invalid_argument);
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
    EXPECT_NO_THROW(kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installOverlapPolynomial(grid);
            },
            "test.pmmm.installRetryFarField"),
        &sums));
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, FarFailureDrainsNearWorkWithoutKickingAndCanRetry) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, true, true, true);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto const parameters = overlapParameters();
    KickSums sums;
    EXPECT_THROW(kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
                     hpx::annotated_function(
                         [] {
                             PROFILE();
                             throw std::runtime_error("test far-field failure");
                         },
                         "test.pmmm.throwFarFieldFailure"),
                     &sums),
        std::runtime_error);
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
    EXPECT_NO_THROW(kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installOverlapPolynomial(grid);
            },
            "test.pmmm.installFarFailureRetry"),
        &sums));
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, NearRangeFailureStillRunsFarCallbackAndPreservesParticleState) {
    constexpr GridIndices kLastCell{7, 0, 0};
    PROFILE();

    if constexpr (!validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, true, true, true);
    auto const original = grid.cell(kLastCell).localParticles;
    grid.setInteriorRange(kLastCell, {original.begin, original.end + 1});
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto const parameters = overlapParameters();
    size_t farCalls = 0;
    KickSums sums;
    EXPECT_ANY_THROW(kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                farCalls++;
                installOverlapPolynomial(grid);
            },
            "test.pmmm.installRangeFailureFarField"),
        &sums));
    EXPECT_EQ(farCalls, 1);
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
    grid.setInteriorRange(kLastCell, original);
    EXPECT_NO_THROW(kickPmmmParticlesOverlappedCpu(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installOverlapPolynomial(grid);
            },
            "test.pmmm.installRangeFailureRetry"),
        &sums));
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, MovedNearFieldAcceptsOneKickAndRejectsUnstagedPotentialBeforeMutation) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, false, false, true);
    installOverlapPolynomial(grid);
    auto parameters = overlapParameters();
    parameters.sumPotential = false;
    PmmmNearField original(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, false);
    auto moved = std::move(original);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    KickSums sums;
    EXPECT_THROW(original.kick(grid, particles, parameters, &sums), std::logic_error);
    auto requestedPotential = parameters;
    requestedPotential.sumPotential = true;
    EXPECT_THROW(moved.kick(grid, particles, requestedPotential, &sums), std::invalid_argument);
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
    EXPECT_NO_THROW(moved.kick(grid, particles, parameters, &sums));
    EXPECT_EQ(sums.particleCount, 8);
    auto const afterKick = particleSnapshot(particles);
    EXPECT_THROW(moved.kick(grid, particles, parameters, &sums), std::logic_error);
    EXPECT_EQ(particleSnapshot(particles), afterKick);
    EXPECT_EQ(sums.particleCount, 8);
}

TEST_F(PmmmKickTest, FailedKickConsumesNearFieldBeforeASecondApplicationCanChangeParticles) {
    PROFILE();
    if constexpr (!validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedOverlapParticles(grid, particles, true, true, true);
    installOverlapPolynomial(grid);
    PmmmNearField field(grid, particles, makeFerrersSoftening(0.02f), 2, 0.125f, 1.0, 2, true);
    auto const beforeParticles = particleSnapshot(particles);
    auto const beforeCells = cellSnapshot(grid);
    auto parameters = overlapParameters();
    parameters.timeTick = 1;
    KickSums sums;
    EXPECT_ANY_THROW(field.kick(grid, particles, parameters, &sums));
    parameters.timeTick = 0;
    EXPECT_THROW(field.kick(grid, particles, parameters, &sums), std::logic_error);
    EXPECT_EQ(particleSnapshot(particles), beforeParticles);
    EXPECT_EQ(cellSnapshot(grid), beforeCells);
    EXPECT_EQ(sums.particleCount, 0);
}

TEST_F(PmmmKickTest, OverlappedCollectiveReportsOneCombinedOwnedContribution) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    CartesianGrid grid(3, 0, 0);
    CartesianGrid expectedGrid(3, 0, 0);
    ParticleStorage particles;
    ParticleStorage expected;
    seedOverlapParticles(grid, particles, true, true, true);
    seedOverlapParticles(expectedGrid, expected, true, true, true);
    auto const parameters = overlapParameters();
    auto const softening = makeFerrersSoftening(0.02f);
    KickSums expectedSums;
    installOverlapPolynomial(expectedGrid);
    kickPmmmParticlesCpu(expectedGrid, expected, softening, 2, 0.125f, 1.0, 2, parameters, &expectedSums);
    static std::atomic<std::uint64_t> sequence{0x4f504d4d4b000000ULL};
    auto const sums = kickPmmmParticlesOverlapped(grid, particles, softening, 2, 0.125f, 1.0, 2, parameters,
        hpx::annotated_function(
            [&] {
                PROFILE();
                installOverlapPolynomial(grid);
            },
            "test.pmmm.installCollectiveFarField"),
        sequence.fetch_add(1));
    EXPECT_EQ(sums.particleCount, 8);
    expectMatchingKick(grid, particles, expectedGrid, expected, sums, expectedSums);
}

#if COSMICTIGER_MULTIPOLE_ORDER >= 1 && COSMICTIGER_EXPANSION_ORDER >= 2
TEST_F(PmmmKickTest, RealPeriodicConvolutionAndNearOverlapMatchTheSerialKick) {
    float constexpr kMass = 0.125f;
    PROFILE();

    for (int ghostWidth : {1, 2}) {
        CartesianGrid grid(3, 0, 0, ghostWidth);
        CartesianGrid expectedGrid(3, 0, 0, ghostWidth);
        ParticleStorage particles;
        ParticleStorage expected;
        seedOverlapParticles(grid, particles, true, true, true);
        seedOverlapParticles(expectedGrid, expected, true, true, true);
        grid.computeMultipoles(particles, kMass);
        expectedGrid.computeMultipoles(expected, kMass);
        // Low orders and zero image/wave cutoffs keep this integration check small;
        // the second pass also exercises reuse of the cached Green spectra.
        PmmmPeriodicConvolution<1, 2> convolution(grid.layout(), 0, 0, 1.5, ghostWidth);
        PmmmPeriodicConvolution<1, 2> expectedConvolution(expectedGrid.layout(), 0, 0, 1.5, ghostWidth);
        auto noExchange = hpx::annotated_function(
            [](size_t, auto, auto) {
                PROFILE();
                throw std::logic_error("single-locality convolution attempted an exchange");
            },
            "test.pmmm.rejectUnexpectedExchange");
        auto parameters = overlapParameters(KickPhase::diagnostics);
        parameters.ghostWidth = ghostWidth;
        auto const softening = makeFerrersSoftening(0.02f);
        for (size_t repetition = 0; repetition < 2; repetition++) {
            KickSums expectedSums;
            KickSums sums;
            expectedConvolution.computeExpansions(expectedGrid, noExchange);
            kickPmmmParticlesCpu(expectedGrid, expected, softening, 2, kMass, 1.0, 2, parameters, &expectedSums);
            kickPmmmParticlesOverlappedCpu(grid, particles, softening, 2, kMass, 1.0, 2, parameters,
                hpx::annotated_function(
                    [&] {
                        PROFILE();
                        convolution.computeExpansions(grid, noExchange);
                    },
                    "test.pmmm.computePeriodicFarField"),
                &sums);
            SCOPED_TRACE(::testing::Message() << "ghostWidth=" << ghostWidth << " repetition=" << repetition);
            expectMatchingKick(grid, particles, expectedGrid, expected, sums, expectedSums);
        }
    }
}
#endif

TEST_F(PmmmKickTest, RemoteNearCellsMatchPairwiseFieldsAcrossEightLogicalLocalities) {
    size_t constexpr kSiteCount = size_t(1) << NDIM;
    float constexpr kMass = 0.125f;
    PROFILE();

    for (int ghostWidth : {1, 2}) {
        std::vector<CartesianGrid> grids;
        std::vector<ParticleStorage> storage;
        std::array<Position, kSiteCount> positions{};
        std::vector<hpx::id_type> localities(kSiteCount, hpx::find_here());
        for (size_t site = 0; site < kSiteCount; site++) {
            grids.emplace_back(3, 1, site, ghostWidth);
            storage.emplace_back();
            auto& particles = storage.back();
            particles.configureGravityFields(true, true);
            std::array<Coordinate, NDIM> coordinates{};
            auto const block = grids.back().layout().localityCoordinates();
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const cell = block[axis] == 0 ? 3.0 : 3.0 + ghostWidth;
                coordinates[axis] = Coordinate((cell + 0.5) / 8.0);
            }
            particles.append(coordinates, {}, 0);
            positions[site] = position(particles, 0);
            assignRanges(grids.back(), particles);
        }
        static std::atomic<std::uint64_t> sequence{0x504b49434b000000ULL};
        auto const operationId = sequence.fetch_add(1);
        std::vector<hpx::future<void>> exchanges;
        for (size_t site = 0; site < kSiteCount; site++) {
            exchanges.push_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, site] {
                        PROFILE();
                        exchangeGhostParticles(storage[site], grids[site], localities, operationId);
                    },
                    "test.pmmm.exchangeLogicalGhosts")));
        }
        for (auto& exchange : exchanges) {
            EXPECT_NO_THROW(exchange.get());
        }
        auto const softening = makeFerrersSoftening(0.01f);
        KickParameters parameters;
        parameters.phase = KickPhase::diagnostics;
        parameters.sumPotential = true;
        parameters.ghostWidth = ghostWidth;
        for (size_t site = 0; site < kSiteCount; site++) {
            KickSums sums;
            kickPmmmParticlesCpu(grids[site], storage[site], softening, 3, kMass, 1.0, 0, parameters, &sums);
            EXPECT_EQ(sums.particleCount, 1);
            std::array<double, NDIM> expected{};
            double expectedPotential = 0.0;
            for (size_t source = 0; source < kSiteCount; source++) {
                if (source != site) {
                    auto const direct = ferrersAccelerationAndPotential<3>(positions[site], positions[source], softening);
                    expected[0] += kMass * direct.acceleration.x;
                    expected[1] += kMass * direct.acceleration.y;
                    expected[2] += kMass * direct.acceleration.z;
                    expectedPotential += kMass * direct.potential;
                }
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(storage[site].acceleration(axis, 0), expected[axis], 2e-5);
                EXPECT_NEAR(sums.acceleration[axis], expected[axis], 2e-5);
            }
            EXPECT_NEAR(storage[site].potential(0), expectedPotential, 2e-5);
            EXPECT_NEAR(sums.potential, expectedPotential, 2e-5);
            // Cached owned ranges and the appended remote ghost suffix must produce
            // the same complete sink field through the staged near calculation.
            auto const fused = particleSnapshot(storage[site]);
            KickSums overlapped;
            kickPmmmParticlesOverlappedCpu(grids[site], storage[site], softening, 3, kMass, 1.0, 0, parameters,
                hpx::annotated_function([] { PROFILE(); }, "test.pmmm.emptyFarField"), &overlapped);
            EXPECT_EQ(particleSnapshot(storage[site]), fused);
            EXPECT_EQ(overlapped.particleCount, sums.particleCount);
            EXPECT_EQ(overlapped.acceleration, sums.acceleration);
            EXPECT_DOUBLE_EQ(overlapped.potential, sums.potential);
        }
    }
}

}    // namespace
