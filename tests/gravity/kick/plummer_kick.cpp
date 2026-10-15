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

#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace {
namespace gravity = cosmictiger::gravity;
using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::ParticleStorage;

class PlummerKickTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        if (hpx::find_all_localities().size() != 1) {
            GTEST_SKIP() << "requires one physical HPX locality";
        }
#ifdef COSMICTIGER_ENABLE_CUDA
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires a CUDA device";
            }
            auto const properties = cosmictiger::cuda::deviceProperties(cosmictiger::cuda::device().get()).get();
            if (!properties.managedMemory) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires managed memory";
            }
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires a device and driver";
            }
            throw;
        }
#endif
    }
};
}    // namespace

TEST_F(PlummerKickTest, CompleteFieldMatchesPeriodicReferenceAcrossAvailableBackends) {
    constexpr size_t kCount = 12;
    constexpr size_t kSide = 8;
    constexpr int kGhostWidth = 2;
    constexpr float kMass = 0.0625f;
    constexpr float kEpsilon = 0.0625f;
    constexpr double kRelativeTolerance = 2.0e-5;
    constexpr double kBackendTolerance = 4.0e-6;
    constexpr int kEwaldCutoff = 3;
    constexpr double kAlphaBox = 2.0;
    constexpr std::array<float, NDIM> kMomentum{0.25f, -0.125f, 0.5f};
    PROFILE();
    CartesianGrid grid(3, 0, 0, kGhostWidth);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    std::array<cosmictiger::Position, kCount> positions{};
    std::array<size_t, kCount> sinks{};
    for (size_t index = 0; index < kCount; index++) {
        size_t linear = (index * 31 + 5) % (kSide * kSide * kSide);
        cosmictiger::GridIndices coordinates{};
        std::array<Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            coordinates[axis] = static_cast<std::int64_t>(linear % kSide);
            linear /= kSide;
            double const centerOffset = 0.375 + double((index + axis) % 4) * 0.0625;
            position[axis] = Coordinate((double(coordinates[axis]) + centerOffset) / double(kSide));
        }
        particles.append(position, kMomentum, index % 4);
        grid.setInteriorRange(coordinates, {index, index + 1});
        positions[index] = {position[0], position[1], position[2]};
        sinks[index] = index;
    }
    std::array<gravity::DirectSolveField, kCount> reference{};
    gravity::DirectSolveRequest const request{
        positions.data(), kCount, sinks.data(), kCount, reference.data(), kMass, kEpsilon, 2, gravity::SofteningModel::plummer};
    gravity::directSolveCpu(request);
    gravity::PmmmPeriodicConvolution<> convolution(
        grid.layout(), kEwaldCutoff, kEwaldCutoff, kAlphaBox, kGhostWidth, gravity::SofteningModel::plummer, kEpsilon);
    auto exchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("unexpected remote Plummer FFT exchange");
        },
        "test.plummerKick.localExchange");
    gravity::KickParameters parameters;
    parameters.phase = gravity::KickPhase::diagnostics;
    parameters.sumPotential = true;
    parameters.ghostWidth = kGhostWidth;
    auto const softening = gravity::makePlummerSoftening(kEpsilon);
    grid.computeMultipoles(particles, kMass, true);
    convolution.computeExpansions(grid, exchange);
    std::array<gravity::DirectSolveField, kCount> cpuFields{};
    auto checkFields = [&](bool compareCpu) {
        PROFILE();
        for (size_t index = 0; index < kCount; index++) {
            EXPECT_NEAR(particles.potential(index), reference[index].potential, kRelativeTolerance * (1.0 + std::abs(reference[index].potential)));
            if (compareCpu) {
                EXPECT_NEAR(particles.potential(index), cpuFields[index].potential, kBackendTolerance * (1.0 + std::abs(cpuFields[index].potential)));
            } else {
                cpuFields[index].potential = particles.potential(index);
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const actual = particles.acceleration(axis, index);
                EXPECT_NEAR(actual, reference[index].acceleration[axis], kRelativeTolerance * (1.0 + std::abs(reference[index].acceleration[axis])));
                if (compareCpu) {
                    EXPECT_NEAR(actual, cpuFields[index].acceleration[axis], kBackendTolerance * (1.0 + std::abs(cpuFields[index].acceleration[axis])));
                } else {
                    cpuFields[index].acceleration[axis] = actual;
                }
                EXPECT_FLOAT_EQ(particles.velocity(axis, index), kMomentum[axis]);
            }
            EXPECT_EQ(particles.flags(index).rung(), index % 4);
        }
    };
    gravity::KickSums cpuSums;
    gravity::kickPmmmParticlesCpu(grid, particles, softening, 2, kMass, 1.0, 0, parameters, &cpuSums);
    EXPECT_EQ(cpuSums.particleCount, kCount);
    checkFields(false);
#ifdef COSMICTIGER_ENABLE_CUDA
    // CPU P2M with CUDA P2P/L2P uses exactly the same completed CPU locals.
    {
        gravity::PmmmNearField field(grid, particles, softening, 2, kMass, 1.0, 0, true, true, {}, kGhostWidth);
        gravity::KickSums sums;
        field.kickCuda(grid, particles, parameters, &sums);
        EXPECT_EQ(sums.particleCount, kCount);
        EXPECT_NEAR(sums.potential, cpuSums.potential, kBackendTolerance * (1.0 + std::abs(cpuSums.potential)));
        checkFields(true);
    }
    grid.computeMultipolesCuda(particles, kMass, 0, true);
    convolution.computeExpansions(grid, exchange);
    // CUDA P2M with CPU P2P/L2P, followed by the complete CUDA selection.
    gravity::kickPmmmParticlesCpu(grid, particles, softening, 2, kMass, 1.0, 0, parameters);
    checkFields(true);
    {
        gravity::PmmmNearField field(grid, particles, softening, 2, kMass, 1.0, 0, true, true, {}, kGhostWidth);
        gravity::KickSums sums;
        field.kickCuda(grid, particles, parameters, &sums);
        EXPECT_EQ(sums.particleCount, kCount);
        checkFields(true);
    }
#endif
}

#ifdef COSMICTIGER_ENABLE_CUDA
TEST_F(PlummerKickTest, CudaP2mRetainsAllTraceComponentsInCrowdedAndSparseCells) {
    constexpr size_t kCrowdedCount = 41;
    constexpr float kMass = 0.015625f;
    constexpr std::array<cosmictiger::GridIndices, 2> kCells{{{1, 2, 3}, {5, 6, 7}}};
    constexpr size_t kSide = 8;
    PROFILE();
    CartesianGrid grid(3, 0, 0, 2);
    ParticleStorage particles;
    for (size_t index = 0; index <= kCrowdedCount; index++) {
        auto const& cell = kCells[index == kCrowdedCount ? 1 : 0];
        std::array<Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const offset = 0.25 + double((index * 3 + axis * 5) % 9) / 16.0;
            position[axis] = Coordinate((double(cell[axis]) + offset) / double(kSide));
        }
        particles.append(position, {}, 0);
    }
    grid.setInteriorRange(kCells[0], {0, kCrowdedCount});
    grid.setInteriorRange(kCells[1], {kCrowdedCount, kCrowdedCount + 1});
    grid.computeMultipoles(particles, kMass, true);
    std::array<cosmictiger::FullCellMultipole, kCells.size()> expected{};
    for (size_t cell = 0; cell < kCells.size(); cell++) {
        expected[cell] = grid.cell(kCells[cell]).fullMultipole;
    }
    grid.computeMultipolesCuda(particles, kMass, 0, true);
    for (size_t cell = 0; cell < kCells.size(); cell++) {
        auto const& actual = grid.cell(kCells[cell]).fullMultipole;
        for (size_t coefficient = 0; coefficient < actual.size(); coefficient++) {
            EXPECT_FLOAT_EQ(actual[coefficient], expected[cell][coefficient]);
        }
    }
    for (float coefficient : grid.cell({0, 0, 0}).fullMultipole) {
        EXPECT_FLOAT_EQ(coefficient, 0.0f);
    }
}
#endif
