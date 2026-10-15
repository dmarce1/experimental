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

#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/profiler.hpp>
#include <gtest/gtest.h>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <iostream>

TEST(DirectComparison, PeriodicPmmmFieldsAgreeWithIndependentDirectReference) {
    constexpr size_t kParticles = 16;
    constexpr size_t kSide = 8;
    constexpr float kMass = 0.0625f;
    constexpr float kSoftening = 0.01f;
    constexpr double kRelativeTolerance = 2.0e-3;
    constexpr std::uint64_t kOperation = 0x444952504d4d4d00ULL;
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "Full PMMM reference comparison runs in the CPU-only build";
#endif
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    cosmictiger::CartesianGrid grid(3, 0, 0);
    cosmictiger::ParticleStorage particles;
    particles.configureGravityFields(true, true);
    for (size_t index = 0; index < kParticles; index++) {
        size_t linear = (index * 31 + 5) % (kSide * kSide * kSide);
        cosmictiger::GridIndices coordinates{};
        std::array<cosmictiger::Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            coordinates[axis] = static_cast<std::int64_t>(linear % kSide);
            linear /= kSide;
            position[axis] = cosmictiger::Coordinate((double(coordinates[axis]) + 0.35 + 0.1 * double((index + axis) % 4)) / double(kSide));
        }
        particles.append(position, {0.25f, -0.125f, 0.5f}, index % 4);
        grid.setInteriorRange(coordinates, {index, index + 1});
    }
    grid.computeMultipoles(particles, kMass);
    cosmictiger::gravity::PmmmPeriodicConvolution<> convolution(grid.layout());
    auto exchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("unexpected remote FFT exchange");
        },
        "directComparison.singleLocalityFft");
    convolution.computeExpansions(grid, exchange);
    cosmictiger::gravity::KickParameters const parameters{0.025, 0.0, 1.0, 0, cosmictiger::gravity::KickPhase::diagnostics, true};
    cosmictiger::gravity::kickPmmmParticlesCpu(grid, particles, cosmictiger::gravity::makeFerrersSoftening(kSoftening), 2, kMass, 1.0, 0, parameters);
    cosmictiger::GlobalConstantsData constants;
    constants.globalParticleCount = kParticles;
    constants.gravityParticleMass = kMass;
    constants.softeningLength = kSoftening;
    constants.directSinkFraction = 0.5;
    auto const result = cosmictiger::gravity::compareDirectGravity(particles, grid, constants, kOperation);
    std::cout << cosmictiger::gravity::formatDirectComparison(result);
    EXPECT_EQ(result.sourceCount, kParticles);
    EXPECT_EQ(result.sinkCount, kParticles / 2);
    EXPECT_LT(result.acceleration.relativeL1, kRelativeTolerance);
    EXPECT_LT(result.acceleration.relativeL2, kRelativeTolerance);
    EXPECT_LT(result.acceleration.relativeLinf, kRelativeTolerance);
    EXPECT_LT(result.potential.relativeLinf, kRelativeTolerance);
    for (size_t index = 0; index < kParticles; index++) {
        EXPECT_EQ(particles.flags(index).rung(), index % 4);
        EXPECT_FLOAT_EQ(particles.velocity(0, index), 0.25f);
        EXPECT_FLOAT_EQ(particles.velocity(1, index), -0.125f);
        EXPECT_FLOAT_EQ(particles.velocity(2, index), 0.5f);
    }
}
