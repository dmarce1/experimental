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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/gravity/symmetric/multipole.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stddef.h>

namespace {

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleStorage;
namespace gravity = cosmictiger::gravity;
namespace stf = cosmictiger::gravity::stf;
namespace symmetric = cosmictiger::gravity::symmetric;

class P2mCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable";
            }
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        if (!cosmictiger::cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed particle storage";
        }
    }
};

void appendRaw(ParticleStorage& particles, std::array<std::uint32_t, NDIM> const& raw) {
    PROFILE();
    particles.append({Coordinate::fromRaw(raw[0]), Coordinate::fromRaw(raw[1]), Coordinate::fromRaw(raw[2])}, {0.0f, 0.0f, 0.0f}, 0);
}

void expectGridMomentsNear(CartesianGrid const& actual, CartesianGrid const& expected, bool fullMoments = false) {
    constexpr double kAbsoluteTolerance = 1.0e-8;
    constexpr double kRelativeTolerance = 2.0e-6;
    constexpr double kTraceTolerance = 1.0e-11;
    PROFILE();
    ASSERT_EQ(actual.layout().storedCellCount(), expected.layout().storedCellCount());
    for (size_t index = 0; index < actual.layout().storedCellCount(); index++) {
        auto const coordinates = actual.layout().cellCoordinates(index);
        auto const& actualCell = actual.cell(coordinates);
        auto const& expectedCell = expected.cell(coordinates);
        if (fullMoments) {
            for (size_t coefficient = 0; coefficient < actualCell.fullMultipole.size(); coefficient++) {
                double const reference = double(expectedCell.fullMultipole[coefficient]);
                double const tolerance = kAbsoluteTolerance + kRelativeTolerance * std::abs(reference);
                EXPECT_NEAR(double(actualCell.fullMultipole[coefficient]), reference, tolerance) << "cell " << index << ", coefficient " << coefficient;
            }
        } else {
            for (size_t coefficient = 0; coefficient < actualCell.multipole.size(); coefficient++) {
                double const reference = double(expectedCell.multipole[coefficient]);
                double const tolerance = kAbsoluteTolerance + kRelativeTolerance * std::abs(reference);
                EXPECT_NEAR(double(actualCell.multipole[coefficient]), reference, tolerance) << "cell " << index << ", coefficient " << coefficient;
            }
        }
        double const traceScale = std::abs(expectedCell.secondMomentTrace) > 1.0 ? std::abs(expectedCell.secondMomentTrace) : 1.0;
        EXPECT_NEAR(actualCell.secondMomentTrace, expectedCell.secondMomentTrace, kTraceTolerance * traceScale) << "cell " << index;
        EXPECT_EQ(actualCell.expansion, expectedCell.expansion) << "cell " << index;
        EXPECT_EQ(actualCell.fullExpansion, expectedCell.fullExpansion) << "cell " << index;
        EXPECT_EQ(actualCell.localParticles.begin, expectedCell.localParticles.begin) << "cell " << index;
        EXPECT_EQ(actualCell.localParticles.end, expectedCell.localParticles.end) << "cell " << index;
        EXPECT_EQ(actualCell.ownerParticles.begin, expectedCell.ownerParticles.begin) << "cell " << index;
        EXPECT_EQ(actualCell.ownerParticles.end, expectedCell.ownerParticles.end) << "cell " << index;
    }
}

TEST_F(P2mCudaTest, FullRawMomentsPreserveTracesForSparseCrowdedEmptyAndGhostCells) {
    constexpr size_t kCrowdedParticles = 65;
    constexpr size_t kSparseParticles = 8;
    constexpr size_t kTotalParticles = kCrowdedParticles + kSparseParticles + 1;
    constexpr std::uint32_t kCenter = 0x20000000U;
    constexpr std::uint32_t kStep = 0x00400000U;
    constexpr float kMass = 0.75f;
    constexpr GridIndices kGhost{-1, 0, 0};
    constexpr double kTraceTolerance = 2e-6;
    PROFILE();
    ParticleStorage particles;
    particles.reserve(kTotalParticles);
    for (size_t particle = 0; particle < kCrowdedParticles; particle++) {
        appendRaw(particles,
            {kCenter + static_cast<std::uint32_t>((particle % 17 + 1) * kStep), kCenter + static_cast<std::uint32_t>((particle % 13 + 1) * kStep / 2),
                kCenter + static_cast<std::uint32_t>((particle % 11 + 1) * kStep / 3)});
    }
    for (size_t particle = 0; particle < kSparseParticles; particle++) {
        appendRaw(particles, {3 * kCenter + static_cast<std::uint32_t>((particle + 1) * kStep), kCenter + kStep, kCenter - kStep});
    }
    appendRaw(particles, {0xf0000000U, 0x30000000U, 0x10000000U});
    CartesianGrid gpuGrid(2, 1, 0);
    gpuGrid.setInteriorRange({0, 0, 0}, {0, kCrowdedParticles});
    gpuGrid.setInteriorRange({1, 0, 0}, {kCrowdedParticles, kCrowdedParticles + kSparseParticles});
    gpuGrid.setGhostRanges(kGhost, {kTotalParticles - 1, kTotalParticles}, {1000000, 1000001});
    gpuGrid.fullExpansion({0, 0, 0}).fill(3.25f);
    gpuGrid.fullExpansion(kGhost).fill(-7.5f);
    CartesianGrid cpuGrid = gpuGrid;
    cpuGrid.computeMultipoles(particles, kMass, true);
    gpuGrid.computeMultipolesCuda(particles, kMass, 0, true);
    expectGridMomentsNear(gpuGrid, cpuGrid, true);
    auto const& crowded = gpuGrid.cell({0, 0, 0});
    EXPECT_FLOAT_EQ(crowded.fullMultipole[0], float(kCrowdedParticles) * kMass);
    EXPECT_FLOAT_EQ(gpuGrid.cell({1, 0, 0}).fullMultipole[0], float(kSparseParticles) * kMass);
    EXPECT_FLOAT_EQ(gpuGrid.cell(kGhost).fullMultipole[0], kMass);
    EXPECT_FLOAT_EQ(gpuGrid.cell({1, 1, 1}).fullMultipole[0], 0.0f);
    if constexpr (gravity::kMultipoleOrder >= 2) {
        double const trace = double(crowded.fullMultipole[symmetric::index(2, 0, 0)]) + double(crowded.fullMultipole[symmetric::index(0, 2, 0)]) +
            double(crowded.fullMultipole[symmetric::index(0, 0, 2)]);
        EXPECT_GT(trace, 0.0);
        EXPECT_NEAR(trace, crowded.secondMomentTrace, kTraceTolerance * crowded.secondMomentTrace);
    }
}

TEST_F(P2mCudaTest, CrowdedEmptyAndGhostCellsMatchCpu) {
    std::uint32_t constexpr kCenter = 0x20000000U;
    std::uint32_t constexpr kStep = 0x00400000U;
    constexpr GridIndices kGhost{-1, 0, 0};
    PROFILE();

    ParticleStorage particles;
    particles.reserve(258);
    for (size_t particle = 0; particle < 257; particle++) {
        appendRaw(particles,
            {kCenter + static_cast<std::uint32_t>((particle % 17 + 1) * kStep), kCenter + static_cast<std::uint32_t>((particle % 13 + 1) * kStep / 2),
                kCenter + static_cast<std::uint32_t>((particle % 11 + 1) * kStep / 3)});
    }
    appendRaw(particles, {0xf0000000U, 0x30000000U, 0x10000000U});

    CartesianGrid gpuGrid(2, 1, 0);
    gpuGrid.setInteriorRange({0, 0, 0}, {0, 257});
    gpuGrid.setGhostRanges(kGhost, {257, 258}, {1000000, 1000001});
    gpuGrid.expansion({0, 0, 0}).fill(3.25f);
    gpuGrid.expansion(kGhost).fill(-7.5f);
    gpuGrid.expansion({1, 1, 1}).fill(2.0f);
    CartesianGrid cpuGrid = gpuGrid;

    cpuGrid.computeMultipoles(particles, 0.75f);
    gpuGrid.computeMultipolesCuda(particles, 0.75f);
    expectGridMomentsNear(gpuGrid, cpuGrid);
    EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[0], 257.0f * 0.75f);
    EXPECT_GT(gpuGrid.cell({0, 0, 0}).multipole[stf::index(1, 0, 0)], 0.0f);
    if constexpr (gravity::kMultipoleOrder >= 5) {
        EXPECT_NE(gpuGrid.cell({0, 0, 0}).multipole[stf::index(5, 0, 0)], 0.0f);
    }
    EXPECT_GT(gpuGrid.cell({0, 0, 0}).secondMomentTrace, 0.0);
    EXPECT_FLOAT_EQ(gpuGrid.cell(kGhost).multipole[0], 0.75f);
    EXPECT_FLOAT_EQ(gpuGrid.cell({1, 1, 1}).multipole[0], 0.0f);
    EXPECT_DOUBLE_EQ(gpuGrid.cell({1, 1, 1}).secondMomentTrace, 0.0);

    particles.position(0, 0) = Coordinate::fromRaw(kCenter + 2 * kStep);
    cpuGrid.computeMultipoles(particles, 1.25f);
    gpuGrid.computeMultipolesCuda(particles, 1.25f);
    expectGridMomentsNear(gpuGrid, cpuGrid);
    EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[0], 257.0f * 1.25f);
}

TEST_F(P2mCudaTest, SparseThresholdAndFirstCooperativeCellMatchCpu) {
    PROFILE();
    ParticleStorage particles;
    particles.reserve(17);
    for (std::uint32_t particle = 0; particle < 8; particle++) {
        appendRaw(particles, {0x40000000U + (particle + 1) * 0x01000000U, 0x40000000U, 0x40000000U});
    }
    for (std::uint32_t particle = 0; particle < 9; particle++) {
        appendRaw(particles, {0xc0000000U + (particle + 1) * 0x01000000U, 0x40000000U, 0x40000000U});
    }
    CartesianGrid gpuGrid(1, 0, 0);
    gpuGrid.setInteriorRange({0, 0, 0}, {0, 8});
    gpuGrid.setInteriorRange({1, 0, 0}, {8, 17});
    CartesianGrid cpuGrid = gpuGrid;
    cpuGrid.computeMultipoles(particles, 1.25f);
    gpuGrid.computeMultipolesCuda(particles, 1.25f);
    expectGridMomentsNear(gpuGrid, cpuGrid);
    EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[0], 10.0f);
    EXPECT_FLOAT_EQ(gpuGrid.cell({1, 0, 0}).multipole[0], 11.25f);
}

TEST_F(P2mCudaTest, RawFixedPositionOffsetsSurviveFineCellNormalization) {
    size_t constexpr kSide = size_t(1) << 20;
    size_t constexpr kHalf = kSide / 2;
    size_t constexpr kRank = kHalf + kSide * (kHalf + kSide * kHalf);
    std::uint32_t constexpr kRawCenter = 0x80000800U;
    PROFILE();

    CartesianGrid gpuGrid(20, 20, kRank);
    ParticleStorage particles;
    appendRaw(particles, {kRawCenter + 1, kRawCenter + 17, kRawCenter - 19});
    gpuGrid.setInteriorRange({0, 0, 0}, {0, 1});
    CartesianGrid cpuGrid = gpuGrid;
    cpuGrid.computeMultipoles(particles);
    gpuGrid.computeMultipolesCuda(particles);
    expectGridMomentsNear(gpuGrid, cpuGrid);
    if constexpr (gravity::kMultipoleOrder >= 1) {
        EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[stf::index(1, 0, 0)], 1.0f / 4096.0f);
        EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[stf::index(0, 1, 0)], 17.0f / 4096.0f);
        EXPECT_FLOAT_EQ(gpuGrid.cell({0, 0, 0}).multipole[stf::index(0, 0, 1)], -19.0f / 4096.0f);
    }
}

}    // namespace
