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

// Newton's third law for complete gravity evaluations: with equal masses the
// owned accelerations must sum to zero up to floating-point rounding. Each
// case reports |sum a| / sum |a| for the complete field and for its near
// (direct) part alone, so a violation can be attributed to one branch.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stddef.h>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_runtime_convolution.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/profiler.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

namespace {

using namespace cosmictiger;
using namespace cosmictiger::gravity;

constexpr size_t kMeshExponent = 3;
constexpr size_t kUniformCount = 1500;
constexpr size_t kClusterCount = 1500;
constexpr double kClusterWidth = 0.03;
constexpr double kSofteningCellFraction = 0.05;
constexpr int kFerrersOrder = 2;
constexpr std::uint64_t kSeed = 42;

struct SymmetryCase {
    int ghostWidth;
    int fmmOrder;
    SofteningModel model;
    bool cuda;
    NearFieldMethod nearField = NearFieldMethod::direct;
};

struct SymmetryResult {
    double complete;
    double nearOnly;
};

// Append particles in interior cell order and assign each cell its range.
void seedParticles(CartesianGrid& grid, ParticleStorage& particles) {
    PROFILE();
    auto const& layout = grid.layout();
    std::mt19937_64 generator(kSeed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, kClusterWidth);
    std::vector<std::array<Coordinate, NDIM>> positions;
    for (size_t index = 0; index < kUniformCount + kClusterCount; index++) {
        std::array<Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            double value = index < kUniformCount ? uniform(generator) : 0.3 + normal(generator);
            value -= std::floor(value);
            position[axis] = Coordinate(value);
        }
        positions.push_back(position);
    }
    std::stable_sort(positions.begin(), positions.end(), [&layout](auto const& first, auto const& second) {
        return layout.cellIndex(layout.particleOwner(first).coordinates) < layout.cellIndex(layout.particleOwner(second).coordinates);
    });
    particles.configureGravityFields(true, false);
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

double imbalance(ParticleStorage const& particles) {
    PROFILE();
    std::array<double, NDIM> sum{};
    double magnitude = 0.0;
    for (ParticleCount index = 0; index < particles.size(); index++) {
        double squared = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const value = particles.acceleration(axis, index);
            sum[axis] += value;
            squared += value * value;
        }
        magnitude += std::sqrt(squared);
    }
    return std::hypot(sum[0], sum[1], sum[2]) / magnitude;
}

// Evaluate every owned particle once without changing momenta or rungs.
// The near-only pass leaves every cell expansion at zero.
SymmetryResult measure(SymmetryCase const& testCase, std::uint64_t operationId) {
    PROFILE();
    CartesianGrid grid(kMeshExponent, 0, 0, static_cast<size_t>(testCase.ghostWidth));
    ParticleStorage particles;
    seedParticles(grid, particles);
    double const spacing = grid.layout().spacing();
    double const softeningLength = kSofteningCellFraction * spacing;
    float const particleMass = static_cast<float>(1.0 / double(particles.size()));
    auto const softening = makeSoftening(static_cast<float>(softeningLength), testCase.model);
    bool const fullMoments = testCase.model == SofteningModel::plummer;
    KickParameters parameters;
    parameters.phase = KickPhase::diagnostics;
    parameters.ghostWidth = testCase.ghostWidth;
    parameters.nearFieldMethod = testCase.nearField;
    parameters.fmmOrder = testCase.fmmOrder;
    PmmmRuntimeConvolution convolution(grid.layout(), testCase.fmmOrder, 4, 4, 1.5, testCase.ghostWidth, testCase.model, softeningLength);
    FftHpxExchange exchange({hpx::find_here()}, operationId, 0);
    SymmetryResult result{};
    for (bool const far : {false, true}) {
        grid.computeMultipoles(particles, particleMass, fullMoments);
        auto farField = hpx::annotated_function(
            [&] {
                PROFILE();
                if (far) {
                    convolution.computeExpansions(grid, exchange);
                }
            },
            "test.forceSymmetry.farField");
#ifdef COSMICTIGER_ENABLE_CUDA
        if (testCase.cuda) {
            kickPmmmParticlesOverlappedCuda(grid, particles, softening, kFerrersOrder, particleMass, 1.0, 0, parameters, farField, operationId + 1 + far);
        } else
#endif
        {
            kickPmmmParticlesOverlapped(grid, particles, softening, kFerrersOrder, particleMass, 1.0, 0, parameters, farField, operationId + 1 + far);
        }
        (far ? result.complete : result.nearOnly) = imbalance(particles);
    }
    return result;
}

class ForceSymmetryTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        try {
            if (cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA-enabled ParticleStorage requires a CUDA device";
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

void runCases(bool cuda, std::uint64_t operationBase, NearFieldMethod nearField = NearFieldMethod::direct) {
    PROFILE();
    // Single-precision epsilon. Measured rounding is about 1e-9 for these
    // cases, while truncation-level asymmetry would be near the FMM error.
    constexpr double kTolerance = 1.0e-7;
    std::uint64_t operationId = operationBase;
    for (int const ghostWidth : {1, 2}) {
        for (int const fmmOrder : {4, 6}) {
            for (auto const model : {SofteningModel::ferrers, SofteningModel::plummer}) {
                // The tree near field requires Ferrers softening.
                if (nearField == NearFieldMethod::tree && model != SofteningModel::ferrers) {
                    continue;
                }
                SymmetryCase const testCase{ghostWidth, fmmOrder, model, cuda, nearField};
                auto const result = measure(testCase, operationId);
                operationId += 8;
                std::printf("force symmetry: near=%s backend=%s width=%d order=%d model=%s complete=%.3e near=%.3e\n",
                    nearField == NearFieldMethod::tree ? "tree" : "direct", cuda ? "cuda" : "cpu", ghostWidth, fmmOrder,
                    model == SofteningModel::ferrers ? "ferrers" : "plummer", result.complete, result.nearOnly);
                std::fflush(stdout);
                SCOPED_TRACE(::testing::Message() << "width=" << ghostWidth << " order=" << fmmOrder << " plummer=" << (model == SofteningModel::plummer));
                EXPECT_LT(result.nearOnly, kTolerance);
                EXPECT_LT(result.complete, kTolerance);
            }
        }
    }
}

TEST_F(ForceSymmetryTest, CpuCompleteAndNearFieldsConserveMomentum) {
    PROFILE();
    runCases(false, 0x5100);
}

TEST_F(ForceSymmetryTest, CpuTreeNearFieldConservesMomentum) {
    PROFILE();
    runCases(false, 0x5300, NearFieldMethod::tree);
}

#ifdef COSMICTIGER_ENABLE_CUDA
TEST_F(ForceSymmetryTest, CudaCompleteAndNearFieldsConserveMomentum) {
    PROFILE();
    runCases(true, 0x5200);
}
#endif

}    // namespace
