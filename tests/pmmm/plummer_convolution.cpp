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
#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_plummer_green.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <array>
#include <cmath>
#include <stddef.h>
#include <stdexcept>

namespace {
namespace gravity = cosmictiger::gravity;
namespace symmetric = gravity::symmetric;

struct PlummerSource {
    cosmictiger::GridIndices cell;
    std::array<double, NDIM> offset;
    float mass;
};

template <int width>
void checkPlummerConvolution() {
    constexpr size_t kSide = 8;
    constexpr int kSourceOrder = 2;
    constexpr int kLocalOrder = 3;
    constexpr int kImageCutoff = 2;
    constexpr int kWaveCutoff = 2;
    constexpr double kAlphaBox = 1.5;
    constexpr double kSpacing = 1.0 / double(kSide);
    constexpr double kEpsilon = 0.0625;
    constexpr float kGhostSentinel = -71.0f;
    constexpr double kTolerance = 3.0e-6;
    constexpr std::array<double, kSourceOrder + 1> kFactorials{1.0, 1.0, 2.0};
    constexpr auto kSources = std::array{PlummerSource{{0, 0, 0}, {0.125, -0.25, 0.375}, 0.5f}, PlummerSource{{3, 1, 6}, {-0.375, 0.25, -0.125}, 1.0f},
        PlummerSource{{7, 6, 2}, {0.25, 0.125, -0.25}, 0.25f}};
    constexpr std::array<cosmictiger::GridIndices, 8> kTargets{{{0, 0, 0}, {7, 7, 7}, {3, 1, 6}, {7, 6, 2}, {2, 2, 2}, {4, 0, 7}, {1, 5, 3}, {6, 3, 5}}};
    constexpr auto kSourceExponents = symmetric::exponents<kSourceOrder>();
    constexpr auto kLocalExponents = symmetric::exponents<kLocalOrder>();
    PROFILE();

    cosmictiger::CartesianGrid grid(3, 0, 0, width);
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        grid.fullExpansion(coordinates).fill(kGhostSentinel);
    }
    for (auto const& source : kSources) {
        auto& moments = grid.fullMultipole(source.cell);
        std::array<std::array<double, kSourceOrder + 1>, NDIM> powers{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            powers[axis][0] = 1.0;
            for (int order = 1; order <= kSourceOrder; order++) {
                powers[axis][order] = powers[axis][order - 1] * source.offset[axis];
            }
        }
        for (size_t beta = 0; beta < kSourceExponents.size(); beta++) {
            auto const b = kSourceExponents[beta];
            moments[beta] = static_cast<float>(source.mass * powers[0][b.nx] * powers[1][b.ny] * powers[2][b.nz]);
        }
    }
    gravity::PmmmPlummerGreenEvaluator<kSourceOrder + kLocalOrder, width> const evaluator(kSide, kSpacing, kEpsilon, kImageCutoff, kWaveCutoff, kAlphaBox);
    std::array<std::array<double, symmetric::coefficientCount(kLocalOrder)>, kTargets.size()> expected{};
    for (size_t target = 0; target < kTargets.size(); target++) {
        for (auto const& source : kSources) {
            std::array<int, NDIM> displacement{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                displacement[axis] = static_cast<int>(kTargets[target][axis] - source.cell[axis]);
            }
            auto const green = evaluator.evaluate(displacement);
            auto const& moments = grid.fullMultipole(source.cell);
            for (size_t alpha = 0; alpha < kLocalExponents.size(); alpha++) {
                auto const a = kLocalExponents[alpha];
                for (size_t beta = 0; beta < kSourceExponents.size(); beta++) {
                    auto const b = kSourceExponents[beta];
                    double const sign = b.rank % 2 == 0 ? 1.0 : -1.0;
                    double const factor = sign / (kFactorials[b.nx] * kFactorials[b.ny] * kFactorials[b.nz]);
                    expected[target][alpha] += factor * double(moments[beta]) * green[symmetric::index(a.nx + b.nx, a.ny + b.ny, a.nz + b.nz)];
                }
            }
        }
    }
    gravity::PmmmPeriodicConvolution<kSourceOrder, kLocalOrder> convolution(
        grid.layout(), kImageCutoff, kWaveCutoff, kAlphaBox, width, gravity::SofteningModel::plummer, kEpsilon);
    auto exchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("unexpected remote Plummer FFT exchange");
        },
        "test.plummerConvolution.localExchange");
    for (int pass = 1; pass <= 2; pass++) {
        convolution.computeExpansions(grid, exchange);
        for (size_t target = 0; target < kTargets.size(); target++) {
            auto const& actual = grid.cell(kTargets[target]).fullExpansion;
            for (size_t coefficient = 0; coefficient < kLocalExponents.size(); coefficient++) {
                double const reference = double(pass) * expected[target][coefficient];
                EXPECT_NEAR(actual[coefficient], reference, kTolerance * (1.0 + std::abs(reference)));
            }
            for (size_t coefficient = kLocalExponents.size(); coefficient < actual.size(); coefficient++) {
                EXPECT_FLOAT_EQ(actual[coefficient], 0.0f);
            }
        }
        for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
            auto const coordinates = grid.layout().cellCoordinates(index);
            if (!grid.layout().isInterior(coordinates)) {
                for (float coefficient : grid.cell(coordinates).fullExpansion) {
                    EXPECT_FLOAT_EQ(coefficient, kGhostSentinel);
                }
            }
        }
        if (pass == 1) {
            for (auto const& source : kSources) {
                for (float& moment : grid.fullMultipole(source.cell)) {
                    moment *= 2.0f;
                }
            }
        }
    }
}
}    // namespace

TEST(PlummerConvolution, FourierContractionMatchesAllCartesianDerivativeIndices) {
    PROFILE();
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    checkPlummerConvolution<1>();
    checkPlummerConvolution<2>();
}
