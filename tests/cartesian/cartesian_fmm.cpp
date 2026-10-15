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
#include <cmath>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/cartesian/fmm.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <gtest/gtest.h>
#include <hpx/include/async.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/type_support/bit_cast.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace {
namespace cartesian = cosmictiger::gravity::cartesian;
namespace stf = cosmictiger::gravity::stf;

void expectFieldBitsEqual(cartesian::Field const& actual, cartesian::Field const& expected) {
    PROFILE();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t target = 0; target < actual.size(); target++) {
        ASSERT_EQ(actual[target].size(), expected[target].size());
        for (size_t coefficient = 0; coefficient < actual[target].size(); coefficient++) {
            EXPECT_EQ(hpx::bit_cast<std::uint64_t>(actual[target][coefficient]), hpx::bit_cast<std::uint64_t>(expected[target][coefficient]))
                << "target=" << target << " coefficient=" << coefficient;
        }
    }
}

TEST(CartesianFmm, StorageAndPointTranslationsAtEveryOrder) {
    constexpr cartesian::Vector kOffset{.13, -.21, .07};
    constexpr cartesian::Vector kShift{-.11, .03, .09};
    constexpr double kRatio = .7;
    PROFILE();

    for (int p = 0; p <= cartesian::kMaxOrder; p++) {
        SCOPED_TRACE(p);
        EXPECT_EQ(cartesian::coefficientCount(p), (p + 1) * (p + 1));
        auto const& indices = cartesian::indices(p);
        EXPECT_EQ(indices.size(), static_cast<size_t>((p + 1) * (p + 1)));
        for (size_t i = 0; i < indices.size(); i++) {
            EXPECT_EQ(stf::index(indices[i].x(), indices[i].y(), indices[i].z()), i);
            EXPECT_LE(indices[i].degree(), p);
        }
        auto const point = cartesian::point(2.3, kOffset, p);
        cartesian::Vector translated{};
        for (size_t d = 0; d < NDIM; d++) {
            translated[d] = kRatio * kOffset[d] + kShift[d];
        }
        auto const expected = cartesian::point(2.3, translated, p);
        auto const actual = cartesian::shiftMultipole(point, kShift, kRatio, p);
        auto const massOnly = cartesian::shiftMultipole({2.3}, kShift, kRatio, p);
        auto const shiftedMass = cartesian::point(2.3, kShift, p);
        for (size_t i = 0; i < actual.size(); i++) {
            EXPECT_NEAR(actual[i], expected[i], 2e-13 * std::max(1., std::abs(expected[i])));
            EXPECT_NEAR(massOnly[i], shiftedMass[i], 2e-13);
        }
    }
}

TEST(CartesianFmm, MonopoleM2LMatchesDirectAndPreservesTraceAtEveryOrder) {
    constexpr cartesian::Offset kSeparation{3, -2, 4};
    constexpr double kWidth = .37;
    constexpr double kMass = 2.1;
    PROFILE();

    for (int p = cartesian::kMinOrder; p <= cartesian::kMaxOrder; p++) {
        SCOPED_TRACE(p);
        size_t const count = cartesian::coefficientCount(p);
        cartesian::Coefficients local(count + 1, 0), direct(5, 0), physical(count + 1, 0);
        local.back() = direct.back() = physical.back() = -3.5;
        cartesian::Operator op(p, kSeparation);
        op.add(local, {kMass}, kWidth);
        cartesian::addDirect(direct, kMass, kSeparation, kWidth);
        cartesian::Vector r{};
        for (size_t d = 0; d < NDIM; d++) {
            r[d] = kWidth * kSeparation[d];
        }
        cartesian::addM2L(physical, {kMass}, r, kWidth, kWidth, p);
        for (size_t i = 0; i < 4; i++) {
            EXPECT_NEAR(local[i], direct[i], 2e-14 * std::max(1., std::abs(direct[i])));
        }
        for (size_t i = 0; i < local.size(); i++) {
            EXPECT_NEAR(local[i], physical[i], 2e-13 * std::max(1., std::abs(physical[i])));
        }
        EXPECT_EQ(local.back(), -3.5);
        cartesian::Coefficients leaf(5, 0);
        leaf.back() = -7;
        op.add(leaf, {kMass}, kWidth);
        for (size_t i = 0; i < 4; i++) {
            EXPECT_NEAR(leaf[i], direct[i], 2e-14 * std::max(1., std::abs(direct[i])));
        }
        EXPECT_EQ(leaf.back(), -7);
        if (p >= 2) {
            EXPECT_NEAR(cartesian::derivative(local, 2, 0, 0) + cartesian::derivative(local, 0, 2, 0) + cartesian::derivative(local, 0, 0, 2), -3.5, 2e-14);
        }
    }
}

TEST(CartesianFmm, DipolesScaleChangesAndCenteredM2LAtEveryOrder) {
    constexpr cartesian::Offset kSeparation{3, -2, 4};
    constexpr double kWidth = .37;
    PROFILE();

    cartesian::Vector r{};
    for (size_t d = 0; d < NDIM; d++) {
        r[d] = kWidth * kSeparation[d];
    }
    for (int p = 1; p <= cartesian::kMaxOrder; p++) {
        SCOPED_TRACE(p);
        auto source = cartesian::point(2.1, {.13, -.07, .11}, p);
        auto const rescaled = cartesian::shiftMultipole(source, {}, .7, p);
        cartesian::Coefficients cached(source.size(), 0), physical(source.size(), 0);
        cartesian::Operator(p, kSeparation).add(cached, source, kWidth);
        cartesian::addM2L(physical, rescaled, r, kWidth / .7, kWidth, p);
        for (size_t i = 0; i < cached.size(); i++) {
            EXPECT_NEAR(cached[i], physical[i], 3e-13 * std::max(1., std::abs(cached[i])));
        }
        cartesian::Coefficients centered(source.size(), 0), zeroDipole(source.size(), 0);
        cartesian::addM2L(centered, source, r, kWidth, kWidth, p, true);
        for (size_t i = 1; i < 4; i++) {
            source[i] = 0;
        }
        cartesian::addM2L(zeroDipole, source, r, kWidth, kWidth, p);
        for (size_t i = 0; i < centered.size(); i++) {
            EXPECT_NEAR(centered[i], zeroDipole[i], 2e-13);
        }
    }
}

TEST(CartesianFmm, BatchedTargetsMatchScalarM2LBitwiseAndPreserveEmptyRowsAndTraces) {
    constexpr size_t kTargetCount = 37;
    PROFILE();

    for (int const order : {1, 3, cartesian::kMaxOrder}) {
        cartesian::Field source{{2.3}, cartesian::point(1.7, {.13, -.07, .11}, order), cartesian::point(.4, {-.08, .17, .03}, order)};
        source.back().push_back(-4.5);    // source trace is irrelevant to exterior M2L
        auto const originalSource = source;
        size_t const coefficientCount = cartesian::coefficientCount(order);
        cartesian::Field local(kTargetCount, cartesian::Coefficients(coefficientCount + 1));
        std::vector<std::vector<cartesian::M2lInteraction>> interactions(kTargetCount);
        std::vector<cartesian::M2lTarget> targets(kTargetCount);
        for (size_t target = 0; target < kTargetCount; target++) {
            for (size_t coefficient = 0; coefficient < coefficientCount; coefficient++) {
                local[target][coefficient] = .017 * static_cast<double>(target + coefficient + 1);
            }
            local[target].back() = -3.5 - .125 * static_cast<double>(target);
            if (target % 7 != 0) {
                double const shift = .01 * static_cast<double>(target);
                interactions[target] = {{1, {2.5 + shift, -1.75, .5}, .75, false}, {0, {-1.25, 3.5 + shift, .8}, 1., false},
                    {2, {.5, -2.25, 3.75 + shift}, 1.25, target % 2 == 0}, {1, {3.5, .75 + shift, -2.25}, .6, false}};
            }
            targets[target] = {interactions[target], .5 + .0625 * static_cast<double>(target % 5)};
        }
        auto expected = local;
        for (size_t target = 0; target < targets.size(); target++) {
            for (auto const& interaction : targets[target].interactions) {
                cartesian::addM2L(expected[target], source[interaction.sourceIndex], interaction.separation, interaction.sourceScale,
                    targets[target].targetScale, order, interaction.centered);
            }
        }
        cartesian::addM2LTargets(local, source, targets, order);
        expectFieldBitsEqual(local, expected);
        expectFieldBitsEqual(source, originalSource);
        for (size_t target = 0; target < kTargetCount; target++) {
            EXPECT_EQ(local[target].back(), -3.5 - .125 * static_cast<double>(target));
        }
    }

    cartesian::Field emptyLocal;
    cartesian::Field const emptySource;
    EXPECT_NO_THROW(cartesian::addM2LTargets(emptyLocal, emptySource, {}, 1));
    EXPECT_TRUE(emptyLocal.empty());
}

TEST(CartesianFmm, BatchedTargetsPreserveCancellationSensitiveSourceOrder) {
    constexpr size_t kTargetCount = 33;
    constexpr std::array<cartesian::M2lInteraction, 3> kLostUnit{{{0, {1., 0., 0.}, 1.}, {1, {1., 0., 0.}, 1.}, {2, {1., 0., 0.}, 1.}}};
    constexpr std::array<cartesian::M2lInteraction, 3> kRetainedUnit{{kLostUnit[0], kLostUnit[2], kLostUnit[1]}};
    PROFILE();

    cartesian::Field const source{{1e16}, {1.}, {-1e16}};
    cartesian::Field local(kTargetCount, cartesian::Coefficients(cartesian::coefficientCount(1), 0.));
    auto expected = local;
    std::vector<cartesian::M2lTarget> targets;
    targets.reserve(kTargetCount);
    for (size_t target = 0; target < kTargetCount; target++) {
        targets.push_back({target % 2 == 0 ? kLostUnit : kRetainedUnit, 1.});
        for (auto const& interaction : targets.back().interactions) {
            cartesian::addM2L(expected[target], source[interaction.sourceIndex], interaction.separation, interaction.sourceScale, 1., 1);
        }
    }
    ASSERT_NE(expected[0][0], expected[1][0]);
    cartesian::addM2LTargets(local, source, targets, 1);
    expectFieldBitsEqual(local, expected);
}

TEST(CartesianFmm, BatchedTargetsValidateConsumedInputsAndRejectCrossTargetAliasing) {
    constexpr std::array<cartesian::M2lInteraction, 2> kCross{{{1, {1., 2., 3.}, 1.}, {0, {-1., -2., -3.}, 1.}}};
    PROFILE();

    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    cartesian::Field source{{2.}};
    cartesian::Field local(1, cartesian::Coefficients(cartesian::coefficientCount(2), .25));
    std::array<cartesian::M2lInteraction, 1> interactions{{{0, {1., 2., 3.}, .75}}};
    std::array<cartesian::M2lTarget, 1> targets{{{interactions, .5}}};
    auto const original = local;
    EXPECT_THROW(cartesian::addM2LTargets(local, source, {}, 2), std::runtime_error);
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 0), std::runtime_error);
    interactions[0].sourceIndex = source.size();
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);
    interactions[0].sourceIndex = 0;
    interactions[0].sourceScale = 0.;
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);
    interactions[0].sourceScale = .75;
    targets[0].targetScale = 0.;
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);
    targets[0].targetScale = .5;
    interactions[0].separation = {};
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);
    interactions[0].separation = {1., 2., 3.};
    source[0] = {1., 2.};
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);
    expectFieldBitsEqual(local, original);
    source[0] = {2.};
    local[0].resize(2);
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 2), std::runtime_error);

    cartesian::Field alias(2, cartesian::Coefficients(cartesian::coefficientCount(2), .25));
    auto const originalAlias = alias;
    std::array<cartesian::M2lTarget, 2> const crossTargets{{{std::span(kCross).first(1), 1.}, {std::span(kCross).last(1), 1.}}};
    EXPECT_THROW(cartesian::addM2LTargets(alias, alias, crossTargets, 2), std::runtime_error);
    expectFieldBitsEqual(alias, originalAlias);
}

TEST(CartesianFmm, BatchedTargetsDrainOtherTasksBeforeRethrowing) {
    constexpr size_t kTargetCount = 32;
    constexpr cartesian::M2lInteraction kValid{0, {2., -3., 4.}, .75};
    PROFILE();

    if (!cosmictiger::validationChecksEnabled() || hpx::get_os_thread_count() < 2) {
        GTEST_SKIP() << "requires validation checks and multiple HPX workers";
    }
    cartesian::Field const source{{2.}};
    cartesian::Field local(kTargetCount, cartesian::Coefficients(cartesian::coefficientCount(3), 0.));
    std::array<cartesian::M2lInteraction, 2> const failing{{kValid, {source.size(), {2., -3., 4.}, .75}}};
    std::vector<cartesian::M2lInteraction> const longList(256, kValid);
    std::vector<cartesian::M2lTarget> targets(kTargetCount, {std::span(&kValid, 1), .5});
    targets.front().interactions = failing;
    targets.back().interactions = longList;
    cartesian::Coefficients expectedFirst = local.front();
    cartesian::Coefficients expectedLast = local.back();
    cartesian::addM2L(expectedFirst, source[0], kValid.separation, kValid.sourceScale, .5, 3);
    for (auto const& interaction : longList) {
        cartesian::addM2L(expectedLast, source[0], interaction.separation, interaction.sourceScale, .5, 3);
    }
    EXPECT_THROW(cartesian::addM2LTargets(local, source, targets, 3), std::runtime_error);
    // The first row demonstrates partial mutation at the failing interaction;
    // the last row belongs to a separate task and must finish before rethrow.
    expectFieldBitsEqual({local.front(), local.back()}, {expectedFirst, expectedLast});
}

TEST(CartesianFmm, LocalPolynomialShiftCompositionAndScaling) {
    constexpr cartesian::Vector kA{.1, -.2, .05};
    constexpr cartesian::Vector kB{-.03, .04, .09};
    constexpr double kRatio = .7;
    PROFILE();

    for (int p = 0; p <= cartesian::kMaxOrder; p++) {
        cartesian::Coefficients local(cartesian::coefficientCount(p) + 1, 0);
        for (size_t i = 0; i + 1 < local.size(); i++) {
            local[i] = .017 * (i + 1);
        }
        local.back() = p >= 2 ? -2.7 : 0;
        cartesian::Vector combined{};
        for (size_t d = 0; d < NDIM; d++) {
            combined[d] = kA[d] + kRatio * kB[d];
        }
        auto const first = cartesian::shiftLocal(local, kA, kRatio, p);
        auto const composed = cartesian::shiftLocal(first, kB, .9, p);
        auto const direct = cartesian::shiftLocal(local, combined, kRatio * .9, p);
        for (size_t i = 0; i < composed.size(); i++) {
            EXPECT_NEAR(composed[i], direct[i], 3e-13 * std::max(1., std::abs(direct[i])));
        }
        auto const value = cartesian::evaluate(local, combined, p);
        EXPECT_NEAR(value[0], direct[0], 3e-13);
        if (p > 0) {
            for (size_t d = 0; d < NDIM; d++) {
                EXPECT_NEAR(value[d + 1] * kRatio * .9, direct[stf::index(d == 0, d == 1, d == 2)], 3e-13);
            }
        }
    }
}

TEST(CartesianFmm, CacheReusesGeometryAcrossHpxTasks) {
    constexpr cartesian::Offset kOffset{13, -7, 11};
    PROFILE();

    size_t const before = cartesian::cachedOperatorCount();
    std::vector<hpx::future<std::shared_ptr<cartesian::Operator const>>> requests;
    for (int i = 0; i < 32; i++) {
        requests.push_back(hpx::async(hpx::annotated_function(
            [kOffset] {
                PROFILE();
                return cartesian::getOperator(3, kOffset);
            },
            "test.cartesian.getOperator")));
    }
    auto const first = requests.front().get();
    for (size_t i = 1; i < requests.size(); i++) {
        EXPECT_EQ(requests[i].get(), first);
    }
    EXPECT_EQ(cartesian::getOperator(3, kOffset), first);
    EXPECT_EQ(cartesian::cachedOperatorCount(), before + 1);
}

TEST(CartesianFmm, RejectsInvalidShapesGeometryAndAliasing) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::coefficientCount(-1), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::point(1, {}, cartesian::kMaxOrder + 1), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::Operator(0, {1, 0, 0}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::Operator(3, {}), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::shiftMultipole({1, 2}, {}, 1, 3), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::shiftLocal({1, 2}, {}, 1, 3), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::shiftMultipole({1}, {}, 0, 3), std::runtime_error);
    }
    cartesian::Coefficients local(16, .25);
    auto const before = local;
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::Operator(3, {1, 0, 0}).add(local, local, 1), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::addM2L(local, {1}, {}, 1, 1, 3), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::derivative(local, -1, 0, 0), std::runtime_error);
    }
    if constexpr (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(cartesian::derivative(local, 0, 0, 4), std::runtime_error);
    }
    EXPECT_EQ(local, before);
}
}    // namespace
