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

#include <cosmictiger/gravity/kick/kick_sums.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cosmictiger::ParticleCount;
using cosmictiger::gravity::addKickSums;
using cosmictiger::gravity::allReduceKickSums;
using cosmictiger::gravity::KickSums;

std::uint64_t nextOperationId() {
    PROFILE();
    static std::atomic<std::uint64_t> sequence{0x4b49434b53554d00ULL};
    return sequence.fetch_add(1);
}

KickSums contribution(size_t siteIndex) {
    PROFILE();
    KickSums result;
    double const value = static_cast<double>(siteIndex + 1);
    for (size_t axis = 0; axis < NDIM; axis++) {
        result.acceleration[axis] = (axis % 2 == 0 ? value : -value) * static_cast<double>(axis + 1);
    }
    result.potential = -2.0 * value;
    result.particleCount = static_cast<ParticleCount>(siteIndex + 1);
    result.momentumSquared = 3.0 * value;
    result.maximumRung = static_cast<std::uint32_t>(siteIndex % (cosmictiger::Rung::maximumRung() + 1));
    return result;
}

void expectTotals(KickSums const& result, size_t siteCount, bool includesPotential) {
    PROFILE();
    double const sum = static_cast<double>(siteCount) * static_cast<double>(siteCount + 1) / 2.0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_DOUBLE_EQ(result.acceleration[axis], (axis % 2 == 0 ? sum : -sum) * static_cast<double>(axis + 1));
    }
    EXPECT_DOUBLE_EQ(result.potential, includesPotential ? -2.0 * sum : 0.0);
    EXPECT_EQ(result.particleCount, siteCount * (siteCount + 1) / 2);
    EXPECT_DOUBLE_EQ(result.momentumSquared, 3.0 * sum);
    EXPECT_EQ(result.maximumRung, siteCount > cosmictiger::Rung::maximumRung() ? cosmictiger::Rung::maximumRung() : siteCount - 1);
}

}    // namespace

TEST(KickSums, AddsCompleteFieldsAndEmptyContributions) {
    PROFILE();
    auto const first = contribution(0);
    auto const second = contribution(1);
    expectTotals(addKickSums(first, second), 2, true);
    expectTotals(addKickSums(addKickSums(first, second), KickSums{}), 2, true);
}

TEST(KickSums, SerializationPreservesKineticAndRungDiagnostics) {
    PROFILE();
    auto expected = addKickSums(contribution(0), contribution(1));
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    KickSums restored;
    {
        hpx::serialization::input_archive archive(bytes, bytes.size());
        archive >> restored;
    }
    expectTotals(restored, 2, true);
}

TEST(KickSums, RejectsParticleCountOverflow) {
    PROFILE();
    KickSums first;
    first.particleCount = std::numeric_limits<ParticleCount>::max();
    KickSums second;
    second.particleCount = 1;
    EXPECT_THROW(addKickSums(first, second), std::overflow_error);
}

TEST(KickSums, LogicalLocalityReduction) {
    size_t constexpr kSiteCount = 4;
    PROFILE();

    for (bool includesPotential : {false, true}) {
        auto const operationId = nextOperationId();
        std::vector<hpx::future<KickSums>> futures;
        for (size_t siteIndex = 0; siteIndex < kSiteCount; siteIndex++) {
            futures.push_back(hpx::async(hpx::annotated_function(
                [=] {
                    PROFILE();
                    return allReduceKickSums(contribution(siteIndex), includesPotential, operationId, kSiteCount, siteIndex);
                },
                "test.kick.reduceLogicalSums")));
        }
        for (auto& future : futures) {
            expectTotals(future.get(), kSiteCount, includesPotential);
        }
    }
}

TEST(KickSums, EmptyLogicalLocalitiesParticipate) {
    size_t constexpr kSiteCount = 4;
    PROFILE();

    auto const operationId = nextOperationId();
    std::vector<hpx::future<KickSums>> futures;
    for (size_t siteIndex = 0; siteIndex < kSiteCount; siteIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [=] {
                PROFILE();
                return allReduceKickSums(siteIndex == 0 ? contribution(0) : KickSums{}, true, operationId, kSiteCount, siteIndex);
            },
            "test.kick.reduceEmptyLogicalSites")));
    }
    for (auto& future : futures) {
        expectTotals(future.get(), 1, true);
    }
}

TEST(KickSums, PotentialSelectionMismatchReachesEveryLogicalLocality) {
    size_t constexpr kSiteCount = 4;
    PROFILE();

    auto const operationId = nextOperationId();
    std::vector<hpx::future<KickSums>> futures;
    for (size_t siteIndex = 0; siteIndex < kSiteCount; siteIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [=] {
                PROFILE();
                return allReduceKickSums(contribution(siteIndex), siteIndex == 0, operationId, kSiteCount, siteIndex);
            },
            "test.kick.reduceMismatchedPotential")));
    }
    for (auto& future : futures) {
        EXPECT_THROW(future.get(), std::invalid_argument);
    }
}

TEST(KickSums, LocalFailureReachesEveryLogicalLocality) {
    size_t constexpr kSiteCount = 4;
    PROFILE();

    auto const operationId = nextOperationId();
    std::vector<hpx::future<KickSums>> futures;
    for (size_t siteIndex = 0; siteIndex < kSiteCount; siteIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [=] {
                PROFILE();
                return allReduceKickSums(
                    contribution(siteIndex), true, operationId, kSiteCount, siteIndex, siteIndex == 2 ? "injected local kick failure" : "");
            },
            "test.kick.reduceLocalFailure")));
    }
    for (auto& future : futures) {
        try {
            (void) future.get();
            ADD_FAILURE() << "local kick failure was not propagated";
        } catch (std::runtime_error const& error) {
            EXPECT_NE(std::string(error.what()).find("injected local kick failure"), std::string::npos);
        }
    }
}

TEST(KickSums, ReductionFailureReachesEveryLogicalLocality) {
    size_t constexpr kSiteCount = 2;
    PROFILE();

    auto const operationId = nextOperationId();
    std::vector<hpx::future<KickSums>> futures;
    for (size_t siteIndex = 0; siteIndex < kSiteCount; siteIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [=] {
                PROFILE();
                KickSums local;
                local.particleCount = std::numeric_limits<ParticleCount>::max();
                return allReduceKickSums(local, false, operationId, kSiteCount, siteIndex);
            },
            "test.kick.reduceOverflow")));
    }
    for (auto& future : futures) {
        try {
            (void) future.get();
            ADD_FAILURE() << "kick reduction overflow was not propagated";
        } catch (std::runtime_error const& error) {
            EXPECT_NE(std::string(error.what()).find("particle count overflow"), std::string::npos);
        }
    }
}

TEST(KickSums, RejectsInvalidParticipationMetadata) {
    PROFILE();
    EXPECT_THROW(allReduceKickSums({}, false, nextOperationId(), 0, 0), std::invalid_argument);
    EXPECT_THROW(allReduceKickSums({}, false, nextOperationId(), 2, 2), std::invalid_argument);
}

namespace cosmictiger::detail {

gravity::KickSums reduceRemoteKickSums(gravity::KickSums local, bool includesPotential, std::uint64_t operationId) {
    PROFILE();
    return gravity::allReduceKickSums(local, includesPotential, operationId);
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::reduceRemoteKickSums, CosmictigerReduceRemoteKickSumsAction)

TEST(KickSums, PhysicalLocalityReduction) {
    PROFILE();
    auto const localities = hpx::find_all_localities();
    for (bool includesPotential : {false, true}) {
        auto const operationId = nextOperationId();
        std::vector<hpx::future<KickSums>> futures;
        for (size_t siteIndex = 0; siteIndex < localities.size(); siteIndex++) {
            futures.push_back(
                hpx::async<CosmictigerReduceRemoteKickSumsAction>(localities[siteIndex], contribution(siteIndex), includesPotential, operationId));
        }
        for (auto& future : futures) {
            expectTotals(future.get(), localities.size(), includesPotential);
        }
    }
}

TEST(KickSums, PotentialSelectionMismatchReachesEveryPhysicalLocality) {
    PROFILE();
    auto const localities = hpx::find_all_localities();
    if (localities.size() < 2) {
        GTEST_SKIP() << "requires at least two HPX localities";
    }
    auto const operationId = nextOperationId();
    std::vector<hpx::future<KickSums>> futures;
    for (size_t siteIndex = 0; siteIndex < localities.size(); siteIndex++) {
        futures.push_back(hpx::async<CosmictigerReduceRemoteKickSumsAction>(localities[siteIndex], contribution(siteIndex), siteIndex == 0, operationId));
    }
    for (auto& future : futures) {
        EXPECT_THROW(future.get(), std::invalid_argument);
    }
}
