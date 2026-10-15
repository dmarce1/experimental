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

#include <cosmictiger/fft/fft_communication_plan.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stddef.h>

namespace {

using cosmictiger::FftCommunicationStrategy;
using cosmictiger::FftCommunicationWorkload;
using cosmictiger::findOptimalFftCommunicationPlan;

constexpr size_t kTotalValues = size_t(1) << 39;
constexpr size_t kLocalityCount = size_t(1) << 12;
constexpr double kGibibyte = static_cast<double>(size_t(1) << 30);
constexpr double kTebibyte = kGibibyte * static_cast<double>(size_t(1) << 10);

TEST(FftCommunicationPlan, DirectRedistributionMinimizesBytes) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(kTotalValues, kLocalityCount, 1.0, 0.0);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::pencilSlab);
    ASSERT_EQ(plan.blockToPencil.size(), 1);
    EXPECT_EQ(plan.blockToPencil[0].sourceGroupSize, 16);
    EXPECT_EQ(plan.blockToPencil[0].destinationGroupSize, 16);
    ASSERT_EQ(plan.pencilToSlab.size(), 1);
    EXPECT_EQ(plan.pencilToSlab[0].sourceGroupSize, 4096);
    EXPECT_EQ(plan.pencilToSlab[0].destinationGroupSize, 4096);
    EXPECT_EQ(plan.communicationPhases, 8220);
    EXPECT_EQ(plan.messagesSent, 33'669'120);
    EXPECT_EQ(plan.bytesSent, 15.498046875 * kTebibyte);
    EXPECT_EQ(plan.weightedCost, plan.bytesSent);
}

TEST(FftCommunicationPlan, IntermediateWeightsChooseIntermediateExchangeGroups) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 1.0 / 32.0);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::pencilSlab);
    ASSERT_EQ(plan.blockToPencil.size(), 1);
    EXPECT_EQ(plan.blockToPencil[0].sourceGroupSize, 16);
    ASSERT_EQ(plan.pencilToSlab.size(), 3);
    for (auto const& stage : plan.pencilToSlab) {
        EXPECT_EQ(stage.sourceGroupSize, 16);
        EXPECT_EQ(stage.destinationGroupSize, 16);
    }
    EXPECT_EQ(plan.communicationPhases, 120);
    EXPECT_EQ(plan.messagesSent, 491'520);
    EXPECT_EQ(plan.bytesSent, 30.0 * kTebibyte);
}

TEST(FftCommunicationPlan, BinaryRedistributionsMinimizeMessagesWithinPencilSlabFamily) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 0.5);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::pencilSlab);
    ASSERT_EQ(plan.blockToPencil.size(), 4);
    ASSERT_EQ(plan.pencilToSlab.size(), 12);
    for (auto const& stage : plan.blockToPencil) {
        EXPECT_EQ(stage.sourceGroupSize, 2);
        EXPECT_EQ(stage.destinationGroupSize, 2);
    }
    for (auto const& stage : plan.pencilToSlab) {
        EXPECT_EQ(stage.sourceGroupSize, 2);
        EXPECT_EQ(stage.destinationGroupSize, 2);
    }
    EXPECT_EQ(plan.communicationPhases, 32);
    EXPECT_EQ(plan.messagesSent, 131'072);
    EXPECT_EQ(plan.bytesSent, 64.0 * kTebibyte);
}

TEST(FftCommunicationPlan, PackedVectorRadixWinsBetweenBinaryRoutingAndFullSpectrum) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 4.0);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::packedVectorRadix);
    EXPECT_TRUE(plan.blockToPencil.empty());
    EXPECT_TRUE(plan.pencilToSlab.empty());
    EXPECT_EQ(plan.vectorRadixLevels, 4);
    EXPECT_EQ(plan.communicationPhases, 25);
    EXPECT_EQ(plan.messagesSent, 102'392);
    EXPECT_EQ(plan.bytesSent, 99.9921875 * kTebibyte);
}

TEST(FftCommunicationPlan, FullSpectrumVectorRadixMinimizesMessagesWithoutReplication) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 32.0);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::fullSpectrumVectorRadix);
    EXPECT_EQ(plan.vectorRadixLevels, 4);
    EXPECT_EQ(plan.communicationPhases, 24);
    EXPECT_EQ(plan.messagesSent, 98'304);
    EXPECT_EQ(plan.bytesSent, 168.0 * kTebibyte);
}

TEST(FftCommunicationPlan, HandlesMinimumStorageBoundary) {
    PROFILE();
    auto const plan = findOptimalFftCommunicationPlan(size_t(1) << 9, size_t(1) << 3, 1.0, 0.0);

    ASSERT_EQ(plan.strategy, FftCommunicationStrategy::pencilSlab);
    EXPECT_EQ(plan.spectralLocalityCount, 8);
    ASSERT_EQ(plan.blockToPencil.size(), 1);
    EXPECT_EQ(plan.blockToPencil[0].sourceGroupSize, 4);
    EXPECT_EQ(plan.blockToPencil[0].destinationGroupSize, 4);
    ASSERT_EQ(plan.pencilToSlab.size(), 1);
    EXPECT_EQ(plan.pencilToSlab[0].sourceGroupSize, 8);
    EXPECT_EQ(plan.pencilToSlab[0].destinationGroupSize, 8);
    EXPECT_EQ(plan.communicationPhases, 22);
    EXPECT_EQ(plan.messagesSent, 168);
    EXPECT_EQ(plan.bytesSent, 17'408.0);
}

TEST(FftCommunicationPlan, UsesTheRealTypeSize) {
    PROFILE();
    auto const doublePlan = findOptimalFftCommunicationPlan<double>(kTotalValues, kLocalityCount, 1.0, 0.0);
    auto const floatPlan = findOptimalFftCommunicationPlan<float>(kTotalValues, kLocalityCount, 1.0, 0.0);
    EXPECT_EQ(floatPlan.bytesSent * 2.0, doublePlan.bytesSent);
    EXPECT_EQ(floatPlan.messagesSent, doublePlan.messagesSent);
}

TEST(FftCommunicationPlan, BatchPacksValuesIntoEachMessage) {
    PROFILE();
    auto const single = findOptimalFftCommunicationPlan<double>(kTotalValues, kLocalityCount, 1.0, 0.0);
    auto const batched = findOptimalFftCommunicationPlan<double, 4>(kTotalValues, kLocalityCount, 1.0, 0.0);

    EXPECT_EQ(batched.strategy, single.strategy);
    EXPECT_EQ(batched.communicationPhases, single.communicationPhases);
    EXPECT_EQ(batched.messagesSent, single.messagesSent);
    EXPECT_EQ(batched.bytesSent, 4.0 * single.bytesSent);
    EXPECT_EQ(batched.weightedCost, 4.0 * single.weightedCost);

    auto const singleWeighted = findOptimalFftCommunicationPlan<double>(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 4.0);
    auto const batchedWeighted = findOptimalFftCommunicationPlan<double, 4>(kTotalValues, kLocalityCount, 1.0 / kGibibyte, 4.0);
    EXPECT_EQ(singleWeighted.strategy, FftCommunicationStrategy::packedVectorRadix);
    EXPECT_EQ(batchedWeighted.strategy, FftCommunicationStrategy::pencilSlab);
}

TEST(FftCommunicationPlan, TwoInputWorkloadCountsBothForwardTransforms) {
    PROFILE();
    auto const prepositioned = findOptimalFftCommunicationPlan<double, 2>(16 * 16 * 16, 8, 1.0, 0.0);
    auto const twoInput = findOptimalFftCommunicationPlan<double, 2>(16 * 16 * 16, 8, 1.0, 0.0, FftCommunicationWorkload::twoForwardInputs);
    ASSERT_EQ(prepositioned.strategy, FftCommunicationStrategy::pencilSlab);
    ASSERT_EQ(twoInput.strategy, FftCommunicationStrategy::pencilSlab);
    EXPECT_EQ(prepositioned.workload, FftCommunicationWorkload::prepositionedMultiplier);
    EXPECT_EQ(twoInput.workload, FftCommunicationWorkload::twoForwardInputs);
    EXPECT_EQ(twoInput.bytesSent, 1.5 * prepositioned.bytesSent);
    EXPECT_EQ(twoInput.messagesSent, 3 * prepositioned.messagesSent / 2);
    EXPECT_EQ(twoInput.communicationPhases, 3 * prepositioned.communicationPhases / 2);

    auto const boundary = findOptimalFftCommunicationPlan<double, 2>(512, 8, 1.0, 0.0, FftCommunicationWorkload::twoForwardInputs);
    ASSERT_EQ(boundary.strategy, FftCommunicationStrategy::pencilSlab);
    EXPECT_EQ(boundary.bytesSent, 26'112.0 * 2.0);
    EXPECT_EQ(boundary.messagesSent, 252);
    EXPECT_EQ(boundary.communicationPhases, 33);

    auto const packed = findOptimalFftCommunicationPlan<double, 2>(16 * 16 * 16, 8, 0.0, 1.0, FftCommunicationWorkload::twoForwardInputs);
    ASSERT_EQ(packed.strategy, FftCommunicationStrategy::packedVectorRadix);
    EXPECT_EQ(packed.bytesSent, 9.0 * 16 * 16 * 16 * sizeof(double) * 2);
    EXPECT_EQ(packed.messagesSent, 72);
    EXPECT_EQ(packed.communicationPhases, 9);
}

TEST(FftCommunicationPlan, RejectsInputsOutsideTheCostModel) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    EXPECT_THROW(findOptimalFftCommunicationPlan(16, 8, 1.0, 1.0), std::runtime_error);
    EXPECT_THROW(findOptimalFftCommunicationPlan(512, 4, 1.0, 1.0), std::runtime_error);
    EXPECT_THROW(findOptimalFftCommunicationPlan(512, 64, 1.0, 1.0), std::runtime_error);
    EXPECT_THROW(findOptimalFftCommunicationPlan(512, 8, -1.0, 1.0), std::runtime_error);
    EXPECT_THROW(findOptimalFftCommunicationPlan(512, 8, 1.0, -1.0), std::runtime_error);
}

}    // namespace
