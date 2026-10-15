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

#pragma once

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cosmictiger/dimensions.hpp>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger {

enum class FftCommunicationStrategy { pencilSlab, packedVectorRadix, fullSpectrumVectorRadix };
enum class FftCommunicationWorkload { prepositionedMultiplier, twoForwardInputs };

// A forward all-to-all within independent locality groups. Balanced plans use
// equal source and destination group sizes; both are retained so the movement
// is explicit in the returned plan.
struct FftRedistributionStage {
    size_t sourceGroupSize;
    size_t destinationGroupSize;
};

// Counts cover the chosen workload: either one forward real FFT with a
// prepositioned multiplier, or two forward FFTs, followed by one inverse.
// A message is one directed, nonempty interlocality send.
// The two redistribution lists describe the forward path; the inverse follows
// them in reverse order. bytesSent and messagesSent are global totals. A batch
// packs all its values into each interlocality message, so batching scales
// bytesSent while leaving messagesSent and communicationPhases unchanged.
// A phase counts one peer exchange in the serialized cost model. Bounded
// asynchronous transports may overlap these exchanges without changing the
// payload/message counts. A g-way redistribution
// contributes g-1 phases per transform.
struct FftCommunicationPlan {
    FftCommunicationStrategy strategy;
    FftCommunicationWorkload workload;
    std::vector<FftRedistributionStage> blockToPencil;
    std::vector<FftRedistributionStage> pencilToSlab;
    size_t spectralLocalityCount;
    size_t vectorRadixLevels;
    size_t communicationPhases;
    size_t messagesSent;
    double bytesSent;
    double weightedCost;
};

namespace detail {

    struct FftRoute {
        std::vector<FftRedistributionStage> stages;
        size_t messages = 0;
        size_t rounds = 0;
        double bytes = 0.0;
        double comparisonCost = 0.0;
        bool valid = false;
    };

    inline bool betterFftMetrics(double leftCost, double leftBytes, size_t leftMessages, size_t leftPhases, double rightCost, double rightBytes,
        size_t rightMessages, size_t rightPhases) noexcept {
        PROFILE();
        if (leftCost != rightCost) {
            return leftCost < rightCost;
        }
        if (leftBytes != rightBytes) {
            return leftBytes < rightBytes;
        }
        if (leftMessages != rightMessages) {
            return leftMessages < rightMessages;
        }
        return leftPhases < rightPhases;
    }

    inline bool betterFftRoute(FftRoute const& left, FftRoute const& right) noexcept {
        PROFILE();
        return !right.valid ||
            betterFftMetrics(left.comparisonCost, left.bytes, left.messages, left.rounds, right.comparisonCost, right.bytes, right.messages, right.rounds);
    }

    inline FftRoute findFftRoute(size_t bitCount, size_t activeLocalities, double totalBytes, double byteWeight, double messageWeight) {
        PROFILE();
        // A 2^a-way ownership exchange sends B(1-2^-a) bytes in
        // P(2^a-1) directed messages. Dynamic programming considers every
        // ordered factorization whose exponents sum to bitCount.
        std::vector<FftRoute> best(bitCount + 1);
        best[0].valid = true;
        for (size_t bits = 1; bits <= bitCount; bits++) {
            for (size_t stageBits = 1; stageBits <= bits; stageBits++) {
                size_t const groupSize = size_t(1) << stageBits;
                double const stageBytes = totalBytes * (1.0 - 1.0 / static_cast<double>(groupSize));
                size_t const stageMessages = activeLocalities * (groupSize - 1);
                FftRoute candidate = best[bits - stageBits];
                candidate.stages.push_back({groupSize, groupSize});
                candidate.bytes += stageBytes;
                candidate.messages += stageMessages;
                candidate.rounds += groupSize - 1;
                candidate.comparisonCost += byteWeight * stageBytes + messageWeight * static_cast<double>(stageMessages);
                candidate.valid = true;
                if (betterFftRoute(candidate, best[bits])) {
                    best[bits] = std::move(candidate);
                }
            }
        }
        hpx::sort(hpx::execution::seq, best[bitCount].stages.begin(), best[bitCount].stages.end(),
            hpx::annotated_function(
                [](auto const& left, auto const& right) {
                    PROFILE();
                    return left.sourceGroupSize > right.sourceGroupSize;
                },
                "FFT order redistribution stages"));
        return best[bitCount];
    }

    inline bool isPositivePowerOfEight(size_t value) noexcept {
        PROFILE();
        return value >= 8 && std::has_single_bit(value) && std::countr_zero(value) % NDIM == 0;
    }

    inline bool betterFftPlan(
        FftCommunicationPlan const& left, double leftComparisonCost, FftCommunicationPlan const& right, double rightComparisonCost) noexcept {
        PROFILE();
        return betterFftMetrics(leftComparisonCost, left.bytesSent, left.messagesSent, left.communicationPhases, rightComparisonCost, right.bytesSent,
            right.messagesSent, right.communicationPhases);
    }

}    // namespace detail

// Find the least weighted communication among the nonreplicating algorithms
// developed for CosmicTiger's blocked cubic layout:
//
//  * compact real block-to-pencil-to-Fourier-slab transforms, including every
//    power-of-two factorization of both ownership redistributions;
//  * an adjacent-real packed vector-radix transform, with one conjugate-partner
//    exchange around the pointwise multiplication; and
//  * a full-Hermitian-spectrum vector-radix transform.
//
// Real defaults to double and batchSize to one, so the four-argument call counts
// eight bytes per value. totalValues is the value count for one batch member;
// packed batches share message counts. The default workload assumes that the
// multiplier is prepositioned in the chosen spectral layout; select
// twoForwardInputs when both real operands need forward transforms. Forward
// DIF and inverse DIT
// retain the same digit-reversed frequency order around the multiplication,
// so no bit-reversal pass is included. Deliberate replication and redundant FFT
// work are excluded because their optimum also requires a workspace or
// computation-cost limit.
template <std::floating_point Real = double, size_t batchSize = 1>
FftCommunicationPlan findOptimalFftCommunicationPlan(size_t totalValues, size_t localityCount, double weightPerByte, double weightPerMessage,
    FftCommunicationWorkload workload = FftCommunicationWorkload::prepositionedMultiplier) {
    constexpr size_t kSelfConjugateLocalities = size_t(1) << NDIM;
    static_assert(batchSize > 0, "FFT batch size must be positive");
    PROFILE();
    expect(detail::isPositivePowerOfEight(totalValues));
    expect(detail::isPositivePowerOfEight(localityCount));
    expectNonNegative(weightPerByte);
    expectNonNegative(weightPerMessage);
    FpeGuard fpeGuard;

    size_t const valueExponent = std::countr_zero(totalValues);
    size_t const sideExponent = valueExponent / NDIM;
    size_t const localityExponent = std::countr_zero(localityCount);
    size_t const localitySideExponent = localityExponent / NDIM;
    size_t const sideLength = size_t(1) << sideExponent;
    expectRange(size_t(1), localityCount, sideLength);

    double const weightScale = std::max(weightPerByte, weightPerMessage);
    double const comparisonByteWeight = weightScale == 0.0 ? 0.0 : weightPerByte / weightScale;
    double const comparisonMessageWeight = weightScale == 0.0 ? 0.0 : weightPerMessage / weightScale;
    double const totalBytes = static_cast<double>(totalValues) * static_cast<double>(sizeof(Real)) * static_cast<double>(batchSize);
    size_t const transformCount = workload == FftCommunicationWorkload::twoForwardInputs ? 3 : 2;

    // At the limiting P=M storage case, M/2 compact complex planes cannot be
    // assigned whole to all P localities. Split each plane across two
    // localities and fuse that extra ownership bit into the first route. The
    // two halves then gather to one locality for the local x/y FFT and scatter
    // again afterward, adding B bytes, P messages, and two phases overall.
    size_t const blockToPencilBits = localitySideExponent + (localityCount == sideLength ? 1 : 0);
    detail::FftRoute const blockToPencil = detail::findFftRoute(blockToPencilBits, localityCount, totalBytes, comparisonByteWeight, comparisonMessageWeight);
    detail::FftRoute const pencilToSlab = detail::findFftRoute(localityExponent, localityCount, totalBytes, comparisonByteWeight, comparisonMessageWeight);

    FftCommunicationPlan best{
        FftCommunicationStrategy::pencilSlab,
        workload,
        blockToPencil.stages,
        pencilToSlab.stages,
        localityCount,
        0,
        transformCount * (blockToPencil.rounds + pencilToSlab.rounds),
        transformCount * (blockToPencil.messages + pencilToSlab.messages),
        static_cast<double>(transformCount) * (blockToPencil.bytes + pencilToSlab.bytes),
        0.0,
    };
    if (localityCount == sideLength) {
        best.communicationPhases += transformCount;
        best.messagesSent += transformCount * localityCount / 2;
        best.bytesSent += static_cast<double>(transformCount) * totalBytes / 2.0;
    }
    double bestComparisonCost = comparisonByteWeight * best.bytesSent + comparisonMessageWeight * static_cast<double>(best.messagesSent);

    size_t const vectorRadixLevels = localitySideExponent;
    FftCommunicationPlan packedVectorRadix{
        FftCommunicationStrategy::packedVectorRadix,
        workload,
        {},
        {},
        localityCount,
        vectorRadixLevels,
        NDIM * transformCount * vectorRadixLevels + (localityCount == kSelfConjugateLocalities ? 0 : 1),
        NDIM * transformCount * vectorRadixLevels * localityCount + localityCount - kSelfConjugateLocalities,
        totalBytes *
            (static_cast<double>(NDIM) * static_cast<double>(transformCount) * static_cast<double>(vectorRadixLevels) +
                static_cast<double>(transformCount - 1) * (1.0 - static_cast<double>(kSelfConjugateLocalities) / static_cast<double>(localityCount))),
        0.0,
    };
    double const packedComparisonCost =
        comparisonByteWeight * packedVectorRadix.bytesSent + comparisonMessageWeight * static_cast<double>(packedVectorRadix.messagesSent);
    if (detail::betterFftPlan(packedVectorRadix, packedComparisonCost, best, bestComparisonCost)) {
        best = std::move(packedVectorRadix);
        bestComparisonCost = packedComparisonCost;
    }

    FftCommunicationPlan fullSpectrumVectorRadix{
        FftCommunicationStrategy::fullSpectrumVectorRadix,
        workload,
        {},
        {},
        localityCount,
        vectorRadixLevels,
        NDIM * transformCount * vectorRadixLevels,
        NDIM * transformCount * vectorRadixLevels * localityCount,
        totalBytes * static_cast<double>(transformCount) *
            (2.0 * static_cast<double>(NDIM) * static_cast<double>(vectorRadixLevels) - static_cast<double>(NDIM)),
        0.0,
    };
    double const fullSpectrumComparisonCost =
        comparisonByteWeight * fullSpectrumVectorRadix.bytesSent + comparisonMessageWeight * static_cast<double>(fullSpectrumVectorRadix.messagesSent);
    if (detail::betterFftPlan(fullSpectrumVectorRadix, fullSpectrumComparisonCost, best, bestComparisonCost)) {
        best = std::move(fullSpectrumVectorRadix);
    }

    best.weightedCost = weightPerByte * best.bytesSent + weightPerMessage * static_cast<double>(best.messagesSent);
    return best;
}

}    // namespace cosmictiger
