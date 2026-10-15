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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/gravity/direct/direct_solve_kernel.hpp>
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cosmictiger::gravity {

namespace detail {
    void validateDirectSolveRequest(DirectSolveRequest const& request, DirectEwaldParameters const& parameters) {
        PROFILE();
        if (request.softeningModel != SofteningModel::ferrers && request.softeningModel != SofteningModel::plummer) {
            throw std::invalid_argument("unknown direct-solve softening model");
        }
        expectPositive(request.particleMass);
        expectPositive(request.softeningLength);
        expectRange(0, request.ferrersOrder, 3);
        expectPositive(parameters.alpha);
        expectRange(0, parameters.imageCutoff, std::numeric_limits<int>::max() - 1);
        expectRange(0, parameters.waveCutoff, std::numeric_limits<int>::max() - 1);
        if (request.sinkCount != 0 && (request.sourceCount == 0 || request.sources == nullptr || request.sinks == nullptr || request.fields == nullptr)) {
            throw std::invalid_argument("a nonempty direct solve requires sources, sink indices, and output fields");
        }
    }

    std::uint64_t directSamplingHash(std::uint64_t counter) noexcept {
        constexpr std::uint64_t kIncrement = 0x9e3779b97f4a7c15ULL;
        constexpr std::uint64_t kFirstMultiplier = 0xbf58476d1ce4e5b9ULL;
        constexpr std::uint64_t kSecondMultiplier = 0x94d049bb133111ebULL;
        constexpr unsigned kFirstShift = 30;
        constexpr unsigned kSecondShift = 27;
        constexpr unsigned kFinalShift = 31;
        PROFILE();
        counter += kIncrement;
        counter = (counter ^ (counter >> kFirstShift)) * kFirstMultiplier;
        counter = (counter ^ (counter >> kSecondShift)) * kSecondMultiplier;
        return counter ^ (counter >> kFinalShift);
    }

    size_t directSamplingIndex(size_t maximum, std::uint64_t& counter) noexcept {
        static_assert(sizeof(size_t) <= sizeof(std::uint64_t));
        PROFILE();
        std::uint64_t const bound = static_cast<std::uint64_t>(maximum) + 1;
        if (bound == 0) {
            return static_cast<size_t>(directSamplingHash(counter++));
        }
        // Reject the short interval so modulo reduction is exactly uniform.
        std::uint64_t const threshold = (std::uint64_t(0) - bound) % bound;
        std::uint64_t value;
        do {
            value = directSamplingHash(counter++);
        } while (value < threshold);
        return static_cast<size_t>(value % bound);
    }
}    // namespace detail

void directSolveCpu(DirectSolveRequest const& request, DirectEwaldParameters const& parameters) {
    constexpr size_t kMinimumSinksPerChunk = 1;
    PROFILE();
    detail::validateDirectSolveRequest(request, parameters);
    auto const weights = detail::prepareDirectEwaldWeights(request, parameters);
    ::cosmictiger::detail::ParallelChunkPlan const plan(request.sinkCount, kMinimumSinksPerChunk);
    ::cosmictiger::detail::parallelForChunks(plan,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t sinkOrdinal = begin; sinkOrdinal < end; sinkOrdinal++) {
                    request.fields[sinkOrdinal] = detail::directSolveSink(request, parameters, sinkOrdinal, &weights);
                }
            },
            "directSolveCpu.evaluateSinks"));
}

std::vector<size_t> sampleDirectSinks(size_t sourceCount, double fraction, std::uint64_t seed) {
    constexpr std::uint64_t kDirectSamplingDomain = 0x6469726563747369ULL;
    PROFILE();
    if (!(fraction >= 0.0 && fraction <= 1.0)) {
        throw std::invalid_argument("direct-solve sink fraction must be between zero and one");
    }
    if (sourceCount == 0 || fraction == 0.0) {
        return {};
    }
    size_t sinkCount;
    {
        FpeGuard fpeGuard;
        double const count = ::ceil(fraction * static_cast<double>(sourceCount));
        sinkCount = count >= static_cast<double>(sourceCount) ? sourceCount : static_cast<size_t>(count);
    }
    std::vector<size_t> sinks;
    sinks.reserve(sinkCount);
    if (sinkCount == sourceCount) {
        for (size_t index = 0; index < sourceCount; index++) {
            sinks.push_back(index);
        }
        return sinks;
    }
    std::unordered_set<size_t> selected;
    selected.reserve(sinkCount);
    std::uint64_t counter = detail::directSamplingHash(seed ^ kDirectSamplingDomain);
    // Floyd's algorithm uses O(sinkCount) work and storage, independent of how
    // many unselected source particles there are.
    for (size_t maximum = sourceCount - sinkCount; maximum < sourceCount; maximum++) {
        size_t candidate = detail::directSamplingIndex(maximum, counter);
        if (!selected.insert(candidate).second) {
            candidate = maximum;
            selected.insert(candidate);
        }
        sinks.push_back(candidate);
    }
    hpx::sort(hpx::execution::seq, sinks.begin(), sinks.end(),
        hpx::annotated_function(
            [](size_t left, size_t right) {
                PROFILE();
                return left < right;
            },
            "sampleDirectSinks.sortIndices"));
    return sinks;
}

}    // namespace cosmictiger::gravity
