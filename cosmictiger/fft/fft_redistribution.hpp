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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/fft/fft_parallel.hpp>

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/futures/future.hpp>

#include <algorithm>
#include <bit>
#include <exception>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger::detail {

// A layout is a bit permutation from a global array index to a locality
// address and a dense local index. Both bit lists are ordered least first.
class FftBitLayout {
public:
    FftBitLayout(size_t globalBitCount, std::vector<size_t> localityBits)
      : globalBitCount_(globalBitCount)
      , localityBits_(std::move(localityBits)) {
        PROFILE();
        std::vector<bool> owned(globalBitCount_, false);
        for (size_t const bit : localityBits_) {
            expectIndex(bit, globalBitCount_);
            expect(!owned[bit]);
            owned[bit] = true;
        }
        for (size_t bit = 0; bit != globalBitCount_; bit++) {
            if (!owned[bit]) {
                localBits_.push_back(bit);
            }
        }
    }

    size_t globalBitCount() const noexcept {
        PROFILE();
        return globalBitCount_;
    }

    size_t localityBitCount() const noexcept {
        PROFILE();
        return localityBits_.size();
    }

    std::vector<size_t> const& localityBits() const noexcept {
        PROFILE();
        return localityBits_;
    }

    size_t localityOf(size_t globalIndex) const noexcept {
        PROFILE();
        size_t locality = 0;
        for (size_t bit = 0; bit != localityBits_.size(); bit++) {
            locality |= ((globalIndex >> localityBits_[bit]) & size_t(1)) << bit;
        }
        return locality;
    }

    size_t globalIndex(size_t locality, size_t localIndex) const noexcept {
        PROFILE();
        size_t result = 0;
        for (size_t bit = 0; bit != localityBits_.size(); bit++) {
            result |= ((locality >> bit) & size_t(1)) << localityBits_[bit];
        }
        for (size_t bit = 0; bit != localBits_.size(); bit++) {
            result |= ((localIndex >> bit) & size_t(1)) << localBits_[bit];
        }
        return result;
    }

private:
    size_t globalBitCount_;
    std::vector<size_t> localityBits_;
    std::vector<size_t> localBits_;
};

template <typename Value, typename Exchange>
std::vector<Value> redistributeFftBits(std::span<Value const> input, size_t locality, FftBitLayout const& from, FftBitLayout const& to, size_t firstAddressBit,
    size_t exchangedBits, Exchange&& exchange) {
    [[maybe_unused]] constexpr size_t kMaximumInFlight = 4;
    PROFILE();
    expect(!(from.globalBitCount() != to.globalBitCount() || from.localityBitCount() != to.localityBitCount() ||
        firstAddressBit + exchangedBits > from.localityBitCount() || input.size() != (size_t(1) << (from.globalBitCount() - from.localityBitCount()))));

    size_t const groupSize = size_t(1) << exchangedBits;
    ParallelChunkPlan const chunks(input.size(), 4096);
    size_t const chunkCount = chunks.size();
    std::vector<std::vector<size_t>> sendCounts(chunkCount, std::vector<size_t>(groupSize, 0));
    std::vector<std::vector<size_t>> receiveCounts(chunkCount, std::vector<size_t>(groupSize, 0));
    parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t chunk, size_t begin, size_t end) {
                PROFILE();
                for (size_t localIndex = begin; localIndex != end; localIndex++) {
                    size_t const outgoingGlobalIndex = from.globalIndex(locality, localIndex);
                    size_t const destination = to.localityOf(outgoingGlobalIndex);
                    expect(!((destination ^ locality) & ~((groupSize - 1) << firstAddressBit)));
                    sendCounts[chunk][(destination ^ locality) >> firstAddressBit]++;

                    size_t const incomingGlobalIndex = to.globalIndex(locality, localIndex);
                    size_t const source = from.localityOf(incomingGlobalIndex);
                    expect(!((source ^ locality) & ~((groupSize - 1) << firstAddressBit)));
                    receiveCounts[chunk][(source ^ locality) >> firstAddressBit]++;
                }
            },
            "FFT redistribution count peer payloads"));

    std::vector<std::vector<Value>> outgoing(groupSize);
    std::vector<std::vector<size_t>> incomingIndices(groupSize);
    std::vector<std::vector<size_t>> sendOffsets(chunkCount, std::vector<size_t>(groupSize, 0));
    std::vector<std::vector<size_t>> receiveOffsets(chunkCount, std::vector<size_t>(groupSize, 0));
    for (size_t mask = 0; mask != groupSize; mask++) {
        size_t sendTotal = 0;
        size_t receiveTotal = 0;
        for (size_t chunk = 0; chunk != chunkCount; chunk++) {
            sendOffsets[chunk][mask] = sendTotal;
            receiveOffsets[chunk][mask] = receiveTotal;
            sendTotal += sendCounts[chunk][mask];
            receiveTotal += receiveCounts[chunk][mask];
        }
        outgoing[mask].resize(sendTotal);
        incomingIndices[mask].resize(receiveTotal);
    }
    parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t chunk, size_t begin, size_t end) {
                PROFILE();
                std::vector<size_t> sendPosition = sendOffsets[chunk];
                std::vector<size_t> receivePosition = receiveOffsets[chunk];
                for (size_t localIndex = begin; localIndex != end; localIndex++) {
                    size_t const outgoingGlobalIndex = from.globalIndex(locality, localIndex);
                    size_t const destination = to.localityOf(outgoingGlobalIndex);
                    size_t const sendMask = (destination ^ locality) >> firstAddressBit;
                    outgoing[sendMask][sendPosition[sendMask]++] = input[localIndex];

                    size_t const incomingGlobalIndex = to.globalIndex(locality, localIndex);
                    size_t const source = from.localityOf(incomingGlobalIndex);
                    size_t const receiveMask = (source ^ locality) >> firstAddressBit;
                    incomingIndices[receiveMask][receivePosition[receiveMask]++] = localIndex;
                }
            },
            "FFT redistribution pack peer payloads"));

    std::vector<Value> output(input.size());
    auto scatter = [&](size_t mask, std::span<Value const> received) {
        PROFILE();
        expectEqual(received.size(), incomingIndices[mask].size());
        for (size_t index = 0; index != received.size(); index++) {
            output[incomingIndices[mask][index]] = received[index];
        }
    };
    if constexpr (requires { exchange.template exchangeAsync<Value>(size_t(0), std::vector<Value>(), size_t(0)); }) {
        // All endpoints submit masks in the same order, preserving matching
        // tags. Ready transfers scatter into disjoint destinations while the
        // other transfers in this bounded window remain in flight. The send
        // vectors belong to the transport until its completion future is ready.
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(kMaximumInFlight);
        bool localScattered = false;
        for (size_t firstMask = 1; firstMask < groupSize; firstMask += kMaximumInFlight) {
            tasks.clear();
            std::exception_ptr failure;
            size_t const endMask = std::min(groupSize, firstMask + kMaximumInFlight);
            for (size_t mask = firstMask; mask < endMask; mask++) {
                try {
                    size_t const peer = locality ^ (mask << firstAddressBit);
                    auto received = exchange.template exchangeAsync<Value>(peer, std::move(outgoing[mask]), incomingIndices[mask].size()).share();
                    try {
                        tasks.push_back(received.then(hpx::launch::async,
                            hpx::annotated_function(
                                [&, mask](hpx::shared_future<std::vector<Value>> ready) {
                                    PROFILE();
                                    scatter(mask, ready.get());
                                },
                                "FFT redistribution scatter received peer payload")));
                    } catch (...) {
                        failure = std::current_exception();
                        if (received.valid()) {
                            try {
                                received.get();
                            } catch (...) {
                            }
                        }
                        break;
                    }
                } catch (...) {
                    failure = std::current_exception();
                    break;
                }
            }
            if (!localScattered && !failure) {
                try {
                    scatter(0, outgoing[0]);
                    localScattered = true;
                } catch (...) {
                    failure = std::current_exception();
                }
            }
            // Output and index arrays stay live until every launched scatter
            // completes, including a partial launch or a failed transfer.
            for (auto& task : tasks) {
                try {
                    task.get();
                } catch (...) {
                    if (!failure) {
                        failure = std::current_exception();
                    }
                }
            }
            if (failure) {
                std::rethrow_exception(failure);
            }
        }
        if (!localScattered) {
            scatter(0, outgoing[0]);
        }
    } else {
        // User-supplied synchronous callbacks retain their existing ordering.
        for (size_t mask = 0; mask != groupSize; mask++) {
            std::vector<Value> received;
            if (mask == 0) {
                received = std::move(outgoing[0]);
            } else {
                size_t const peer = locality ^ (mask << firstAddressBit);
                received.resize(incomingIndices[mask].size());
                exchange(peer, std::span<Value const>(outgoing[mask]), std::span<Value>(received));
            }
            scatter(mask, received);
        }
    }
    return output;
}

inline FftBitLayout intermediateFftLayout(FftBitLayout const& first, FftBitLayout const& last, size_t exchangedAddressBits) {
    PROFILE();
    expect(!(first.globalBitCount() != last.globalBitCount() || first.localityBitCount() != last.localityBitCount() ||
        exchangedAddressBits > first.localityBitCount()));
    std::vector<size_t> bits = first.localityBits();
    for (size_t index = 0; index != exchangedAddressBits; index++) {
        bits[index] = last.localityBits()[index];
    }
    return FftBitLayout(first.globalBitCount(), std::move(bits));
}

template <typename Value, typename Stages, typename Exchange>
std::vector<Value> routeFftBits(
    std::vector<Value> input, size_t locality, FftBitLayout const& first, FftBitLayout const& last, Stages const& stages, Exchange&& exchange) {
    PROFILE();
    size_t exchangedBits = 0;
    FftBitLayout current = first;
    for (auto const& stage : stages) {
        expect(std::has_single_bit(stage.sourceGroupSize));
        expectEqual(stage.sourceGroupSize, stage.destinationGroupSize);
        size_t const stageBits = std::countr_zero(stage.sourceGroupSize);
        FftBitLayout const next = intermediateFftLayout(first, last, exchangedBits + stageBits);
        input = redistributeFftBits(std::span<Value const>(input), locality, current, next, exchangedBits, stageBits, exchange);
        current = next;
        exchangedBits += stageBits;
    }
    expect(current.localityBits() == last.localityBits());
    return input;
}

template <typename Value, typename Stages, typename Exchange>
std::vector<Value> routeFftBitsBackward(
    std::vector<Value> input, size_t locality, FftBitLayout const& first, FftBitLayout const& last, Stages const& stages, Exchange&& exchange) {
    PROFILE();
    std::vector<FftBitLayout> layouts;
    std::vector<size_t> stageBits;
    layouts.push_back(first);
    size_t exchangedBits = 0;
    for (auto const& stage : stages) {
        expect(std::has_single_bit(stage.sourceGroupSize));
        expectEqual(stage.sourceGroupSize, stage.destinationGroupSize);
        size_t const bits = std::countr_zero(stage.sourceGroupSize);
        exchangedBits += bits;
        layouts.push_back(intermediateFftLayout(first, last, exchangedBits));
        stageBits.push_back(bits);
    }
    expect(layouts.back().localityBits() == last.localityBits());
    for (size_t stage = stages.size(); stage != 0; stage--) {
        size_t const firstAddressBit = exchangedBits - stageBits[stage - 1];
        input =
            redistributeFftBits(std::span<Value const>(input), locality, layouts[stage], layouts[stage - 1], firstAddressBit, stageBits[stage - 1], exchange);
        exchangedBits = firstAddressBit;
    }
    return input;
}

}    // namespace cosmictiger::detail
