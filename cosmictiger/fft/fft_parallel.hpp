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

#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <stddef.h>

namespace cosmictiger::detail {

// Approximate work units: scalar batch entries for element loops, or n*log2(n)
// for a single FFT line/lane. This is a scheduling heuristic, not a flop count.
inline constexpr size_t kFftChunkWork = 4096;

inline size_t fftMinimumChunkSize(size_t workPerItem) {
    PROFILE();
    expectPositive(workPerItem);
    return 1 + (kFftChunkWork - 1) / workPerItem;
}

inline size_t fftLineWork(size_t length) {
    PROFILE();
    expectPositive(length);
    size_t levels = 0;
    for (size_t remaining = length; remaining > 1; remaining /= 2) {
        levels++;
    }
    levels = levels == 0 ? 1 : levels;
    // Only the cutoff matters; saturation avoids overflow for large lengths.
    return length > kFftChunkWork / levels ? kFftChunkWork : length * levels;
}

// At most four contiguous chunks per HPX worker. Small ranges, single-worker
// runtimes and callers outside HPX execute directly. All tasks are drained
// before return/rethrow; callers must not hold locks or FpeGuard across this call.
template <typename Function>
void parallelForFft(size_t count, Function&& function, size_t workPerItem = 1) {
    PROFILE();
    ParallelChunkPlan const plan(count, fftMinimumChunkSize(workPerItem));
    parallelForChunks(plan,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    function(index);
                }
            },
            "FFT iterate element chunk"));
}

// One private scratch object per chunk, reused by every line/lane in that chunk.
// Scratch belongs to this invocation, so simultaneous transforms never share it.
template <typename MakeScratch, typename Function>
void parallelForFftWithScratch(size_t count, size_t workPerItem, MakeScratch&& makeScratch, Function&& function) {
    PROFILE();
    ParallelChunkPlan const plan(count, fftMinimumChunkSize(workPerItem));
    parallelForChunks(plan,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                auto scratch = makeScratch();
                for (size_t index = begin; index < end; index++) {
                    function(index, scratch);
                }
            },
            "FFT iterate line chunk with scratch"));
}

}    // namespace cosmictiger::detail
