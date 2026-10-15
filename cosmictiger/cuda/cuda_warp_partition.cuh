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

/** @file
 * @brief Warp-wide partition of one entry per lane among several output lists.
 *
 * This replaces the old tree walk's per-list shuffle prefix scan
 * (compute_indices_array) for splitting a warp's checklist entries among the
 * cc, cp, pc, leaf, and next lists. Each lane names at most one list, so one
 * __ballot_sync per list yields every count, and the population count of the
 * ballot below the lane yields the lane's exclusive offset in a stable order.
 * Five lists need five votes instead of about 35 dependent shuffles.
 *
 * Every function here must be called convergently by all 32 lanes of a warp.
 * Keep loops over lists fully unrolled: dynamically indexing the per-list
 * arrays places them in local memory, which costs more than the votes save.
 */
#pragma once

#include <stddef.h>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::cuda {

inline constexpr unsigned kWarpSize = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;
// A lane that appends to no list passes this list index.
inline constexpr int kNoWarpList = -1;

// The per-lane outcome of partitionWarp. offset is the number of lower lanes
// that chose the same list, or zero for kNoWarpList. counts are warp-uniform.
template <int listCount>
struct WarpPartition {
    unsigned offset;
    unsigned counts[listCount];
};

namespace detail {

    // Lanes below this one, independent of the block shape.
    __device__ inline unsigned lanesBelow() {
        PROFILE();
        unsigned mask;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
        return mask;
    }

}    // namespace detail

// Partition one entry per lane. list is in [0, listCount) or kNoWarpList.
template <int listCount>
__device__ inline WarpPartition<listCount> partitionWarp(int list) {
    static_assert(listCount > 0, "A warp partition needs at least one list");
    PROFILE();
    expect(list >= kNoWarpList && list < listCount);
    unsigned const below = detail::lanesBelow();
    WarpPartition<listCount> result{};
#pragma unroll
    for (int candidate = 0; candidate < listCount; candidate++) {
        unsigned const members = __ballot_sync(kFullWarpMask, list == candidate);
        result.counts[candidate] = __popc(members);
        if (list == candidate) {
            result.offset = __popc(members & below);
        }
    }
    return result;
}

// Append each lane's value to the end of its chosen list and advance the
// warp-uniform sizes, preserving lane order within each list. Every lane must
// pass identical lists, sizes, and capacities. If any list would exceed its
// capacity, nothing is written, sizes are unchanged, and every lane receives
// false so that the caller can grow the lists and retry the same entries.
template <typename Value, int listCount>
__device__ inline bool appendWarp(
    int list, Value const& value, Value* const (&lists)[listCount], size_t (&sizes)[listCount], size_t const (&capacities)[listCount]) {
    PROFILE();
    WarpPartition<listCount> const partition = partitionWarp<listCount>(list);
    bool fits = true;
#pragma unroll
    for (int candidate = 0; candidate < listCount; candidate++) {
        expect(sizes[candidate] <= capacities[candidate]);
        fits = fits && capacities[candidate] - sizes[candidate] >= partition.counts[candidate];
    }
    if (!fits) {
        return false;
    }
    Value* destination = nullptr;
#pragma unroll
    for (int candidate = 0; candidate < listCount; candidate++) {
        if (list == candidate) {
            destination = lists[candidate] + sizes[candidate] + partition.offset;
        }
        sizes[candidate] += partition.counts[candidate];
    }
    if (destination != nullptr) {
        *destination = value;
    }
    return true;
}

}    // namespace cosmictiger::cuda
