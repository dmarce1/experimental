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

#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/thread.hpp>

#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

TEST(ParallelChunks, BoundariesCoverLargestAddressableRangeWithoutOverflow) {
    constexpr size_t kCount = std::numeric_limits<size_t>::max();
    PROFILE();

    cosmictiger::detail::ParallelChunkPlan const plan(kCount, 1);
    size_t previous = 0;
    for (size_t chunk = 0; chunk < plan.size(); chunk++) {
        EXPECT_EQ(plan.begin(chunk), previous);
        EXPECT_GT(plan.end(chunk), previous);
        previous = plan.end(chunk);
    }
    EXPECT_EQ(previous, kCount);
}

TEST(ParallelChunks, CallbackFailureDrainsEveryLaunchedTask) {
    PROFILE();
    cosmictiger::detail::ParallelChunkPlan const plan(8192, 1024);
    std::vector<unsigned char> completed(plan.size(), 0);
    EXPECT_THROW(cosmictiger::detail::parallelForChunks(plan,
                     hpx::annotated_function(
                         [&](size_t chunk, size_t, size_t) {
                             PROFILE();
                             if (chunk == 0) {
                                 throw std::runtime_error("chunk failure");
                             }
                             hpx::this_thread::yield();
                             completed[chunk] = 1;
                         },
                         "test.parallel.failureChunk")),
        std::runtime_error);
    for (size_t chunk = 1; chunk < completed.size(); chunk++) {
        EXPECT_EQ(completed[chunk], 1);
    }
}
