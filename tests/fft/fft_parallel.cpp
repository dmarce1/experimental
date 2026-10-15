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

#include <cosmictiger/fft/fft_parallel.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/thread.hpp>

#include <array>
#include <atomic>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

TEST(FftParallel, SmallLoopsRunInlineAndEmptyLoopsDoNotAllocate) {
    PROFILE();
    auto const caller = hpx::this_thread::get_id();
    size_t visited = 0;
    cosmictiger::detail::parallelForFft(17,
        hpx::annotated_function(
            [&](size_t index) {
                PROFILE();
                EXPECT_EQ(hpx::this_thread::get_id(), caller);
                EXPECT_EQ(index, visited++);
            },
            "FFT parallel test inline element"));
    EXPECT_EQ(visited, 17U);
    size_t allocations = 0;
    cosmictiger::detail::parallelForFftWithScratch(0, 1,
        hpx::annotated_function(
            [&] {
                PROFILE();
                allocations++;
                return std::array<size_t, 1>{};
            },
            "FFT parallel test empty scratch factory"),
        hpx::annotated_function(
            [&](size_t, auto&) {
                PROFILE();
                ADD_FAILURE() << "empty range executed a line";
            },
            "FFT parallel test reject empty line execution"));
    EXPECT_EQ(allocations, 0U);
    EXPECT_EQ(cosmictiger::detail::fftLineWork(1), 1U);
    EXPECT_EQ(cosmictiger::detail::fftMinimumChunkSize(std::numeric_limits<size_t>::max()), 1U);
    EXPECT_EQ(cosmictiger::detail::fftLineWork(std::numeric_limits<size_t>::max()), cosmictiger::detail::kFftChunkWork);
}

TEST(FftParallel, ScratchAllocationsScaleWithChunksAndAreReusedAcrossTailLines) {
    constexpr size_t kCount = 4099;
    PROFILE();

    std::atomic<size_t> allocations{0};
    std::vector<size_t> owners(kCount, kCount);
    std::vector<size_t> visits(kCount, 0);
    cosmictiger::detail::parallelForFftWithScratch(kCount, cosmictiger::detail::fftLineWork(32),
        hpx::annotated_function(
            [&] {
                PROFILE();
                return std::array<size_t, 2>{allocations.fetch_add(1), 0};
            },
            "FFT parallel test reusable scratch factory"),
        hpx::annotated_function(
            [&](size_t index, auto& scratch) {
                PROFILE();
                owners[index] = scratch[0];
                visits[index] = ++scratch[1];
            },
            "FFT parallel test visit reusable scratch line"));
    size_t const allocated = allocations.load();
    EXPECT_LE(allocated, 4 * hpx::get_os_thread_count());
    EXPECT_LT(allocated, kCount);
    if (hpx::get_os_thread_count() > 1) {
        EXPECT_GT(allocated, 1U);
        if (hpx::get_os_thread_count() > 2) {
            EXPECT_GT(allocated, 8U);
        }
    } else {
        EXPECT_EQ(allocated, 1U);
    }
    std::vector<size_t> counts(allocated, 0);
    for (size_t index = 0; index < kCount; index++) {
        ASSERT_LT(owners[index], allocated);
        EXPECT_EQ(visits[index], counts[owners[index]]++ + 1);
    }
    for (size_t const used : counts) {
        EXPECT_GT(used, 1U);
    }
}

namespace {
class ScratchLifetime {
public:
    explicit ScratchLifetime(std::atomic<size_t>& live)
      : live_(&live) {
        PROFILE();
        live_->fetch_add(1);
    }
    ScratchLifetime(ScratchLifetime const&) = delete;
    ScratchLifetime& operator=(ScratchLifetime const&) = delete;
    ~ScratchLifetime() {
        PROFILE();
        live_->fetch_sub(1);
    }

private:
    std::atomic<size_t>* live_;
};
}    // namespace

TEST(FftParallel, FailureDrainsOtherChunksAndReleasesAllScratch) {
    constexpr size_t kCount = 113;
    PROFILE();

    std::atomic<size_t> live{0};
    std::vector<unsigned char> finished(kCount, 0);
    EXPECT_THROW(cosmictiger::detail::parallelForFftWithScratch(kCount, cosmictiger::detail::kFftChunkWork,
                     hpx::annotated_function(
                         [&] {
                             PROFILE();
                             return ScratchLifetime(live);
                         },
                         "FFT parallel test lifetime scratch factory"),
                     hpx::annotated_function(
                         [&](size_t index, auto&) {
                             PROFILE();
                             if (index == 0) {
                                 throw std::runtime_error("injected FFT line failure");
                             }
                             hpx::this_thread::yield();
                             finished[index] = 1;
                         },
                         "FFT parallel test failing line")),
        std::runtime_error);
    EXPECT_EQ(live.load(), 0U);
    if (hpx::get_os_thread_count() > 1) {
        // The tail belongs to another launched chunk, even when the first fails.
        EXPECT_EQ(finished.back(), 1);
    }
}
