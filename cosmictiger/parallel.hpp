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

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <exception>
#include <stddef.h>
#include <vector>

namespace cosmictiger::detail {

// A bounded number of contiguous chunks, with no extra OS threads. Integer
// quotient/remainder boundaries avoid count*chunk overflow at large sizes.
class ParallelChunkPlan {
public:
    explicit ParallelChunkPlan(size_t count, size_t minimumChunkSize = 4096)
      : count_(count) {
        PROFILE();
        expectPositive(minimumChunkSize);
        if (count_ == 0) {
            return;
        }
        chunkCount_ = 1;
        if (hpx::threads::get_self_ptr() == nullptr) {
            return;
        }
        size_t const workers = hpx::get_os_thread_count();
        if (workers < 2) {
            return;
        }
        size_t const maximumChunks = workers > size_t(-1) / 4 ? size_t(-1) : workers * 4;
        size_t const usefulChunks = count_ / minimumChunkSize;
        if (usefulChunks > 1) {
            chunkCount_ = usefulChunks < maximumChunks ? usefulChunks : maximumChunks;
        }
    }

    size_t size() const noexcept {
        PROFILE();
        return chunkCount_;
    }

    size_t begin(size_t chunk) const {
        PROFILE();
        expectIndex(chunk, chunkCount_);
        size_t const quotient = count_ / chunkCount_;
        size_t const remainder = count_ % chunkCount_;
        return chunk * quotient + (chunk < remainder ? chunk : remainder);
    }

    size_t end(size_t chunk) const {
        PROFILE();
        expectIndex(chunk, chunkCount_);
        size_t const first = begin(chunk);
        return first + count_ / chunkCount_ + (chunk < count_ % chunkCount_ ? 1 : 0);
    }

private:
    size_t count_;
    size_t chunkCount_ = 0;
};

// References captured by callbacks remain alive through every launched task,
// even if a task or a later launch fails. Do not call while holding an
// application lock or carrying an FpeGuard across scheduling/suspension.
template <typename Function>
void parallelForChunks(ParallelChunkPlan const& plan, Function&& function) {
    PROFILE();
    if (plan.size() == 0) {
        return;
    }
    if (plan.size() == 1) {
        function(size_t(0), plan.begin(0), plan.end(0));
        return;
    }
    std::vector<hpx::future<void>> tasks;
    tasks.reserve(plan.size());
    std::exception_ptr failure;
    for (size_t chunk = 0; chunk < plan.size(); chunk++) {
        try {
            tasks.push_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&function, plan, chunk] {
                        PROFILE();
                        function(chunk, plan.begin(chunk), plan.end(chunk));
                    },
                    "parallel.async")));
        } catch (...) {
            failure = std::current_exception();
            break;
        }
    }
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

}    // namespace cosmictiger::detail
