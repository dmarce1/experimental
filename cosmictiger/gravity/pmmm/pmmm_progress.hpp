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

#include <cosmictiger/lock_guards.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>
#include <stddef.h>

namespace cosmictiger::gravity {

enum class PmmmProgressStage {
    directGravity,
    cudaPreparation,
    ewaldFilters,
    filterFft,
    multipoleFft,
    contraction,
    expansionFft,
    periodicBackground,
    particleKick,
    cudaKick,
    particleSort,
    diagnosticReduction
};

struct PmmmProgress {
    PmmmProgressStage stage;
    size_t completed;
    size_t total;
};

// Local callbacks may run concurrently in the near and far branches. They must
// not enter collectives; callers retain the observer until all work has joined.
using PmmmProgressObserver = hpx::function<void(PmmmProgress const&)>;

namespace detail {

    inline void reportPmmmProgress(PmmmProgressObserver const& observer, PmmmProgressStage stage, size_t completed, size_t total) {
        PROFILE();
        if (observer) {
            observer({stage, completed, total});
        }
    }

    // Count completed local work, with updates at most once every two seconds.
    // Invoke outside arithmetic guards and locks. finish() follows joined work,
    // so completion is never reported while a worker can still emit an update.
    class PmmmProgressCounter {
    public:
        PmmmProgressCounter(PmmmProgressObserver const& observer, PmmmProgressStage stage, size_t total)
          : observer_(observer)
          , stage_(stage)
          , total_(total) {
            PROFILE();
            if (observer_) {
                lastReport_ = hpx::chrono::high_resolution_clock::now();
                reportPmmmProgress(observer_, stage_, 0, total_);
            }
        }

        void advance(size_t amount) {
            constexpr std::uint64_t kReportInterval = 2000000000ULL;
            PROFILE();

            if (!observer_) {
                return;
            }
            auto const now = hpx::chrono::high_resolution_clock::now();
            size_t completed = 0;
            bool report = false;
            {
                LockGuard lock(mutex_);
                completed_ += amount;
                completed = completed_;
                if (completed < total_ && now >= lastReport_ && now - lastReport_ >= kReportInterval) {
                    lastReport_ = now;
                    report = true;
                }
            }
            if (report) {
                reportPmmmProgress(observer_, stage_, completed, total_);
            }
        }

        void finish() const {
            PROFILE();
            reportPmmmProgress(observer_, stage_, total_, total_);
        }

    private:
        PmmmProgressObserver const& observer_;
        PmmmProgressStage stage_;
        size_t total_;
        size_t completed_ = 0;
        std::uint64_t lastReport_ = 0;
        hpx::spinlock mutex_;
    };

}    // namespace detail
}    // namespace cosmictiger::gravity
