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

#include <cosmictiger/lock_guards.hpp>

#include <fftw3.h>
#include <hpx/mutex.hpp>

#include <climits>
#include <complex>
#include <map>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger::detail {

inline hpx::mutex& fftwPlannerMutex() {
    PROFILE();
    static hpx::mutex mutex;
    return mutex;
}

enum class FftwTransformKind { complexForward, complexBackward, realToComplex, complexToReal };

struct FftRealScratch {
    std::vector<double> real;
    std::vector<std::complex<double>> frequency;
};

// All cached transforms have unit stride and FFTW_ESTIMATE | FFTW_UNALIGNED.
// Complex transforms are in-place; real transforms are out-of-place. Together
// with the length, the kind therefore identifies every plan compatibility rule.
class FftwPlanCache {
public:
    FftwPlanCache() = default;
    FftwPlanCache(FftwPlanCache const&) = delete;
    FftwPlanCache& operator=(FftwPlanCache const&) = delete;

    ~FftwPlanCache() {
        PROFILE();
        // The process-wide cache outlives runtime work. No HPX operations are
        // needed at static teardown, when no executions or planner calls remain.
        for (auto const& entry : plans_) {
            fftw_destroy_plan(entry.second);
        }
    }

    fftw_plan get(size_t length, FftwTransformKind kind) {
        constexpr unsigned kFlags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        PROFILE();
        expectRange(size_t(1), length, size_t(INT_MAX));
        bool realTransform = false;
        switch (kind) {
        case FftwTransformKind::complexForward:
        case FftwTransformKind::complexBackward:
            break;
        case FftwTransformKind::realToComplex:
        case FftwTransformKind::complexToReal:
            realTransform = true;
            break;
        default:
            expect(false);
            break;
        }

        auto const key = std::make_pair(length, kind);
        {
            LockGuard lock(fftwPlannerMutex());
            auto const found = plans_.find(key);
            if (found != plans_.end()) {
                return found->second;
            }
        }

        // Allocate only on a cache miss, outside the planner lock. New-array
        // execution never uses these temporary planning buffers afterward.
        std::vector<double> realScratch(realTransform ? length : 0);
        std::vector<std::complex<double>> complexScratch(realTransform ? length / 2 + 1 : length);
        auto* complexArray = reinterpret_cast<fftw_complex*>(complexScratch.data());

        LockGuard lock(fftwPlannerMutex());
        auto const found = plans_.find(key);
        if (found != plans_.end()) {
            return found->second;
        }
        // Allocate the map node before creating a raw plan, so a failed map
        // allocation cannot leak it. Failed planning removes the placeholder.
        auto const entry = plans_.emplace(key, nullptr).first;
        int const transformLength = static_cast<int>(length);
        FpeGuard fpeGuard;
        switch (kind) {
        case FftwTransformKind::complexForward:
            entry->second = fftw_plan_dft_1d(transformLength, complexArray, complexArray, FFTW_FORWARD, kFlags);
            break;
        case FftwTransformKind::complexBackward:
            entry->second = fftw_plan_dft_1d(transformLength, complexArray, complexArray, FFTW_BACKWARD, kFlags);
            break;
        case FftwTransformKind::realToComplex:
            entry->second = fftw_plan_dft_r2c_1d(transformLength, realScratch.data(), complexArray, kFlags);
            break;
        case FftwTransformKind::complexToReal:
            entry->second = fftw_plan_dft_c2r_1d(transformLength, complexArray, realScratch.data(), kFlags);
            break;
        default:
            break;
        }
        if (entry->second == nullptr) {
            plans_.erase(entry);
            throw std::runtime_error("FFTW could not create a memoized transform plan");
        }
        return entry->second;
    }

private:
    std::map<std::pair<size_t, FftwTransformKind>, fftw_plan> plans_;
};

// Borrowed handles remain valid across FFT object lifetimes until process exit.
// Do not destroy them or call fftw_cleanup. Concurrent execution is supported
// through the matching new-array execute function with independent arrays.
inline fftw_plan memoizedFftwPlan(size_t length, FftwTransformKind kind) {
    PROFILE();
    static FftwPlanCache cache;
    return cache.get(length, kind);
}

}    // namespace cosmictiger::detail
