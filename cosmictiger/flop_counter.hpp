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

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define COSMICTIGER_FLOP_COUNTER_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_FLOP_COUNTER_HOST_DEVICE
#endif

namespace cosmictiger {

namespace detail {

    struct DisabledFlopCount {};

}    // namespace detail

// A local operation count for one CPU worker or CUDA thread. This class does
// not infer arithmetic costs, synchronize threads, or accumulate global totals.
// Values must be nonnegative and representable in Count; additions must fit and
// subtractions must not exceed the current count. Validation builds check these
// contracts before arithmetic or narrowing. Release retains caller preconditions.
//
// FlopCounter<Count, false> discards updates and always reads zero. It contains
// no Count storage and needs no build-wide macro, so enabled and disabled
// counters can coexist without conflicting definitions across translation units.
template <typename Count = std::uint64_t, bool enabled = true>
    requires(std::integral<Count> && !std::same_as<Count, bool> && !std::is_const_v<Count> && !std::is_volatile_v<Count>)
class FlopCounter {
public:
    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter() noexcept {
        PROFILE();
    }
    // NVCC infers host/device execution for explicitly defaulted copy operations.
    constexpr FlopCounter(FlopCounter const&) noexcept = default;
    constexpr FlopCounter& operator=(FlopCounter const&) noexcept = default;

    template <std::integral Value>
    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE explicit constexpr FlopCounter(Value value) {
        PROFILE();
        *this = value;
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE static constexpr bool countingEnabled() noexcept {
        PROFILE();
        return enabled;
    }

    template <std::integral Value>
    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter& operator=(Value value) {
        PROFILE();
        if constexpr (enabled) {
            expectRange(Count{0}, value, kMaximumCount_);
            count_ = static_cast<Count>(value);
        }
        return *this;
    }

    template <std::integral Value>
    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter& operator+=(Value value) {
        PROFILE();
        if constexpr (enabled) {
            expectRange(Count{0}, value, kMaximumCount_ - count_);
            count_ += static_cast<Count>(value);
        }
        return *this;
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter& operator+=(FlopCounter const& other) {
        PROFILE();
        return *this += other.read();
    }

    template <std::integral Value>
    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter& operator-=(Value value) {
        PROFILE();
        if constexpr (enabled) {
            expectRange(Count{0}, value, count_);
            count_ -= static_cast<Count>(value);
        }
        return *this;
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr FlopCounter& operator-=(FlopCounter const& other) {
        PROFILE();
        return *this -= other.read();
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr void reset() noexcept {
        PROFILE();
        if constexpr (enabled) {
            count_ = Count{0};
        }
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr Count read() const noexcept {
        PROFILE();
        if constexpr (enabled) {
            return count_;
        } else {
            return Count{0};
        }
    }

    COSMICTIGER_FLOP_COUNTER_HOST_DEVICE constexpr operator Count() const noexcept {
        PROFILE();
        return read();
    }

private:
    static constexpr Count kMaximumCount_ = std::numeric_limits<Count>::max();
    [[no_unique_address]] std::conditional_t<enabled, Count, detail::DisabledFlopCount> count_{};
};

}    // namespace cosmictiger

#undef COSMICTIGER_FLOP_COUNTER_HOST_DEVICE
