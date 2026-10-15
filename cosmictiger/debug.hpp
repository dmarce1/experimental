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

#include <cfenv>
#include <cosmictiger/profiler.hpp>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__CUDACC__)
#define COSMICTIGER_DEBUG_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_DEBUG_HOST_DEVICE
#endif

namespace cosmictiger {

// Floating-point masks belong to the calling OS thread. Keep guards in local
// arithmetic scopes: never carry one across HPX scheduling or suspension.
// CUDA has no floating-point exception flags or trap masks; on the device the
// guard is inert, while the expectation primitives below still diagnose/trap.
enum class Fpe : int { invalid = FE_INVALID, divByZero = FE_DIVBYZERO, overflow = FE_OVERFLOW, underflow = FE_UNDERFLOW, inexact = FE_INEXACT };

COSMICTIGER_DEBUG_HOST_DEVICE constexpr int toMask(std::initializer_list<Fpe> exceptions) noexcept {
    PROFILE();
    int mask = 0;
    for (auto const exception : exceptions) {
        mask |= static_cast<int>(exception);
    }
    return mask;
}

COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool validationChecksEnabled() noexcept {
    PROFILE();
#ifdef COSMICTIGER_VALIDATION_CHECKS
    return true;
#else
    return false;
#endif
}

class FpeGuard {
public:
    COSMICTIGER_DEBUG_HOST_DEVICE static constexpr bool supportsHardwareTraps() noexcept {
        PROFILE();
#if defined(COSMICTIGER_VALIDATION_CHECKS) && defined(__GLIBC__) && !defined(__CUDA_ARCH__)
        return true;
#else
        return false;
#endif
    }

    COSMICTIGER_DEBUG_HOST_DEVICE explicit FpeGuard(int mask = FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID) noexcept {
        PROFILE();
#if defined(COSMICTIGER_VALIDATION_CHECKS) && defined(__GLIBC__) && !defined(__CUDA_ARCH__)
        previousEnabled_ = fegetexcept();
        if (previousEnabled_ < 0) {
            return;
        }
        enabledHere_ = (mask & FE_ALL_EXCEPT) & ~previousEnabled_;
        if (enabledHere_ != 0) {
            std::fegetexceptflag(&previousFlags_, enabledHere_);
            std::feclearexcept(enabledHere_);
            feenableexcept(enabledHere_);
        }
#else
        (void) mask;
#endif
    }

    FpeGuard(FpeGuard const&) = delete;
    FpeGuard(FpeGuard&&) = delete;
    FpeGuard& operator=(FpeGuard const&) = delete;
    FpeGuard& operator=(FpeGuard&&) = delete;

    COSMICTIGER_DEBUG_HOST_DEVICE ~FpeGuard() noexcept {
        PROFILE();
#if defined(COSMICTIGER_VALIDATION_CHECKS) && defined(__GLIBC__) && !defined(__CUDA_ARCH__)
        if (previousEnabled_ < 0) {
            return;
        }
        int const nowEnabled = fegetexcept();
        int const toDisable = nowEnabled & ~previousEnabled_;
        int const toEnable = previousEnabled_ & ~nowEnabled;
        if (toDisable != 0) {
            fedisableexcept(toDisable);
        }
        if (toEnable != 0) {
            std::feclearexcept(toEnable);
            feenableexcept(toEnable);
        }
        if (enabledHere_ != 0) {
            std::fesetexceptflag(&previousFlags_, enabledHere_);
        }
#endif
    }

private:
#ifdef COSMICTIGER_VALIDATION_CHECKS
    int previousEnabled_ = -1;
    int enabledHere_ = 0;
    std::fexcept_t previousFlags_{};
#endif
};

namespace detail {

    // Preserve integer ordering across signed/unsigned bounds, including
    // negative indices and signed bounds of a narrower type.
    template <typename Left, typename Right>
    COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool expectationLess(Left left, Right right) {
        PROFILE();
        if constexpr (std::is_integral_v<Left> && std::is_integral_v<Right> && std::is_signed_v<Left> != std::is_signed_v<Right>) {
            if constexpr (std::is_signed_v<Left>) {
                return left < 0 || static_cast<std::make_unsigned_t<Left>>(left) < right;
            } else {
                return right >= 0 && left < static_cast<std::make_unsigned_t<Right>>(right);
            }
        } else {
            return left < right;
        }
    }

    template <typename Left, typename Right>
    COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool expectationLessEqual(Left left, Right right) {
        PROFILE();
        if constexpr (std::is_integral_v<Left> && std::is_integral_v<Right>) {
            return !expectationLess(right, left);
        } else {
            return left <= right;
        }
    }

    template <typename Left, typename Right>
    COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool expectationEqual(Left left, Right right) {
        PROFILE();
        if constexpr (std::is_integral_v<Left> && std::is_integral_v<Right>) {
            return !expectationLess(left, right) && !expectationLess(right, left);
        } else {
            return left == right;
        }
    }

    [[noreturn]] COSMICTIGER_DEBUG_HOST_DEVICE inline void expectationFailed(char const* expression, char const* requirement, char const* file, int line) {
        PROFILE();
#if defined(__CUDA_ARCH__)
        printf("CosmicTiger expectation failed: %s (%s), %s:%d\n", expression, requirement, file, line);
        asm volatile("trap;");
        __builtin_unreachable();
#else
        throw std::runtime_error(std::string("CosmicTiger expectation failed: ") + expression + " (" + requirement + "), " + file + ":" + std::to_string(line));
#endif
    }

}    // namespace detail

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value identityExpectation(Value value) {
    PROFILE();
    return value;
}

COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool uncheckedExpectation() noexcept {
    PROFILE();
    return true;
}

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectPositiveAt(Value value, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(value > Value{})) {
        detail::expectationFailed(expression, "positive", file, line);
    }
    return value;
}

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectNegativeAt(Value value, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(value < Value{})) {
        detail::expectationFailed(expression, "negative", file, line);
    }
    return value;
}

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectNonNegativeAt(Value value, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(value >= Value{})) {
        detail::expectationFailed(expression, "nonnegative", file, line);
    }
    return value;
}

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectNonPositiveAt(Value value, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(value <= Value{})) {
        detail::expectationFailed(expression, "nonpositive", file, line);
    }
    return value;
}

template <typename Value>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectNonZeroAt(Value value, char const* expression, char const* file, int line) {
    PROFILE();
    if (value == Value{}) {
        detail::expectationFailed(expression, "nonzero", file, line);
    }
    return value;
}

template <typename Lower, typename Value, typename Upper>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectRangeAt(Lower lower, Value value, Upper upper, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(detail::expectationLessEqual(lower, value) && detail::expectationLessEqual(value, upper))) {
        detail::expectationFailed(expression, "within inclusive range", file, line);
    }
    return value;
}

template <typename Index, typename Size>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Index expectIndexAt(Index index, Size size, char const* expression, char const* file, int line) {
    PROFILE();
    if (!(index >= Index{} && detail::expectationLess(index, size))) {
        detail::expectationFailed(expression, "within exclusive index range", file, line);
    }
    return index;
}

COSMICTIGER_DEBUG_HOST_DEVICE constexpr bool expectAt(bool condition, char const* expression, char const* file, int line) {
    PROFILE();
    if (!condition) {
        detail::expectationFailed(expression, "true", file, line);
    }
    return condition;
}

template <typename Value, typename Expected>
COSMICTIGER_DEBUG_HOST_DEVICE constexpr Value expectEqualAt(Value value, Expected expected, char const* expression, char const* file, int line) {
    PROFILE();
    if (!detail::expectationEqual(value, expected)) {
        detail::expectationFailed(expression, "equal to expected value", file, line);
    }
    return value;
}

}    // namespace cosmictiger

#ifdef COSMICTIGER_VALIDATION_CHECKS
#define expectPositive(v) expectPositiveAt((v), #v, __FILE__, __LINE__)
#define expectNegative(v) expectNegativeAt((v), #v, __FILE__, __LINE__)
#define expectNonNegative(v) expectNonNegativeAt((v), #v, __FILE__, __LINE__)
#define expectNonPositive(v) expectNonPositiveAt((v), #v, __FILE__, __LINE__)
#define expectNonZero(v) expectNonZeroAt((v), #v, __FILE__, __LINE__)
#define expectRange(a, v, b) expectRangeAt((a), (v), (b), #v, __FILE__, __LINE__)
#define expectIndex(v, size) expectIndexAt((v), (size), #v, __FILE__, __LINE__)
#define expect(condition) expectAt((condition), #condition, __FILE__, __LINE__)
#define expectEqual(v, expected) expectEqualAt((v), (expected), #v, __FILE__, __LINE__)
#else
#define expectPositive(v) identityExpectation((v))
#define expectNegative(v) identityExpectation((v))
#define expectNonNegative(v) identityExpectation((v))
#define expectNonPositive(v) identityExpectation((v))
#define expectNonZero(v) identityExpectation((v))
#define expectRange(a, v, b) identityExpectation((v))
#define expectIndex(v, size) identityExpectation((v))
#define expect(condition) uncheckedExpectation()
#define expectEqual(v, expected) identityExpectation((v))
#endif
