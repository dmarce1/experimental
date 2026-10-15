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

#include <cstdint>
#include <string>
#include <type_traits>

#if defined(__CUDACC__)
#define COSMICTIGER_PROFILE_HOST_FORCE_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define COSMICTIGER_PROFILE_HOST_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define COSMICTIGER_PROFILE_HOST_FORCE_INLINE __forceinline
#else
#define COSMICTIGER_PROFILE_HOST_FORCE_INLINE inline
#endif

namespace profile {

// Start before hpx::init to sample every newly created OS worker. A late start
// attaches the caller and future workers; registration can attach other callers.
// The interval measures each worker's CPU time, not elapsed wall time. A stopped
// sampler cannot be restarted in the same process.
bool profileStart(std::uint64_t intervalNanoseconds = 10000000, std::string const& filename = {}) noexcept;
void profileStop() noexcept;
// PROFILE() registers optional signature metadata once per function. Repeated
// call addresses do not identify inlining: samples use exact ELF function ranges,
// and inlined work belongs to the enclosing compiled function.
bool profileRegisterFunction(char const* functionName) noexcept;
std::string profileReport();
bool writeProfileReport(std::string const& filename = {}) noexcept;

namespace detail {

    // Keep the constant-evaluation query outside consteval callers, where a
    // direct query would produce a tautological-condition warning under NVCC.
    COSMICTIGER_PROFILE_HOST_FORCE_INLINE constexpr bool registrationIsRuntime() noexcept {
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
        return !__builtin_is_constant_evaluated();
#else
        return !std::is_constant_evaluated();
#endif
    }

    // The macro's unique lambda type gives every function specialization its
    // own once flag. Its static storage belongs here, so callers remain valid
    // C++20 constexpr functions. Force inlining keeps the registration hook's
    // return address inside the original caller, including in Debug builds.
    template <typename FunctionTag>
    COSMICTIGER_PROFILE_HOST_FORCE_INLINE void registerFunctionOnce(char const* functionName) noexcept {
        [[maybe_unused]] static bool const functionRegistered = profileRegisterFunction(functionName);
    }

}    // namespace detail

}    // namespace profile

#undef COSMICTIGER_PROFILE_HOST_FORCE_INLINE

#if defined(__clang__) || defined(__GNUC__)
#define CURRENT_FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define CURRENT_FUNCTION_SIGNATURE __FUNCSIG__
#else
#define CURRENT_FUNCTION_SIGNATURE __func__
#endif

#if !defined(PROFILE_DISABLE) && !defined(__CUDA_ARCH__)
#define PROFILE()                                                                                 \
    do {                                                                                          \
        if (::profile::detail::registrationIsRuntime()) {                                         \
            ::profile::detail::registerFunctionOnce<decltype([] {})>(CURRENT_FUNCTION_SIGNATURE); \
        }                                                                                         \
    } while (false)
#else
#define PROFILE() ((void) 0)
#endif
