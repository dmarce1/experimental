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
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/spatial_types.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__CUDACC__)
#define COSMICTIGER_FERRERS_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_FERRERS_HOST_DEVICE
#endif

namespace cosmictiger::gravity {

struct FerrersAccelerationPotential {
    Acceleration acceleration;
    float potential;
};

namespace detail {

    // Integer subtraction wraps modulo 2^32. C++20 conversion to int32_t
    // selects the signed nearest image, including -1/2 at the half-box tie.
    COSMICTIGER_FERRERS_HOST_DEVICE inline float ferrersDisplacement(Coordinate target, Coordinate source) noexcept {
        constexpr float kCoordinateScale = 0x1p-32f;
        PROFILE();
        std::uint32_t const wrapped = target.raw() - source.raw();
        return static_cast<float>(static_cast<std::int32_t>(wrapped)) * kCoordinateScale;
    }

    template <int order>
    COSMICTIGER_FERRERS_HOST_DEVICE inline float ferrersForcePolynomial(float t) noexcept {
        [[maybe_unused]] constexpr float kLinearCoefficient = 1.5f;
        [[maybe_unused]] constexpr float kQuadraticCoefficient = 1.875f;
        [[maybe_unused]] constexpr float kCubicCoefficient = 2.1875f;
        static_assert(order >= 0 && order <= 3, "Ferrers order must be F0, F1, F2, or F3");
        PROFILE();
        if constexpr (order == 0) {
            return 1.0f;
        } else {
            if constexpr (order == 1) {
                return 1.0f + t * kLinearCoefficient;
            } else {
                if constexpr (order == 2) {
                    return 1.0f + t * (kLinearCoefficient + t * kQuadraticCoefficient);
                } else {
                    return 1.0f + t * (kLinearCoefficient + t * (kQuadraticCoefficient + t * kCubicCoefficient));
                }
            }
        }
    }

    template <int order>
    COSMICTIGER_FERRERS_HOST_DEVICE inline float ferrersPotentialPolynomial(float t) noexcept {
        [[maybe_unused]] constexpr float kLinearCoefficient = 0.5f;
        [[maybe_unused]] constexpr float kQuadraticCoefficient = 0.375f;
        [[maybe_unused]] constexpr float kCubicCoefficient = 0.3125f;
        [[maybe_unused]] constexpr float kQuarticCoefficient = 0.2734375f;
        static_assert(order >= 0 && order <= 3, "Ferrers order must be F0, F1, F2, or F3");
        PROFILE();
        if constexpr (order == 0) {
            return 1.0f + t * kLinearCoefficient;
        } else {
            if constexpr (order == 1) {
                return 1.0f + t * (kLinearCoefficient + t * kQuadraticCoefficient);
            } else {
                if constexpr (order == 2) {
                    return 1.0f + t * (kLinearCoefficient + t * (kQuadraticCoefficient + t * kCubicCoefficient));
                } else {
                    return 1.0f + t * (kLinearCoefficient + t * (kQuadraticCoefficient + t * (kCubicCoefficient + t * kQuarticCoefficient)));
                }
            }
        }
    }

    template <int order, bool computePotential>
    COSMICTIGER_FERRERS_HOST_DEVICE inline auto ferrersInteraction(Position target, Position source, SofteningLength softening) noexcept {
        constexpr Acceleration kZeroAcceleration{0.0f, 0.0f, 0.0f};
        static_assert(order >= 0 && order <= 3, "Ferrers order must be F0, F1, F2, or F3");
        PROFILE();

        FpeGuard fpeGuard;
        float const dx = ferrersDisplacement(target.x, source.x);
        float const dy = ferrersDisplacement(target.y, source.y);
        float const dz = ferrersDisplacement(target.z, source.z);
        float const radiusSquared = dx * dx + dy * dy + dz * dz;
        if (radiusSquared == 0.0f) {
            if constexpr (computePotential) {
                return FerrersAccelerationPotential{kZeroAcceleration, -softening.inverseQ * ferrersPotentialPolynomial<order>(1.0f)};
            } else {
                return kZeroAcceleration;
            }
        }
        float accelerationScale;
        float potential = 0.0f;

        if (radiusSquared < softening.qSquared) {
            float const t = 1.0f - radiusSquared * softening.inverseQSquared;
            accelerationScale = -softening.inverseQCubed * ferrersForcePolynomial<order>(t);
            if constexpr (computePotential) {
                potential = -softening.inverseQ * ferrersPotentialPolynomial<order>(t);
            }
        } else {
            float const inverseRadius = 1.0f / sqrtf(radiusSquared);
            accelerationScale = -inverseRadius * inverseRadius * inverseRadius;
            if constexpr (computePotential) {
                potential = -inverseRadius;
            }
        }

        Acceleration const acceleration{accelerationScale * dx, accelerationScale * dy, accelerationScale * dz};
        if constexpr (computePotential) {
            return FerrersAccelerationPotential{acceleration, potential};
        } else {
            return acceleration;
        }
    }

}    // namespace detail

// Unit-mass density is C_n/q^3 (1-r^2/q^2)^n for r<q, zero outside.
// The source mass and gravitational constant are both one; potential is zero at infinity.
// The shared implementation removes all potential arithmetic in this mode.
template <int order>
COSMICTIGER_FERRERS_HOST_DEVICE inline Acceleration ferrersAcceleration(Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    return detail::ferrersInteraction<order, false>(target, source, softening);
}

template <int order>
COSMICTIGER_FERRERS_HOST_DEVICE inline FerrersAccelerationPotential ferrersAccelerationAndPotential(
    Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    return detail::ferrersInteraction<order, true>(target, source, softening);
}

}    // namespace cosmictiger::gravity

#undef COSMICTIGER_FERRERS_HOST_DEVICE
