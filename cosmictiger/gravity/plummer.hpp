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

#include <cosmictiger/gravity/ferrers.hpp>
#include <cosmictiger/profiler.hpp>

#if defined(__CUDACC__)
#define COSMICTIGER_PLUMMER_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_PLUMMER_HOST_DEVICE
#endif

namespace cosmictiger::gravity {

struct PlummerAccelerationPotential {
    Acceleration acceleration;
    float potential;
};

namespace detail {
    template <bool computePotential>
    COSMICTIGER_PLUMMER_HOST_DEVICE inline auto plummerInteraction(Position target, Position source, SofteningLength softening) noexcept {
        PROFILE();
        FpeGuard fpeGuard;
        float const dx = ferrersDisplacement(target.x, source.x);
        float const dy = ferrersDisplacement(target.y, source.y);
        float const dz = ferrersDisplacement(target.z, source.z);
        float const radiusSquared = dx * dx + dy * dy + dz * dz;
        float const inverseRadius = 1.0f / sqrtf(radiusSquared + softening.qSquared);
        float const accelerationScale = -inverseRadius * inverseRadius * inverseRadius;
        Acceleration const acceleration{accelerationScale * dx, accelerationScale * dy, accelerationScale * dz};
        if constexpr (computePotential) {
            return PlummerAccelerationPotential{acceleration, -inverseRadius};
        } else {
            return acceleration;
        }
    }
}    // namespace detail

// Unit mass, unit G, with the isolated potential zero at infinity.
COSMICTIGER_PLUMMER_HOST_DEVICE inline Acceleration plummerAcceleration(Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    return detail::plummerInteraction<false>(target, source, softening);
}

COSMICTIGER_PLUMMER_HOST_DEVICE inline PlummerAccelerationPotential plummerAccelerationAndPotential(
    Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    return detail::plummerInteraction<true>(target, source, softening);
}

template <int order>
COSMICTIGER_PLUMMER_HOST_DEVICE inline Acceleration softenedAcceleration(Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    if (softening.model == SofteningModel::plummer) {
        return plummerAcceleration(target, source, softening);
    }
    return ferrersAcceleration<order>(target, source, softening);
}

template <int order>
COSMICTIGER_PLUMMER_HOST_DEVICE inline FerrersAccelerationPotential softenedAccelerationAndPotential(
    Position target, Position source, SofteningLength softening) noexcept {
    PROFILE();
    if (softening.model == SofteningModel::plummer) {
        auto const field = plummerAccelerationAndPotential(target, source, softening);
        return {field.acceleration, field.potential};
    }
    return ferrersAccelerationAndPotential<order>(target, source, softening);
}

// The timestep criterion uses the equivalent central-potential length for
// Ferrers, and the actual epsilon for Plummer.
template <int order>
inline double softeningEpsilon(SofteningLength softening) {
    PROFILE();
    FpeGuard fpeGuard;
    double const scale = softening.model == SofteningModel::plummer ? 1.0 : double(detail::ferrersPotentialPolynomial<order>(1.0f));
    return 1.0 / (double(softening.inverseQ) * scale);
}

}    // namespace cosmictiger::gravity

#undef COSMICTIGER_PLUMMER_HOST_DEVICE
