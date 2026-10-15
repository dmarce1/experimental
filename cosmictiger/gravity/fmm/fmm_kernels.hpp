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

/** @file
 * @brief Single-interaction FMM arithmetic shared by the CPU and CUDA backends.
 *
 * Implementation detail of fmm_interactions. Each function adds one source to
 * a caller-owned accumulator and calls the generated STF kernel directly, so
 * the host and device paths evaluate the same polynomial.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/spatial_types.hpp>

#if defined(__CUDACC__)
#define COSMICTIGER_FMM_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_FMM_HOST_DEVICE
#endif

namespace cosmictiger::gravity::fmm::detail {

static_assert(NDIM == 3, "Generated STF operators require three spatial dimensions");

// Host-side request validation shared by the CPU and CUDA entry points.
inline void requireOrderAndSpacing(int fmmOrder, double spacing) {
    PROFILE();
    if (fmmOrder < kMinimumFmmOrder || fmmOrder > kMaximumFmmOrder) {
        throw std::invalid_argument("FMM order is outside the supported local/source rank range");
    }
    if (!(spacing > 0.0)) {
        throw std::invalid_argument("FMM cell spacing must be positive");
    }
}

template <typename Request>
inline void requireTables(Request const& request) {
    PROFILE();
    if (request.targetCount != 0 && request.targets == nullptr) {
        throw std::invalid_argument("FMM targets are required for a nonempty batch");
    }
    if (request.sourceCount != 0 && request.sources == nullptr) {
        throw std::invalid_argument("FMM sources are required for a nonempty source table");
    }
}

template <typename Request>
inline void requirePositions(Request const& request) {
    PROFILE();
    for (size_t axis = 0; axis < NDIM; axis++) {
        if (request.positions[axis] == nullptr) {
            throw std::invalid_argument("FMM particle positions are required for a nonempty batch");
        }
    }
}

// Independent STF coefficients through a rank, matching stf::coefficientCount
// but usable in device constant expressions.
COSMICTIGER_FMM_HOST_DEVICE constexpr size_t stfCount(int rank) noexcept {
    PROFILE();
    return static_cast<size_t>((rank + 1) * (rank + 1));
}

// The local rank p accumulates through stfCount(p) coefficients.
template <int fmmOrder>
COSMICTIGER_FMM_HOST_DEVICE constexpr size_t localCount() noexcept {
    PROFILE();
    return stfCount(fmmOrder);
}

// Integer subtraction wraps modulo 2^32; the signed conversion selects the
// nearest periodic image, as in the Ferrers direct kernel.
template <typename Real>
COSMICTIGER_FMM_HOST_DEVICE inline Real minimumImageOffset(Coordinate position, Coordinate center) noexcept {
    constexpr Real kCoordinateScale = Real(0x1p-32);
    PROFILE();
    std::uint32_t const wrapped = position.raw() - center.raw();
    return static_cast<Real>(static_cast<std::int32_t>(wrapped)) * kCoordinateScale;
}

template <typename Real>
COSMICTIGER_FMM_HOST_DEVICE inline Real inverseLength(Real const (&separation)[NDIM]) {
    PROFILE();
    Real const radiusSquared = separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2];
    expectPositive(radiusSquared);
#if defined(__CUDA_ARCH__)
    if constexpr (sizeof(Real) == sizeof(float)) {
        return rsqrtf(radiusSquared);
    } else {
        return rsqrt(radiusSquared);
    }
#else
    return Real(1) / std::sqrt(radiusSquared);
#endif
}

// cc: separation = target center - source center. Both scales are spacing.
template <typename Real, int fmmOrder>
COSMICTIGER_FMM_HOST_DEVICE inline void addM2l(Real (&local)[localCount<fmmOrder>()], float const* multipole, Real const (&separation)[NDIM], Real spacing) {
    constexpr int kSourceOrder = fmmOrder - 1;
    PROFILE();
    Real const inverseRadius = inverseLength(separation);
    Real const ratio = spacing * inverseRadius;
    stf::detail::M2lKernel<fmmOrder, false>::template add<Real, Real[localCount<fmmOrder>()], float const*, kSourceOrder>(
        local, multipole, separation[0] * inverseRadius, separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius, ratio, ratio);
}

// cp: separation = target center - particle position.
template <typename Real, int fmmOrder>
COSMICTIGER_FMM_HOST_DEVICE inline void addP2l(Real (&local)[localCount<fmmOrder>()], Real mass, Real const (&separation)[NDIM], Real spacing) {
    PROFILE();
    Real const inverseRadius = inverseLength(separation);
    stf::detail::P2lKernel<fmmOrder>::add(
        local, mass, separation[0] * inverseRadius, separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius, spacing * inverseRadius);
}

// pc: separation = particle position - source center. field holds the
// potential followed by the acceleration. Every retained source rank
// contributes to both, unlike an M2L truncated to a linear local.
template <typename Real, int fmmOrder>
COSMICTIGER_FMM_HOST_DEVICE inline void addM2p(Real (&field)[NDIM + 1], float const* multipole, Real const (&separation)[NDIM], Real spacing) {
    constexpr int kSourceOrder = fmmOrder - 1;
    PROFILE();
    Real const inverseRadius = inverseLength(separation);
    stf::detail::M2pKernel<kSourceOrder, false>::add(
        field, multipole, separation[0] * inverseRadius, separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius, spacing * inverseRadius);
}

}    // namespace cosmictiger::gravity::fmm::detail

#undef COSMICTIGER_FMM_HOST_DEVICE
