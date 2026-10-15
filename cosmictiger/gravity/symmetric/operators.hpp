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

#include <cosmictiger/gravity/symmetric/multipole.hpp>
#include <cosmictiger/profiler.hpp>

#include <cmath>
#include <type_traits>

namespace cosmictiger::gravity::symmetric {

namespace detail {
    template <typename Real, int order>
    COSMICTIGER_SYMMETRIC_HOST_DEVICE auto powers(Real value) {
        PROFILE();
        std::array<Real, order + 1> result{};
        result[0] = Real(1);
        for (int rank = 1; rank <= order; rank++) {
            result[rank] = result[rank - 1] * value;
        }
        return result;
    }

    template <typename Real, int order>
    COSMICTIGER_SYMMETRIC_HOST_DEVICE auto coordinatePowers(std::array<Real, NDIM> const& offset, Real scale) {
        PROFILE();
        expectPositive(scale);
        std::array<std::array<Real, order + 1>, NDIM> result{};
        for (int d = 0; d < NDIM; d++) {
            result[d] = powers<Real, order>(offset[d] / scale);
        }
        return result;
    }
}    // namespace detail

/// Derivatives of -1/sqrt(r^2+softening^2), multiplied by targetScale^rank.
/// Zero softening gives Newtonian gravity. Positive softening is regular at r=0.
/// The recurrence follows (r^2+eps^2) d_i G = -r_i G and retains all traces.
/// It uses O(coefficientCount(order)) arithmetic and local, memoized powers.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Expansion<Real, order> greenDerivatives(
    std::array<Real, NDIM> const& separation, Real softening = Real(0), Real targetScale = Real(1)) {
    constexpr auto kExponents = exponents<order>();
    static_assert(std::is_floating_point_v<Real>);
    PROFILE();
    FpeGuard fpeGuard;
    expectNonNegative(softening);
    expectPositive(targetScale);
    Real radiusSquared = softening * softening;
    for (int d = 0; d < NDIM; d++) {
        radiusSquared += separation[d] * separation[d];
    }
    expectPositive(radiusSquared);
    Real const inverseRadius = Real(1) / std::sqrt(radiusSquared);
    std::array<Real, NDIM> unit{};
    for (int d = 0; d < NDIM; d++) {
        unit[d] = separation[d] * inverseRadius;
    }
    auto const scalePowers = detail::powers<Real, order>(targetScale * inverseRadius);
    Expansion<Real, order> result{};
    result[0] = Real(-1);
    for (size_t coefficient = 1; coefficient < result.size(); coefficient++) {
        auto const a = kExponents[coefficient];
        std::array<int, NDIM> beta{a.nx, a.ny, a.nz};
        int direction = 0;
        while (beta[direction] == 0) {
            direction++;
        }
        beta[direction]--;
        Real value = -Real(2 * beta[direction] + 1) * unit[direction] * result[index(beta[0], beta[1], beta[2])];
        if (beta[direction] > 0) {
            auto lower = beta;
            lower[direction]--;
            value -= Real(beta[direction] * beta[direction]) * result[index(lower[0], lower[1], lower[2])];
        }
        for (int d = 0; d < NDIM; d++) {
            if (d != direction && beta[d] > 0) {
                auto lower = beta;
                lower[d]--;
                lower[direction]++;
                value -= Real(2 * beta[d]) * unit[d] * result[index(lower[0], lower[1], lower[2])];
                if (beta[d] > 1) {
                    lower[d]--;
                    value -= Real(beta[d] * (beta[d] - 1)) * result[index(lower[0], lower[1], lower[2])];
                }
            }
        }
        result[coefficient] = value;
    }
    for (size_t coefficient = 0; coefficient < result.size(); coefficient++) {
        result[coefficient] *= inverseRadius * scalePowers[kExponents[coefficient].rank];
    }
    return result;
}

/// Accumulate all raw moments. Return |offset/sourceScale|^2 for callers.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Real addPoint(Multipole<Real, order>& target, Real mass, std::array<Real, NDIM> const& offset, Real sourceScale = Real(1)) {
    constexpr auto kExponents = exponents<order>();
    PROFILE();
    FpeGuard fpeGuard;
    auto const powers = detail::coordinatePowers<Real, order>(offset, sourceScale);
    Real radiusSquared = 0;
    for (int d = 0; d < NDIM; d++) {
        if constexpr (order >= 2) {
            radiusSquared += powers[d][2];
        } else if constexpr (order >= 1) {
            radiusSquared += powers[d][1] * powers[d][1];
        } else {
            Real const coordinate = offset[d] / sourceScale;
            radiusSquared += coordinate * coordinate;
        }
    }
    for (size_t coefficient = 0; coefficient < target.size(); coefficient++) {
        auto const a = kExponents[coefficient];
        target[coefficient] += mass * powers[0][a.nx] * powers[1][a.ny] * powers[2][a.nz];
    }
    return radiusSquared;
}

/// displacement=oldCenter-newCenter. The source and target must be distinct.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE void addM2M(Multipole<Real, order>& target, Multipole<Real, order> const& source, std::array<Real, NDIM> const& displacement,
    Real sourceScale = Real(1), Real targetScale = Real(1)) {
    constexpr auto kExponents = exponents<order>();
    constexpr Combinatorics<order> kCombinatorics{};
    PROFILE();
    FpeGuard fpeGuard;
    expectPositive(sourceScale);
    expectPositive(targetScale);
    expect(target.data() != source.data());
    auto const powers = detail::coordinatePowers<Real, order>(displacement, targetScale);
    auto const scalePowers = detail::powers<Real, order>(sourceScale / targetScale);
    for (size_t targetIndex = 0; targetIndex < target.size(); targetIndex++) {
        auto const a = kExponents[targetIndex];
        Real value = 0;
        for (size_t sourceIndex = 0; sourceIndex <= targetIndex; sourceIndex++) {
            auto const b = kExponents[sourceIndex];
            if (b.nx <= a.nx && b.ny <= a.ny && b.nz <= a.nz) {
                Real const weight = Real(kCombinatorics.binomial(a.nx, b.nx) * kCombinatorics.binomial(a.ny, b.ny) * kCombinatorics.binomial(a.nz, b.nz));
                value += weight * source[sourceIndex] * scalePowers[b.rank] * powers[0][a.nx - b.nx] * powers[1][a.ny - b.ny] * powers[2][a.nz - b.nz];
            }
        }
        target[targetIndex] += value;
    }
}

/// Exact translation of the retained polynomial, displacement=newCenter-oldCenter.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE void addL2L(Expansion<Real, order>& target, Expansion<Real, order> const& source, std::array<Real, NDIM> const& displacement,
    Real sourceScale = Real(1), Real targetScale = Real(1)) {
    constexpr auto kExponents = exponents<order>();
    constexpr Combinatorics<order> kCombinatorics{};
    PROFILE();
    FpeGuard fpeGuard;
    expectPositive(sourceScale);
    expectPositive(targetScale);
    expect(target.data() != source.data());
    auto const powers = detail::coordinatePowers<Real, order>(displacement, sourceScale);
    auto const scalePowers = detail::powers<Real, order>(targetScale / sourceScale);
    for (size_t targetIndex = 0; targetIndex < target.size(); targetIndex++) {
        auto const a = kExponents[targetIndex];
        Real value = 0;
        for (size_t sourceIndex = targetIndex; sourceIndex < source.size(); sourceIndex++) {
            auto const b = kExponents[sourceIndex];
            if (b.nx >= a.nx && b.ny >= a.ny && b.nz >= a.nz) {
                int const nx = b.nx - a.nx;
                int const ny = b.ny - a.ny;
                int const nz = b.nz - a.nz;
                Real const weight = Real(kCombinatorics.inverseFactorial(nx) * kCombinatorics.inverseFactorial(ny) * kCombinatorics.inverseFactorial(nz));
                value += weight * source[sourceIndex] * powers[0][nx] * powers[1][ny] * powers[2][nz];
            }
        }
        target[targetIndex] += value * scalePowers[a.rank];
    }
}

/// separation=targetCenter-particle. Softening applies to every retained rank.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE void addP2L(
    Expansion<Real, order>& local, Real mass, std::array<Real, NDIM> const& separation, Real targetScale = Real(1), Real softening = Real(0)) {
    PROFILE();
    FpeGuard fpeGuard;
    auto const green = greenDerivatives<Real, order>(separation, softening, targetScale);
    for (size_t coefficient = 0; coefficient < local.size(); coefficient++) {
        local[coefficient] += mass * green[coefficient];
    }
}

/// Full source-by-local contraction, retaining every pair of ranks independently.
/// This is the rectangular truncation needed by PMMM, not total-rank truncation.
/// separation=targetCenter-sourceCenter; buffers must be distinct.
template <typename Real, int sourceOrder, int localOrder = sourceOrder>
COSMICTIGER_SYMMETRIC_HOST_DEVICE void addM2L(Expansion<Real, localOrder>& local, Multipole<Real, sourceOrder> const& source,
    std::array<Real, NDIM> const& separation, Real sourceScale = Real(1), Real targetScale = Real(1), Real softening = Real(0)) {
    constexpr auto kSourceExponents = exponents<sourceOrder>();
    constexpr auto kLocalExponents = exponents<localOrder>();
    PROFILE();
    FpeGuard fpeGuard;
    expectPositive(sourceScale);
    expectPositive(targetScale);
    expect(local.data() != source.data());
    auto const green = greenDerivatives<Real, sourceOrder + localOrder>(separation, softening, targetScale);
    auto const scalePowers = detail::powers<Real, sourceOrder>(-sourceScale / targetScale);
    for (size_t localIndex = 0; localIndex < local.size(); localIndex++) {
        auto const a = kLocalExponents[localIndex];
        Real value = 0;
        for (size_t sourceIndex = 0; sourceIndex < source.size(); sourceIndex++) {
            auto const b = kSourceExponents[sourceIndex];
            value += source[sourceIndex] * Real(b.inverseFactorial) * scalePowers[b.rank] * green[index(a.nx + b.nx, a.ny + b.ny, a.nz + b.nz)];
        }
        local[localIndex] += value;
    }
}

/// Evaluate the finite polynomial at offset=position-localCenter.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Field<Real> evaluateL2P(
    Expansion<Real, order> const& local, std::array<Real, NDIM> const& offset, Real targetScale = Real(1)) {
    constexpr auto kExponents = exponents<order>();
    PROFILE();
    FpeGuard fpeGuard;
    auto const powers = detail::coordinatePowers<Real, order>(offset, targetScale);
    Field<Real> result{};
    for (size_t coefficient = 0; coefficient < local.size(); coefficient++) {
        auto const a = kExponents[coefficient];
        Real const monomial = Real(a.inverseFactorial) * powers[0][a.nx] * powers[1][a.ny] * powers[2][a.nz];
        result.potential += local[coefficient] * monomial;
        if (a.rank < order) {
            std::array<int, NDIM> next{a.nx, a.ny, a.nz};
            for (int d = 0; d < NDIM; d++) {
                next[d]++;
                result.acceleration[d] -= local[index(next[0], next[1], next[2])] * monomial;
                next[d]--;
            }
        }
    }
    for (int d = 0; d < NDIM; d++) {
        result.acceleration[d] /= targetScale;
    }
    return result;
}

/// All source ranks contribute to both potential and acceleration.
template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Field<Real> evaluateM2P(
    Multipole<Real, order> const& source, std::array<Real, NDIM> const& separation, Real sourceScale = Real(1), Real softening = Real(0)) {
    constexpr int kLocalOrder = 1;
    constexpr std::array<Real, NDIM> kZero{};
    PROFILE();
    Expansion<Real, kLocalOrder> local{};
    addM2L<Real, order, kLocalOrder>(local, source, separation, sourceScale, Real(1), softening);
    return evaluateL2P<Real, kLocalOrder>(local, kZero);
}

/// separation=sink-source. The acceleration is the negative potential gradient.
template <typename Real>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Field<Real> evaluateP2P(Real mass, std::array<Real, NDIM> const& separation, Real softening = Real(0)) {
    PROFILE();
    FpeGuard fpeGuard;
    expectNonNegative(softening);
    Real radiusSquared = softening * softening;
    for (int d = 0; d < NDIM; d++) {
        radiusSquared += separation[d] * separation[d];
    }
    expectPositive(radiusSquared);
    Real const inverseRadius = Real(1) / std::sqrt(radiusSquared);
    Real const forceFactor = -mass * inverseRadius * inverseRadius * inverseRadius;
    Field<Real> result{};
    result.potential = -mass * inverseRadius;
    for (int d = 0; d < NDIM; d++) {
        result.acceleration[d] = forceFactor * separation[d];
    }
    return result;
}

}    // namespace cosmictiger::gravity::symmetric
