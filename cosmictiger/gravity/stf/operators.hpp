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
 * @brief Generated compact STF translations and particle interactions.
 *
 * All add operators accumulate into their destination. Moments are raw STF
 * tensors divided by scale^rank; locals are potential derivatives multiplied
 * by scale^rank. The gravitational constant is one. See m2l.hpp for storage.
 */
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>

namespace cosmictiger::gravity::stf {

template <typename Real = double>
struct Field {
    Real potential{};
    std::array<Real, NDIM> acceleration{};
};

namespace detail {
    template <int order>
    constexpr void requireGeneratedOrder() {
        static_assert(order >= kMinOrder && order <= kMaxOrder, "Operator order is outside the configured generated range");
        PROFILE();
    }

    template <typename Real>
    std::array<Real, NDIM> normalizedShift(std::array<Real, NDIM> const& shift, Real scale) {
        PROFILE();
        FpeGuard fpeGuard;
        std::array<Real, NDIM> result{};
        for (int d = 0; d < NDIM; d++) {
            result[d] = shift[d] / scale;
        }
        return result;
    }

    template <typename Real>
    Real inverseDistance(std::array<Real, NDIM> const& separation) {
        static_assert(std::is_floating_point_v<Real>, "The checked STF interface requires a floating-point scalar");
        PROFILE();
        FpeGuard fpeGuard;
        Real const r2 = separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2];
        expectPositive(r2);
        return Real(1) / std::sqrt(r2);
    }
}    // namespace detail

/// Shift source moments to a new center. displacement = oldCenter-newCenter.
/// Every source rank, including the dipole, participates. Zero shift is valid.
/// sourceScale and targetScale belong to the old and new centers respectively.
/// Buffers must be distinct; arithmetic powers must be representable in Real.
template <typename Real, int order>
void addM2M(Multipole<Real, order>& target, Multipole<Real, order> const& source, std::array<Real, NDIM> const& displacement, Real sourceScale = Real(1),
    Real targetScale = Real(1)) {
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireGeneratedOrder<order>();
    detail::requireScale(sourceScale);
    detail::requireScale(targetScale);
    expect(target.data() != source.data());
    auto const d = detail::normalizedShift(displacement, targetScale);
    Real const ratio = sourceScale / targetScale;
    detail::requireScale(ratio);
    detail::M2mKernel<order>::add(target, source, d[0], d[1], d[2], ratio);
}

/// Shift a local polynomial to a new center. displacement = newCenter-oldCenter.
/// The finite polynomial translation is exact, including its constant Laplacian
/// when withTrace=true. Its normalized trace changes by (targetScale/sourceScale)^2.
template <typename Real, int order, bool withTrace = false>
void addL2L(Expansion<Real, order, withTrace>& target, Expansion<Real, order, withTrace> const& source, std::array<Real, NDIM> const& displacement,
    Real sourceScale = Real(1), Real targetScale = Real(1)) {
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireGeneratedOrder<order>();
    static_assert(!withTrace || order >= 2, "A translated background Laplacian requires a quadratic local expansion");
    detail::requireScale(sourceScale);
    detail::requireScale(targetScale);
    expect(target.data() != source.data());
    auto const d = detail::normalizedShift(displacement, sourceScale);
    Real const ratio = targetScale / sourceScale;
    detail::requireScale(ratio);
    detail::L2lKernel<order, withTrace>::add(target, source, d[0], d[1], d[2], ratio);
}

/// Add the local potential derivatives of one particle; separation=targetCenter-particle.
/// Every local rank through order is generated. The optional background trace
/// is preserved because this exterior Newtonian contribution is harmonic.
template <typename Real, int order, bool withTrace = false>
void addP2L(Expansion<Real, order, withTrace>& local, Real mass, std::array<Real, NDIM> const& separation, Real targetScale = Real(1)) {
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireGeneratedOrder<order>();
    detail::requireScale(targetScale);

    Real const invr = detail::inverseDistance(separation);
    Real const ratio = targetScale * invr;
    detail::requireScale(ratio);
    detail::P2lKernel<order>::add(local, mass, separation[0] * invr, separation[1] * invr, separation[2] * invr, invr, ratio);
}

/// Evaluate potential and acceleration at separation=particlePosition-sourceCenter.
/// All retained source ranks contribute to BOTH potential and acceleration.
/// This differentiates the source-truncated potential, unlike triangular M2L
/// whose highest source rank contributes only to its constant local term.
template <typename Real, int order, bool centered = false>
Field<Real> evaluateM2P(Multipole<Real, order> const& source, std::array<Real, NDIM> const& separation, Real sourceScale = Real(1)) {
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireGeneratedOrder<order>();
    detail::requireScale(sourceScale);
    Real const invr = detail::inverseDistance(separation);
    Real const ratio = sourceScale * invr;
    detail::requireScale(ratio);
    std::array<Real, NDIM + 1> values{};
    detail::M2pKernel<order, centered>::add(values, source, separation[0] * invr, separation[1] * invr, separation[2] * invr, invr, ratio);
    return {values[0], {values[1], values[2], values[3]}};
}

}    // namespace cosmictiger::gravity::stf
