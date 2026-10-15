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
 * @brief Compact symmetric trace-free Cartesian multipoles and generated M2L.
 *
 * Standalone C++17 module. No CosmicTiger runtime, HPX, Boost, or allocation.
 * Source tensors are raw STF moments divided by sourceScale^rank, without
 * factorial normalization. Expansion tensors are derivatives of the negative
 * Newtonian potential, with G=1, multiplied by targetScale^rank.
 */
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <cmath>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/stf/combinatorics.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>

namespace cosmictiger::gravity::stf {

/// Number of independent symmetric trace-free components through a degree.
constexpr size_t coefficientCount(int order) {
    PROFILE();
    return size_t((order + 1) * (order + 1));
}

/// A constant-background expansion needs only one additional Laplacian scalar.
constexpr size_t localCoefficientCount(int order, bool withTrace = false) {
    PROFILE();
    return coefficientCount(order) + (withTrace ? 1 : 0);
}

/// Rank n starts at n*n; nz=0 precedes nz=1, with nx increasing in each plane.
/// Preconditions: nx,ny >= 0 and nz is 0 or 1.
constexpr size_t index(int nx, int ny, int nz) {
    PROFILE();
    int const n = nx + ny + nz;
    return size_t(n * n + (nz == 0 ? nx : n + 1 + nx));
}

namespace detail {
    template <int order, bool withTrace = false>
    class StorageSize {
    public:
        static_assert(order >= 0 && order <= kMaxOrder, "STF storage order exceeds the configured maximum");
        static constexpr size_t value() noexcept {
            PROFILE();
            return kValue_;
        }

    private:
        static constexpr size_t kValue_ = localCoefficientCount(order, withTrace);
    };

    inline constexpr Combinatorics<kMaxOrder> kCombinatorics{};

    template <typename Real>
    void requireScale(Real scale) {
        static_assert(std::is_floating_point_v<Real>, "The checked STF interface requires a floating-point scalar");
        PROFILE();
        expectPositive(scale);
    }
}    // namespace detail

/// Only nx+ny+nz=n components with nz=0 or 1 are stored: 2n+1 per rank.
template <typename Real, int order>
using Multipole = std::array<Real, detail::StorageSize<order>::value()>;

/// withTrace appends the normalized-coordinate Laplacian (101 entries at order
/// nine). Store only nz=0,1 derivatives. Rank-two zz is recovered with the
/// supplied trace; all higher traces vanish for a constant background density.
template <typename Real, int order, bool withTrace = false>
using Expansion = std::array<Real, detail::StorageSize<order, withTrace>::value()>;

template <int order>
constexpr size_t localIndex(int nx, int ny, int nz) {
    PROFILE();
    expect(!(nx < 0 || ny < 0 || nz < 0 || nx + ny + nz > order || nz > 1));
    return index(nx, ny, nz);
}

/// Recover unstored components by symmetry and trace. For a constant-background
/// expansion only the undifferentiated Laplacian, at (0,0,2), is nonzero.
template <typename Real, int order, bool withTrace = false>
Real component(Expansion<Real, order, withTrace> const& tensor, int nx, int ny, int nz) {
    PROFILE();
    FpeGuard fpeGuard;
    expect(!(nx < 0 || ny < 0 || nz < 0 || nx + ny + nz > order));
    if (nz < 2) {
        return tensor[index(nx, ny, nz)];
    }
    Real value = -component<Real, order, withTrace>(tensor, nx + 2, ny, nz - 2) - component<Real, order, withTrace>(tensor, nx, ny + 2, nz - 2);
    if constexpr (withTrace) {
        if (nx == 0 && ny == 0 && nz == 2) {
            value += tensor[coefficientCount(order)];
        }
    }
    return value;
}

/// Set the normalized-coordinate Laplacian, preserving the supplied xx and yy.
/// A periodic gravity background uses -4*pi*G*rho_background*targetScale^2.
/// This supplies a trace constraint; it does not compute a periodic Green function.
template <typename Real, int order>
void setLaplacian(Expansion<Real, order, true>& local, Real normalizedLaplacian) {
    static_assert(order >= 2, "The Laplacian requires a quadratic local expansion");
    PROFILE();
    local[coefficientCount(order)] = normalizedLaplacian;
}

template <typename Real, int order>
Real laplacian(Expansion<Real, order, true> const& local) {
    static_assert(order >= 2, "The Laplacian requires a quadratic local expansion");
    PROFILE();
    return local[coefficientCount(order)];
}

/// Accumulate mass*STF((offset/sourceScale)^n) for every source rank n<=order.
/// offset = particle position - source expansion center. This includes dipoles;
/// centered=true M2L is appropriate only when the total source dipole is zero.
/// The explicit finite STF projection is used here, outside the M2L hot path.
/// Return the normalized squared radius so trace accumulation can reuse it.
template <typename Real, int order>
Real addPoint(Multipole<Real, order>& moment, Real mass, std::array<Real, NDIM> const& offset, Real sourceScale = Real(1)) {
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireScale(sourceScale);

    // Memoize powers once per particle; every projection term reuses them.
    std::array<std::array<Real, order + 1>, NDIM> powers{};
    Real r2 = 0;
    for (int d = 0; d < NDIM; d++) {
        Real const x = offset[d] / sourceScale;

        Real const square = x * x;
        r2 += square;
        powers[d][0] = Real(1);
        if constexpr (order > 0) {
            powers[d][1] = x;
        }
        if constexpr (order > 1) {
            powers[d][2] = square;
        }
        for (int n = 3; n <= order; n++) {
            powers[d][n] = powers[d][n - 1] * x;
        }
    }

    std::array<Real, order / 2 + 1> radiusPowers{};
    radiusPowers[0] = Real(1);
    for (int k = 1; k <= order / 2; k++) {
        radiusPowers[k] = radiusPowers[k - 1] * r2;
    }
    for (int n = 0; n <= order; n++) {
        for (int nz = 0; nz <= 1 && nz <= n; nz++) {
            for (int nx = 0; nx <= n - nz; nx++) {
                int const ny = n - nx - nz;
                Real value = 0;
                for (int bx = 0; bx <= nx / 2; bx++) {
                    for (int by = 0; by <= ny / 2; by++) {
                        int const k = bx + by;    // nz<=1, so there is no z-index pair.
                        double const sign = k % 2 ? -1 : 1;
                        double const pairs = detail::kCombinatorics.factorial(nx) * detail::kCombinatorics.factorial(ny) /
                            (detail::kCombinatorics.factorial(nx - 2 * bx) * detail::kCombinatorics.factorial(ny - 2 * by) *
                                detail::kCombinatorics.factorial(bx) * detail::kCombinatorics.factorial(by) * detail::kCombinatorics.powerOfTwo(k));
                        Real const factor =
                            Real(sign * pairs * detail::kCombinatorics.oddFactorial(2 * n - 2 * k - 1) / detail::kCombinatorics.oddFactorial(2 * n - 1));
                        value += factor * radiusPowers[k] * powers[0][nx - 2 * bx] * powers[1][ny - 2 * by] * powers[2][nz];
                    }
                }
                moment[index(nx, ny, nz)] += mass * value;
            }
        }
    }
    return r2;
}

/// Add an exterior Newtonian M2L with source rank n and local rank k satisfying
/// n+k<=expansionOrder. sourceOrder may be expansionOrder or expansionOrder-1.
/// separation=target center-source center. Source moments use sourceScale; output local
/// derivatives use targetScale. Both scale arguments are physical lengths.
/// centered=true explicitly omits a known-zero source dipole (default: false).
/// withTrace preserves the appended uniform-background Laplacian unchanged.
/// The highest source rank affects potential only when sourceOrder=expansionOrder.
/// The buffers must be distinct. Squared separation, normalized scales, and
/// intermediate powers must be representable in Real.
/// No omitted force shell is added; this is an exterior Newtonian translation.
template <typename Real, int sourceOrder, int expansionOrder = sourceOrder, bool centered = false, bool withTrace = false>
void addM2L(Expansion<Real, expansionOrder, withTrace>& local, Multipole<Real, sourceOrder> const& source, std::array<Real, NDIM> const& separation,
    Real sourceScale = Real(1), Real targetScale = Real(1)) {
    static_assert(expansionOrder >= kMinOrder && expansionOrder <= kMaxOrder, "Unsupported expansion order");
    static_assert(sourceOrder == expansionOrder || sourceOrder + 1 == expansionOrder, "Source order must equal expansion order or be one less");
    PROFILE();
    FpeGuard fpeGuard;
    detail::requireScale(sourceScale);
    detail::requireScale(targetScale);
    expect(local.data() != source.data());
    Real const r2 = separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2];
    expectPositive(r2);
    Real const inverseRadius = Real(1) / std::sqrt(r2);
    Real const sourceRatio = sourceScale * inverseRadius;
    Real const targetRatio = targetScale * inverseRadius;
    detail::requireScale(sourceRatio);
    detail::requireScale(targetRatio);
    detail::M2lKernel<expansionOrder, centered>::template add<Real, Expansion<Real, expansionOrder, withTrace>, Multipole<Real, sourceOrder>, sourceOrder>(
        local, source, separation[0] * inverseRadius, separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius, sourceRatio, targetRatio);
}

}    // namespace cosmictiger::gravity::stf
