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
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <stddef.h>

#if defined(__CUDACC__)
#define COSMICTIGER_SYMMETRIC_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_SYMMETRIC_HOST_DEVICE
#endif

namespace cosmictiger::gravity::symmetric {

/// All symmetric Cartesian components, including every trace, through order.
COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr size_t coefficientCount(int order) {
    constexpr int kCartesianDimensionCount = 3;
    constexpr size_t kDimensionFactorial = 6;
    static_assert(NDIM == kCartesianDimensionCount, "Cartesian tensor indexing requires three spatial dimensions");
    PROFILE();
    expectNonNegative(order);
    return size_t(order + 1) * size_t(order + 2) * size_t(order + NDIM) / kDimensionFactorial;
}

/// Rank-major layout, then increasing nz and increasing nx; ny=rank-nz-nx.
COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr size_t index(int nx, int ny, int nz) {
    constexpr size_t kDimensionFactorial = 6;
    PROFILE();
    expectNonNegative(nx);
    expectNonNegative(ny);
    expectNonNegative(nz);
    int const n = nx + ny + nz;
    return size_t(n) * size_t(n + 1) * size_t(n + 2) / kDimensionFactorial + size_t(nz * (n + 1) - nz * (nz - 1) / 2 + nx);
}

struct Exponent {
    int nx;
    int ny;
    int nz;
    int rank;
    double inverseFactorial;
};

/// Compile-time factorial data also usable with runtime indices on the device.
template <int order>
class Combinatorics {
public:
    static_assert(order >= 0);

    COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr Combinatorics() {
        PROFILE();
        factorials_[0] = 1;
        inverseFactorials_[0] = 1;
        for (int n = 1; n <= order; n++) {
            factorials_[n] = factorials_[n - 1] * double(n);
            inverseFactorials_[n] = inverseFactorials_[n - 1] / double(n);
        }
        for (int n = 0; n <= order; n++) {
            binomials_[n][0] = 1;
            for (int k = 1; k <= n; k++) {
                binomials_[n][k] = binomials_[n][k - 1] * double(n - k + 1) / double(k);
            }
        }
    }

    COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr double factorial(int n) const {
        PROFILE();
        return factorials_[expectIndex(n, order + 1)];
    }

    COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr double inverseFactorial(int n) const {
        PROFILE();
        return inverseFactorials_[expectIndex(n, order + 1)];
    }

    COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr double binomial(int n, int k) const {
        PROFILE();
        return binomials_[expectIndex(n, order + 1)][expectRange(0, k, n)];
    }

private:
    double factorials_[order + 1]{};
    double inverseFactorials_[order + 1]{};
    double binomials_[order + 1][order + 1]{};
};

template <int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE constexpr auto exponents() {
    constexpr Combinatorics<order> kCombinatorics{};
    PROFILE();
    std::array<Exponent, coefficientCount(order)> result{};
    for (int rank = 0; rank <= order; rank++) {
        for (int nz = 0; nz <= rank; nz++) {
            for (int nx = 0; nx <= rank - nz; nx++) {
                int const ny = rank - nz - nx;
                result[index(nx, ny, nz)] = {
                    nx, ny, nz, rank, kCombinatorics.inverseFactorial(nx) * kCombinatorics.inverseFactorial(ny) * kCombinatorics.inverseFactorial(nz)};
            }
        }
    }
    return result;
}

namespace detail {
    template <int order>
    class StorageSize {
    public:
        static_assert(order >= 0);
        COSMICTIGER_SYMMETRIC_HOST_DEVICE static constexpr size_t value() {
            PROFILE();
            return coefficientCount(order);
        }
    };
}    // namespace detail

/// Raw mass moments of (position-center)/sourceScale, without factorials.
template <typename Real, int order>
using Multipole = std::array<Real, detail::StorageSize<order>::value()>;

/// Raw potential derivatives multiplied by targetScale^rank, without factorials.
/// Every trace is stored; no harmonic or uniform-density constraint is imposed.
template <typename Real, int order>
using Expansion = std::array<Real, detail::StorageSize<order>::value()>;

template <typename Real = double>
struct Field {
    Real potential{};
    std::array<Real, NDIM> acceleration{};
};

template <typename Real, int order>
COSMICTIGER_SYMMETRIC_HOST_DEVICE Real component(Expansion<Real, order> const& tensor, int nx, int ny, int nz) {
    PROFILE();
    expectNonNegative(nx);
    expectNonNegative(ny);
    expectNonNegative(nz);
    expectRange(0, nx + ny + nz, order);
    return tensor[index(nx, ny, nz)];
}

}    // namespace cosmictiger::gravity::symmetric
