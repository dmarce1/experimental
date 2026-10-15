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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger::gravity {

// A rank n occupies [n*n,(n+1)*(n+1)): nz=0 with nx increasing, then
// nz=1 with nx increasing. This is the indexing of stf::Multipole/Expansion.
// The entries are spacing^n times the independent STF derivatives of -1/r.
// PMMM uses ranks through sourceOrder+expansionOrder, not the triangular
// n+k<=expansionOrder truncation of the ordinary FMM M2L operator.
template <int maxRank>
using PmmmGreenCoefficients = std::array<double, (maxRank + 1) * (maxRank + 1)>;

template <int neighborWidth = kPmmmNeighborWidth>
constexpr bool pmmmNear(std::array<int, NDIM> const& offset) {
    static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
    PROFILE();
    for (int coordinate : offset) {
        if (coordinate < -neighborWidth || coordinate > neighborWidth) {
            return false;
        }
    }
    return true;
}

namespace detail {
    constexpr size_t pmmmIndex(int nx, int ny, int nz) {
        PROFILE();
        int const rank = nx + ny + nz;
        return static_cast<size_t>(rank * rank + (nz == 0 ? nx : rank + 1 + nx));
    }

    constexpr double pmmmFactorial(int n) {
        PROFILE();
        double result = 1.0;
        for (int i = 2; i <= n; i++) {
            result *= static_cast<double>(i);
        }
        return result;
    }

    constexpr double pmmmOddFactorial(int n) {
        PROFILE();
        double result = 1.0;
        for (int i = n; i > 1; i -= 2) {
            result *= static_cast<double>(i);
        }
        return result;
    }

    template <int maxRank>
    PmmmGreenCoefficients<maxRank> pmmmStfPowers(std::array<double, NDIM> const& vector) {
        PROFILE();
        PmmmGreenCoefficients<maxRank> result{};
        std::array<std::array<double, maxRank + 1>, NDIM> powers{};
        for (int axis = 0; axis < NDIM; axis++) {
            powers[axis][0] = 1.0;
            for (int n = 1; n <= maxRank; n++) {
                powers[axis][n] = powers[axis][n - 1] * vector[axis];
            }
        }
        double const radiusSquared = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
        std::array<double, maxRank / 2 + 1> radiusPowers{};
        radiusPowers[0] = 1.0;
        for (int k = 1; k <= maxRank / 2; k++) {
            radiusPowers[k] = radiusPowers[k - 1] * radiusSquared;
        }
        for (int rank = 0; rank <= maxRank; rank++) {
            double const oddFactorial = pmmmOddFactorial(2 * rank - 1);
            for (int nz = 0; nz <= 1 && nz <= rank; nz++) {
                for (int nx = 0; nx <= rank - nz; nx++) {
                    int const ny = rank - nz - nx;
                    double value = 0.0;
                    for (int bx = 0; bx <= nx / 2; bx++) {
                        for (int by = 0; by <= ny / 2; by++) {
                            int const k = bx + by;
                            double const pairs = pmmmFactorial(nx) * pmmmFactorial(ny) /
                                (pmmmFactorial(nx - 2 * bx) * pmmmFactorial(ny - 2 * by) * pmmmFactorial(bx) * pmmmFactorial(by) * std::ldexp(1.0, k));
                            double const factor = (k % 2 ? -1.0 : 1.0) * pairs * pmmmOddFactorial(2 * rank - 2 * k - 1) / oddFactorial;
                            value += factor * radiusPowers[k] * powers[0][nx - 2 * bx] * powers[1][ny - 2 * by] * powers[2][nz];
                        }
                    }
                    result[pmmmIndex(nx, ny, nz)] = value;
                }
            }
        }
        return result;
    }
}    // namespace detail

// Separation is target center minus source center, in physical length units.
// No near-field mask is applied here. For rank n, the result is the STF part of
// spacing^n * d^n(-1/|separation|), using the same sign and scale convention
// as stf::addM2L. A full PMMM translation uses all ranks through n+k.
template <int maxRank>
PmmmGreenCoefficients<maxRank> pmmmFreeSpaceGreen(std::array<double, NDIM> const& separation, double spacing) {
    PROFILE();
    FpeGuard fpeGuard;
    static_assert(maxRank >= 0 && maxRank <= 18, "PMMM Green rank must be in the supported STF range");
    expectPositive(spacing);
    double radiusSquared = 0.0;
    for (double coordinate : separation) {
        radiusSquared += coordinate * coordinate;
    }
    expectPositive(radiusSquared);
    double const radius = std::sqrt(radiusSquared);
    double const inverseRadius = 1.0 / radius;
    double const ratio = spacing * inverseRadius;

    std::array<double, NDIM> const unit = {separation[0] * inverseRadius, separation[1] * inverseRadius, separation[2] * inverseRadius};
    auto result = detail::pmmmStfPowers<maxRank>(unit);
    double ratioPower = 1.0;
    for (int rank = 0; rank <= maxRank; rank++) {
        double const prefactor = (rank % 2 ? 1.0 : -1.0) * detail::pmmmOddFactorial(2 * rank - 1) * inverseRadius * ratioPower;
        for (size_t index = static_cast<size_t>(rank * rank); index < static_cast<size_t>((rank + 1) * (rank + 1)); index++) {
            result[index] *= prefactor;
        }
        ratioPower *= ratio;
    }
    return result;
}

// The directly evaluated cells are the 3^3 or 5^3 cube, including the target
// cell. For an open mesh, all of their Green coefficients are exactly zero.
template <int maxRank, int neighborWidth = kPmmmNeighborWidth>
PmmmGreenCoefficients<maxRank> pmmmOpenGreen(std::array<int, NDIM> const& offset, double spacing) {
    PROFILE();
    FpeGuard fpeGuard;
    static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
    expectPositive(spacing);
    if (pmmmNear<neighborWidth>(offset)) {
        return {};
    }
    std::array<double, NDIM> const separation = {
        static_cast<double>(offset[0]) * spacing, static_cast<double>(offset[1]) * spacing, static_cast<double>(offset[2]) * spacing};
    return pmmmFreeSpaceGreen<maxRank>(separation, spacing);
}

// Build one local FFT input block. The physical open grid has sideLength^3
// cells and uses a padded (2*sideLength)^3 convolution. Locality IDs are
// [x][y][z] with z fastest, while each local block is [z][y][x] with x
// fastest, matching FftFilterBank. Consecutive coefficient indices occupy
// batch lanes; unused final lanes are zero. The alias plane at sideLength on
// each axis is zero.
template <int maxRank, int neighborWidth = kPmmmNeighborWidth, size_t batchSize = 1>
std::vector<std::array<double, batchSize>> pmmmOpenGreenFilterBlock(
    size_t sideLength, size_t localitySide, size_t localityId, double spacing, size_t firstCoefficient) {
    static_assert(batchSize > 0, "PMMM FFT batch must have at least one lane");
    static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
    PROFILE();
    expectPositive(spacing);
    expectRange(size_t(2), sideLength, static_cast<size_t>(std::numeric_limits<int>::max() / 2));
    expect(std::has_single_bit(sideLength));
    expectRange(size_t(1), localitySide, 2 * sideLength);
    expect(std::has_single_bit(localitySide));
    expect((2 * sideLength) % localitySide == 0);
    expect(localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expect(localitySide * localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expectIndex(localityId, localitySide * localitySide * localitySide);
    expectIndex(firstCoefficient, PmmmGreenCoefficients<maxRank>{}.size());
    size_t const fftSide = 2 * sideLength;
    size_t const blockSide = fftSide / localitySide;
    expect(!(blockSide > std::numeric_limits<size_t>::max() / blockSide || blockSide * blockSide > std::numeric_limits<size_t>::max() / blockSide));
    std::vector<std::array<double, batchSize>> block(blockSide * blockSide * blockSide);
    size_t const blockX = localityId / (localitySide * localitySide);
    size_t const blockY = (localityId / localitySide) % localitySide;
    size_t const blockZ = localityId % localitySide;
    for (size_t z = 0; z < blockSide; z++) {
        int const globalZ = static_cast<int>(blockZ * blockSide + z);
        for (size_t y = 0; y < blockSide; y++) {
            int const globalY = static_cast<int>(blockY * blockSide + y);
            for (size_t x = 0; x < blockSide; x++) {
                int const globalX = static_cast<int>(blockX * blockSide + x);
                if (globalX == static_cast<int>(sideLength) || globalY == static_cast<int>(sideLength) || globalZ == static_cast<int>(sideLength)) {
                    continue;
                }
                std::array<int, NDIM> const offset = {globalX < static_cast<int>(sideLength) ? globalX : globalX - static_cast<int>(fftSide),
                    globalY < static_cast<int>(sideLength) ? globalY : globalY - static_cast<int>(fftSide),
                    globalZ < static_cast<int>(sideLength) ? globalZ : globalZ - static_cast<int>(fftSide)};
                auto const coefficients = pmmmOpenGreen<maxRank, neighborWidth>(offset, spacing);
                auto& values = block[(z * blockSide + y) * blockSide + x];
                for (size_t lane = 0; lane < batchSize && lane < coefficients.size() - firstCoefficient; lane++) {
                    values[lane] = coefficients[firstCoefficient + lane];
                }
            }
        }
    }
    return block;
}

}    // namespace cosmictiger::gravity
