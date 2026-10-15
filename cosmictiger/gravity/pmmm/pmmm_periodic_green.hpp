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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_progress.hpp>
#include <cosmictiger/parallel.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger::gravity {

namespace detail {
    constexpr double kPmmmPi = 3.141592653589793238462643383279502884;

    // Q(rank+1/2,x) = Gamma(rank+1/2,x)/Gamma(rank+1/2).
    template <int maxRank>
    std::array<double, maxRank + 1> pmmmUpperGammaRatios(double x) {
        PROFILE();
        std::array<double, maxRank + 1> ratios{};
        ratios[0] = std::erfc(std::sqrt(x));
        double term = (2.0 / std::sqrt(kPmmmPi)) * std::sqrt(x) * std::exp(-x);
        for (int rank = 1; rank <= maxRank; rank++) {
            ratios[rank] = ratios[rank - 1] + term;
            if (ratios[rank] > 1.0) {
                ratios[rank] = 1.0;
            }
            term *= x / (static_cast<double>(rank) + 0.5);
        }
        return ratios;
    }

    // Evaluate P(a,x)=gamma(a,x)/Gamma(a) from its positive series when x is
    // small. This avoids cancellation in 1-Q(a,x) for a high harmonic rank.
    template <int maxRank>
    std::array<double, maxRank + 1> pmmmLowerGammaRatios(double x) {
        PROFILE();
        std::array<double, maxRank + 1> ratios{};
        if (x == 0.0) {
            return ratios;
        }
        if (x > 32.0) {
            auto const upper = pmmmUpperGammaRatios<maxRank>(x);
            for (int rank = 0; rank <= maxRank; rank++) {
                ratios[rank] = 1.0 - upper[rank];
            }
            return ratios;
        }
        double leading = (2.0 / std::sqrt(kPmmmPi)) * std::sqrt(x) * std::exp(-x);
        for (int rank = 0; rank <= maxRank; rank++) {
            double term = 1.0;
            double sum = 1.0;
            for (int j = 1; j <= 256; j++) {
                term *= x / (static_cast<double>(rank) + static_cast<double>(j) + 0.5);
                sum += term;
                if (term <= std::numeric_limits<double>::epsilon() * sum) {
                    break;
                }
            }
            ratios[rank] = leading * sum;
            if (ratios[rank] > 1.0) {
                ratios[rank] = 1.0;
            }
            leading *= x / (static_cast<double>(rank) + 1.5);
        }
        return ratios;
    }

    // The odd-rank Fourier derivatives are imaginary before pairing +/-k.
    inline double pmmmFourierPhase(int rank, double cosine, double sine) {
        PROFILE();
        switch (rank % 4) {
        case 0:
            return cosine;
        case 1:
            return -sine;
        case 2:
            return -cosine;
        default:
            return sine;
        }
    }
}    // namespace detail

// Normalized local Laplacian from one unit source mass in a cubic periodic
// gravity box. The STF Green coefficients below exclude this trace. Use
// pmmmAddPeriodicBackground to add the complete quadratic polynomial.
inline double pmmmPeriodicBackgroundLaplacian(size_t sideLength, double spacing) {
    PROFILE();
    FpeGuard fpeGuard;
    expect(!(sideLength == 0 || !(spacing > 0.0)));
    double const period = static_cast<double>(sideLength) * spacing;
    expectPositive(period);
    double const inverseSide = 1.0 / static_cast<double>(sideLength);
    double const laplacian = -4.0 * detail::kPmmmPi * inverseSide * inverseSide / period;

    return laplacian;
}

// The periodic Ewald sum of the harmonic derivatives of the gravitational
// Green function -1/r. Indices and spacing^rank scaling match
// pmmmFreeSpaceGreen. The displacement is reduced to its minimum-image cell
// offset. A directly evaluated cell removes only that primary-image term;
// its periodic mirror contributions remain in this filter. The origin keeps
// the finite regular limit after subtracting the singular primary term.
//
// These coefficients are STF. For periodic gravity, the uniform-background
// Laplacian and its nonharmonic quadratic correction must be supplied
// separately (the optional STF local trace supports that). In particular,
// the raw source second-moment trace contributes an extra constant potential.
// The rank-zero constant fixes the Ewald potential to the zero-mean gravity
// convention.
template <int maxRank, int neighborWidth = kPmmmNeighborWidth>
PmmmGreenCoefficients<maxRank> pmmmPeriodicGreen(
    std::array<int, NDIM> const& displacement, size_t sideLength, double spacing, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5) {
    PROFILE();
    FpeGuard fpeGuard;
    static_assert(maxRank >= 0 && maxRank <= 18, "PMMM Green rank must be in the supported STF range");
    static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
    expect(!(sideLength < static_cast<size_t>(2 * neighborWidth + 1) || sideLength > static_cast<size_t>(std::numeric_limits<int>::max()) || !(spacing > 0.0) ||
        imageCutoff < 0 || waveCutoff < 0 || !(alphaBox > 0.0)));
    double const period = static_cast<double>(sideLength) * spacing;
    expectPositive(period);

    std::array<int, NDIM> offset{};
    int64_t const integerSide = static_cast<int64_t>(sideLength);
    for (int axis = 0; axis < NDIM; axis++) {
        int64_t wrapped = static_cast<int64_t>(displacement[axis]) % integerSide;
        if (wrapped < 0) {
            wrapped += integerSide;
        }
        if (wrapped > integerSide / 2) {
            wrapped -= integerSide;
        }
        offset[axis] = static_cast<int>(wrapped);
    }
    bool const near = pmmmNear<neighborWidth>(offset);
    bool const origin = offset[0] == 0 && offset[1] == 0 && offset[2] == 0;

    PmmmGreenCoefficients<maxRank> result{};
    for (int imageZ = -imageCutoff; imageZ <= imageCutoff; imageZ++) {
        for (int imageY = -imageCutoff; imageY <= imageCutoff; imageY++) {
            for (int imageX = -imageCutoff; imageX <= imageCutoff; imageX++) {
                bool const primary = imageX == 0 && imageY == 0 && imageZ == 0;
                if (primary && origin) {
                    continue;
                }
                std::array<double, NDIM> const cellSeparation = {static_cast<double>(offset[0]) + static_cast<double>(imageX) * sideLength,
                    static_cast<double>(offset[1]) + static_cast<double>(imageY) * sideLength,
                    static_cast<double>(offset[2]) + static_cast<double>(imageZ) * sideLength};
                double const radiusCells = std::hypot(cellSeparation[0], cellSeparation[1], cellSeparation[2]);
                double const alphaRadius = alphaBox * (radiusCells / static_cast<double>(sideLength));
                double const gammaArgument = alphaRadius * alphaRadius;
                expectPositive(radiusCells);
                bool const subtractPrimary = primary && near;
                auto const gammaRatios =
                    subtractPrimary ? detail::pmmmLowerGammaRatios<maxRank>(gammaArgument) : detail::pmmmUpperGammaRatios<maxRank>(gammaArgument);
                std::array<double, NDIM> const physicalSeparation = {cellSeparation[0] * spacing, cellSeparation[1] * spacing, cellSeparation[2] * spacing};
                auto const freeSpace = pmmmFreeSpaceGreen<maxRank>(physicalSeparation, spacing);
                for (int rank = 0; rank <= maxRank; rank++) {
                    double const weight = subtractPrimary ? -gammaRatios[rank] : gammaRatios[rank];
                    for (size_t index = static_cast<size_t>(rank * rank); index < static_cast<size_t>((rank + 1) * (rank + 1)); index++) {
                        result[index] += weight * freeSpace[index];
                    }
                }
            }
        }
    }
    if (origin) {
        result[0] += (2.0 * alphaBox) / (std::sqrt(detail::kPmmmPi) * period);
    }
    result[0] += detail::kPmmmPi / (period * alphaBox * alphaBox);

    double const waveRatio = 2.0 * detail::kPmmmPi / static_cast<double>(sideLength);
    double rankScale = -2.0 / (detail::kPmmmPi * period);    // Pair k with -k.
    std::array<double, maxRank + 1> reciprocalScales{};
    for (int rank = 0; rank <= maxRank; rank++) {
        reciprocalScales[rank] = rankScale;
        rankScale *= waveRatio;
    }
    for (int waveZ = -waveCutoff; waveZ <= waveCutoff; waveZ++) {
        for (int waveY = -waveCutoff; waveY <= waveCutoff; waveY++) {
            for (int waveX = -waveCutoff; waveX <= waveCutoff; waveX++) {
                // One representative of each +/- wave-number pair.
                if (waveX < 0 || (waveX == 0 && waveY < 0) || (waveX == 0 && waveY == 0 && waveZ <= 0)) {
                    continue;
                }
                std::array<double, NDIM> const wave = {static_cast<double>(waveX), static_cast<double>(waveY), static_cast<double>(waveZ)};
                double const waveSquared = wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2];
                double const exponent = -detail::kPmmmPi * detail::kPmmmPi * waveSquared / (alphaBox * alphaBox);
                double const damping = std::exp(exponent) / waveSquared;
                double const phase = waveRatio * (wave[0] * offset[0] + wave[1] * offset[1] + wave[2] * offset[2]);
                double const cosine = std::cos(phase);
                double const sine = std::sin(phase);
                auto const stfWave = detail::pmmmStfPowers<maxRank>(wave);
                for (int rank = 0; rank <= maxRank; rank++) {
                    double const factor = reciprocalScales[rank] * damping * detail::pmmmFourierPhase(rank, cosine, sine);
                    for (size_t index = static_cast<size_t>(rank * rank); index < static_cast<size_t>((rank + 1) * (rank + 1)); index++) {
                        result[index] += factor * stfWave[index];
                    }
                }
            }
        }
    }
    return result;
}

// Complete the nonharmonic part omitted by the STF Green filters. The sums
// are over every source cell: totalDipole is sum(m*s/spacing), and
// totalSecondMomentTrace is sum(m*|s/spacing|^2), with s relative to each
// source cell center. The local coordinates have the same spacing scale.
// This adds (b/6) sum(m*|t-s|^2), b=-4*pi*spacing^2/volume, including the
// source trace and the cross term. Locals with a trace store xx and yy explicitly,
// so both receive b*mass/3 in addition to the appended Laplacian b*mass.
template <typename Real, int expansionOrder>
void pmmmAddPeriodicBackground(std::array<Real, (expansionOrder + 1) * (expansionOrder + 1) + 1>& local, Real totalMass,
    std::array<Real, NDIM> const& totalDipole, Real totalSecondMomentTrace, size_t sideLength, double spacing) {
    PROFILE();
    FpeGuard fpeGuard;
    static_assert(expansionOrder >= 2, "PMMM periodic background requires a quadratic local expansion");
    double const background = pmmmPeriodicBackgroundLaplacian(sideLength, spacing);

    local[0] += static_cast<Real>(background * static_cast<double>(totalSecondMomentTrace) / (2.0 * NDIM));
    local[detail::pmmmIndex(0, 1, 0)] -= static_cast<Real>(background * static_cast<double>(totalDipole[1]) / static_cast<double>(NDIM));
    local[detail::pmmmIndex(1, 0, 0)] -= static_cast<Real>(background * static_cast<double>(totalDipole[0]) / static_cast<double>(NDIM));
    local[detail::pmmmIndex(0, 0, 1)] -= static_cast<Real>(background * static_cast<double>(totalDipole[2]) / static_cast<double>(NDIM));
    local[detail::pmmmIndex(0, 2, 0)] += static_cast<Real>(background * static_cast<double>(totalMass) / static_cast<double>(NDIM));
    local[detail::pmmmIndex(2, 0, 0)] += static_cast<Real>(background * static_cast<double>(totalMass) / static_cast<double>(NDIM));
    local[(expansionOrder + 1) * (expansionOrder + 1)] += static_cast<Real>(background * static_cast<double>(totalMass));
}

// Build one local FFT input block on the unpadded periodic mesh. Locality IDs
// are [x][y][z] with z fastest, while local blocks are [z][y][x] with x
// fastest, matching FftFilterBank. Consecutive coefficients occupy batch
// lanes; unused final lanes remain zero.
template <int maxRank, int neighborWidth = kPmmmNeighborWidth, size_t batchSize = 1>
std::vector<std::array<double, batchSize>> pmmmPeriodicGreenFilterBlock(size_t sideLength, size_t localitySide, size_t localityId, double spacing,
    size_t firstCoefficient, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5, PmmmProgressObserver const& observer = {}) {
    constexpr size_t kEwaldChunkSize = 4;
    static_assert(batchSize > 0, "PMMM FFT batch must have at least one lane");
    static_assert(neighborWidth == 1 || neighborWidth == 2, "PMMM neighbor width must be one or two cells");
    PROFILE();
    expectRange(static_cast<size_t>(2 * neighborWidth + 1), sideLength, static_cast<size_t>(std::numeric_limits<int>::max()));
    expect(std::has_single_bit(sideLength));
    expectRange(size_t(1), localitySide, sideLength);
    expect(std::has_single_bit(localitySide));
    expect(sideLength % localitySide == 0);
    expect(localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expect(localitySide * localitySide <= std::numeric_limits<size_t>::max() / localitySide);
    expectIndex(localityId, localitySide * localitySide * localitySide);
    expectIndex(firstCoefficient, PmmmGreenCoefficients<maxRank>{}.size());
    expectPositive(spacing);
    expectNonNegative(imageCutoff);
    expectNonNegative(waveCutoff);
    expectPositive(alphaBox);
    size_t const blockSide = sideLength / localitySide;
    expect(!(blockSide > std::numeric_limits<size_t>::max() / blockSide || blockSide * blockSide > std::numeric_limits<size_t>::max() / blockSide));
    std::vector<std::array<double, batchSize>> block(blockSide * blockSide * blockSide);
    size_t const blockX = localityId / (localitySide * localitySide);
    size_t const blockY = (localityId / localitySide) % localitySide;
    size_t const blockZ = localityId % localitySide;
    // Each cell contains an expensive Ewald sum, so four cells already provide
    // useful task granularity. Keep each sum and its FpeGuard inside one task.
    detail::PmmmProgressCounter progress(observer, PmmmProgressStage::ewaldFilters, block.size());
    ::cosmictiger::detail::ParallelChunkPlan const chunks(block.size(), kEwaldChunkSize);
    ::cosmictiger::detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t index = begin; index < end; index++) {
                    size_t const x = index % blockSide;
                    size_t const y = (index / blockSide) % blockSide;
                    size_t const z = index / (blockSide * blockSide);
                    std::array<int, NDIM> const offset = {
                        static_cast<int>(blockX * blockSide + x), static_cast<int>(blockY * blockSide + y), static_cast<int>(blockZ * blockSide + z)};
                    auto const coefficients = pmmmPeriodicGreen<maxRank, neighborWidth>(offset, sideLength, spacing, imageCutoff, waveCutoff, alphaBox);
                    auto& values = block[index];
                    for (size_t lane = 0; lane < batchSize && lane < coefficients.size() - firstCoefficient; lane++) {
                        values[lane] = coefficients[firstCoefficient + lane];
                    }
                    progress.advance(1);
                }
            },
            "pmmmPeriodicGreen.parallelForChunks"));
    progress.finish();
    return block;
}

}    // namespace cosmictiger::gravity
