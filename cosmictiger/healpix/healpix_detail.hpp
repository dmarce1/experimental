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

/*
 *  This file is part of Healpix_cxx.
 *
 *  Healpix_cxx is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpix_cxx is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpix_cxx; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Copyright (C) 2003-2016 Max-Planck-Society
 *  Author: Martin Reinecke
 */

/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2010 Krzysztof M. Gorski, Eric Hivon,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *
 *  This file is part of HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.jpl.nasa.gov
 *
 *----------------------------------------------------------------------------- */
// HEALPix reference equations adapted from Healpix_cxx and the original
// HEALPix/JPL implementation by Gorski, Hivon, Wandelt, Banday, Bartelmann,
// Ansari and Ganga. Scalar/SIMD integration and integer safety are project code.
// https://healpix.sourceforge.io/  https://irsa.ipac.caltech.edu/healpix/

#pragma once

#include <cosmictiger/debug.hpp>
#include <cosmictiger/healpix/healpix.hpp>
#include <cosmictiger/profiler.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>

namespace cosmictiger::healpix::detail {

inline constexpr std::uint64_t kFaceCount = 12;
inline constexpr double kPolarBoundary = 2.0 / 3.0;
inline constexpr double kStablePoleBoundary = 0.99;
inline constexpr double kInverseHalfPi = 2.0 / std::numbers::pi;
inline constexpr double kHalfPi = std::numbers::pi / 2.0;
inline constexpr double kQuarterPi = std::numbers::pi / 4.0;
inline constexpr std::uint32_t kMaximumNside = std::uint32_t{1} << 29;
inline constexpr std::array<int, kFaceCount> kFaceRings{2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
inline constexpr std::array<int, kFaceCount> kFaceLongitudes{1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

struct FaceCoordinates {
    std::int64_t x;
    std::int64_t y;
    int face;
};

inline double longitude(std::array<double, NDIM> const& direction) {
    constexpr double kQuadrants = 4.0;
    PROFILE();
    if (direction[0] == 0.0 && direction[1] == 0.0) {
        return 0.0;
    }
    double value = std::atan2(direction[1], direction[0]) * kInverseHalfPi;
    if (value < 0.0) {
        value += kQuadrants;
    }
    // Adding four can round a tiny negative longitude to exactly four.
    if (value >= kQuadrants) {
        value = 0.0;
    }
    return value;
}

inline double magnitudeScale(std::array<double, NDIM> const& direction) {
    PROFILE();
    double scale = 0.0;
    for (size_t dimension = 0; dimension < NDIM; dimension++) {
        double const component = std::abs(direction[dimension]);
        if (component > scale) {
            scale = component;
        }
    }
    return expectPositive(scale);
}

// Morton bit interleaving with unsigned arithmetic, independent of ISA.
inline std::uint64_t spreadBits(std::uint64_t value) {
    constexpr std::uint64_t kMasks[] = {0x0000ffff0000ffffULL, 0x00ff00ff00ff00ffULL, 0x0f0f0f0f0f0f0f0fULL, 0x3333333333333333ULL, 0x5555555555555555ULL};
    PROFILE();
    value = (value | (value << 16)) & kMasks[0];
    value = (value | (value << 8)) & kMasks[1];
    value = (value | (value << 4)) & kMasks[2];
    value = (value | (value << 2)) & kMasks[3];
    return (value | (value << 1)) & kMasks[4];
}

inline std::uint64_t compactBits(std::uint64_t value) {
    constexpr std::uint64_t kMasks[] = {
        0x5555555555555555ULL, 0x3333333333333333ULL, 0x0f0f0f0f0f0f0f0fULL, 0x00ff00ff00ff00ffULL, 0x0000ffff0000ffffULL, 0x00000000ffffffffULL};
    PROFILE();
    value &= kMasks[0];
    value = (value | (value >> 1)) & kMasks[1];
    value = (value | (value >> 2)) & kMasks[2];
    value = (value | (value >> 4)) & kMasks[3];
    value = (value | (value >> 8)) & kMasks[4];
    return (value | (value >> 16)) & kMasks[5];
}

inline std::uint64_t faceToNested(std::uint32_t nside, FaceCoordinates const& coordinates) {
    PROFILE();
    auto const faceSize = std::uint64_t{nside} * nside;
    return static_cast<std::uint64_t>(coordinates.face) * faceSize + spreadBits(static_cast<std::uint64_t>(coordinates.x)) +
        (spreadBits(static_cast<std::uint64_t>(coordinates.y)) << 1);
}

// Edge coordinates are nonnegative; their integer conversion is floor.
// Keeping the integer finish shared gives scalar and SIMD identical indexing.
inline std::uint64_t finishPixel(
    std::uint32_t nside, Ordering ordering, double z, double longitudeValue, double ascending, double descending, bool equatorial) {
    constexpr std::int64_t kQuadrants = 4;
    constexpr int kSouthernFaces = 8;
    PROFILE();
    auto const side = static_cast<std::int64_t>(nside);
    auto const faceSize = std::uint64_t{nside} * nside;
    auto const count = kFaceCount * faceSize;
    auto const cap = 2 * (faceSize - nside);
    auto const quadrants = kQuadrants * side;
    auto ascendingIndex = static_cast<std::int64_t>(expectNonNegative(ascending));
    auto descendingIndex = static_cast<std::int64_t>(expectNonNegative(descending));
    if (equatorial) {
        if (ordering == Ordering::ring) {
            auto const ring = side + 1 + ascendingIndex - descendingIndex;
            auto const shift = 1 - (ring & 1);
            auto const azimuth = ((ascendingIndex + descendingIndex - side + shift + 1 + 2 * quadrants) / 2) % quadrants;
            return cap + static_cast<std::uint64_t>((ring - 1) * quadrants + azimuth);
        }
        auto const ascendingFace = ascendingIndex / side;
        auto const descendingFace = descendingIndex / side;
        int const face = static_cast<int>(ascendingFace == descendingFace ? (ascendingFace | kQuadrants) :
                                                                            (ascendingFace < descendingFace ? ascendingFace : descendingFace + kSouthernFaces));
        return faceToNested(nside, {descendingIndex % side, side - ascendingIndex % side - 1, face});
    }
    if (ordering == Ordering::ring) {
        auto const ring = ascendingIndex + descendingIndex + 1;
        auto const azimuth = static_cast<std::int64_t>(longitudeValue * static_cast<double>(ring));
        return z >= 0.0 ? static_cast<std::uint64_t>(2 * ring * (ring - 1) + azimuth) : count - static_cast<std::uint64_t>(2 * ring * (ring + 1) - azimuth);
    }
    if (ascendingIndex >= side) {
        ascendingIndex = side - 1;
    }
    if (descendingIndex >= side) {
        descendingIndex = side - 1;
    }
    int const quadrant = static_cast<int>(longitudeValue);
    return z >= 0.0 ? faceToNested(nside, {side - descendingIndex - 1, side - ascendingIndex - 1, quadrant}) :
                      faceToNested(nside, {ascendingIndex, descendingIndex, quadrant + kSouthernFaces});
}

}    // namespace cosmictiger::healpix::detail
