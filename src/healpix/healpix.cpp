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

#include <cosmictiger/healpix/healpix_detail.hpp>
#include <cosmictiger/profiler.hpp>

#include <bit>
#include <stdexcept>

namespace cosmictiger::healpix {
namespace detail {

    std::uint64_t integerSquareRoot(std::uint64_t value) {
        PROFILE();
        auto root = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(value)));
        // Correct double rounding at the largest supported pixel indices using
        // integer division; neither the comparison nor correction can overflow.
        while (root != 0 && root > value / root) {
            root--;
        }
        while (root + 1 <= value / (root + 1)) {
            root++;
        }
        return root;
    }

    FaceCoordinates nestedToFace(std::uint32_t nside, std::uint64_t pixel) {
        PROFILE();
        auto const faceSize = std::uint64_t{nside} * nside;
        auto const local = pixel % faceSize;
        return {static_cast<std::int64_t>(compactBits(local)), static_cast<std::int64_t>(compactBits(local >> 1)), static_cast<int>(pixel / faceSize)};
    }

    FaceCoordinates ringToFace(std::uint32_t nside, std::uint64_t pixel) {
        constexpr std::int64_t kQuadrants = 4;
        constexpr std::int64_t kSouthernFaces = 8;
        PROFILE();
        auto const side = static_cast<std::int64_t>(nside);
        auto const faceSize = std::uint64_t{nside} * nside;
        auto const cap = 2 * (faceSize - nside);
        auto const count = kFaceCount * faceSize;
        std::int64_t ring;
        std::int64_t azimuth;
        std::int64_t shift;
        std::int64_t radius;
        int face;
        if (pixel < cap) {
            ring = static_cast<std::int64_t>((1 + integerSquareRoot(1 + 2 * pixel)) / 2);
            azimuth = static_cast<std::int64_t>(pixel + 1) - 2 * ring * (ring - 1);
            shift = 0;
            radius = ring;
            face = static_cast<int>((azimuth - 1) / radius);
        } else if (pixel < count - cap) {
            auto const local = static_cast<std::int64_t>(pixel - cap);
            auto const row = local / (kQuadrants * side);
            ring = row + side;
            azimuth = local - row * kQuadrants * side + 1;
            shift = (ring + side) & 1;
            radius = side;
            auto const ascending = row + 1;
            auto const descending = 2 * side + 1 - row;
            auto const descendingFace = (azimuth - ascending / 2 + side - 1) / side;
            auto const ascendingFace = (azimuth - descending / 2 + side - 1) / side;
            face = static_cast<int>(ascendingFace == descendingFace ? (ascendingFace | kQuadrants) :
                                                                      (ascendingFace < descendingFace ? ascendingFace : descendingFace + kSouthernFaces));
        } else {
            auto const local = static_cast<std::int64_t>(count - pixel);
            ring = static_cast<std::int64_t>((1 + integerSquareRoot(2 * local - 1)) / 2);
            azimuth = kQuadrants * ring + 1 - (local - 2 * ring * (ring - 1));
            shift = 0;
            radius = ring;
            ring = kQuadrants * side - ring;
            face = static_cast<int>((azimuth - 1) / radius + kSouthernFaces);
        }
        auto const vertical = ring - (2 + (face >> 2)) * side + 1;
        auto horizontal = 2 * azimuth - kFaceLongitudes[face] * radius - shift - 1;
        if (horizontal >= 2 * side) {
            horizontal -= 2 * kQuadrants * side;
        }
        return {(horizontal - vertical) / 2, (-horizontal - vertical) / 2, face};
    }

    std::uint64_t faceToRing(std::uint32_t nside, FaceCoordinates const& coordinates) {
        constexpr std::int64_t kQuadrants = 4;
        constexpr std::int64_t kSouthernBoundary = 3;
        PROFILE();
        auto const side = static_cast<std::int64_t>(nside);
        auto const faceSize = std::uint64_t{nside} * nside;
        auto const count = kFaceCount * faceSize;
        auto const ring = kFaceRings[coordinates.face] * side - coordinates.x - coordinates.y - 1;
        std::int64_t radius;
        std::int64_t shift;
        std::uint64_t preceding;
        if (ring < side) {
            radius = ring;
            shift = 0;
            preceding = static_cast<std::uint64_t>(2 * radius * (radius - 1));
        } else if (ring > kSouthernBoundary * side) {
            radius = kQuadrants * side - ring;
            shift = 0;
            preceding = count - static_cast<std::uint64_t>(2 * radius * (radius + 1));
        } else {
            radius = side;
            shift = (ring - side) & 1;
            preceding = 2 * (faceSize - nside) + static_cast<std::uint64_t>((ring - side) * kQuadrants * side);
        }
        auto azimuth = (kFaceLongitudes[coordinates.face] * radius + coordinates.x - coordinates.y + 1 + shift) / 2;
        if (azimuth < 1) {
            azimuth += kQuadrants * radius;
        }
        if (azimuth > kQuadrants * radius) {
            azimuth -= kQuadrants * radius;
        }
        return preceding + static_cast<std::uint64_t>(azimuth - 1);
    }

}    // namespace detail

Healpix::Healpix(std::uint32_t nside, Ordering ordering)
  : nside_(nside)
  , ordering_(ordering)
  , pixelCount_(detail::kFaceCount * std::uint64_t{nside} * nside) {
    PROFILE();
    if (nside == 0 || nside > detail::kMaximumNside) {
        throw std::invalid_argument("HEALPix nside must be in [1, 2^29]");
    }
    if (ordering != Ordering::nested && ordering != Ordering::ring) {
        throw std::invalid_argument("Unknown HEALPix ordering");
    }
    if (ordering == Ordering::nested && !std::has_single_bit(nside)) {
        throw std::invalid_argument("NESTED HEALPix requires a power-of-two nside");
    }
}

std::uint64_t Healpix::vectorToPixel(std::array<double, NDIM> const& direction) const {
    constexpr double kHalf = 0.5;
    constexpr double kEquatorialSlope = 0.75;
    constexpr double kPolarFactor = 3.0;
    static_assert(NDIM == 3, "HEALPix maps the two-sphere in three spatial dimensions");
    PROFILE();
    FpeGuard guard;
    double const scale = detail::magnitudeScale(direction);
    double const x = direction[0] / scale;
    double const y = direction[1] / scale;
    double const z = direction[2] / scale;
    double const transverseSquared = x * x + y * y;
    double const inverseLength = 1.0 / std::sqrt(transverseSquared + z * z);
    double const normalizedZ = z * inverseLength;
    double const absoluteZ = std::abs(normalizedZ);
    double const longitude = detail::longitude(direction);
    if (absoluteZ <= detail::kPolarBoundary) {
        double const middle = nside_ * (kHalf + longitude);
        double const offset = nside_ * (normalizedZ * kEquatorialSlope);
        return detail::finishPixel(nside_, ordering_, normalizedZ, longitude, middle - offset, middle + offset, true);
    }
    double const transverse = std::sqrt(transverseSquared) * inverseLength;
    double const radial = absoluteZ < detail::kStablePoleBoundary ? nside_ * std::sqrt(kPolarFactor * (1.0 - absoluteZ)) :
                                                                    nside_ * transverse / std::sqrt((1.0 + absoluteZ) / kPolarFactor);
    double const fraction = longitude - static_cast<int>(longitude);
    return detail::finishPixel(nside_, ordering_, normalizedZ, longitude, fraction * radial, (1.0 - fraction) * radial, false);
}

std::uint64_t Healpix::convertOrdering(std::uint64_t pixel, Ordering target) const {
    PROFILE();
    if (pixel >= pixelCount_) {
        throw std::out_of_range("HEALPix pixel exceeds map size");
    }
    if (target != Ordering::nested && target != Ordering::ring) {
        throw std::invalid_argument("Unknown HEALPix ordering");
    }
    if (target == ordering_) {
        return pixel;
    }
    if (!std::has_single_bit(nside_)) {
        throw std::invalid_argument("NESTED HEALPix requires a power-of-two nside");
    }
    if (target == Ordering::nested) {
        return detail::faceToNested(nside_, detail::ringToFace(nside_, pixel));
    }
    return detail::faceToRing(nside_, detail::nestedToFace(nside_, pixel));
}

std::array<double, NDIM> Healpix::pixelToVector(std::uint64_t pixel) const {
    constexpr std::int64_t kQuadrants = 4;
    constexpr double kPolarFactor = 3.0;
    constexpr double kHalf = 0.5;
    PROFILE();
    if (pixel >= pixelCount_) {
        throw std::out_of_range("HEALPix pixel exceeds map size");
    }
    FpeGuard guard;
    if (ordering_ == Ordering::nested) {
        pixel = detail::faceToRing(nside_, detail::nestedToFace(nside_, pixel));
    }
    auto const side = static_cast<std::int64_t>(nside_);
    auto const cap = std::uint64_t{2} * nside_ * (nside_ - 1);
    double z;
    double transverse;
    double phi;
    if (pixel < cap || pixel >= pixelCount_ - cap) {
        bool const north = pixel < cap;
        auto const distance = north ? pixel : pixelCount_ - pixel - 1;
        auto const ring = static_cast<std::int64_t>((1 + detail::integerSquareRoot(1 + 2 * distance)) / 2);
        auto const within = static_cast<std::int64_t>(distance + 1) - 2 * ring * (ring - 1);
        auto const azimuth = north ? within : kQuadrants * ring + 1 - within;
        double const ratio = static_cast<double>(ring) / nside_;
        double const polarOffset = ratio * ratio / kPolarFactor;
        z = north ? 1.0 - polarOffset : polarOffset - 1.0;
        // Unlike sqrt(1-z*z), this preserves the polar radius at nside=2^29.
        transverse = std::sqrt(polarOffset * (2.0 - polarOffset));
        phi = (static_cast<double>(azimuth) - kHalf) * detail::kHalfPi / static_cast<double>(ring);
    } else {
        auto const local = static_cast<std::int64_t>(pixel - cap);
        auto const row = local / (kQuadrants * side);
        auto const ring = row + side;
        auto const azimuth = local - row * kQuadrants * side + 1;
        double const offset = ((ring + side) & 1) != 0 ? 1.0 : kHalf;
        z = static_cast<double>(2 * side - ring) * (2.0 / (kPolarFactor * nside_));
        transverse = std::sqrt((1.0 - z) * (1.0 + z));
        phi = (static_cast<double>(azimuth) - offset) * detail::kHalfPi / nside_;
    }
    return {transverse * std::cos(phi), transverse * std::sin(phi), z};
}

}    // namespace cosmictiger::healpix
