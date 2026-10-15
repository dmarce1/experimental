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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <stddef.h>

namespace cosmictiger::healpix {

enum class Ordering { nested, ring };

// Equal-area HEALPix pixels. Directions need not be normalized, but must be
// finite and nonzero. Pixel indices are zero-based. Polar longitudes are zero
// for an exactly axial vector; points on edges follow the HEALPix convention.
class Healpix {
public:
    explicit Healpix(std::uint32_t nside, Ordering ordering = Ordering::nested);

    constexpr std::uint32_t nside() const noexcept {
        PROFILE();
        return nside_;
    }
    constexpr Ordering ordering() const noexcept {
        PROFILE();
        return ordering_;
    }
    constexpr std::uint64_t pixelCount() const noexcept {
        PROFILE();
        return pixelCount_;
    }

    std::uint64_t vectorToPixel(std::array<double, NDIM> const& direction) const;
    std::array<double, NDIM> pixelToVector(std::uint64_t pixel) const;

    // Explicit native SIMD arithmetic for complete packs; scalar tails and a
    // scalar implementation on targets without an enabled SIMD instruction set.
    // atan2 and final integer indexing are evaluated per lane.
    void vectorsToPixels(std::span<std::array<double, NDIM> const> directions, std::span<std::uint64_t> pixels) const;
    static size_t nativeLaneCount() noexcept;

    // Exact integer conversion, without a floating-point direction round trip.
    // Converting to NESTED requires a power-of-two nside.
    std::uint64_t convertOrdering(std::uint64_t pixel, Ordering target) const;

private:
    std::uint32_t nside_;
    Ordering ordering_;
    std::uint64_t pixelCount_;
};

}    // namespace cosmictiger::healpix
