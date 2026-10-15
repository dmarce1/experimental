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

#include <cosmictiger/healpix/healpix_detail.hpp>
#include <cosmictiger/profiler.hpp>

#include <stdexcept>

#if defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace cosmictiger::healpix {
namespace detail {

#if defined(__AVX__) || defined(__SSE2__)
    // The installed HPX build does not enable HPX_HAVE_DATAPAR. Keep this small
    // arithmetic pack private: no integer lane aliasing or implicit conversion,
    // no approximate reciprocal/square-root, and no dependency on a vector ABI.
    class NativeDouble {
    public:
        static constexpr size_t laneCount() noexcept {
            PROFILE();
#if defined(__AVX__)
            return 4;
#else
            return 2;
#endif
        }
        explicit NativeDouble(double value) {
            PROFILE();
#if defined(__AVX__)
            values_ = _mm256_set1_pd(value);
#else
            values_ = _mm_set1_pd(value);
#endif
        }
        static NativeDouble load(double const* values) {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_loadu_pd(values);
#else
            result.values_ = _mm_loadu_pd(values);
#endif
            return result;
        }
        void store(double* values) const {
            PROFILE();
#if defined(__AVX__)
            _mm256_storeu_pd(values, values_);
#else
            _mm_storeu_pd(values, values_);
#endif
        }
        NativeDouble operator+(NativeDouble other) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_add_pd(values_, other.values_);
#else
            result.values_ = _mm_add_pd(values_, other.values_);
#endif
            return result;
        }
        NativeDouble operator-(NativeDouble other) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_sub_pd(values_, other.values_);
#else
            result.values_ = _mm_sub_pd(values_, other.values_);
#endif
            return result;
        }
        NativeDouble operator*(NativeDouble other) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_mul_pd(values_, other.values_);
#else
            result.values_ = _mm_mul_pd(values_, other.values_);
#endif
            return result;
        }
        NativeDouble operator/(NativeDouble other) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_div_pd(values_, other.values_);
#else
            result.values_ = _mm_div_pd(values_, other.values_);
#endif
            return result;
        }
        NativeDouble squareRoot() const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_sqrt_pd(values_);
#else
            result.values_ = _mm_sqrt_pd(values_);
#endif
            return result;
        }
        NativeDouble absolute() const {
            PROFILE();
            NativeDouble const sign(-0.0);
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_andnot_pd(sign.values_, values_);
#else
            result.values_ = _mm_andnot_pd(sign.values_, values_);
#endif
            return result;
        }
        NativeDouble maximum(NativeDouble other) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            result.values_ = _mm256_max_pd(values_, other.values_);
#else
            result.values_ = _mm_max_pd(values_, other.values_);
#endif
            return result;
        }
        // Pick per lane using this <= boundary. Only bitwise mask operations
        // touch the mask; it is never reinterpreted as arithmetic data.
        NativeDouble selectBelow(NativeDouble boundary, NativeDouble below, NativeDouble above) const {
            PROFILE();
            NativeDouble result(0.0);
#if defined(__AVX__)
            auto const mask = _mm256_cmp_pd(values_, boundary.values_, _CMP_LE_OQ);
            result.values_ = _mm256_or_pd(_mm256_and_pd(mask, below.values_), _mm256_andnot_pd(mask, above.values_));
#else
            auto const mask = _mm_cmple_pd(values_, boundary.values_);
            result.values_ = _mm_or_pd(_mm_and_pd(mask, below.values_), _mm_andnot_pd(mask, above.values_));
#endif
            return result;
        }

    private:
#if defined(__AVX__)
        __m256d values_;
#else
        __m128d values_;
#endif
    };
#endif

}    // namespace detail

size_t Healpix::nativeLaneCount() noexcept {
    PROFILE();
#if defined(__AVX__) || defined(__SSE2__)
    return detail::NativeDouble::laneCount();
#else
    return 1;
#endif
}

void Healpix::vectorsToPixels(std::span<std::array<double, NDIM> const> directions, std::span<std::uint64_t> pixels) const {
#if defined(__AVX__) || defined(__SSE2__)
    constexpr double kHalf = 0.5;
    constexpr double kEquatorialSlope = 0.75;
    constexpr double kPolarFactor = 3.0;
    constexpr size_t kLanes = detail::NativeDouble::laneCount();
    // Scalar uses '< 0.99'; converting that threshold to the immediately lower
    // representable value lets the pack's <= selection express the same test.
    constexpr double kBelowStablePoleBoundary = 0x1.fae147ae147adp-1;
    PROFILE();
#endif
    if (pixels.size() != directions.size()) {
        throw std::invalid_argument("HEALPix batch input and output sizes must match");
    }
    FpeGuard guard;
    size_t first = 0;
#if defined(__AVX__) || defined(__SSE2__)
    using detail::NativeDouble;
    NativeDouble const zero(0.0);
    NativeDouble const one(1.0);
    NativeDouble const half(kHalf);
    NativeDouble const slope(kEquatorialSlope);
    NativeDouble const polarFactor(kPolarFactor);
    NativeDouble const side(nside_);
    NativeDouble const boundary(detail::kPolarBoundary);
    NativeDouble const stableBoundary(kBelowStablePoleBoundary);
    for (; directions.size() - first >= kLanes; first += kLanes) {
        std::array<double, kLanes> xValues;
        std::array<double, kLanes> yValues;
        std::array<double, kLanes> zValues;
        std::array<double, kLanes> longitudeValues;
        std::array<double, kLanes> fractionValues;
        for (size_t lane = 0; lane < kLanes; lane++) {
            auto const& direction = directions[first + lane];
            xValues[lane] = direction[0];
            yValues[lane] = direction[1];
            zValues[lane] = direction[2];
            longitudeValues[lane] = detail::longitude(direction);
            fractionValues[lane] = longitudeValues[lane] - static_cast<int>(longitudeValues[lane]);
        }
        auto const inputX = NativeDouble::load(xValues.data());
        auto const inputY = NativeDouble::load(yValues.data());
        auto const inputZ = NativeDouble::load(zValues.data());
        auto const scale = inputX.absolute().maximum(inputY.absolute()).maximum(inputZ.absolute());
        scale.store(xValues.data());
        for (size_t lane = 0; lane < kLanes; lane++) {
            expectPositive(xValues[lane]);
        }
        auto const x = inputX / scale;
        auto const y = inputY / scale;
        auto const z = inputZ / scale;
        auto const transverseSquared = x * x + y * y;
        auto const inverseLength = one / (transverseSquared + z * z).squareRoot();
        auto const normalizedZ = z * inverseLength;
        auto const absoluteZ = normalizedZ.absolute();
        auto const transverse = transverseSquared.squareRoot() * inverseLength;
        auto const longitude = NativeDouble::load(longitudeValues.data());
        auto const fraction = NativeDouble::load(fractionValues.data());
        auto const middle = side * (half + longitude);
        auto const offset = side * (normalizedZ * slope);
        // Both branches are evaluated; clamp the unused polar square root so
        // inactive lanes cannot raise an arithmetic exception.
        auto const directRadial = side * (polarFactor * (one - absoluteZ).maximum(zero)).squareRoot();
        auto const stableRadial = side * transverse / ((one + absoluteZ) / polarFactor).squareRoot();
        auto const radial = absoluteZ.selectBelow(stableBoundary, directRadial, stableRadial);
        auto const ascending = absoluteZ.selectBelow(boundary, middle - offset, fraction * radial);
        auto const descending = absoluteZ.selectBelow(boundary, middle + offset, (one - fraction) * radial);
        ascending.store(xValues.data());
        descending.store(yValues.data());
        normalizedZ.store(zValues.data());
        for (size_t lane = 0; lane < kLanes; lane++) {
            bool const equatorial = std::abs(zValues[lane]) <= detail::kPolarBoundary;
            pixels[first + lane] = detail::finishPixel(nside_, ordering_, zValues[lane], longitudeValues[lane], xValues[lane], yValues[lane], equatorial);
        }
    }
#endif
    for (; first < directions.size(); first++) {
        pixels[first] = vectorToPixel(directions[first]);
    }
}

}    // namespace cosmictiger::healpix
