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
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/spatial_types.hpp>

#include <hpx/type_support/bit_cast.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cosmictiger {

namespace detail {

    template <typename Real>
    inline constexpr bool kIsFixed = false;

    template <std::integral Integer, auto dxValue>
    inline constexpr bool kIsFixed<Fixed<Integer, dxValue>> = true;

}    // namespace detail

template <typename Real>
concept BoxReal = std::floating_point<Real> || detail::kIsFixed<Real>;

// A Cartesian product of NDIM half-open intervals [min,max). Touching means
// that the closed bounds meet but the interiors do not overlap. Fixed's upper
// periodic endpoint cannot be stored as a value; fromPeriodicBounds marks an
// upper bound equal to Fixed::min() as that exclusive endpoint instead.
template <BoxReal Real>
class Box {
public:
    using Point = std::array<Real, NDIM>;

    Box() = default;

    Box(Point lower, Point upper) {
        PROFILE();
        FpeGuard const fpeGuard;
        for (size_t axis = 0; axis < NDIM; axis++) {
            minimum_[axis] = lower[axis];
            maximum_[axis] = upper[axis];
            expect(maximum_[axis] >= minimum_[axis]);
        }
    }

    static Box fromPeriodicBounds(Point lower, Point upper)
        requires(detail::kIsFixed<Real>)
    {
        PROFILE();
        std::array<bool, NDIM> periodicUpper{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            periodicUpper[axis] = upper[axis] == Real::min();
        }
        return Box(lower, upper, periodicUpper);
    }

    Real min(size_t axis) const {
        PROFILE();
        expectIndex(axis, NDIM);
        return minimum_[axis];
    }

    Real max(size_t axis) const {
        PROFILE();
        expectIndex(axis, NDIM);
        return maximum_[axis];
    }

    Point const& minCorner() const noexcept {
        PROFILE();
        return minimum_;
    }

    Point const& maxCorner() const noexcept {
        PROFILE();
        return maximum_;
    }

    bool hasPeriodicUpper(size_t axis) const {
        PROFILE();
        expectIndex(axis, NDIM);
        return periodicUpper_[axis];
    }

    bool isEmpty() const noexcept {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (!periodicUpper_[axis] && minimum_[axis] == maximum_[axis]) {
                return true;
            }
        }
        return false;
    }

    bool contains(Point const& point) const noexcept {
        PROFILE();
        if (isEmpty()) {
            return false;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (point[axis] < minimum_[axis] || (!periodicUpper_[axis] && !(point[axis] < maximum_[axis]))) {
                return false;
            }
        }
        return true;
    }

    // Checks unit-period images against this box's unwrapped bounds. Each
    // coordinate may be any floating-point or Fixed format.
    template <BoxReal CoordinateReal>
    bool containsPeriodic(std::array<CoordinateReal, NDIM> const& point) const
        requires(std::floating_point<Real>)
    {
        PROFILE();
        FpeGuard const fpeGuard;
        if (isEmpty()) {
            return false;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if constexpr (detail::kIsFixed<CoordinateReal>) {
                if (!containsPeriodicDyadicAxis(periodicFraction(point[axis]), minimum_[axis], maximum_[axis])) {
                    return false;
                }
            } else {
                using Comparison = std::common_type_t<Real, CoordinateReal>;
                Comparison const value = static_cast<Comparison>(point[axis]);
                if (!containsPeriodicDyadicAxis(periodicFraction(value), static_cast<Comparison>(minimum_[axis]), static_cast<Comparison>(maximum_[axis]))) {
                    return false;
                }
            }
        }
        return true;
    }

    bool containsPeriodic(Point const& point) const
        requires(std::floating_point<Real>)
    {
        PROFILE();
        return containsPeriodic<Real>(point);
    }

    bool containsPeriodic(Position const& point) const
        requires(std::floating_point<Real>)
    {
        PROFILE();
        return containsPeriodic(std::array<Coordinate, NDIM>{point.x, point.y, point.z});
    }

    // A positive-volume overlap in every dimension.
    bool intersects(Box const& other) const noexcept {
        PROFILE();
        if (isEmpty() || other.isEmpty()) {
            return false;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if ((!periodicUpper_[axis] && !(other.minimum_[axis] < maximum_[axis])) ||
                (!other.periodicUpper_[axis] && !(minimum_[axis] < other.maximum_[axis]))) {
                return false;
            }
        }
        return true;
    }

    // A strict gap on at least one axis; boundary contact is not a gap.
    bool isDisjoint(Box const& other) const noexcept {
        PROFILE();
        if (isEmpty() || other.isEmpty()) {
            return true;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if ((!periodicUpper_[axis] && maximum_[axis] < other.minimum_[axis]) || (!other.periodicUpper_[axis] && other.maximum_[axis] < minimum_[axis])) {
                return true;
            }
        }
        return false;
    }

    bool touches(Box const& other) const noexcept {
        PROFILE();
        return !isEmpty() && !other.isEmpty() && !isDisjoint(other) && !intersects(other);
    }

    // Margin is applied to both faces of each axis. Fixed bounds may not wrap
    // past their representable interval; validation builds diagnose wrapping.
    Box expanded(Real margin) const {
        PROFILE();
        return expanded(uniformMargin(margin));
    }

    Box expanded(Point const& margin) const {
        PROFILE();
        FpeGuard const fpeGuard;
        Point lower = minimum_;
        Point upper = maximum_;
        auto upperFlags = periodicUpper_;
        for (size_t axis = 0; axis < NDIM; axis++) {
            expectNonNegative(margin[axis]);
            lower[axis] = lowerDown(minimum_[axis], margin[axis]);
            auto const result = upperUp(maximum_[axis], periodicUpper_[axis], margin[axis]);
            upper[axis] = result.first;
            upperFlags[axis] = result.second;
        }
        return Box(lower, upper, upperFlags);
    }

    Box contracted(Real margin) const {
        PROFILE();
        return contracted(uniformMargin(margin));
    }

    Box contracted(Point const& margin) const {
        PROFILE();
        FpeGuard const fpeGuard;
        Point lower = minimum_;
        Point upper = maximum_;
        auto upperFlags = periodicUpper_;
        for (size_t axis = 0; axis < NDIM; axis++) {
            expectNonNegative(margin[axis]);
            lower[axis] = lowerUp(minimum_[axis], margin[axis]);
            auto const result = upperDown(maximum_[axis], periodicUpper_[axis], margin[axis]);
            upper[axis] = result.first;
            upperFlags[axis] = result.second;
        }
        return Box(lower, upper, upperFlags);
    }

    std::pair<Box, Box> split(size_t axis, Real position) const {
        PROFILE();
        expectIndex(axis, NDIM);
        FpeGuard const fpeGuard;
        expect(!isEmpty());
        expect(minimum_[axis] < position);
        expect(periodicUpper_[axis] || position < maximum_[axis]);
        Box lowerHalf = *this;
        Box upperHalf = *this;
        lowerHalf.maximum_[axis] = position;
        lowerHalf.periodicUpper_[axis] = false;
        upperHalf.minimum_[axis] = position;
        return {lowerHalf, upperHalf};
    }

    bool operator==(Box const&) const = default;

private:
    struct DyadicFraction {
        detail::WideUnsigned numerator;
        int exponent;
        bool negative;
    };

    Point minimum_{};
    Point maximum_{};
    std::array<bool, NDIM> periodicUpper_{};

    Box(Point lower, Point upper, std::array<bool, NDIM> upperFlags) {
        PROFILE();
        FpeGuard const fpeGuard;
        for (size_t axis = 0; axis < NDIM; axis++) {
            minimum_[axis] = lower[axis];
            maximum_[axis] = upper[axis];
            periodicUpper_[axis] = upperFlags[axis];
            if constexpr (detail::kIsFixed<Real>) {
                if (periodicUpper_[axis]) {
                    expectEqual(maximum_[axis], Real::min());
                } else {
                    expect(maximum_[axis] >= minimum_[axis]);
                }
            } else {
                expect(!periodicUpper_[axis]);
                expect(maximum_[axis] >= minimum_[axis]);
            }
        }
    }

    static Point uniformMargin(Real margin) {
        PROFILE();
        Point result{};
        result.fill(margin);
        return result;
    }

    template <std::floating_point Comparison>
    static DyadicFraction periodicFraction(Comparison coordinate) {
        constexpr int kDigits = std::numeric_limits<Comparison>::digits;
        PROFILE();
        Comparison integralPart = 0;
        Comparison const remainder = std::modf(coordinate, &integralPart);
        if (remainder == Comparison(0)) {
            return {detail::WideUnsigned{}, 0, false};
        }
        int exponent = 0;
        Comparison const significand = std::frexp(std::abs(remainder), &exponent);

        static_assert(kDigits <= 2 * detail::WideUnsigned::wordBits());
        detail::WideUnsigned numerator;
        if constexpr (kDigits <= std::numeric_limits<std::uintmax_t>::digits) {
            numerator = detail::WideUnsigned(static_cast<std::uintmax_t>(std::ldexp(significand, kDigits)));
        } else {
            Comparison bits = significand;
            for (int bit = 0; bit < kDigits; bit++) {
                bits *= Comparison(2);
                bool const high = bits >= Comparison(1);
                numerator.shiftLeft(high);
                if (high) {
                    bits -= Comparison(1);
                }
            }
        }
        return {numerator, exponent - kDigits, remainder < Comparison(0)};
    }

    static int bitLength(detail::WideUnsigned const& value) {
        PROFILE();
        for (int bit = 3 * detail::WideUnsigned::wordBits() - 1; bit >= 0; bit--) {
            if (value.bitAt(bit)) {
                return bit + 1;
            }
        }
        return 0;
    }

    template <std::floating_point Comparison>
    static int compareMagnitude(detail::WideUnsigned amount, int exponent, Comparison bound) {
        constexpr int kDigits = std::numeric_limits<Comparison>::digits;
        PROFILE();
        int const amountBits = bitLength(amount);
        if (bound == Comparison(0)) {
            return amountBits == 0 ? 0 : 1;
        }
        if (amountBits == 0) {
            return -1;
        }
        int boundExponent = 0;
        Comparison const fraction = std::frexp(bound, &boundExponent);

        static_assert(kDigits <= 2 * detail::WideUnsigned::wordBits());
        detail::WideUnsigned reference;
        if constexpr (kDigits <= std::numeric_limits<std::uintmax_t>::digits) {
            reference = detail::WideUnsigned(static_cast<std::uintmax_t>(std::ldexp(fraction, kDigits)));
        } else {
            Comparison bits = fraction;
            for (int bit = 0; bit < kDigits; bit++) {
                bits *= Comparison(2);
                bool const high = bits >= Comparison(1);
                reference.shiftLeft(high);
                if (high) {
                    bits -= Comparison(1);
                }
            }
        }
        boundExponent -= kDigits;
        int const amountTop = amountBits + exponent;
        int const boundTop = kDigits + boundExponent;
        if (amountTop != boundTop) {
            return amountTop < boundTop ? -1 : 1;
        }
        if (exponent > boundExponent) {
            for (int shift = 0; shift < exponent - boundExponent; shift++) {
                amount.shiftLeft();
            }
        } else {
            for (int shift = 0; shift < boundExponent - exponent; shift++) {
                reference.shiftLeft();
            }
        }
        if (!amount.greaterEqual(reference)) {
            return -1;
        }
        return reference.greaterEqual(amount) ? 0 : 1;
    }

    template <std::floating_point Comparison>
    static int compareOneMinusMagnitudeToBound(DyadicFraction const& fraction, Comparison bound) {
        PROFILE();
        // With bound >= 1/2, 1-bound is exact by Sterbenz's lemma. Otherwise,
        // a magnitude below 1/2 already settles the comparison. A magnitude
        // above 1/2 has at most two raw words of fractional bits, so its
        // complement can be constructed exactly.
        if (bound >= Comparison(0.5)) {
            return -compareMagnitude(fraction.numerator, fraction.exponent, Comparison(1) - bound);
        }
        if (compareMagnitude(fraction.numerator, fraction.exponent, Comparison(0.5)) < 0) {
            return 1;
        }
        detail::WideUnsigned complement(1);
        for (int bit = 0; bit < -fraction.exponent; bit++) {
            complement.shiftLeft();
        }
        complement.subtract(fraction.numerator);
        return compareMagnitude(complement, fraction.exponent, bound);
    }

    template <std::floating_point Comparison>
    static int compareImageToBound(DyadicFraction const& fraction, Comparison imageInteger, Comparison bound) {
        PROFILE();
        Comparison whole = 0;
        Comparison const remainder = std::modf(bound, &whole);
        Comparison const integerDifference = imageInteger - whole;
        if (integerDifference >= Comparison(2)) {
            return 1;
        }
        if (integerDifference <= Comparison(-2)) {
            return -1;
        }
        bool const negative = fraction.negative && bitLength(fraction.numerator) != 0;
        if (integerDifference == Comparison(1)) {
            if (!negative || remainder <= Comparison(0)) {
                return 1;
            }
            return compareOneMinusMagnitudeToBound(fraction, remainder);
        }
        if (integerDifference == Comparison(-1)) {
            if (negative || remainder >= Comparison(0) || bitLength(fraction.numerator) == 0) {
                return -1;
            }
            return -compareOneMinusMagnitudeToBound(fraction, -remainder);
        }
        if (negative) {
            return remainder >= Comparison(0) ? -1 : -compareMagnitude(fraction.numerator, fraction.exponent, -remainder);
        }
        return remainder < Comparison(0) ? 1 : compareMagnitude(fraction.numerator, fraction.exponent, remainder);
    }

    template <std::integral Integer, auto dxValue>
    static DyadicFraction periodicFraction(Fixed<Integer, dxValue> coordinate) {
        constexpr int kCapacity = 2 * detail::WideUnsigned::wordBits();
        PROFILE();
        bool negative = false;
        if constexpr (std::signed_integral<Integer>) {
            negative = coordinate.raw() < 0;
        }
        std::uintmax_t const magnitude =
            negative ? std::uintmax_t(0) - static_cast<std::uintmax_t>(coordinate.raw()) : static_cast<std::uintmax_t>(coordinate.raw());
        std::uintmax_t dxSignificand = 0;
        int dxExponent = 0;
        if constexpr (std::integral<decltype(dxValue)>) {
            dxSignificand = static_cast<std::uintmax_t>(dxValue);
        } else {
            auto const fraction = std::frexp(dxValue, &dxExponent);
            dxSignificand = static_cast<std::uintmax_t>(std::ldexp(fraction, std::numeric_limits<decltype(dxValue)>::digits));
            dxExponent -= std::numeric_limits<decltype(dxValue)>::digits;
        }
        detail::WideUnsigned product(magnitude);
        product.multiply(dxSignificand);
        int const exponent = dxExponent - Fixed<Integer, dxValue>::fractionalBits();
        detail::WideUnsigned fraction;
        if (exponent < 0) {
            int const retainedBits = -exponent < kCapacity ? -exponent : kCapacity;
            for (int bit = retainedBits - 1; bit >= 0; bit--) {
                fraction.shiftLeft(product.bitAt(bit));
            }
        }
        return {fraction, exponent, negative};
    }

    template <std::floating_point Comparison>
    static bool containsPeriodicDyadicAxis(DyadicFraction const& fraction, Comparison lower, Comparison upper) {
        PROFILE();
        // Any interval wider than two periods contains every periodic image.
        // Past the integer-resolution threshold, distinct floating bounds
        // are at least one full period apart as well.
        Comparison const threshold = std::ldexp(Comparison(1), std::numeric_limits<Comparison>::digits - 1);
        if (lower >= threshold || lower < -threshold) {
            return true;
        }
        // Handle large bounds first so subtracting two finite endpoints cannot
        // overflow for a valid interval spanning the floating-point range.
        if (upper - lower > Comparison(2)) {
            return true;
        }
        Comparison const firstInteger = std::floor(lower);
        for (int offset = 0; offset <= 2; offset++) {
            Comparison const imageInteger = firstInteger + static_cast<Comparison>(offset);
            if (compareImageToBound(fraction, imageInteger, lower) >= 0 && compareImageToBound(fraction, imageInteger, upper) < 0) {
                return true;
            }
        }
        return false;
    }

    static Real lowerDown(Real value, Real margin) {
        PROFILE();
        Real const result = value - margin;
        if constexpr (detail::kIsFixed<Real>) {
            expect(result <= value);
        }
        return result;
    }

    static Real lowerUp(Real value, Real margin) {
        PROFILE();
        Real const result = value + margin;
        if constexpr (detail::kIsFixed<Real>) {
            expect(result >= value);
        }
        return result;
    }

    static std::pair<Real, bool> upperUp(Real value, bool atPeriod, Real margin) {
        PROFILE();
        if (atPeriod) {
            expectEqual(margin, Real{});
            return {value, true};
        }
        Real const result = value + margin;
        if constexpr (detail::kIsFixed<Real>) {
            if (margin != Real{} && result == Real::min() && value != Real::min()) {
                return {result, true};
            }
            expect(result >= value);
        }
        return {result, false};
    }

    static std::pair<Real, bool> upperDown(Real value, bool atPeriod, Real margin) {
        PROFILE();
        if (atPeriod && margin == Real{}) {
            return {value, true};
        }
        Real const result = value - margin;
        if constexpr (detail::kIsFixed<Real>) {
            expect(atPeriod || result <= value);
        }
        return {result, false};
    }
};

}    // namespace cosmictiger
