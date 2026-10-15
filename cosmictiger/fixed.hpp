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

#include <array>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__)
#define COSMICTIGER_FIXED_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_FIXED_HOST_DEVICE
#endif

namespace cosmictiger {

namespace detail {

    // Three native words hold two raw values and the binary range significand.
    class WideUnsigned {
    public:
        using Word = std::uintmax_t;
        static constexpr int wordBits() noexcept {
            PROFILE();
            return kWordBits_;
        }

        explicit constexpr WideUnsigned(Word value = 0) noexcept
          : words_{value, 0, 0} {
            PROFILE();
        }

        constexpr bool bitAt(int bit) const noexcept {
            PROFILE();
            return bit >= 0 && bit < 3 * kWordBits_ && ((words_[static_cast<size_t>(bit / kWordBits_)] >> (bit % kWordBits_)) & Word(1)) != 0;
        }

        constexpr void shiftLeft(bool lowBit = false) noexcept {
            PROFILE();
            for (size_t word = 2; word > 0; word--) {
                words_[word] = (words_[word] << 1) | (words_[word - 1] >> (kWordBits_ - 1));
            }
            words_[0] = (words_[0] << 1) | Word(lowBit);
        }

        constexpr void multiply(Word factor) noexcept {
            PROFILE();
            WideUnsigned result;
            WideUnsigned addend = *this;
            while (factor != 0) {
                if ((factor & Word(1)) != 0) {
                    result.add(addend);
                }
                addend.shiftLeft();
                factor >>= 1;
            }
            *this = result;
        }

        constexpr bool greaterEqual(WideUnsigned const& other) const noexcept {
            PROFILE();
            for (size_t index = 3; index > 0; index--) {
                size_t const word = index - 1;
                if (words_[word] != other.words_[word]) {
                    return words_[word] > other.words_[word];
                }
            }
            return true;
        }

        constexpr void subtract(WideUnsigned const& other) noexcept {
            PROFILE();
            Word borrow = 0;
            for (size_t word = 0; word < 3; word++) {
                Word const difference = words_[word] - other.words_[word];
                Word const nextBorrow = Word(words_[word] < other.words_[word] || difference < borrow);
                words_[word] = difference - borrow;
                borrow = nextBorrow;
            }
        }

    private:
        static constexpr int kWordBits_ = std::numeric_limits<Word>::digits;
        std::array<Word, 3> words_;

        constexpr void add(WideUnsigned const& other) noexcept {
            PROFILE();
            Word carry = 0;
            for (size_t word = 0; word < 3; word++) {
                Word const sum = words_[word] + other.words_[word];
                Word const nextSum = sum + carry;
                carry = Word(sum < words_[word] || nextSum < sum);
                words_[word] = nextSum;
            }
        }
    };

}    // namespace detail

// A value is raw() * dx / 2^W, with period dx (default 1).
// Signed ranges are [-dx/2, dx/2); unsigned ranges are [0, dx).
// Conversions and products/quotients truncate toward zero before wrapping.
// Floating output can round to the upper endpoint if its precision is too low.
template <std::integral T, auto dxValue = 1>
    requires(!std::same_as<std::remove_cv_t<T>, bool> && std::same_as<T, std::remove_cv_t<T>>)
class Fixed {
public:
    using Integer = T;
    using Unsigned = std::make_unsigned_t<T>;
    COSMICTIGER_FIXED_HOST_DEVICE static constexpr int fractionalBits() noexcept {
        PROFILE();
        return kFractionalBits_;
    }

    static_assert(std::is_arithmetic_v<decltype(dxValue)>, "Fixed dx must be a real number");
    static_assert(dxValue > 0 && dxValue <= std::numeric_limits<long double>::max(), "Fixed dx must be positive and finite");
    static_assert(std::numeric_limits<decltype(dxValue)>::digits <= std::numeric_limits<std::uintmax_t>::digits, "Fixed dx significand must fit in uintmax_t");
    static constexpr auto dx() noexcept {
        PROFILE();
        return kDx_;
    }

    COSMICTIGER_FIXED_HOST_DEVICE constexpr Fixed() noexcept = default;

    template <std::floating_point F>
    explicit Fixed(F value) {
        PROFILE();
        long double const number = static_cast<long double>(value);
        FpeGuard const fpeGuard;
        if constexpr (std::numeric_limits<F>::digits <= std::numeric_limits<std::uintmax_t>::digits) {
            // Split the input into an exact binary integer and exponent, then
            // quantize with integer division rather than a rounded value / dx.
            int exponent = 0;
            long double const fraction = std::frexp(std::fabs(number), &exponent);
            auto const significand = static_cast<std::uintmax_t>(std::ldexp(fraction, std::numeric_limits<F>::digits));
            Unsigned const result = divideScaled(significand, detail::WideUnsigned(kBinaryScale_.significand),
                kFractionalBits_ + exponent - std::numeric_limits<F>::digits - kBinaryScale_.exponent);
            bits_ = number < 0 ? negateBits(result) : result;
        } else {
            long double const fraction = std::fmod(number, kDx_) / kDx_;
            // A ratio immediately below one can round to one; avoid an
            // out-of-range floating-to-integer cast in that case.
            auto const magnitude =
                std::fabs(fraction) >= 1.0L ? std::numeric_limits<Unsigned>::max() : static_cast<Unsigned>(std::ldexp(std::fabs(fraction), kFractionalBits_));
            bits_ = fraction < 0 ? negateBits(magnitude) : magnitude;
        }
    }

    template <std::integral I>
    explicit constexpr Fixed(I value) noexcept {
        PROFILE();
        bool const negative = value < 0;
        auto const magnitude = negative ? std::uintmax_t(0) - static_cast<std::uintmax_t>(value) : static_cast<std::uintmax_t>(value);
        Unsigned const result = divideScaled(magnitude, detail::WideUnsigned(kBinaryScale_.significand), kFractionalBits_ - kBinaryScale_.exponent);
        bits_ = negative ? negateBits(result) : result;
    }

    COSMICTIGER_FIXED_HOST_DEVICE static constexpr Fixed fromRaw(T value) noexcept {
        PROFILE();
        return fromBits(static_cast<Unsigned>(value));
    }

    COSMICTIGER_FIXED_HOST_DEVICE constexpr T raw() const noexcept {
        PROFILE();
        // C++20 integer conversion preserves the value modulo 2^W.
        return static_cast<T>(bits_);
    }

    static constexpr Fixed min() noexcept {
        PROFILE();
        return fromRaw(std::numeric_limits<T>::min());
    }

    static constexpr Fixed max() noexcept {
        PROFILE();
        return fromRaw(std::numeric_limits<T>::max());
    }

    static constexpr Fixed epsilon() noexcept {
        PROFILE();
        return fromRaw(T(1));
    }

    template <std::floating_point Real>
    Real toReal() const noexcept {
        PROFILE();
        return static_cast<Real>(std::ldexp(static_cast<long double>(raw()), -kFractionalBits_) * kDx_);
    }

    long double toLongDouble() const noexcept {
        PROFILE();
        return toReal<long double>();
    }

    double toDouble() const noexcept {
        PROFILE();
        return toReal<double>();
    }

    float toFloat() const noexcept {
        PROFILE();
        return toReal<float>();
    }

    explicit operator long double() const noexcept {
        PROFILE();
        return toLongDouble();
    }
    explicit operator double() const noexcept {
        PROFILE();
        return toDouble();
    }
    explicit operator float() const noexcept {
        PROFILE();
        return toFloat();
    }

    constexpr bool operator==(Fixed const&) const noexcept = default;

    constexpr std::strong_ordering operator<=>(Fixed const& other) const noexcept {
        PROFILE();
        return raw() <=> other.raw();
    }

    constexpr Fixed operator+() const noexcept {
        PROFILE();
        return *this;
    }

    constexpr Fixed operator-() const noexcept {
        PROFILE();
        return fromBits(negateBits(bits_));
    }

    constexpr Fixed& operator+=(Fixed other) noexcept {
        PROFILE();
        bits_ = static_cast<Unsigned>(std::uintmax_t(bits_) + std::uintmax_t(other.bits_));
        return *this;
    }

    constexpr Fixed& operator-=(Fixed other) noexcept {
        PROFILE();
        bits_ = static_cast<Unsigned>(std::uintmax_t(bits_) - std::uintmax_t(other.bits_));
        return *this;
    }

    constexpr Fixed& operator*=(Fixed other) noexcept {
        PROFILE();
        if constexpr (dxValue == 1) {
            // Compute the upper W bits of the unsigned 2W-bit product without
            // relying on a nonstandard double-width integer or floating point.
            Unsigned low = 0;
            Unsigned high = 0;
            Unsigned addLow = magnitudeBits();
            Unsigned addHigh = 0;
            Unsigned multiplier = other.magnitudeBits();
            for (int bit = 0; bit < kFractionalBits_; bit++) {
                if ((multiplier & 1U) != 0) {
                    Unsigned const nextLow = static_cast<Unsigned>(std::uintmax_t(low) + std::uintmax_t(addLow));
                    high = static_cast<Unsigned>(std::uintmax_t(high) + std::uintmax_t(addHigh) + (nextLow < low ? 1U : 0U));
                    low = nextLow;
                }
                addHigh = static_cast<Unsigned>((std::uintmax_t(addHigh) << 1) | (std::uintmax_t(addLow) >> (kFractionalBits_ - 1)));
                addLow = static_cast<Unsigned>(std::uintmax_t(addLow) << 1);
                multiplier = static_cast<Unsigned>(multiplier >> 1);
            }
            bits_ = isNegative() != other.isNegative() ? negateBits(high) : high;
        } else {
            detail::WideUnsigned product(magnitudeBits());
            product.multiply(other.magnitudeBits());
            product.multiply(kBinaryScale_.significand);
            Unsigned result = 0;
            for (int bit = kFractionalBits_ - 1; bit >= 0; bit--) {
                result = static_cast<Unsigned>((std::uintmax_t(result) << 1) | std::uintmax_t(product.bitAt(bit + kFractionalBits_ - kBinaryScale_.exponent)));
            }
            bits_ = isNegative() != other.isNegative() ? negateBits(result) : result;
        }
        return *this;
    }

    constexpr Fixed& operator/=(Fixed other) {
        PROFILE();
        Unsigned const divisor = other.magnitudeBits();
        expectNonZero(divisor);
        if constexpr (dxValue == 1) {
            // The integer part of a / b wraps away. Generate W fractional bits
            // from the remainder without ever forming a * 2^W or overflowing 2r.
            Unsigned remainder = static_cast<Unsigned>(magnitudeBits() % divisor);
            Unsigned quotient = 0;
            for (int bit = 0; bit < kFractionalBits_; bit++) {
                quotient = static_cast<Unsigned>(std::uintmax_t(quotient) << 1);
                Unsigned const complement = static_cast<Unsigned>(divisor - remainder);
                if (remainder >= complement) {
                    remainder = static_cast<Unsigned>(remainder - complement);
                    quotient = static_cast<Unsigned>(quotient | 1U);
                } else {
                    remainder = static_cast<Unsigned>(std::uintmax_t(remainder) << 1);
                }
            }
            bits_ = isNegative() != other.isNegative() ? negateBits(quotient) : quotient;
        } else {
            detail::WideUnsigned denominator(divisor);
            denominator.multiply(kBinaryScale_.significand);
            Unsigned const quotient = divideScaled(magnitudeBits(), denominator, kFractionalBits_ - kBinaryScale_.exponent);
            bits_ = isNegative() != other.isNegative() ? negateBits(quotient) : quotient;
        }
        return *this;
    }

    friend constexpr Fixed operator+(Fixed left, Fixed right) noexcept {
        PROFILE();
        return left += right;
    }
    friend constexpr Fixed operator-(Fixed left, Fixed right) noexcept {
        PROFILE();
        return left -= right;
    }
    friend constexpr Fixed operator*(Fixed left, Fixed right) noexcept {
        PROFILE();
        return left *= right;
    }
    friend constexpr Fixed operator/(Fixed left, Fixed right) {
        PROFILE();
        return left /= right;
    }

private:
    static constexpr int kFractionalBits_ = std::numeric_limits<Unsigned>::digits;
    static constexpr long double kDx_ = static_cast<long double>(dxValue);

    // dx = significand * 2^exponent, exactly as specified by the template.
    struct BinaryScale {
        std::uintmax_t significand;
        int exponent;
    };

    static constexpr BinaryScale rangeScale() noexcept {
        PROFILE();
        long double significand = kDx_;
        int exponent = 0;
        while (significand >= 2.0L) {
            significand *= 0.5L;
            ++exponent;
        }
        while (significand < 1.0L) {
            significand *= 2.0L;
            --exponent;
        }
        while (significand != static_cast<long double>(static_cast<std::uintmax_t>(significand))) {
            significand *= 2.0L;
            --exponent;
        }
        return {static_cast<std::uintmax_t>(significand), exponent};
    }

    static constexpr BinaryScale kBinaryScale_ = rangeScale();
    Unsigned bits_ = 0;

    // floor(numerator * 2^shift / denominator), retaining only W low bits.
    // Stream shifted numerator bits through long division so even very small
    // dx needs no dynamically allocated or unbounded-width integer.
    static constexpr Unsigned divideScaled(std::uintmax_t numerator, detail::WideUnsigned const& denominator, int shift) noexcept {
        PROFILE();
        if (shift < 0) {
            if (-shift >= detail::WideUnsigned::wordBits()) {
                return 0;
            }
            numerator >>= -shift;
            shift = 0;
        }
        detail::WideUnsigned remainder;
        Unsigned quotient = 0;
        for (int bit = detail::WideUnsigned::wordBits() - 1; bit >= -shift; bit--) {
            bool const inputBit = bit >= 0 && ((numerator >> bit) & std::uintmax_t(1)) != 0;
            remainder.shiftLeft(inputBit);
            quotient = static_cast<Unsigned>(std::uintmax_t(quotient) << 1);
            if (remainder.greaterEqual(denominator)) {
                remainder.subtract(denominator);
                quotient = static_cast<Unsigned>(quotient | 1U);
            }
        }
        return quotient;
    }

    static constexpr Unsigned negateBits(Unsigned value) noexcept {
        PROFILE();
        return static_cast<Unsigned>(std::uintmax_t(0) - std::uintmax_t(value));
    }

    COSMICTIGER_FIXED_HOST_DEVICE static constexpr Fixed fromBits(Unsigned value) noexcept {
        PROFILE();
        Fixed result;
        result.bits_ = value;
        return result;
    }

    constexpr bool isNegative() const noexcept {
        PROFILE();
        if constexpr (std::is_signed_v<T>) {
            return raw() < 0;
        } else {
            return false;
        }
    }

    constexpr Unsigned magnitudeBits() const noexcept {
        PROFILE();
        return isNegative() ? negateBits(bits_) : bits_;
    }
};

}    // namespace cosmictiger

#undef COSMICTIGER_FIXED_HOST_DEVICE
