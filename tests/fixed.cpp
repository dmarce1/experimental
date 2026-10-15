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

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stddef.h>
#include <type_traits>

namespace {

using cosmictiger::Fixed;

template <typename T>
class FixedTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long,
    unsigned long long, wchar_t, char8_t, char16_t, char32_t>;
TYPED_TEST_SUITE(FixedTest, IntegerTypes);

TYPED_TEST(FixedTest, RangeAndResolution) {
    using Number = Fixed<TypeParam>;
    PROFILE();
    EXPECT_EQ(Number().raw(), TypeParam(0));
    EXPECT_EQ(Number::min().raw(), std::numeric_limits<TypeParam>::min());
    EXPECT_EQ(Number::max().raw(), std::numeric_limits<TypeParam>::max());
    EXPECT_EQ(Number::epsilon().toLongDouble(), std::ldexp(1.0L, -Number::fractionalBits()));
    EXPECT_EQ(Number::min().toLongDouble(), std::is_signed_v<TypeParam> ? -0.5L : 0.0L);
    if constexpr (std::numeric_limits<long double>::digits >= Number::fractionalBits()) {
        long double const upper = std::is_signed_v<TypeParam> ? 0.5L : 1.0L;
        EXPECT_EQ(Number::max().toLongDouble(), upper - Number::epsilon().toLongDouble());
        EXPECT_EQ(Number(Number::max().toLongDouble()).raw(), Number::max().raw());
    }
}

TYPED_TEST(FixedTest, WrapsAtBothEnds) {
    using Number = Fixed<TypeParam>;
    constexpr Number kStep = Number::epsilon();

    static_assert(Number::max() + kStep == Number::min());
    static_assert(Number::min() - kStep == Number::max());
    PROFILE();
    Number value = Number::max();
    EXPECT_EQ(&(value += kStep), &value);
    EXPECT_EQ(value, Number::min());
    EXPECT_EQ(&(value -= kStep), &value);
    EXPECT_EQ(value, Number::max());
    EXPECT_EQ(-kStep + kStep, Number());
    EXPECT_EQ(-Number::min(), Number::min());
}

TYPED_TEST(FixedTest, ConvertsPeriodicRealValues) {
    using Number = Fixed<TypeParam>;
    PROFILE();
    Number const quarter(0.25);
    static_assert(std::same_as<decltype(quarter.template toReal<float>()), float>);
    static_assert(std::same_as<decltype(quarter.template toReal<double>()), double>);
    static_assert(std::same_as<decltype(quarter.template toReal<long double>()), long double>);
    static_assert(noexcept(quarter.template toReal<double>()));
    EXPECT_EQ(quarter.template toReal<float>(), 0.25f);
    EXPECT_EQ(quarter.template toReal<double>(), 0.25);
    EXPECT_EQ(quarter.template toReal<long double>(), 0.25L);
    EXPECT_EQ(Number(0.25).toDouble(), 0.25);
    EXPECT_EQ(Number(0.25f).toFloat(), 0.25f);
    EXPECT_EQ(static_cast<long double>(Number(0.25L)), 0.25L);
    EXPECT_EQ(static_cast<double>(Number(0.25)), 0.25);
    EXPECT_EQ(static_cast<float>(Number(0.25f)), 0.25f);
    EXPECT_EQ(Number(1.25), Number(0.25));
    EXPECT_EQ(Number(-0.75), Number(0.25));
    EXPECT_EQ(Number(1.0), Number());
    EXPECT_EQ(Number(-1.0), Number());
    EXPECT_EQ(Number(std::numeric_limits<unsigned long long>::max()), Number());
    EXPECT_EQ(Number(std::numeric_limits<long long>::min()), Number());
    EXPECT_EQ(Number(std::numeric_limits<double>::max()), Number());
    if constexpr (std::is_signed_v<TypeParam>) {
        EXPECT_EQ(Number(0.5).toDouble(), -0.5);
        EXPECT_EQ(Number(-0.25).toDouble(), -0.25);
    } else {
        EXPECT_EQ(Number(0.5).toDouble(), 0.5);
        EXPECT_EQ(Number(-0.25).toDouble(), 0.75);
    }
    long double const step = Number::epsilon().toLongDouble();
    EXPECT_EQ(Number(step * 1.75L).raw(), TypeParam(1));
    EXPECT_EQ(Number(-step * 1.75L), -Number::epsilon());
    EXPECT_EQ(Number(step * 0.75L), Number());
    EXPECT_EQ(Number(-step * 0.75L), Number());
}

TYPED_TEST(FixedTest, OrdersRepresentativesAndPreservesRawValues) {
    using Number = Fixed<TypeParam>;
    PROFILE();
    EXPECT_LT(Number::min(), Number::max());
    EXPECT_LE(Number::min(), Number::min());
    EXPECT_GT(Number(0.25), Number());
    EXPECT_GE(Number(0.25), Number(0.25));
    EXPECT_NE(Number(0.25), Number());
    EXPECT_EQ(+Number(0.25), Number(0.25));
    EXPECT_EQ(Number::fromRaw(std::numeric_limits<TypeParam>::min()).raw(), std::numeric_limits<TypeParam>::min());
    EXPECT_EQ(Number::fromRaw(std::numeric_limits<TypeParam>::max()).raw(), std::numeric_limits<TypeParam>::max());
}

TYPED_TEST(FixedTest, ProductsAndQuotients) {
    using Number = Fixed<TypeParam>;
    PROFILE();
    EXPECT_EQ((Number(0.25) * Number(0.25)).toDouble(), 0.0625);
    EXPECT_EQ(Number(0.25) / Number(0.25), Number());                 // 1 wraps to 0.
    EXPECT_EQ((Number(0.25) / Number(0.125)).raw(), TypeParam(0));    // 2 wraps to 0.
    EXPECT_EQ(Number(0.1875) / Number(0.25), Number(0.75));
    EXPECT_EQ(Number::epsilon() * Number::epsilon(), Number());
    EXPECT_EQ(Number::min() * Number::min(), std::is_signed_v<TypeParam> ? Number(0.25) : Number());
    if constexpr (std::is_signed_v<TypeParam>) {
        EXPECT_EQ(Number(-0.25) * Number(0.25), Number(-0.0625));
        EXPECT_EQ(Number(-0.0625) / Number(0.25), Number(-0.25));
        EXPECT_EQ(Number(-0.0625) / Number(-0.25), Number(0.25));
        EXPECT_EQ(Number::min() / Number::min(), Number());
    }
    Number value(0.25);
    EXPECT_EQ(&(value *= Number(0.25)), &value);
    EXPECT_EQ(value, Number(0.0625));
    EXPECT_EQ(&(value /= Number(0.25)), &value);
    EXPECT_EQ(value, Number(0.25));
}

TYPED_TEST(FixedTest, RejectsZeroDivisors) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation is disabled in Release";
    }
    using Number = Fixed<TypeParam>;

    Number value(0.25);
    EXPECT_THROW(value / Number(), std::runtime_error);
    EXPECT_THROW(value /= Number(), std::runtime_error);
    EXPECT_EQ(value, Number(0.25));
}

template <typename T>
void checkEveryEightBitPair() {
    using Number = Fixed<T>;
    PROFILE();
    for (int a = std::numeric_limits<T>::min(); a <= std::numeric_limits<T>::max(); a++) {
        for (int b = std::numeric_limits<T>::min(); b <= std::numeric_limits<T>::max(); b++) {
            Number const left = Number::fromRaw(static_cast<T>(a));
            Number const right = Number::fromRaw(static_cast<T>(b));
            SCOPED_TRACE(::testing::Message() << "a=" << a << ", b=" << b);
            // Wider signed arithmetic supplies an independent exact reference;
            // convert via unsigned char to obtain the periodic 8-bit result.
            auto const expected = [](int raw) {
                PROFILE();
                return static_cast<T>(static_cast<unsigned char>(raw));
            };
            EXPECT_EQ((left + right).raw(), expected(a + b));
            EXPECT_EQ((left - right).raw(), expected(a - b));
            EXPECT_EQ((left * right).raw(), expected(a * b / 256));
            if (b != 0) {
                EXPECT_EQ((left / right).raw(), expected(a * 256 / b));
            }
        }
    }
}

TEST(FixedExhaustive, SignedEightBitArithmetic) {
    PROFILE();
    checkEveryEightBitPair<std::int8_t>();
}
TEST(FixedExhaustive, UnsignedEightBitArithmetic) {
    PROFILE();
    checkEveryEightBitPair<std::uint8_t>();
}

TEST(FixedWide, ExactUnsigned64BitBoundaries) {
    using Number = Fixed<std::uint64_t>;
    constexpr Number kMaximum = Number::max();
    constexpr Number kStep = Number::epsilon();

    static_assert((kMaximum * kMaximum).raw() == std::numeric_limits<std::uint64_t>::max() - 1);
    static_assert(kMaximum / (kMaximum - kStep) == kStep);
    static_assert(kMaximum / kStep == Number());
    static_assert(kMaximum / Number::fromRaw(3) == Number());
    PROFILE();
    EXPECT_EQ((kMaximum * kMaximum).raw(), std::numeric_limits<std::uint64_t>::max() - 1);
}

TEST(FixedWide, ExactSigned64BitBoundaries) {
    constexpr auto kQuarterRaw = std::int64_t(1) << 62;

    using Number = Fixed<std::int64_t>;
    static_assert((Number::min() * Number::min()).raw() == kQuarterRaw);
    static_assert((Number::max() * Number::max()).raw() == kQuarterRaw - 1);
    static_assert((Number::min() / Number::max()).raw() == -2);
    static_assert((Number::max() / Number::min()).raw() == 2);
    static_assert(Number::min() / -Number::epsilon() == Number());
    PROFILE();
    EXPECT_EQ((Number::min() / Number::max()).raw(), -2);
}

TYPED_TEST(FixedTest, CompileTimeRangeChangesScaleAndPeriod) {
    using Number = Fixed<TypeParam, 8>;
    static_assert(Number::dx() == 8.0L);
    PROFILE();
    Number const scaled(3.5);
    EXPECT_EQ(scaled.template toReal<float>(), 3.5f);
    EXPECT_EQ(scaled.template toReal<double>(), 3.5);
    EXPECT_EQ(scaled.template toReal<long double>(), 3.5L);
    if constexpr (std::is_signed_v<TypeParam>) {
        EXPECT_EQ(Number(-3.5).template toReal<double>(), -3.5);
    }
    EXPECT_EQ(Number(1).toDouble(), 1.0);
    EXPECT_EQ(Number(9), Number(1));
    EXPECT_EQ(Number(-7), Number(1));
    EXPECT_EQ(Number(8), Number());
    EXPECT_EQ(Number::epsilon().toLongDouble(), std::ldexp(8.0L, -Number::fractionalBits()));
    EXPECT_EQ(Number::min().toDouble(), std::is_signed_v<TypeParam> ? -4.0 : 0.0);
    EXPECT_EQ(Number::max() + Number::epsilon(), Number::min());
    EXPECT_EQ(Number::min() - Number::epsilon(), Number::max());
    EXPECT_EQ(Number(3.0) + Number(3.0), Number(6.0));
    EXPECT_EQ(Number(1.0) - Number(3.0), Number(-2.0));
    EXPECT_EQ(Number(2.0) * Number(2.0), Number(4.0));
    EXPECT_EQ(Number(1.0) / Number(2.0), Number(0.5));
    Number value(2);
    EXPECT_EQ(&(value *= Number(2)), &value);
    EXPECT_EQ(value, Number(4));
    value = Number(1);
    EXPECT_EQ(&(value /= Number(2)), &value);
    EXPECT_EQ(value, Number(0.5));
}

template <typename T, auto dxValue, int rangeNumerator, int rangeDenominator>
void checkEveryScaledEightBitPair() {
    using Number = Fixed<T, dxValue>;
    PROFILE();
    for (int a = std::numeric_limits<T>::min(); a <= std::numeric_limits<T>::max(); a++) {
        for (int b = std::numeric_limits<T>::min(); b <= std::numeric_limits<T>::max(); b++) {
            Number const left = Number::fromRaw(static_cast<T>(a));
            Number const right = Number::fromRaw(static_cast<T>(b));
            SCOPED_TRACE(::testing::Message() << "dx=" << Number::dx() << ", a=" << a << ", b=" << b);
            auto const expected = [](int raw) {
                PROFILE();
                return static_cast<T>(static_cast<unsigned char>(raw));
            };
            EXPECT_EQ((left * right).raw(), expected(a * b * rangeNumerator / (256 * rangeDenominator)));
            if (b != 0) {
                EXPECT_EQ((left / right).raw(), expected(a * 256 * rangeDenominator / (b * rangeNumerator)));
            }
        }
    }
}

TEST(FixedRangeExhaustive, SignedRangeTwo) {
    PROFILE();
    checkEveryScaledEightBitPair<std::int8_t, 2, 2, 1>();
}
TEST(FixedRangeExhaustive, UnsignedRangeTwo) {
    PROFILE();
    checkEveryScaledEightBitPair<std::uint8_t, 2, 2, 1>();
}
TEST(FixedRangeExhaustive, SignedRangeHalf) {
    PROFILE();
    checkEveryScaledEightBitPair<std::int8_t, 0.5, 1, 2>();
}
TEST(FixedRangeExhaustive, UnsignedRangeHalf) {
    PROFILE();
    checkEveryScaledEightBitPair<std::uint8_t, 0.5, 1, 2>();
}
TEST(FixedRangeExhaustive, SignedRangeOneAndHalf) {
    PROFILE();
    checkEveryScaledEightBitPair<std::int8_t, 1.5, 3, 2>();
}
TEST(FixedRangeExhaustive, UnsignedRangeOneAndHalf) {
    PROFILE();
    checkEveryScaledEightBitPair<std::uint8_t, 1.5, 3, 2>();
}

TEST(FixedRange, FloatingScaleAndSmallWholeNumberPeriods) {
    using Half = Fixed<std::int64_t, 0.5>;
    using Decimal = Fixed<std::uint64_t, 0.1>;
    PROFILE();
    EXPECT_EQ(Half(1), Half());
    EXPECT_EQ(Half(-1), Half());
    EXPECT_EQ(Half(0.25).toDouble(), -0.25);
    EXPECT_EQ(Half(0.125).toDouble(), 0.125);
    EXPECT_EQ(Decimal(0.1), Decimal());
    EXPECT_EQ(Decimal(0.025).toDouble(), 0.025);
    EXPECT_EQ(Decimal(0.075).toDouble(), 0.075);
}

TEST(FixedRange, Exact64BitFloatingInputQuantization) {
    using Decimal = Fixed<std::uint64_t, 0.1>;
    using Ten = Fixed<std::uint64_t, 10>;
    PROFILE();
    // Independent Fraction-based references, including a large input for
    // which dividing by dx in floating point would lose the periodic residue.
    EXPECT_EQ(Decimal(1.0).raw(), 18446744073709541376ULL);
    EXPECT_EQ(Decimal(-1.0).raw(), 10240ULL);
    EXPECT_EQ(Decimal(9.5).raw(), 18446744073709454336ULL);
    EXPECT_EQ(Decimal(0.025).raw(), 4611686018427387904ULL);
    EXPECT_EQ(Decimal(-0.075).raw(), 4611686018427389184ULL);
    EXPECT_EQ(Decimal(1e300).raw(), 20689803522124798ULL);
    EXPECT_EQ(Ten(1.0).raw(), 1844674407370955161ULL);
    EXPECT_EQ(Ten(-1.0).raw(), 16602069666338596455ULL);
    EXPECT_EQ(Ten(9.5).raw(), 17524406870024074035ULL);
    EXPECT_EQ(Ten(1e300).raw(), 0ULL);
    EXPECT_EQ(Decimal(1).raw(), Decimal(1.0).raw());
    EXPECT_EQ(Decimal(-1).raw(), Decimal(-1.0).raw());
}

TEST(FixedRange, ExtremeBinaryScales) {
    using Small = Fixed<std::uint64_t, 0x1p-100>;
    using Large = Fixed<std::uint64_t, 0x1p100>;
    using Subnormal = Fixed<std::uint64_t, std::numeric_limits<double>::denorm_min()>;
    PROFILE();
    EXPECT_EQ(Small(1), Small());
    EXPECT_EQ((Small::max() * Small::max()).raw(), 0U);
    EXPECT_EQ(Small::epsilon() / Small::epsilon(), Small(1));
    EXPECT_EQ((Large::max() * Large::max()).raw(), std::uint64_t(1) << 36);
    EXPECT_EQ((Large::max() / Large::epsilon()).raw(), (std::uint64_t(1) << 28) - 1);
    EXPECT_EQ(Subnormal(1), Subnormal());
    EXPECT_EQ(Subnormal::max() * Subnormal::max(), Subnormal());
}

// Reference constants computed with Python arbitrary-precision integers and
// Fraction, using the exact binary value of each floating range literal.
struct WideCase {
    std::uint64_t left;
    std::uint64_t right;
    std::uint64_t product;
    std::uint64_t quotient;
};

template <typename T, auto dxValue, size_t count>
void checkWideCases(WideCase const (&cases)[count]) {
    using Number = Fixed<T, dxValue>;
    PROFILE();
    for (auto const& item : cases) {
        SCOPED_TRACE(::testing::Message() << "dx=" << Number::dx() << ", a=" << item.left << ", b=" << item.right);
        Number const left = Number::fromRaw(static_cast<T>(item.left));
        Number const right = Number::fromRaw(static_cast<T>(item.right));
        EXPECT_EQ((left * right).raw(), static_cast<T>(item.product));
        EXPECT_EQ((left / right).raw(), static_cast<T>(item.quotient));
    }
}

TEST(FixedRangeWide, UnsignedTwo) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 18446744073709551612ULL, 9223372036854775808ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 9223372036854775807ULL, 9223372036854775809ULL},
        {18446744073709551615ULL, 3ULL, 5ULL, 9223372036854775808ULL},
        {1ULL, 18446744073709551615ULL, 1ULL, 0ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 18446744073709551610ULL, 9223372036854775807ULL},
        {9223372036854775808ULL, 1ULL, 1ULL, 0ULL},
        {998929593307175453ULL, 5585244288925278951ULL, 604905210780147144ULL, 1649614376933366145ULL},
        {5062234686534833358ULL, 3518121024227711814ULL, 1930915744167047662ULL, 13271537144470786028ULL},
        {13710763633731943340ULL, 12385550241370724592ULL, 18411417337888677321ULL, 10210242697242746869ULL},
        {8780762158464141131ULL, 17442300332176919406ULL, 16605281680209979509ULL, 4643208442251541744ULL},
        {11011539873612845040ULL, 3839891830083058778ULL, 4584345272896374995ULL, 8002836655963038476ULL},
        {1744285867361428364ULL, 17811943639232448934ULL, 3368520909273242598ULL, 903225263854228563ULL},
    };
    PROFILE();

    checkWideCases<std::uint64_t, 2>(kCases);
}

TEST(FixedRangeWide, SignedTwo) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 0ULL, 9223372036854775808ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 9223372036854775809ULL, 9223372036854775807ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 15372286728091293014ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 9223372036854775808ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 0ULL, 0ULL},
        {9223372036854775808ULL, 1ULL, 18446744073709551615ULL, 0ULL},
        {4928226861259480634ULL, 4815022708369045688ULL, 2572760174276848659ULL, 9440219200713560233ULL},
        {2860806639266285051ULL, 4569059535921043258ULL, 1417181894358801833ULL, 5774992370314588955ULL},
        {1361400556466137372ULL, 7762624046321871525ULL, 1145789268184369905ULL, 1617584949179393524ULL},
        {4059515958013504113ULL, 17221403190080986684ULL, 17907430376807784883ULL, 6336745553937397661ULL},
        {4223140651205465179ULL, 15571375147974541056ULL, 17130187900006145886ULL, 4900100705128168705ULL},
        {7133966357093129787ULL, 12755359575301483348ULL, 14044650633838932748ULL, 6885545594098809432ULL},
    };
    PROFILE();

    checkWideCases<std::int64_t, 2>(kCases);
}

TEST(FixedRangeWide, UnsignedHalf) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 9223372036854775807ULL, 0ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 2305843009213693951ULL, 4ULL},
        {18446744073709551615ULL, 3ULL, 1ULL, 0ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 2ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 9223372036854775806ULL, 18446744073709551613ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 0ULL},
        {15740670355304351147ULL, 15249846671956151102ULL, 6506373386897795509ULL, 1187434741713150854ULL},
        {16553952142187060526ULL, 6045214064211728692ULL, 2712461990809011563ULL, 8793808295213215483ULL},
        {16500893457944025987ULL, 16520732162473247711ULL, 7389023238218004625ULL, 18402441012367540311ULL},
        {11323255837994161253ULL, 10905772558479681479ULL, 3347172061833906025ULL, 1412317590699900735ULL},
        {2333230964198489936ULL, 10023955818978012332ULL, 633938542410044833ULL, 8587530759051628080ULL},
        {11237317544398566020ULL, 7939826119316990240ULL, 2418376570780184602ULL, 15322244968499195133ULL},
    };
    PROFILE();

    checkWideCases<std::uint64_t, 0.5>(kCases);
}

TEST(FixedRangeWide, SignedHalf) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 0ULL, 0ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 16140901064495857665ULL, 18446744073709551612ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 6148914691236517206ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 0ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 0ULL, 0ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 0ULL},
        {4019472335906099957ULL, 6574493550800588006ULL, 716278028373000793ULL, 4108963653096697367ULL},
        {11849468414742871716ULL, 5354755204798420074ULL, 17489209344784413263ULL, 9885958025506218745ULL},
        {12814312839679806877ULL, 13142047964709104209ULL, 809853921428687806ULL, 2279356197284529347ULL},
        {6911612976257040959ULL, 14206247471982081427ULL, 17652330945269058343ULL, 13654040139386659163ULL},
        {559148261879126049ULL, 1039799853704797622ULL, 15758940401027906ULL, 1392583368765879158ULL},
        {11262412699829206958ULL, 14565144409166345837ULL, 755870470674700846ULL, 12944770689025371106ULL},
    };
    PROFILE();

    checkWideCases<std::int64_t, 0.5>(kCases);
}

TEST(FixedRangeWide, UnsignedOneAndHalf) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 9223372036854775805ULL, 12297829382473034410ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 6917529027641081855ULL, 12297829382473034412ULL},
        {18446744073709551615ULL, 3ULL, 4ULL, 6148914691236517205ULL},
        {1ULL, 18446744073709551615ULL, 1ULL, 0ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 9223372036854775803ULL, 12297829382473034409ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 6148914691236517205ULL},
        {10950252715511163274ULL, 5855554819535048269ULL, 5213912396184501069ULL, 4550963870387732288ULL},
        {9745793909315898640ULL, 5654950650099591934ULL, 4481439926444956856ULL, 2747448079214192662ULL},
        {16123502794081903411ULL, 10517898653300215969ULL, 13789861856926617921ULL, 405318771299001520ULL},
        {7385868289973931903ULL, 8663270842654723647ULL, 5203013915237132308ULL, 10484509802499007352ULL},
        {18407475628130361989ULL, 5453277835776412802ULL, 8162503782866082707ULL, 4617691266851108741ULL},
        {18146980837415418590ULL, 13099082386803134975ULL, 882585490727796047ULL, 17036954769471244687ULL},
    };
    PROFILE();

    checkWideCases<std::uint64_t, 1.5>(kCases);
}

TEST(FixedRangeWide, SignedOneAndHalf) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 0ULL, 12297829382473034410ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 11529215046068469761ULL, 6148914691236517204ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 14347467612885206813ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 6148914691236517206ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 0ULL, 6148914691236517205ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 12297829382473034411ULL},
        {10414000705205598582ULL, 16147311630955762617ULL, 1501952103204327019ULL, 6067246676670154238ULL},
        {7846426141577509189ULL, 9458749756863053513ULL, 12712103319255330624ULL, 7710866129234835330ULL},
        {12157721800005803441ULL, 7930164769889827396ULL, 14391314345349743622ULL, 8693942568469728491ULL},
        {17805810680472122901ULL, 8527809300070866029ULL, 18002295091569177794ULL, 17522463377669843693ULL},
        {5550629300857341188ULL, 10206425393319692793ULL, 14727473550981241783ULL, 10162999869769410293ULL},
        {3007317443374923003ULL, 10639421801488077684ULL, 16537537519846983382ULL, 13709719582079768085ULL},
    };
    PROFILE();

    checkWideCases<std::int64_t, 1.5>(kCases);
}

TEST(FixedRangeWide, UnsignedDecimal) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 1844674407370955263ULL, 18446744073709541376ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 461168601842738815ULL, 18446744073709541396ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 12297829382476533077ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 9ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 1844674407370955263ULL, 18446744073709541366ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 5242879ULL},
        {12813979821553966594ULL, 11649239136978274636ULL, 809211179171947265ULL, 18443842613208008504ULL},
        {9332266206456414061ULL, 9501849481939610490ULL, 480701572401164364ULL, 15154480511793659545ULL},
        {6015417400918193718ULL, 7841260178246596250ULL, 255700695650547537ULL, 12386862907295270584ULL},
        {12279881792086567070ULL, 13734080282953535959ULL, 914269107457622889ULL, 17361617908759892235ULL},
        {15310585154395958050ULL, 12277126371153410548ULL, 1018987350861125039ULL, 8685123769535979460ULL},
        {12308850795267694364ULL, 7799674763532714283ULL, 520444326285017248ULL, 14411260047335326382ULL},
    };
    PROFILE();

    checkWideCases<std::uint64_t, 0.1>(kCases);
}

TEST(FixedRangeWide, SignedDecimal) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 0ULL, 18446744073709541376ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 17985575471866812801ULL, 10220ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 12297829382473037824ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 10240ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 0ULL, 18446744073709531136ULL},
        {9223372036854775808ULL, 1ULL, 0ULL, 18446744073704308737ULL},
        {11473484317045517005ULL, 2754714730474967147ULL, 18342610025476682108ULL, 12656148045166202020ULL},
        {3592434968377434862ULL, 11574258967050604731ULL, 18312904976044605822ULL, 14254313734569719748ULL},
        {15905995322248203310ULL, 4001419017687570871ULL, 18391630822652131072ULL, 11997405896869990328ULL},
        {1951072993108944395ULL, 16547302475611105282ULL, 18426654082818859306ULL, 13432474980725854729ULL},
        {545883981367851529ULL, 12553339541935688075ULL, 18429304057646223053ULL, 1360216169609486321ULL},
        {15267152379243922168ULL, 11733145160159727693ULL, 115719626510781940ULL, 13577662257402993324ULL},
    };
    PROFILE();

    checkWideCases<std::int64_t, 0.1>(kCases);
}

TEST(FixedRangeWide, UnsignedTen) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 18446744073709551596ULL, 1844674407370955161ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 9223372036854775803ULL, 1844674407370955161ULL},
        {18446744073709551615ULL, 3ULL, 29ULL, 9223372036854775808ULL},
        {1ULL, 18446744073709551615ULL, 9ULL, 0ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 18446744073709551586ULL, 1844674407370955161ULL},
        {9223372036854775808ULL, 1ULL, 5ULL, 14757395258967641292ULL},
        {904028189632002061ULL, 4168693019574690857ULL, 2042970829192959758ULL, 400038490991162237ULL},
        {1073464208788221310ULL, 2356877328319313043ULL, 1371528463964000081ULL, 840176079334776579ULL},
        {4688618297484764039ULL, 11746030750134087400ULL, 11408201860292623978ULL, 736331647965640738ULL},
        {8991331671502896835ULL, 13650866832919386084ULL, 11196968494538996938ULL, 1215020234657009890ULL},
        {5433194932103210379ULL, 11165783169585144394ULL, 14440293491046834925ULL, 897606149903477260ULL},
        {18150444119042028283ULL, 17296131969226771780ULL, 4162440007570495793ULL, 1935788869348589876ULL},
    };
    PROFILE();

    checkWideCases<std::uint64_t, 10>(kCases);
}

TEST(FixedRangeWide, SignedTen) {
    constexpr WideCase kCases[] = {
        {18446744073709551615ULL, 18446744073709551615ULL, 0ULL, 1844674407370955161ULL},
        {9223372036854775808ULL, 9223372036854775807ULL, 9223372036854775813ULL, 16602069666338596455ULL},
        {18446744073709551615ULL, 3ULL, 0ULL, 17831852604585899896ULL},
        {1ULL, 18446744073709551615ULL, 0ULL, 16602069666338596455ULL},
        {18446744073709551614ULL, 18446744073709551615ULL, 0ULL, 3689348814741910323ULL},
        {9223372036854775808ULL, 1ULL, 18446744073709551611ULL, 3689348814741910324ULL},
        {466660331447206954ULL, 13255879955926124368ULL, 17133574465011716719ULL, 18280907259101601647ULL},
        {1539509401102946049ULL, 11508973022689226973ULL, 12656690439817613381ULL, 18037406035420736447ULL},
        {13104593986119115677ULL, 3860690845507750558ULL, 7266239884960670639ULL, 15894214502899917076ULL},
        {14813453321645032604ULL, 2172095486403141366ULL, 14168561207500033751ULL, 15361134575036226827ULL},
        {1908726705398110553ULL, 16351941420545876362ULL, 16279203964378885034ULL, 16765927363459630425ULL},
        {13180395784435911569ULL, 12778204134967438352ULL, 16183075717746777991ULL, 1713791913704087195ULL},
    };
    PROFILE();

    checkWideCases<std::int64_t, 10>(kCases);
}

}    // namespace
