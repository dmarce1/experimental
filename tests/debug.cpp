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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <csignal>
#include <cstdlib>
#include <limits>
#include <stddef.h>

using namespace cosmictiger;

TEST(ValidationPrimitives, PreserveValuesAndConstantExpressions) {
    static_assert(expectPositive(2) == 2);
    static_assert(expectNegative(-2) == -2);
    static_assert(expectNonNegative(0) == 0);
    static_assert(expectNonPositive(0) == 0);
    static_assert(expectNonZero(-1) == -1);
    static_assert(expectRange(0, 2, 2) == 2);
    static_assert(expectIndex(size_t(1), size_t(2)) == 1);
    static_assert(expectEqual(2, 2) == 2);
    static_assert(expect(true));
    PROFILE();
    int value = 0;
    EXPECT_EQ(expectPositive(++value), 1);
    EXPECT_EQ(value, 1);
}

TEST(ValidationPrimitives, ReportsContractsWithSourceLocations) {
    PROFILE();
    if (!validationChecksEnabled()) {
        GTEST_SKIP() << "Validation is disabled in Release";
    }
    EXPECT_THROW(expectPositive(0), std::runtime_error);
    EXPECT_THROW(expectNegative(0), std::runtime_error);
    EXPECT_THROW(expectNonNegative(-1), std::runtime_error);
    EXPECT_THROW(expectNonPositive(1), std::runtime_error);
    EXPECT_THROW(expectNonZero(0), std::runtime_error);
    EXPECT_THROW(expectRange(1, 0, 3), std::runtime_error);
    EXPECT_THROW(expectRange(1, 4, 3), std::runtime_error);
    EXPECT_THROW(expectIndex(2, 2), std::runtime_error);
    EXPECT_THROW(expectIndex(-1, 2), std::runtime_error);
    EXPECT_THROW(expectIndex(size_t(0), -1), std::runtime_error);
    EXPECT_THROW(expectRange(size_t(0), -1, size_t(2)), std::runtime_error);
    EXPECT_THROW(expectRange(0, size_t(1), -1), std::runtime_error);
    EXPECT_THROW(expectEqual(-1, size_t(-1)), std::runtime_error);
    EXPECT_THROW(expectEqual(1, 2), std::runtime_error);
    try {
        expect(false);
        FAIL() << "Expected a diagnostic";
    } catch (std::runtime_error const& error) {
        EXPECT_NE(std::string(error.what()).find("tests/debug.cpp:"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("false"), std::string::npos);
    }
}

TEST(ValidationPrimitives, ReleaseOmitsContractsAndBoundEvaluation) {
    PROFILE();
    if (validationChecksEnabled()) {
        GTEST_SKIP() << "Requires disabled validation";
    }
    int boundEvaluations = 0;
    EXPECT_EQ(expectPositive(-1), -1);
    EXPECT_EQ(expectRange(++boundEvaluations, 7, ++boundEvaluations), 7);
    EXPECT_EQ(expectIndex(7, ++boundEvaluations), 7);
    EXPECT_EQ(expectEqual(7, ++boundEvaluations), 7);
    expect(++boundEvaluations > 0);
    EXPECT_EQ(boundEvaluations, 0);
    EXPECT_FALSE(FpeGuard::supportsHardwareTraps());
}

TEST(FpeGuard, RestoresMasksAndNestedScopes) {
    PROFILE();
    if (!FpeGuard::supportsHardwareTraps()) {
        GTEST_SKIP() << "Hardware traps are unavailable or validation is disabled";
    }
#if defined(__GLIBC__)
    int const before = fegetexcept();
    {
        FpeGuard const outer;
        int const enabled = fegetexcept();
        EXPECT_EQ(enabled & (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW), FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
        {
            FpeGuard const inner;
            EXPECT_EQ(fegetexcept(), enabled);
        }
        EXPECT_EQ(fegetexcept(), enabled);
    }
    EXPECT_EQ(fegetexcept(), before);
#endif
}

TEST(FpeGuard, PreservesPendingMaskedExceptions) {
    PROFILE();
    if (!FpeGuard::supportsHardwareTraps()) {
        GTEST_SKIP() << "Hardware traps are unavailable or validation is disabled";
    }
#if defined(__GLIBC__)
    ASSERT_EQ(fegetexcept() & FE_INVALID, 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    {
        FpeGuard const guard;
        EXPECT_EQ(std::fetestexcept(FE_INVALID), 0);
    }
    EXPECT_NE(std::fetestexcept(FE_INVALID), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
#endif
}

TEST(FpeGuard, ValidationBuildPreservesSubnormalArithmetic) {
    PROFILE();
    if (!validationChecksEnabled()) {
        GTEST_SKIP() << "Validation is disabled in Release";
    }
    FpeGuard const guard;
    double volatile smallestNormal = std::numeric_limits<double>::min();
    double volatile divisor = 2.0;
    double const result = smallestNormal / divisor;
    EXPECT_GT(result, 0.0);
    EXPECT_LT(result, std::numeric_limits<double>::min());
}

// This executable has no HPX runtime: death tests fork only a single-threaded
// process, and cannot alter an application's floating-point environment.
TEST(FpeGuardDeathTest, DivisionByZeroTraps) {
    PROFILE();
    if (!FpeGuard::supportsHardwareTraps()) {
        GTEST_SKIP() << "Hardware traps are unavailable or validation is disabled";
    }
    EXPECT_EXIT(
        {
            FpeGuard const guard;
            double volatile numerator = 1.0;
            double volatile denominator = 0.0;
            double volatile result = numerator / denominator;
            (void) result;
            std::exit(0);
        },
        ::testing::KilledBySignal(SIGFPE), "");
}

TEST(FpeGuardDeathTest, InvalidArithmeticTraps) {
    PROFILE();
    if (!FpeGuard::supportsHardwareTraps()) {
        GTEST_SKIP() << "Hardware traps are unavailable or validation is disabled";
    }
    EXPECT_EXIT(
        {
            FpeGuard const guard;
            double volatile zero = 0.0;
            double volatile result = zero / zero;
            (void) result;
            std::exit(0);
        },
        ::testing::KilledBySignal(SIGFPE), "");
}

TEST(FpeGuardDeathTest, OverflowTraps) {
    PROFILE();
    if (!FpeGuard::supportsHardwareTraps()) {
        GTEST_SKIP() << "Hardware traps are unavailable or validation is disabled";
    }
    EXPECT_EXIT(
        {
            FpeGuard const guard;
            double volatile large = 1.0e300;
            double volatile result = large * large;
            (void) result;
            std::exit(0);
        },
        ::testing::KilledBySignal(SIGFPE), "");
}
