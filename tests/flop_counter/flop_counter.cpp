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

#include <cosmictiger/flop_counter.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/debug.hpp>
#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using cosmictiger::FlopCounter;

template <typename Count>
concept ValidCounterCount = requires { typename FlopCounter<Count>; };

static_assert(ValidCounterCount<std::uint64_t>);
static_assert(ValidCounterCount<int>);
static_assert(!ValidCounterCount<bool>);
static_assert(!ValidCounterCount<double>);
static_assert(!ValidCounterCount<int const>);
static_assert(!ValidCounterCount<int volatile>);
static_assert(std::is_empty_v<FlopCounter<std::uint64_t, false>>);
static_assert(std::is_trivially_copyable_v<FlopCounter<>>);
static_assert(std::same_as<decltype(FlopCounter<>{}.read()), std::uint64_t>);
static_assert(FlopCounter<>::countingEnabled());
static_assert(!FlopCounter<std::uint64_t, false>::countingEnabled());
static_assert(FlopCounter<>{}.read() == 0);
static_assert(FlopCounter<std::uint64_t, false>{}.read() == 0);
static_assert(static_cast<std::uint64_t>(FlopCounter<>{7}) == 7);

constexpr std::uint64_t constexprEnabledCount() {
    PROFILE();
    FlopCounter<> counter;
    (counter = 9) += 7;
    counter -= FlopCounter<>{3};
    auto copy = counter;
    counter.reset();
    copy += copy;
    return copy.read() + counter.read();
}

constexpr std::uint64_t constexprDisabledCount() {
    PROFILE();
    FlopCounter<std::uint64_t, false> counter{123};
    (counter = 9) += 7;
    counter -= FlopCounter<std::uint64_t, false>{3};
    auto copy = counter;
    counter.reset();
    copy += copy;
    return copy.read() + counter.read();
}

static_assert(constexprEnabledCount() == 26);
static_assert(constexprDisabledCount() == 0);

TEST(FlopCounter, StartsAtZeroAndReturnsItsReferenceFromMutatingOperators) {
    PROFILE();
    FlopCounter<> counter;
    EXPECT_EQ(counter.read(), 0);
    EXPECT_EQ(&(counter = 11), &counter);
    EXPECT_EQ(&(counter += 7), &counter);
    EXPECT_EQ(&(counter -= 3), &counter);
    EXPECT_EQ(counter.read(), 15);

    (counter = 19) += 5;
    EXPECT_EQ(counter.read(), 24);
    (counter += 8) -= 4;
    EXPECT_EQ(counter.read(), 28);

    FlopCounter<> const increment{3};
    EXPECT_EQ(&(counter += increment), &counter);
    EXPECT_EQ(&(counter -= increment), &counter);
    EXPECT_EQ(counter.read(), 28);
    counter.reset();
    EXPECT_EQ(counter.read(), 0);
}

TEST(FlopCounter, PreservesExactCountsBeyondIntAndDoublePrecision) {
    std::uint64_t constexpr kAboveInt = std::uint64_t(std::numeric_limits<int>::max()) + 1;
    std::uint64_t constexpr kAboveDoublePrecision = (std::uint64_t(1) << 53) + 1;
    PROFILE();

    FlopCounter<> counter{kAboveInt};
    counter += 17;
    counter -= 9;
    EXPECT_EQ(counter.read(), kAboveInt + 8);

    counter = kAboveDoublePrecision;
    counter += FlopCounter<>{2};
    counter -= FlopCounter<>{1};
    EXPECT_EQ(counter.read(), kAboveDoublePrecision + 1);
    counter -= 1;
    EXPECT_EQ(counter.read(), kAboveDoublePrecision);

    counter = std::numeric_limits<std::uint64_t>::max();
    EXPECT_EQ(counter.read(), std::numeric_limits<std::uint64_t>::max());
    counter -= std::numeric_limits<std::uint64_t>::max();
    EXPECT_EQ(counter.read(), 0);
}

TEST(FlopCounter, CopiesAndConstConversionsRetainIndependentValues) {
    PROFILE();
    FlopCounter<> original{41};
    auto copy = original;
    copy += 1;
    EXPECT_EQ(original.read(), 41);
    EXPECT_EQ(copy.read(), 42);
    original = copy;
    copy.reset();
    EXPECT_EQ(original.read(), 42);
    EXPECT_EQ(copy.read(), 0);
    FlopCounter<> const constant = original;
    std::uint64_t const implicitConversion = constant;
    EXPECT_EQ(implicitConversion, 42);

    FlopCounter<int> signedCounter{17};
    signedCounter += 9U;
    signedCounter -= 3;
    EXPECT_EQ(signedCounter.read(), 23);
}

TEST(FlopCounter, DisabledCountersIgnoreUpdatesAndAlwaysConvertToZero) {
    using DisabledCounter = FlopCounter<std::uint64_t, false>;
    PROFILE();
    DisabledCounter counter{std::numeric_limits<std::uint64_t>::max()};
    EXPECT_EQ(&(counter = 17), &counter);
    EXPECT_EQ(&(counter += 19), &counter);
    EXPECT_EQ(&(counter -= 23), &counter);
    EXPECT_EQ(&(counter += DisabledCounter{13}), &counter);
    EXPECT_EQ(&(counter -= DisabledCounter{29}), &counter);
    (counter = 7) += 11;
    auto copy = counter;
    copy.reset();
    DisabledCounter const constant = copy;
    std::uint64_t const implicitConversion = constant;
    EXPECT_EQ(counter.read(), 0);
    EXPECT_EQ(copy.read(), 0);
    EXPECT_EQ(implicitConversion, 0);

    // Disabled instrumentation must elide contracts as well as arithmetic.
    DisabledCounter ignored{-1};
    EXPECT_NO_THROW(ignored = -1);
    EXPECT_NO_THROW(ignored += std::numeric_limits<std::uint64_t>::max());
    EXPECT_NO_THROW(ignored -= -1);
    EXPECT_EQ(ignored.read(), 0);
}

TEST(FlopCounter, ValidationRejectsNegativeAndNarrowingInputsBeforeChangingState) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Requires validation checks";
    }
    using SmallCounter = FlopCounter<std::uint8_t>;
    EXPECT_THROW((SmallCounter{-1}), std::runtime_error);
    EXPECT_THROW((SmallCounter{256}), std::runtime_error);
    EXPECT_THROW((FlopCounter<std::int8_t>{std::numeric_limits<std::uint64_t>::max()}), std::runtime_error);

    SmallCounter counter{37};
    EXPECT_THROW(counter = -1, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
    EXPECT_THROW(counter = 256, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
    EXPECT_THROW(counter += -1, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
    EXPECT_THROW(counter += 256, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
    EXPECT_THROW(counter -= -1, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
    EXPECT_THROW(counter -= 256, std::runtime_error);
    EXPECT_EQ(counter.read(), 37);
}

TEST(FlopCounter, ValidationRejectsOverflowAndUnderflowBeforeChangingState) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Requires validation checks";
    }
    FlopCounter<> counter{std::numeric_limits<std::uint64_t>::max()};
    EXPECT_THROW(counter += 1, std::runtime_error);
    EXPECT_EQ(counter.read(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_THROW(counter += FlopCounter<>{1}, std::runtime_error);
    EXPECT_EQ(counter.read(), std::numeric_limits<std::uint64_t>::max());
    counter = 0;
    EXPECT_THROW(counter -= 1, std::runtime_error);
    EXPECT_EQ(counter.read(), 0);
    EXPECT_THROW(counter -= FlopCounter<>{1}, std::runtime_error);
    EXPECT_EQ(counter.read(), 0);

    FlopCounter<std::int8_t> signedCounter{std::numeric_limits<std::int8_t>::max()};
    EXPECT_THROW(signedCounter += 1U, std::runtime_error);
    EXPECT_EQ(signedCounter.read(), std::numeric_limits<std::int8_t>::max());
    signedCounter = 3;
    EXPECT_THROW(signedCounter -= 4U, std::runtime_error);
    EXPECT_EQ(signedCounter.read(), 3);
}

}    // namespace
