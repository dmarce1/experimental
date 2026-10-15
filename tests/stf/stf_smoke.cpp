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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>

namespace stf = cosmictiger::gravity::stf;

// These must remain constant expressions, including zero-order edge cases.
constexpr stf::Combinatorics<9> kCombinatorics{};
static_assert(kCombinatorics.factorial(0) == 1.0);
static_assert(kCombinatorics.factorial(9) == 362880.0);
static_assert(kCombinatorics.oddFactorial(-1) == 1.0);
static_assert(kCombinatorics.oddFactorial(17) == 34459425.0);
static_assert(kCombinatorics.inverseFactorial(3) == 1.0 / 6.0);
static_assert(kCombinatorics.powerOfTwo(9) == 512.0);
static_assert(kCombinatorics.binomial(4, 2) == 6.0);
static_assert(stf::Combinatorics<0>{}.factorial(0) == 1.0);
static_assert(stf::Combinatorics<0>{}.oddFactorial(-1) == 1.0);
static_assert(stf::Combinatorics<0>{}.inverseFactorial(0) == 1.0);
static_assert(stf::Combinatorics<0>{}.powerOfTwo(0) == 1.0);
static_assert(stf::Combinatorics<0>{}.binomial(0, 0) == 1.0);

template <int sourceOrder, int expansionOrder>
bool checkMonopole() {
    PROFILE();
    stf::Multipole<double, sourceOrder> source{};
    stf::Expansion<double, expansionOrder> local{};
    source[stf::index(0, 0, 0)] = 1.0;
    stf::addM2L<double, sourceOrder, expansionOrder>(local, source, std::array<double, NDIM>{2.0, 0.0, 0.0});

    auto const close = [](double actual, double expected) {
        PROFILE();
        return std::abs(actual - expected) < 1.0e-13;
    };
    bool valid = close(local[stf::index(0, 0, 0)], -0.5);
    if constexpr (expansionOrder > 0) {
        valid = valid && close(local[stf::index(1, 0, 0)], 0.25) && close(local[stf::index(0, 1, 0)], 0.0) && close(local[stf::index(0, 0, 1)], 0.0);
    }
    return valid;
}

template <int expansionOrder>
bool checkRange() {
    PROFILE();
    bool valid = checkMonopole<expansionOrder, expansionOrder>();
    if constexpr (expansionOrder > 0) {
        valid = checkMonopole<expansionOrder - 1, expansionOrder>() && valid;
    }
    if constexpr (expansionOrder < stf::kMaxOrder) {
        valid = checkRange<expansionOrder + 1>() && valid;
    }
    return valid;
}

template <int order>
bool checkPointRadius() {
    PROFILE();
    stf::Multipole<double, order> source{};
    double const radiusSquared = stf::addPoint<double, order>(source, 2.0, {0.5, -0.25, 0.125}, 0.5);
    bool valid = radiusSquared == 1.3125 && source[stf::index(0, 0, 0)] == 2.0;
    if constexpr (order < stf::kMaxOrder) {
        valid = checkPointRadius<order + 1>() && valid;
    }
    return valid;
}

int main() {
    static_assert(
        std::tuple_size<stf::Multipole<double, stf::kMaxOrder>>::value == stf::coefficientCount(stf::kMaxOrder), "Multipoles must have compact STF storage");
    static_assert(
        std::tuple_size<stf::Expansion<double, stf::kMaxOrder>>::value == stf::coefficientCount(stf::kMaxOrder), "Expansions must have compact STF storage");
    PROFILE();
    if (!checkRange<stf::kMinOrder>()) {
        std::cerr << "Standalone STF M2L monopole smoke test failed\n";
        return 1;
    }
    if (!checkPointRadius<stf::kMinOrder>()) {
        std::cerr << "Standalone STF P2M radius reuse test failed\n";
        return 1;
    }
    return 0;
}
