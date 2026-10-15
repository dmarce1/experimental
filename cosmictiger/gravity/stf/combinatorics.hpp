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

#if defined(__CUDACC__)
#define COSMICTIGER_COMBINATORICS_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_COMBINATORICS_HOST_DEVICE
#endif

namespace cosmictiger::gravity::stf {

// Instantiate as constexpr host data or constant-initialized CUDA constant
// memory. Runtime degree arguments then reuse the compile-time results.
template <int maximumOrder>
class Combinatorics {
public:
    static_assert(maximumOrder >= 0);

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr Combinatorics() {
        PROFILE();
        factorials_[0] = 1.0;
        inverseFactorials_[0] = 1.0;
        powersOfTwo_[0] = 1.0;
        for (int n = 1; n <= maximumOrder; n++) {
            factorials_[n] = factorials_[n - 1] * double(n);
            inverseFactorials_[n] = inverseFactorials_[n - 1] / double(n);
            powersOfTwo_[n] = powersOfTwo_[n - 1] * 2.0;
        }
        for (int rank = 0; rank <= maximumOrder; rank++) {
            // Preserve the existing descending product and (-1)!! = 1.
            oddFactorials_[rank] = 1.0;
            for (int n = 2 * rank - 1; n > 1; n -= 2) {
                oddFactorials_[rank] *= double(n);
            }
            binomials_[rank][0] = 1.0;
            for (int k = 1; k <= rank; k++) {
                binomials_[rank][k] = binomials_[rank][k - 1] * (double(rank - k + 1) / double(k));
            }
        }
    }

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr double factorial(int n) const {
        PROFILE();
        return factorials_[expectIndex(n, maximumOrder + 1)];
    }

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr double inverseFactorial(int n) const {
        PROFILE();
        return inverseFactorials_[expectIndex(n, maximumOrder + 1)];
    }

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr double oddFactorial(int n) const {
        PROFILE();
        expectRange(-1, n, 2 * maximumOrder - 1);
        expect(n % 2 != 0);
        return oddFactorials_[(n + 1) / 2];
    }

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr double powerOfTwo(int n) const {
        PROFILE();
        return powersOfTwo_[expectIndex(n, maximumOrder + 1)];
    }

    COSMICTIGER_COMBINATORICS_HOST_DEVICE constexpr double binomial(int n, int k) const {
        PROFILE();
        return binomials_[expectIndex(n, maximumOrder + 1)][expectRange(0, k, n)];
    }

private:
    double factorials_[maximumOrder + 1]{};
    double inverseFactorials_[maximumOrder + 1]{};
    double oddFactorials_[maximumOrder + 1]{};
    double powersOfTwo_[maximumOrder + 1]{};
    double binomials_[maximumOrder + 1][maximumOrder + 1]{};
};

}    // namespace cosmictiger::gravity::stf

#undef COSMICTIGER_COMBINATORICS_HOST_DEVICE
