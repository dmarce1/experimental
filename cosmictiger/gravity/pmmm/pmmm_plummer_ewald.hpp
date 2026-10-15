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

#include <cmath>
#include <stddef.h>

#if defined(__CUDACC__)
#define COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE
#endif

namespace cosmictiger::gravity::detail {

// E_2(z) = integral_1^infinity exp(-z*t)/t^2 dt. Removing the spatial
// average of the real Ewald sum adds pi*E_2(epsilon^2*alpha^2)/(V*alpha^2).
// The series and continued fraction are the incomplete-gamma representations
// of E_2 (DLMF 8.7 and 8.9), evaluated without subtracting large close terms.
COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE inline double plummerZeroModeIntegral(double z) {
    constexpr double kEulerGamma = 0.577215664901532860606512090082402431;
    constexpr double kSeriesThreshold = 1.0;
    constexpr double kTolerance = 4.0e-16;
    constexpr int kMaximumTerms = 256;
    PROFILE();
    expectNonNegative(z);
    if (z == 0.0) {
        return 1.0;
    }
    if (z <= kSeriesThreshold) {
        double term = -z;
        double sum = term;
        for (int n = 2; n <= kMaximumTerms; n++) {
            term *= -z / static_cast<double>(n);
            double const increment = term / static_cast<double>(n);
            sum += increment;
            if (::fabs(increment) <= kTolerance * ::fabs(sum)) {
                break;
            }
        }
        return ::exp(-z) + z * (kEulerGamma + ::log(z) + sum);
    }
    // The continued fraction for Gamma(-1,z), with the leading exp(-z)
    // removed, avoids cancellation in exp(-z)-z*E_1(z) at large z.
    double denominator = z + 2.0;
    double previousNumerator = 0.0;
    double numerator = 1.0;
    double previousDenominator = 1.0;
    double currentDenominator = denominator;
    double value = 1.0 / denominator;
    for (int n = 1; n <= kMaximumTerms; n++) {
        double const coefficient = -static_cast<double>(n) * static_cast<double>(n + 1);
        denominator += 2.0;
        double const nextNumerator = denominator * numerator + coefficient * previousNumerator;
        double const nextDenominator = denominator * currentDenominator + coefficient * previousDenominator;
        double const nextValue = nextNumerator / nextDenominator;
        double const inverseScale = 1.0 / nextDenominator;
        previousNumerator = numerator * inverseScale;
        numerator = nextNumerator * inverseScale;
        previousDenominator = currentDenominator * inverseScale;
        currentDenominator = 1.0;
        if (::fabs(nextValue - value) <= kTolerance * ::fabs(nextValue)) {
            value = nextValue;
            break;
        }
        value = nextValue;
    }
    return ::exp(-z) * value;
}

template <bool highOrder>
COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE double plummerReciprocalPanel(double b, double z, double lower, double upper) {
    constexpr int kHalfOrder = highOrder ? 8 : 4;
    constexpr double kNodes16[8] = {0.095012509837637440185319335424958063, 0.281603550779258913230460501460496106, 0.458016777657227386342419442983577574,
        0.617876244402643748446671764048791019, 0.755404408355003033895101194847442268, 0.865631202387831743880467897712393132,
        0.944575023073232576077988415534608345, 0.989400934991649932596154173450332627};
    constexpr double kWeights16[8] = {0.189450610455068496285396723208283105, 0.182603415044923588866763667969219939, 0.169156519395002538189312079030359962,
        0.149595988816576732081501730547478549, 0.124628971255533872052476282192016420, 0.095158511682492784809925107602246226,
        0.062253523938647892862843836994377694, 0.027152459411754094851780572456018104};
    constexpr double kNodes8[4] = {0.183434642495649804939476142360183981, 0.525532409916328985817739049189246349, 0.796666477413626739591553936475830437,
        0.960289856497536231683560868569472991};
    constexpr double kWeights8[4] = {0.362683783378361982965150449277195612, 0.313706645877887287337962201986601313, 0.222381034453374470544355994426240884,
        0.101228536290376259152531354309962190};
    PROFILE();
    double const midpoint = 0.5 * (lower + upper);
    double const halfWidth = 0.5 * (upper - lower);
    double result = 0.0;
    for (int index = 0; index < kHalfOrder; index++) {
        double const node = highOrder ? kNodes16[index] : kNodes8[index];
        double const weight = highOrder ? kWeights16[index] : kWeights8[index];
        double const first = midpoint - halfWidth * node;
        double const second = midpoint + halfWidth * node;
        result += weight * (::exp(-first - z * (b / (b + first))) + ::exp(-second - z * (b / (b + second))));
    }
    return halfWidth * result;
}

// Positive integral J(b,z) = integral_0^infinity exp(-v-z*b/(b+v)) dv.
// For wave number k, b=k^2/(4*alpha^2) and z=epsilon^2*alpha^2; the
// reciprocal Green coefficient is -4*pi*exp(-b)*J(b,z)/(V*k^2).
// This is the truncated version of the K_1 integral in DLMF 10.32.10,
// not the Newtonian reciprocal damping. Precompute it once per wave length.
// The omitted v>64 tail is bounded by exp(-64), independently of b and z.
COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE inline double plummerReciprocalIntegral(double b, double z) {
    constexpr double kTolerance = 4.0e-15;
    constexpr int kMaximumPanels = 128;
    constexpr int kIntervalCount = 7;
    PROFILE();
    expectPositive(b);
    expectNonNegative(z);
    if (z == 0.0) {
        return 1.0;
    }
    double result = 0.0;
    double lower = 0.0;
    double upper = 1.0;
    for (int interval = 0; interval < kIntervalCount; interval++) {
        double high = 0.0;
        for (int panels = 1; panels <= kMaximumPanels; panels *= 2) {
            double const width = (upper - lower) / static_cast<double>(panels);
            double low = 0.0;
            high = 0.0;
            for (int panel = 0; panel < panels; panel++) {
                double const left = lower + static_cast<double>(panel) * width;
                low += plummerReciprocalPanel<false>(b, z, left, left + width);
                high += plummerReciprocalPanel<true>(b, z, left, left + width);
            }
            if (::fabs(high - low) <= kTolerance * ::exp(-lower)) {
                break;
            }
        }
        result += high;
        lower = upper;
        upper *= 2.0;
    }
    return result;
}

}    // namespace cosmictiger::gravity::detail

#undef COSMICTIGER_PLUMMER_EWALD_HOST_DEVICE
