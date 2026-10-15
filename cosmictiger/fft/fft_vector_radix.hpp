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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/fft/fft_fftw.hpp>
#include <cosmictiger/fft/fft_parallel.hpp>

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger {

// A full-complex 3-D transform on a cubically blocked real mesh. The locality
// address is ordered [x][y][z], with z fastest. Each local array is row-major
// [z][y][x], with x fastest.
// P=q^3, q=2^p, L=M/q, and each locality owns L^3 spatial values. Each value
// contains batchSize independent FFTs. The caller supplies an exchange functor
// with real and complex overloads of the form
//
//   exchange(partnerId, span<Batch const>, span<Batch>);
//
// All localities must enter each exchange phase concurrently. The functor must
// finish copying the partner's send buffer before returning. The first forward
// level and last inverse level use RealBatch; all other levels use ComplexBatch.
// No FFTW MPI library is used.
//
// At physical locality coordinate r and local frequency index a, the natural
// global frequency is k=reverseBits(r,p)+q*a, independently in x, y, and z.
// Thus the forward spectrum retains digit-reversed locality coordinates. A
// pointwise multiplier must use this layout; the inverse consumes it directly.
template <size_t batchSize>
class FftVectorRadix {
public:
    using RealBatch = std::array<double, batchSize>;
    using ComplexBatch = std::array<std::complex<double>, batchSize>;

    FftVectorRadix(size_t sideLength, size_t localityCount, size_t localityId)
      : meshSideLength_(sideLength)
      , globalLocalityCount_(localityCount)
      , currentLocalityId_(localityId) {
        static_assert(batchSize > 0, "FFT batch size must be positive");
        PROFILE();
        expectRange(size_t(2), meshSideLength_, std::numeric_limits<size_t>::max());
        expect(std::has_single_bit(meshSideLength_));
        expect(std::has_single_bit(globalLocalityCount_));
        expect(std::countr_zero(globalLocalityCount_) % NDIM == 0);
        expectIndex(currentLocalityId_, globalLocalityCount_);
        radixLevels_ = std::countr_zero(globalLocalityCount_) / NDIM;
        localitySideLength_ = size_t(1) << radixLevels_;
        expectRange(size_t(1), globalLocalityCount_, meshSideLength_);
        blockSideLength_ = meshSideLength_ / localitySideLength_;
        expectRange(size_t(1), blockSideLength_, size_t(INT_MAX));
        expect(blockSideLength_ <= std::numeric_limits<size_t>::max() / blockSideLength_);
        expect(blockSideLength_ * blockSideLength_ <= std::numeric_limits<size_t>::max() / blockSideLength_);
        blockValueCount_ = blockSideLength_ * blockSideLength_ * blockSideLength_;
        if (blockSideLength_ > 1) {
            forwardPlan_ = detail::memoizedFftwPlan(blockSideLength_, detail::FftwTransformKind::complexForward);
            inversePlan_ = detail::memoizedFftwPlan(blockSideLength_, detail::FftwTransformKind::complexBackward);
        }
    }

    FftVectorRadix(FftVectorRadix const&) = delete;
    FftVectorRadix& operator=(FftVectorRadix const&) = delete;

    size_t localSideLength() const noexcept {
        PROFILE();
        return blockSideLength_;
    }

    size_t localValueCount() const noexcept {
        PROFILE();
        return blockValueCount_;
    }

    // FFTW's forward and backward transforms are unnormalized. The inverse
    // applies 1/M^3, so inverse(forward(input)) recovers the real input.
    template <typename Exchange>
    std::vector<ComplexBatch> forward(std::span<RealBatch const> input, Exchange&& exchange) {
        PROFILE();
        expectEqual(input.size(), blockValueCount_);
        std::vector<RealBatch> realValues(input.begin(), input.end());
        if (radixLevels_ != 0) {
            // These three raw butterflies commute and keep the payload real.
            // Apply their complex twiddles only after all three exchanges.
            for (size_t axis = 0; axis < NDIM; axis++) {
                firstRealStage(realValues, axis, exchange);
            }
        }

        std::complex<double> const firstTwiddle = firstLevelTwiddle(false);
        std::vector<ComplexBatch> values(blockValueCount_);
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        values[cell][lane] = realValues[cell][lane] * firstTwiddle;
                    }
                },
                "FftVectorRadix initialize complex batches"),
            batchSize);

        for (size_t axis = 0; axis < NDIM; axis++) {
            for (size_t level = 1; level < radixLevels_; level++) {
                complexStage(values, axis, level, false, exchange);
            }
            applySplitTwiddle(values, axis, false);
            localTransform(values, axis, false);
        }
        return values;
    }

    // The spectrum must be Hermitian in the documented layout. Numerical
    // roundoff can leave a tiny imaginary residual before the real stages.
    template <typename Exchange>
    std::vector<RealBatch> inverse(std::span<ComplexBatch const> input, Exchange&& exchange) {
        PROFILE();
        expectEqual(input.size(), blockValueCount_);
        std::vector<ComplexBatch> values(input.begin(), input.end());
        for (size_t axis = NDIM; axis-- > 0;) {
            localTransform(values, axis, true);
            applySplitTwiddle(values, axis, true);
            for (size_t level = radixLevels_; level-- > 1;) {
                complexStage(values, axis, level, true, exchange);
            }
        }

        std::complex<double> const firstTwiddle = firstLevelTwiddle(true);
        std::vector<RealBatch> realValues(blockValueCount_);
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        realValues[cell][lane] = (values[cell][lane] * firstTwiddle).real();
                    }
                },
                "FftVectorRadix recover real batches"),
            batchSize);
        if (radixLevels_ != 0) {
            for (size_t axis = NDIM; axis-- > 0;) {
                firstRealStage(realValues, axis, exchange);
            }
        }
        double const inverseScale = 1.0 / static_cast<double>(meshSideLength_) / static_cast<double>(meshSideLength_) / static_cast<double>(meshSideLength_);
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (double& lane : realValues[cell]) {
                        lane *= inverseScale;
                    }
                },
                "FftVectorRadix normalize inverse transform"),
            batchSize);
        return realValues;
    }

private:
    static size_t reverseBits(size_t value, size_t bitCount) noexcept {
        PROFILE();
        size_t result = 0;
        for (size_t bit = 0; bit < bitCount; bit++) {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }

    size_t localityCoordinate(size_t axis) const noexcept {
        PROFILE();
        return (currentLocalityId_ >> ((NDIM - 1 - axis) * radixLevels_)) & (localitySideLength_ - 1);
    }

    size_t partnerId(size_t axis, size_t partnerBit) const noexcept {
        PROFILE();
        return currentLocalityId_ ^ (partnerBit << ((NDIM - 1 - axis) * radixLevels_));
    }

    static std::complex<double> twiddle(size_t numerator, size_t denominator, bool inverse) {
        PROFILE();
        FpeGuard fpeGuard;
        double const sign = inverse ? 1.0 : -1.0;
        double const angle = sign * 2.0 * std::numbers::pi_v<double> * static_cast<double>(numerator) / static_cast<double>(denominator);
        return {std::cos(angle), std::sin(angle)};
    }

    std::complex<double> firstLevelTwiddle(bool inverse) const {
        PROFILE();
        std::complex<double> factor(1.0, 0.0);
        if (radixLevels_ == 0) {
            return factor;
        }
        size_t const half = localitySideLength_ / 2;
        for (size_t axis = 0; axis < NDIM; axis++) {
            size_t const coordinate = localityCoordinate(axis);
            if ((coordinate & half) != 0) {
                factor *= twiddle(coordinate & (half - 1), localitySideLength_, inverse);
            }
        }
        return factor;
    }

    template <typename Exchange>
    void firstRealStage(std::vector<RealBatch>& values, size_t axis, Exchange& exchange) const {
        PROFILE();
        size_t const half = localitySideLength_ / 2;
        std::vector<RealBatch> partnerValues(blockValueCount_);
        exchange(partnerId(axis, half), std::span<RealBatch const>(values), std::span<RealBatch>(partnerValues));
        bool const upper = (localityCoordinate(axis) & half) != 0;
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        values[cell][lane] = upper ? partnerValues[cell][lane] - values[cell][lane] : values[cell][lane] + partnerValues[cell][lane];
                    }
                },
                "FftVectorRadix apply real butterfly"),
            batchSize);
    }

    template <typename Exchange>
    void complexStage(std::vector<ComplexBatch>& values, size_t axis, size_t level, bool inverse, Exchange& exchange) const {
        PROFILE();
        size_t const stageSize = localitySideLength_ >> level;
        size_t const half = stageSize / 2;
        size_t const coordinate = localityCoordinate(axis);
        bool const upper = (coordinate & half) != 0;
        std::complex<double> const stageTwiddle = twiddle(coordinate & (half - 1), stageSize, inverse);
        std::vector<ComplexBatch> partnerValues(blockValueCount_);
        exchange(partnerId(axis, half), std::span<ComplexBatch const>(values), std::span<ComplexBatch>(partnerValues));
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        if (inverse) {
                            values[cell][lane] = upper ? partnerValues[cell][lane] - values[cell][lane] * stageTwiddle :
                                                         values[cell][lane] + partnerValues[cell][lane] * stageTwiddle;
                        } else {
                            values[cell][lane] =
                                upper ? (partnerValues[cell][lane] - values[cell][lane]) * stageTwiddle : values[cell][lane] + partnerValues[cell][lane];
                        }
                    }
                },
                "FftVectorRadix apply complex butterfly"),
            batchSize);
    }

    void applySplitTwiddle(std::vector<ComplexBatch>& values, size_t axis, bool inverse) const {
        PROFILE();
        size_t const frequencyRemainder = reverseBits(localityCoordinate(axis), radixLevels_);
        if (frequencyRemainder == 0) {
            return;
        }
        size_t const stride = axis == 0 ? 1 : axis == 1 ? blockSideLength_ : blockSideLength_ * blockSideLength_;
        std::vector<std::complex<double>> factors(blockSideLength_);
        double const sign = inverse ? 1.0 : -1.0;
        for (size_t localCoordinate = 0; localCoordinate < blockSideLength_; localCoordinate++) {
            // Multiply in floating point to avoid overflow in b*j when M is
            // large enough that the full mesh cannot be represented in size_t.
            double const angle = sign * 2.0 * std::numbers::pi_v<double> * static_cast<double>(frequencyRemainder) * static_cast<double>(localCoordinate) /
                static_cast<double>(meshSideLength_);
            factors[localCoordinate] = {std::cos(angle), std::sin(angle)};
        }
        detail::parallelForFft(blockValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const localCoordinate = (cell / stride) % blockSideLength_;
                    for (std::complex<double>& lane : values[cell]) {
                        lane *= factors[localCoordinate];
                    }
                },
                "FftVectorRadix apply split twiddle"),
            batchSize);
    }

    void localTransform(std::vector<ComplexBatch>& values, size_t axis, bool inverse) {
        PROFILE();
        if (blockSideLength_ == 1) {
            return;
        }
        size_t const stride = axis == 0 ? 1 : axis == 1 ? blockSideLength_ : blockSideLength_ * blockSideLength_;
        size_t const groupLength = stride * blockSideLength_;
        fftw_plan const plan = inverse ? inversePlan_ : forwardPlan_;
        size_t const lineCount = blockValueCount_ / blockSideLength_;
        detail::parallelForFftWithScratch(lineCount * batchSize, detail::fftLineWork(blockSideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(blockSideLength_);
                },
                "FftVectorRadix allocate transform scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& scratch) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const line = item / batchSize;
                    size_t const lane = item % batchSize;
                    size_t const base = (line / stride) * groupLength + line % stride;
                    for (size_t sample = 0; sample < blockSideLength_; sample++) {
                        scratch[sample] = values[base + sample * stride][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(scratch.data());
                    fftw_execute_dft(plan, transformArray, transformArray);
                    for (size_t sample = 0; sample < blockSideLength_; sample++) {
                        values[base + sample * stride][lane] = scratch[sample];
                    }
                },
                "FftVectorRadix execute local transform line"));
    }

    size_t meshSideLength_;
    size_t globalLocalityCount_;
    size_t currentLocalityId_;
    size_t radixLevels_ = 0;
    size_t localitySideLength_ = 1;
    size_t blockSideLength_ = 0;
    size_t blockValueCount_ = 0;
    fftw_plan forwardPlan_ = nullptr;
    fftw_plan inversePlan_ = nullptr;
};

}    // namespace cosmictiger
