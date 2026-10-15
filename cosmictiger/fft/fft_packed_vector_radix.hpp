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
#include <utility>
#include <vector>

namespace cosmictiger {

// A real 3-D FFT represented by the complex FFT of adjacent even/odd x
// samples. Localities form a q-by-q-by-q grid, with P=q^3 and local real blocks
// of side L=M/q. Each locality stores L^3/2 packed complex values. Both the
// real blocks and packed spectra use [z][y][x] order with x fastest. Packed
// locality ID is (r_x*q+r_y)*q+r_z. Frequency coordinates are
// k_a=reverseBits(r_a,p)+q*j_a, where q=2^p and
// the x frequency has period M/2. The unrepresented full-spectrum x half is
// reconstructed when multiplying.
//
// The exchange functor has the form
//   exchange(partnerId, span<ComplexBatch const>, span<ComplexBatch>);
// every locality must enter each phase concurrently, and exchange must finish
// copying the send buffer before it returns. One instance is used sequentially.
// Local transforms use FFTW3, without FFTW3-MPI.
template <size_t batchSize>
class FftPackedVectorRadix {
public:
    using RealBatch = std::array<double, batchSize>;
    using ComplexBatch = std::array<std::complex<double>, batchSize>;

    FftPackedVectorRadix(size_t sideLength, size_t localityCount, size_t localityId)
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
        expectRange(size_t(1), localitySideLength_, meshSideLength_ / 2);
        blockSideLength_ = meshSideLength_ / localitySideLength_;
        packedXLength_ = blockSideLength_ / 2;
        expectRange(size_t(1), blockSideLength_, size_t(INT_MAX));
        expect(blockSideLength_ <= std::numeric_limits<size_t>::max() / blockSideLength_);
        expect(blockSideLength_ * blockSideLength_ <= std::numeric_limits<size_t>::max() / blockSideLength_);
        blockValueCount_ = blockSideLength_ * blockSideLength_ * blockSideLength_;
        packedValueCount_ = blockValueCount_ / 2;

        if (packedXLength_ > 1) {
            xForwardPlan_ = detail::memoizedFftwPlan(packedXLength_, detail::FftwTransformKind::complexForward);
            xInversePlan_ = detail::memoizedFftwPlan(packedXLength_, detail::FftwTransformKind::complexBackward);
        }
        transverseForwardPlan_ = detail::memoizedFftwPlan(blockSideLength_, detail::FftwTransformKind::complexForward);
        transverseInversePlan_ = detail::memoizedFftwPlan(blockSideLength_, detail::FftwTransformKind::complexBackward);
    }

    FftPackedVectorRadix(FftPackedVectorRadix const&) = delete;
    FftPackedVectorRadix& operator=(FftPackedVectorRadix const&) = delete;

    size_t localSideLength() const noexcept {
        PROFILE();
        return blockSideLength_;
    }

    size_t localValueCount() const noexcept {
        PROFILE();
        return blockValueCount_;
    }

    size_t spectralValueCount() const noexcept {
        PROFILE();
        return packedValueCount_;
    }

    template <typename Exchange>
    std::vector<ComplexBatch> forward(std::span<RealBatch const> input, Exchange&& exchange) {
        PROFILE();
        expectEqual(input.size(), blockValueCount_);
        std::vector<ComplexBatch> values(packedValueCount_);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t packedIndex) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const realIndex = (packedIndex / packedXLength_) * blockSideLength_ + 2 * (packedIndex % packedXLength_);
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        values[packedIndex][lane] = {input[realIndex][lane], input[realIndex + 1][lane]};
                    }
                },
                "FftPackedVectorRadix pack adjacent real samples"),
            batchSize);
        transform(values, false, exchange);
        return values;
    }

    // The inverse is normalized so inverse(forward(input)) recovers input.
    template <typename Exchange>
    std::vector<RealBatch> inverse(std::span<ComplexBatch const> input, Exchange&& exchange) {
        PROFILE();
        expectEqual(input.size(), packedValueCount_);
        std::vector<ComplexBatch> values(input.begin(), input.end());
        transform(values, true, exchange);
        double const inverseScale =
            1.0 / static_cast<double>(meshSideLength_ / 2) / static_cast<double>(meshSideLength_) / static_cast<double>(meshSideLength_);
        std::vector<RealBatch> output(blockValueCount_);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t packedIndex) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const realIndex = (packedIndex / packedXLength_) * blockSideLength_ + 2 * (packedIndex % packedXLength_);
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        std::complex<double> const value = values[packedIndex][lane] * inverseScale;
                        output[realIndex][lane] = value.real();
                        output[realIndex + 1][lane] = value.imag();
                    }
                },
                "FftPackedVectorRadix restore real samples"),
            batchSize);
        return output;
    }

    // Multiplier coefficients use the full-complex vector-radix local layout:
    // [z][y][x] with x extent L and k_x=reverseBits(r_x,p)+q*x. For each packed
    // x index, the first and second x halves are k_x and k_x+M/2. The multiplier
    // must be Hermitian so the inverse product is real. Only the packed input
    // crosses the conjugate-partner exchange: one phase and one packed payload.
    template <typename Exchange>
    std::vector<ComplexBatch> multiplyWithPrepositionedMultiplier(
        std::span<ComplexBatch const> packedInput, std::span<ComplexBatch const> fullComplexMultiplier, Exchange&& exchange) const {
        PROFILE();
        expectEqual(packedInput.size(), packedValueCount_);
        expectEqual(fullComplexMultiplier.size(), blockValueCount_);
        std::vector<ComplexBatch> const partnerValues = exchangeConjugatePartners(packedInput, exchange);
        std::vector<ComplexBatch> product(packedValueCount_);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const localX = index % packedXLength_;
                    size_t const fullLowIndex = (index / packedXLength_) * blockSideLength_ + localX;
                    size_t const fullHighIndex = fullLowIndex + packedXLength_;
                    size_t const partnerIndex = conjugatePartnerIndex(index);
                    std::complex<double> const phase = xPhase(index);
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        auto const [low, high] = recoverFullPair(packedInput[index][lane], partnerValues[partnerIndex][lane], phase);
                        product[index][lane] =
                            repackProduct(low * fullComplexMultiplier[fullLowIndex][lane], high * fullComplexMultiplier[fullHighIndex][lane], phase);
                    }
                },
                "FftPackedVectorRadix multiply prepositioned spectrum"),
            batchSize);
        return product;
    }

    // Two packed operands require both partners, sent together in one phase.
    // Its partner payload is twice that of the prepositioned-multiplier model.
    template <typename Exchange>
    std::vector<ComplexBatch> multiply(std::span<ComplexBatch const> left, std::span<ComplexBatch const> right, Exchange&& exchange) const {
        PROFILE();
        expectEqual(left.size(), packedValueCount_);
        expectEqual(right.size(), packedValueCount_);
        std::vector<ComplexBatch> operands(2 * packedValueCount_);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    operands[index] = left[index];
                    operands[packedValueCount_ + index] = right[index];
                },
                "FftPackedVectorRadix combine operand spectra"),
            batchSize);
        std::vector<ComplexBatch> const partnerValues = exchangeConjugatePartners(std::span<ComplexBatch const>(operands), exchange);
        std::vector<ComplexBatch> product(packedValueCount_);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const partnerIndex = conjugatePartnerIndex(index);
                    std::complex<double> const phase = xPhase(index);
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        auto const [leftLow, leftHigh] = recoverFullPair(left[index][lane], partnerValues[partnerIndex][lane], phase);
                        auto const [rightLow, rightHigh] = recoverFullPair(right[index][lane], partnerValues[packedValueCount_ + partnerIndex][lane], phase);
                        product[index][lane] = repackProduct(leftLow * rightLow, leftHigh * rightHigh, phase);
                    }
                },
                "FftPackedVectorRadix multiply conjugate spectra"),
            batchSize);
        return product;
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

    size_t conjugatePartnerLocality() const noexcept {
        PROFILE();
        size_t partner = 0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            size_t const residue = reverseBits(localityCoordinate(axis), radixLevels_);
            size_t const negativeResidue = (localitySideLength_ - residue) & (localitySideLength_ - 1);
            partner |= reverseBits(negativeResidue, radixLevels_) << ((NDIM - 1 - axis) * radixLevels_);
        }
        return partner;
    }

    size_t axisLocalLength(size_t axis) const noexcept {
        PROFILE();
        return axis == 0 ? packedXLength_ : blockSideLength_;
    }

    size_t axisGlobalLength(size_t axis) const noexcept {
        PROFILE();
        return axis == 0 ? meshSideLength_ / 2 : meshSideLength_;
    }

    size_t axisStride(size_t axis) const noexcept {
        PROFILE();
        return axis == 0 ? 1 : axis == 1 ? packedXLength_ : packedXLength_ * blockSideLength_;
    }

    size_t negativeLocalFrequency(size_t axis, size_t localFrequency) const noexcept {
        PROFILE();
        size_t const residue = reverseBits(localityCoordinate(axis), radixLevels_);
        size_t const globalFrequency = residue + localitySideLength_ * localFrequency;
        size_t const negativeFrequency = globalFrequency == 0 ? 0 : axisGlobalLength(axis) - globalFrequency;
        return negativeFrequency / localitySideLength_;
    }

    size_t conjugatePartnerIndex(size_t index) const noexcept {
        PROFILE();
        size_t const x = index % packedXLength_;
        size_t const y = (index / packedXLength_) % blockSideLength_;
        size_t const z = index / (packedXLength_ * blockSideLength_);
        return (negativeLocalFrequency(2, z) * blockSideLength_ + negativeLocalFrequency(1, y)) * packedXLength_ + negativeLocalFrequency(0, x);
    }

    static std::complex<double> twiddle(size_t numerator, size_t denominator, bool inverse) {
        PROFILE();
        FpeGuard fpeGuard;
        double const sign = inverse ? 1.0 : -1.0;
        double const angle = sign * 2.0 * std::numbers::pi_v<double> * static_cast<double>(numerator) / static_cast<double>(denominator);
        return {std::cos(angle), std::sin(angle)};
    }

    std::complex<double> xPhase(size_t index) const {
        PROFILE();
        size_t const residue = reverseBits(localityCoordinate(0), radixLevels_);
        size_t const frequency = residue + localitySideLength_ * (index % packedXLength_);
        return twiddle(frequency, meshSideLength_, false);
    }

    static std::pair<std::complex<double>, std::complex<double>> recoverFullPair(
        std::complex<double> value, std::complex<double> partnerValue, std::complex<double> phase) {
        PROFILE();
        std::complex<double> const mirror = std::conj(partnerValue);
        std::complex<double> const even = 0.5 * (value + mirror);
        std::complex<double> const odd = std::complex<double>(0.0, -0.5) * (value - mirror);
        std::complex<double> const shiftedOdd = phase * odd;
        return {even + shiftedOdd, even - shiftedOdd};
    }

    static std::complex<double> repackProduct(std::complex<double> low, std::complex<double> high, std::complex<double> phase) {
        PROFILE();
        std::complex<double> const even = 0.5 * (low + high);
        std::complex<double> const odd = 0.5 * (low - high) * std::conj(phase);
        return even + std::complex<double>(0.0, 1.0) * odd;
    }

    template <typename Exchange>
    std::vector<ComplexBatch> exchangeConjugatePartners(std::span<ComplexBatch const> values, Exchange& exchange) const {
        PROFILE();
        std::vector<ComplexBatch> partnerValues(values.size());
        // Even self-conjugate localities enter this logical phase so a
        // sequence-numbered exchange stays aligned for the inverse transform.
        exchange(conjugatePartnerLocality(), values, std::span<ComplexBatch>(partnerValues));
        return partnerValues;
    }

    template <typename Exchange>
    void transform(std::vector<ComplexBatch>& values, bool inverse, Exchange& exchange) {
        PROFILE();
        if (inverse) {
            for (size_t axis = NDIM; axis-- > 0;) {
                localTransform(values, axis, true);
                applySplitTwiddle(values, axis, true);
                for (size_t level = radixLevels_; level-- > 0;) {
                    complexStage(values, axis, level, true, exchange);
                }
            }
        } else {
            for (size_t axis = 0; axis < NDIM; axis++) {
                for (size_t level = 0; level < radixLevels_; level++) {
                    complexStage(values, axis, level, false, exchange);
                }
                applySplitTwiddle(values, axis, false);
                localTransform(values, axis, false);
            }
        }
    }

    template <typename Exchange>
    void complexStage(std::vector<ComplexBatch>& values, size_t axis, size_t level, bool inverse, Exchange& exchange) const {
        PROFILE();
        size_t const stageSize = localitySideLength_ >> level;
        size_t const half = stageSize / 2;
        size_t const coordinate = localityCoordinate(axis);
        bool const upper = (coordinate & half) != 0;
        std::complex<double> const stageTwiddle = twiddle(coordinate & (half - 1), stageSize, inverse);
        std::vector<ComplexBatch> partnerValues(packedValueCount_);
        exchange(
            currentLocalityId_ ^ (half << ((NDIM - 1 - axis) * radixLevels_)), std::span<ComplexBatch const>(values), std::span<ComplexBatch>(partnerValues));
        detail::parallelForFft(packedValueCount_,
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
                "FftPackedVectorRadix apply complex butterfly"),
            batchSize);
    }

    void applySplitTwiddle(std::vector<ComplexBatch>& values, size_t axis, bool inverse) const {
        PROFILE();
        size_t const frequencyRemainder = reverseBits(localityCoordinate(axis), radixLevels_);
        if (frequencyRemainder == 0) {
            return;
        }
        size_t const stride = axisStride(axis);
        size_t const localLength = axisLocalLength(axis);
        detail::parallelForFft(packedValueCount_,
            hpx::annotated_function(
                [&](size_t cell) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const localCoordinate = (cell / stride) % localLength;
                    std::complex<double> const factor = twiddle(frequencyRemainder * localCoordinate, axisGlobalLength(axis), inverse);
                    for (std::complex<double>& lane : values[cell]) {
                        lane *= factor;
                    }
                },
                "FftPackedVectorRadix apply split twiddle"),
            batchSize);
    }

    void localTransform(std::vector<ComplexBatch>& values, size_t axis, bool inverse) {
        PROFILE();
        size_t const localLength = axisLocalLength(axis);
        if (localLength == 1) {
            return;
        }
        size_t const stride = axisStride(axis);
        size_t const groupLength = stride * localLength;
        fftw_plan const plan = axis == 0 ? (inverse ? xInversePlan_ : xForwardPlan_) : (inverse ? transverseInversePlan_ : transverseForwardPlan_);
        size_t const lineCount = packedValueCount_ / localLength;
        detail::parallelForFftWithScratch(lineCount * batchSize, detail::fftLineWork(localLength),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(localLength);
                },
                "FftPackedVectorRadix allocate transform scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& scratch) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const line = item / batchSize;
                    size_t const lane = item % batchSize;
                    size_t const base = (line / stride) * groupLength + line % stride;
                    for (size_t sample = 0; sample < localLength; sample++) {
                        scratch[sample] = values[base + sample * stride][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(scratch.data());
                    fftw_execute_dft(plan, transformArray, transformArray);
                    for (size_t sample = 0; sample < localLength; sample++) {
                        values[base + sample * stride][lane] = scratch[sample];
                    }
                },
                "FftPackedVectorRadix execute local transform line"));
    }

    size_t meshSideLength_;
    size_t globalLocalityCount_;
    size_t currentLocalityId_;
    size_t radixLevels_ = 0;
    size_t localitySideLength_ = 1;
    size_t blockSideLength_ = 0;
    size_t packedXLength_ = 0;
    size_t blockValueCount_ = 0;
    size_t packedValueCount_ = 0;
    fftw_plan xForwardPlan_ = nullptr;
    fftw_plan xInversePlan_ = nullptr;
    fftw_plan transverseForwardPlan_ = nullptr;
    fftw_plan transverseInversePlan_ = nullptr;
};

}    // namespace cosmictiger
