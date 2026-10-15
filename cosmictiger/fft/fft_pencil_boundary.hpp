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
#include <cosmictiger/fft/fft_communication_plan.hpp>
#include <cosmictiger/fft/fft_fftw.hpp>
#include <cosmictiger/fft/fft_parallel.hpp>
#include <cosmictiger/fft/fft_redistribution.hpp>

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <complex>
#include <limits>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger {

// The compact real-spectrum pencil/slab transform at P=M. There are M/2
// packed k_z planes but M localities, so an even locality owns one full XY
// spectral plane and its odd partner owns none. The public block layout is
// [z][y][x] locally and [x][y][z] in locality IDs (z fastest).
//
// The first redistribution uses the planner's extra p+1 address bits to make
// z pencils. A logical reversal of the y locality bits places the extra y bit
// at the end of that address group without changing the public block layout.
// The second redistribution leaves each plane split into two half planes. One
// unequal-size complex exchange gathers the odd half on the even locality
// before the local XY FFT; the inverse scatters it afterward. Each gather and
// scatter sends half of the compact complex data globally. The caller's
// exchange functor must accept zero-length send or receive spans.
//
// Active physical locality 2*k_z owns the full packed spectrum for plane k_z.
// Plane zero packs the k_z=0 and k_z=M/2 real endpoints in its real and
// imaginary components. Use multiply() for products of two packed spectra.
template <size_t batchSize>
class FftPencilBoundary {
public:
    static_assert(batchSize > 0);
    using RealBatch = std::array<double, batchSize>;
    using ComplexBatch = std::array<std::complex<double>, batchSize>;

    FftPencilBoundary(size_t sideLength, size_t localityCount, size_t locality, FftCommunicationPlan plan)
      : meshSideLength_(sideLength)
      , globalLocalityCount_(localityCount)
      , currentLocalityId_(locality)
      , communicationPlan_(std::move(plan)) {
        PROFILE();
        expect(!(meshSideLength_ < 8 || !std::has_single_bit(meshSideLength_) || globalLocalityCount_ != meshSideLength_ ||
            currentLocalityId_ >= globalLocalityCount_ || meshSideLength_ > INT_MAX || communicationPlan_.strategy != FftCommunicationStrategy::pencilSlab ||
            communicationPlan_.spectralLocalityCount != globalLocalityCount_));
        sideBits_ = std::countr_zero(meshSideLength_);
        expect(sideBits_ % NDIM == 0);
        expect(NDIM * sideBits_ < std::numeric_limits<size_t>::digits);
        localitySideBits_ = sideBits_ / NDIM;
        localitySideLength_ = size_t(1) << localitySideBits_;
        blockSideLength_ = meshSideLength_ / localitySideLength_;
        blockValueCount_ = blockSideLength_ * blockSideLength_ * blockSideLength_;
        halfPlaneValueCount_ = meshSideLength_ * meshSideLength_ / 2;
    }

    size_t spectralValueCount() const noexcept {
        PROFILE();
        return (currentLocalityId_ & 1) == 0 ? 2 * halfPlaneValueCount_ : 0;
    }

    template <typename Exchange>
    std::vector<ComplexBatch> forward(std::span<RealBatch const> block, Exchange&& exchange) const {
        PROFILE();
        expectEqual(block.size(), blockValueCount_);
        LogicalExchange<Exchange> logicalExchange(*this, exchange);
        detail::FftBitLayout const blockLayout = realBlockLayout();
        detail::FftBitLayout const pencilLayout = realPencilLayout();
        std::vector<RealBatch> pencil = detail::routeFftBits(std::vector<RealBatch>(block.begin(), block.end()), logicalLocality(), blockLayout, pencilLayout,
            communicationPlan_.blockToPencil, logicalExchange);
        std::vector<ComplexBatch> packed = forwardZ(pencil);
        detail::FftBitLayout const packedPencilLayout = complexPencilLayout();
        detail::FftBitLayout const slabLayout = complexSlabLayout();
        std::vector<ComplexBatch> halfPlane =
            detail::routeFftBits(std::move(packed), logicalLocality(), packedPencilLayout, slabLayout, communicationPlan_.pencilToSlab, logicalExchange);

        size_t const peer = currentLocalityId_ ^ size_t(1);
        if ((currentLocalityId_ & 1) != 0) {
            exchange(peer, std::span<ComplexBatch const>(halfPlane), std::span<ComplexBatch>());
            return {};
        }
        std::vector<ComplexBatch> plane(2 * halfPlaneValueCount_);
        detail::parallelForFft(halfPlaneValueCount_,
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    plane[index] = halfPlane[index];
                },
                "FftPencilBoundary assemble spectral plane"),
            batchSize);
        exchange(peer, std::span<ComplexBatch const>(), std::span<ComplexBatch>(plane.data() + halfPlaneValueCount_, halfPlaneValueCount_));
        transformXY(plane, FFTW_FORWARD);
        return plane;
    }

    template <typename Exchange>
    std::vector<RealBatch> inverse(std::span<ComplexBatch const> spectrum, Exchange&& exchange) const {
        PROFILE();
        expectEqual(spectrum.size(), spectralValueCount());
        std::vector<ComplexBatch> halfPlane(halfPlaneValueCount_);
        size_t const peer = currentLocalityId_ ^ size_t(1);
        if ((currentLocalityId_ & 1) == 0) {
            std::vector<ComplexBatch> plane(spectrum.begin(), spectrum.end());
            transformXY(plane, FFTW_BACKWARD);
            detail::parallelForFft(halfPlaneValueCount_,
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        FpeGuard fpeGuard;
                        halfPlane[index] = plane[index];
                    },
                    "FftPencilBoundary extract half plane"),
                batchSize);
            exchange(peer, std::span<ComplexBatch const>(plane.data() + halfPlaneValueCount_, halfPlaneValueCount_), std::span<ComplexBatch>());
        } else {
            exchange(peer, std::span<ComplexBatch const>(), std::span<ComplexBatch>(halfPlane));
        }

        LogicalExchange<Exchange> logicalExchange(*this, exchange);
        detail::FftBitLayout const packedPencilLayout = complexPencilLayout();
        detail::FftBitLayout const slabLayout = complexSlabLayout();
        std::vector<ComplexBatch> packed = detail::routeFftBitsBackward(
            std::move(halfPlane), logicalLocality(), packedPencilLayout, slabLayout, communicationPlan_.pencilToSlab, logicalExchange);
        std::vector<RealBatch> pencil = inverseZ(packed);
        detail::FftBitLayout const blockLayout = realBlockLayout();
        detail::FftBitLayout const pencilLayout = realPencilLayout();
        return detail::routeFftBitsBackward(std::move(pencil), logicalLocality(), blockLayout, pencilLayout, communicationPlan_.blockToPencil, logicalExchange);
    }

    std::vector<ComplexBatch> multiply(std::span<ComplexBatch const> left, std::span<ComplexBatch const> right) const {
        constexpr std::complex<double> kImaginary(0.0, 1.0);
        PROFILE();
        expectEqual(left.size(), spectralValueCount());
        expectEqual(right.size(), spectralValueCount());
        std::vector<ComplexBatch> product(left.size());
        detail::parallelForFft(left.size(),
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane < batchSize; lane++) {
                        product[index][lane] = left[index][lane] * right[index][lane];
                    }
                },
                "FftPencilBoundary multiply spectra"),
            batchSize);
        if (currentLocalityId_ == 0) {
            detail::parallelForFft(meshSideLength_,
                hpx::annotated_function(
                    [&](size_t y) {
                        PROFILE();
                        FpeGuard fpeGuard;
                        for (size_t x = 0; x < meshSideLength_; x++) {
                            size_t const index = y * meshSideLength_ + x;
                            size_t const reflected = ((meshSideLength_ - y) % meshSideLength_) * meshSideLength_ + (meshSideLength_ - x) % meshSideLength_;
                            for (size_t lane = 0; lane < batchSize; lane++) {
                                auto const leftZero = (left[index][lane] + std::conj(left[reflected][lane])) * 0.5;
                                auto const leftNyquist = (left[index][lane] - std::conj(left[reflected][lane])) / (2.0 * kImaginary);
                                auto const rightZero = (right[index][lane] + std::conj(right[reflected][lane])) * 0.5;
                                auto const rightNyquist = (right[index][lane] - std::conj(right[reflected][lane])) / (2.0 * kImaginary);
                                product[index][lane] = leftZero * rightZero + kImaginary * leftNyquist * rightNyquist;
                            }
                        }
                    },
                    "FftPencilBoundary multiply real endpoints"),
                meshSideLength_ * batchSize);
        }
        return product;
    }

private:
    // Preserve optional asynchronous transport support while remapping the
    // boundary layout's logical peer indices to physical FFT localities.
    template <typename Exchange>
    class LogicalExchange {
    public:
        LogicalExchange(FftPencilBoundary const& fft, Exchange& exchange)
          : fft_(fft)
          , exchange_(exchange) {
            PROFILE();
        }

        template <typename Value>
        void operator()(size_t logicalPeer, std::span<Value const> send, std::span<Value> receive) {
            PROFILE();
            exchange_(fft_.physicalLocality(logicalPeer), send, receive);
        }

        template <typename Value>
            requires requires(Exchange& exchange, size_t peer, std::vector<Value> outgoing, size_t receiveCount) {
                exchange.template exchangeAsync<Value>(peer, std::move(outgoing), receiveCount);
            }
        auto exchangeAsync(size_t logicalPeer, std::vector<Value> outgoing, size_t receiveCount) {
            PROFILE();
            return exchange_.template exchangeAsync<Value>(fft_.physicalLocality(logicalPeer), std::move(outgoing), receiveCount);
        }

    private:
        FftPencilBoundary const& fft_;
        Exchange& exchange_;
    };

    static size_t reverseBits(size_t value, size_t bitCount) noexcept {
        PROFILE();
        size_t reversed = 0;
        for (size_t bit = 0; bit < bitCount; bit++) {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        return reversed;
    }

    size_t physicalLocality(size_t logicalRank) const noexcept {
        PROFILE();
        size_t const mask = localitySideLength_ - 1;
        size_t const z = logicalRank & mask;
        size_t const y = reverseBits((logicalRank >> localitySideBits_) & mask, localitySideBits_);
        size_t const x = logicalRank >> (2 * localitySideBits_);
        return (x * localitySideLength_ + y) * localitySideLength_ + z;
    }

    size_t logicalLocality() const noexcept {
        PROFILE();
        return physicalLocality(currentLocalityId_);
    }

    detail::FftBitLayout realBlockLayout() const {
        PROFILE();
        std::vector<size_t> bits;
        for (size_t bit = 0; bit < localitySideBits_; bit++) {
            bits.push_back(NDIM * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = localitySideBits_; bit-- > 0;) {
            bits.push_back(2 * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit < localitySideBits_; bit++) {
            bits.push_back(sideBits_ - localitySideBits_ + bit);
        }
        return detail::FftBitLayout(NDIM * sideBits_, std::move(bits));
    }

    detail::FftBitLayout realPencilLayout() const {
        PROFILE();
        std::vector<size_t> bits;
        for (size_t bit = 0; bit < localitySideBits_; bit++) {
            bits.push_back(sideBits_ - 2 * localitySideBits_ + bit);
        }
        bits.push_back(sideBits_ - 2 * localitySideBits_ - 1);
        for (size_t bit = localitySideBits_ - 1; bit-- > 0;) {
            bits.push_back(2 * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit < localitySideBits_; bit++) {
            bits.push_back(sideBits_ - localitySideBits_ + bit);
        }
        return detail::FftBitLayout(NDIM * sideBits_, std::move(bits));
    }

    detail::FftBitLayout complexPencilLayout() const {
        PROFILE();
        std::vector<size_t> bits = realPencilLayout().localityBits();
        return detail::FftBitLayout(NDIM * sideBits_ - 1, std::move(bits));
    }

    detail::FftBitLayout complexSlabLayout() const {
        PROFILE();
        std::vector<size_t> bits;
        bits.push_back(2 * sideBits_ - 1);
        // Undo the logical y-bit reversal in plane ownership so physical
        // locality 2*k owns packed z-frequency plane k.
        for (size_t addressBit = 1; addressBit < sideBits_; addressBit++) {
            size_t physicalBit = addressBit;
            if (addressBit >= localitySideBits_ && addressBit < 2 * localitySideBits_) {
                physicalBit = NDIM * localitySideBits_ - 1 - addressBit;
            }
            bits.push_back(2 * sideBits_ + physicalBit - 1);
        }
        return detail::FftBitLayout(NDIM * sideBits_ - 1, std::move(bits));
    }

    std::vector<ComplexBatch> forwardZ(std::span<RealBatch const> pencil) const {
        PROFILE();
        size_t const xExtent = meshSideLength_ / (2 * localitySideLength_ * localitySideLength_);
        size_t const yExtent = 2 * blockSideLength_;
        size_t const packedPlanes = meshSideLength_ / 2;
        std::vector<ComplexBatch> packed(xExtent * yExtent * packedPlanes);
        fftw_plan const transform = detail::memoizedFftwPlan(meshSideLength_, detail::FftwTransformKind::realToComplex);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local real-to-complex plan");
        }
        detail::parallelForFftWithScratch(xExtent * yExtent * batchSize, detail::fftLineWork(meshSideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return detail::FftRealScratch{std::vector<double>(meshSideLength_), std::vector<std::complex<double>>(packedPlanes + 1)};
                },
                "FftPencilBoundary allocate forward real scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& scratch) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const line = item / batchSize;
                    size_t const x = line % xExtent;
                    size_t const y = line / xExtent;
                    auto& real = scratch.real;
                    auto& frequency = scratch.frequency;
                    for (size_t z = 0; z < meshSideLength_; z++) {
                        real[z] = pencil[(z * yExtent + y) * xExtent + x][lane];
                    }
                    fftw_execute_dft_r2c(transform, real.data(), reinterpret_cast<fftw_complex*>(frequency.data()));
                    for (size_t k = 0; k < packedPlanes; k++) {
                        packed[(k * yExtent + y) * xExtent + x][lane] =
                            k == 0 ? std::complex<double>(frequency[0].real(), frequency[packedPlanes].real()) : frequency[k];
                    }
                },
                "FftPencilBoundary execute forward real line"));
        return packed;
    }

    std::vector<RealBatch> inverseZ(std::span<ComplexBatch const> packed) const {
        PROFILE();
        size_t const xExtent = meshSideLength_ / (2 * localitySideLength_ * localitySideLength_);
        size_t const yExtent = 2 * blockSideLength_;
        size_t const packedPlanes = meshSideLength_ / 2;
        std::vector<RealBatch> pencil(xExtent * yExtent * meshSideLength_);
        fftw_plan const transform = detail::memoizedFftwPlan(meshSideLength_, detail::FftwTransformKind::complexToReal);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local complex-to-real plan");
        }
        double const normalization = 1.0 / static_cast<double>(meshSideLength_) / static_cast<double>(meshSideLength_) / static_cast<double>(meshSideLength_);
        detail::parallelForFftWithScratch(xExtent * yExtent * batchSize, detail::fftLineWork(meshSideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return detail::FftRealScratch{std::vector<double>(meshSideLength_), std::vector<std::complex<double>>(packedPlanes + 1)};
                },
                "FftPencilBoundary allocate inverse real scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& scratch) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const line = item / batchSize;
                    size_t const x = line % xExtent;
                    size_t const y = line / xExtent;
                    auto& real = scratch.real;
                    auto& frequency = scratch.frequency;
                    auto const endpoints = packed[y * xExtent + x][lane];
                    frequency[0] = std::complex<double>(endpoints.real(), 0.0);
                    frequency[packedPlanes] = std::complex<double>(endpoints.imag(), 0.0);
                    for (size_t k = 1; k < packedPlanes; k++) {
                        frequency[k] = packed[(k * yExtent + y) * xExtent + x][lane];
                    }
                    fftw_execute_dft_c2r(transform, reinterpret_cast<fftw_complex*>(frequency.data()), real.data());
                    for (size_t z = 0; z < meshSideLength_; z++) {
                        pencil[(z * yExtent + y) * xExtent + x][lane] = real[z] * normalization;
                    }
                },
                "FftPencilBoundary execute inverse real line"));
        return pencil;
    }

    void transformXY(std::vector<ComplexBatch>& plane, int direction) const {
        PROFILE();
        fftw_plan const transform = detail::memoizedFftwPlan(
            meshSideLength_, direction == FFTW_FORWARD ? detail::FftwTransformKind::complexForward : detail::FftwTransformKind::complexBackward);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local x/y plan");
        }
        detail::parallelForFftWithScratch(meshSideLength_ * batchSize, detail::fftLineWork(meshSideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(meshSideLength_);
                },
                "FftPencilBoundary allocate row scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& line) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const y = item / batchSize;
                    for (size_t x = 0; x < meshSideLength_; x++) {
                        line[x] = plane[y * meshSideLength_ + x][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(line.data());
                    fftw_execute_dft(transform, transformArray, transformArray);
                    for (size_t x = 0; x < meshSideLength_; x++) {
                        plane[y * meshSideLength_ + x][lane] = line[x];
                    }
                },
                "FftPencilBoundary transform row"));
        detail::parallelForFftWithScratch(meshSideLength_ * batchSize, detail::fftLineWork(meshSideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(meshSideLength_);
                },
                "FftPencilBoundary allocate column scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& line) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const x = item / batchSize;
                    for (size_t y = 0; y < meshSideLength_; y++) {
                        line[y] = plane[y * meshSideLength_ + x][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(line.data());
                    fftw_execute_dft(transform, transformArray, transformArray);
                    for (size_t y = 0; y < meshSideLength_; y++) {
                        plane[y * meshSideLength_ + x][lane] = line[y];
                    }
                },
                "FftPencilBoundary transform column"));
    }

    size_t meshSideLength_;
    size_t globalLocalityCount_;
    size_t currentLocalityId_;
    FftCommunicationPlan communicationPlan_;
    size_t sideBits_ = 0;
    size_t localitySideBits_ = 0;
    size_t localitySideLength_ = 0;
    size_t blockSideLength_ = 0;
    size_t blockValueCount_ = 0;
    size_t halfPlaneValueCount_ = 0;
};

}    // namespace cosmictiger
