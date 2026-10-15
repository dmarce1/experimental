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

#include <array>
#include <bit>
#include <climits>
#include <complex>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger {

// Each locality owns one row-major [z][y][x] block at the public boundary.
// Spectral output is packed by k_z: plane zero encodes both real endpoint
// planes, and the remaining planes are ordinary complex Fourier coefficients.
// Call multiply() on these spectra so the endpoint planes are unpacked before
// multiplication. FFTW's natural x/y order is retained throughout this path.
template <size_t batchSize>
class FftPencilSlab {
public:
    static_assert(batchSize > 0);
    using RealBatch = std::array<double, batchSize>;
    using ComplexBatch = std::array<std::complex<double>, batchSize>;

    FftPencilSlab(size_t sideLength, size_t localityCount, size_t locality, FftCommunicationPlan plan)
      : sideLength_(sideLength)
      , localityCount_(localityCount)
      , locality_(locality)
      , plan_(std::move(plan)) {
        PROFILE();
        expect(!(!std::has_single_bit(sideLength_) || !std::has_single_bit(localityCount_) || localityCount_ < 8 || locality_ >= localityCount_ ||
            sideLength_ > INT_MAX || plan_.strategy != FftCommunicationStrategy::pencilSlab || plan_.spectralLocalityCount != localityCount_ ||
            localityCount_ > sideLength_ / 2 || std::countr_zero(localityCount_) % NDIM != 0));
        sideBits_ = std::countr_zero(sideLength_);
        localitySideBits_ = std::countr_zero(localityCount_) / NDIM;
        localitySide_ = size_t(1) << localitySideBits_;
        localSide_ = sideLength_ / localitySide_;
        expect(!(sideBits_ < NDIM * localitySideBits_));
    }

    template <typename Exchange>
    std::vector<ComplexBatch> forward(std::span<RealBatch const> block, Exchange&& exchange) const {
        PROFILE();
        expectEqual(block.size(), localSide_ * localSide_ * localSide_);
        detail::FftBitLayout const blockLayout = realBlockLayout();
        detail::FftBitLayout const pencilLayout = realPencilLayout();
        std::vector<RealBatch> pencil =
            detail::routeFftBits(std::vector<RealBatch>(block.begin(), block.end()), locality_, blockLayout, pencilLayout, plan_.blockToPencil, exchange);
        std::vector<ComplexBatch> packed = forwardZ(pencil);
        detail::FftBitLayout const packedPencilLayout = complexPencilLayout();
        detail::FftBitLayout const slabLayout = complexSlabLayout();
        std::vector<ComplexBatch> slab = detail::routeFftBits(std::move(packed), locality_, packedPencilLayout, slabLayout, plan_.pencilToSlab, exchange);
        transformXY(slab, FFTW_FORWARD);
        return slab;
    }

    template <typename Exchange>
    std::vector<RealBatch> inverse(std::span<ComplexBatch const> spectrum, Exchange&& exchange) const {
        PROFILE();
        expectEqual(spectrum.size(), localSide_ * localSide_ * localSide_ / 2);
        std::vector<ComplexBatch> slab(spectrum.begin(), spectrum.end());
        transformXY(slab, FFTW_BACKWARD);
        detail::FftBitLayout const packedPencilLayout = complexPencilLayout();
        detail::FftBitLayout const slabLayout = complexSlabLayout();
        std::vector<ComplexBatch> packed =
            detail::routeFftBitsBackward(std::move(slab), locality_, packedPencilLayout, slabLayout, plan_.pencilToSlab, exchange);
        std::vector<RealBatch> pencil = inverseZ(packed);
        detail::FftBitLayout const blockLayout = realBlockLayout();
        detail::FftBitLayout const pencilLayout = realPencilLayout();
        return detail::routeFftBitsBackward(std::move(pencil), locality_, blockLayout, pencilLayout, plan_.blockToPencil, exchange);
    }

    std::vector<ComplexBatch> multiply(std::span<ComplexBatch const> left, std::span<ComplexBatch const> right) const {
        constexpr std::complex<double> kImaginary(0.0, 1.0);
        PROFILE();
        size_t const spectralCount = localSide_ * localSide_ * localSide_ / 2;
        expectEqual(left.size(), spectralCount);
        expectEqual(right.size(), spectralCount);
        std::vector<ComplexBatch> product(spectralCount);
        detail::parallelForFft(spectralCount,
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane != batchSize; lane++) {
                        product[index][lane] = left[index][lane] * right[index][lane];
                    }
                },
                "FftPencilSlab multiply spectra"),
            batchSize);
        if (locality_ == 0) {
            detail::parallelForFft(sideLength_ * sideLength_,
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        FpeGuard fpeGuard;
                        size_t const x = index % sideLength_;
                        size_t const y = index / sideLength_;
                        size_t const reflected = ((sideLength_ - y) % sideLength_) * sideLength_ + (sideLength_ - x) % sideLength_;
                        for (size_t lane = 0; lane != batchSize; lane++) {
                            auto const leftZero = (left[index][lane] + std::conj(left[reflected][lane])) * 0.5;
                            auto const leftNyquist = (left[index][lane] - std::conj(left[reflected][lane])) / (2.0 * kImaginary);
                            auto const rightZero = (right[index][lane] + std::conj(right[reflected][lane])) * 0.5;
                            auto const rightNyquist = (right[index][lane] - std::conj(right[reflected][lane])) / (2.0 * kImaginary);
                            product[index][lane] = leftZero * rightZero + kImaginary * leftNyquist * rightNyquist;
                        }
                    },
                    "FftPencilSlab multiply real endpoints"),
                batchSize);
        }
        return product;
    }

private:
    detail::FftBitLayout realBlockLayout() const {
        PROFILE();
        std::vector<size_t> bits;
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
            bits.push_back(NDIM * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
            bits.push_back(2 * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
            bits.push_back(sideBits_ - localitySideBits_ + bit);
        }
        return detail::FftBitLayout(NDIM * sideBits_, std::move(bits));
    }

    detail::FftBitLayout realPencilLayout() const {
        PROFILE();
        std::vector<size_t> bits;
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
            bits.push_back(sideBits_ - 2 * localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
            bits.push_back(2 * sideBits_ - localitySideBits_ + bit);
        }
        for (size_t bit = 0; bit != localitySideBits_; bit++) {
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
        for (size_t bit = 0; bit != NDIM * localitySideBits_; bit++) {
            bits.push_back(NDIM * sideBits_ - 1 - NDIM * localitySideBits_ + bit);
        }
        return detail::FftBitLayout(NDIM * sideBits_ - 1, std::move(bits));
    }

    std::vector<ComplexBatch> forwardZ(std::span<RealBatch const> pencil) const {
        PROFILE();
        size_t const xExtent = sideLength_ / (localitySide_ * localitySide_);
        size_t const yExtent = localSide_;
        size_t const packedPlanes = sideLength_ / 2;
        std::vector<ComplexBatch> packed(xExtent * yExtent * packedPlanes);
        fftw_plan const transform = detail::memoizedFftwPlan(sideLength_, detail::FftwTransformKind::realToComplex);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local real-to-complex plan");
        }
        detail::parallelForFftWithScratch(xExtent * yExtent * batchSize, detail::fftLineWork(sideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return detail::FftRealScratch{std::vector<double>(sideLength_), std::vector<std::complex<double>>(packedPlanes + 1)};
                },
                "FftPencilSlab allocate forward real scratch"),
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
                    for (size_t z = 0; z < sideLength_; z++) {
                        real[z] = pencil[(z * yExtent + y) * xExtent + x][lane];
                    }
                    fftw_execute_dft_r2c(transform, real.data(), reinterpret_cast<fftw_complex*>(frequency.data()));
                    for (size_t k = 0; k < packedPlanes; k++) {
                        packed[(k * yExtent + y) * xExtent + x][lane] =
                            k == 0 ? std::complex<double>(frequency[0].real(), frequency[packedPlanes].real()) : frequency[k];
                    }
                },
                "FftPencilSlab execute forward real line"));
        return packed;
    }

    std::vector<RealBatch> inverseZ(std::span<ComplexBatch const> packed) const {
        PROFILE();
        size_t const xExtent = sideLength_ / (localitySide_ * localitySide_);
        size_t const yExtent = localSide_;
        size_t const packedPlanes = sideLength_ / 2;
        std::vector<RealBatch> pencil(xExtent * yExtent * sideLength_);
        fftw_plan const transform = detail::memoizedFftwPlan(sideLength_, detail::FftwTransformKind::complexToReal);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local complex-to-real plan");
        }
        double const normalization = 1.0 / static_cast<double>(sideLength_ * sideLength_ * sideLength_);
        detail::parallelForFftWithScratch(xExtent * yExtent * batchSize, detail::fftLineWork(sideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return detail::FftRealScratch{std::vector<double>(sideLength_), std::vector<std::complex<double>>(packedPlanes + 1)};
                },
                "FftPencilSlab allocate inverse real scratch"),
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
                    for (size_t z = 0; z < sideLength_; z++) {
                        pencil[(z * yExtent + y) * xExtent + x][lane] = real[z] * normalization;
                    }
                },
                "FftPencilSlab execute inverse real line"));
        return pencil;
    }

    void transformXY(std::vector<ComplexBatch>& slab, int direction) const {
        PROFILE();
        size_t const planeSize = sideLength_ * sideLength_;
        size_t const planesPerLocality = (sideLength_ / 2) / localityCount_;
        fftw_plan const transform = detail::memoizedFftwPlan(
            sideLength_, direction == FFTW_FORWARD ? detail::FftwTransformKind::complexForward : detail::FftwTransformKind::complexBackward);
        if (transform == nullptr) {
            throw std::runtime_error("FFTW failed to create local x/y plan");
        }
        detail::parallelForFftWithScratch(planesPerLocality * sideLength_ * batchSize, detail::fftLineWork(sideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(sideLength_);
                },
                "FftPencilSlab allocate row scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& line) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const k = (item / batchSize) / sideLength_;
                    size_t const y = (item / batchSize) % sideLength_;
                    for (size_t x = 0; x < sideLength_; x++) {
                        line[x] = slab[k * planeSize + y * sideLength_ + x][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(line.data());
                    fftw_execute_dft(transform, transformArray, transformArray);
                    for (size_t x = 0; x < sideLength_; x++) {
                        slab[k * planeSize + y * sideLength_ + x][lane] = line[x];
                    }
                },
                "FftPencilSlab transform row"));
        detail::parallelForFftWithScratch(planesPerLocality * sideLength_ * batchSize, detail::fftLineWork(sideLength_),
            hpx::annotated_function(
                [&] {
                    PROFILE();
                    return std::vector<std::complex<double>>(sideLength_);
                },
                "FftPencilSlab allocate column scratch"),
            hpx::annotated_function(
                [&](size_t item, auto& line) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    size_t const lane = item % batchSize;
                    size_t const k = (item / batchSize) / sideLength_;
                    size_t const x = (item / batchSize) % sideLength_;
                    for (size_t y = 0; y < sideLength_; y++) {
                        line[y] = slab[k * planeSize + y * sideLength_ + x][lane];
                    }
                    auto* transformArray = reinterpret_cast<fftw_complex*>(line.data());
                    fftw_execute_dft(transform, transformArray, transformArray);
                    for (size_t y = 0; y < sideLength_; y++) {
                        slab[k * planeSize + y * sideLength_ + x][lane] = line[y];
                    }
                },
                "FftPencilSlab transform column"));
    }

    size_t sideLength_;
    size_t localityCount_;
    size_t locality_;
    FftCommunicationPlan plan_;
    size_t sideBits_ = 0;
    size_t localitySideBits_ = 0;
    size_t localitySide_ = 0;
    size_t localSide_ = 0;
};

}    // namespace cosmictiger
