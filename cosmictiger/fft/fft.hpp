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

#include <cosmictiger/fft/fft_communication_plan.hpp>
#include <cosmictiger/fft/fft_packed_vector_radix.hpp>
#include <cosmictiger/fft/fft_parallel.hpp>
#include <cosmictiger/fft/fft_pencil_boundary.hpp>
#include <cosmictiger/fft/fft_pencil_slab.hpp>
#include <cosmictiger/fft/fft_vector_radix.hpp>

#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger {

// One instance operates on the local block of one HPX locality. All
// participating localities construct this with the same M, P, weights, and
// batchSize, then call methods in the same order. The exchange callback
// transfers batched spans to a partner; FftHpxExchange provides this transport.
// Pass that transport directly to enable bounded asynchronous pencil/slab
// redistributions; a callback exposing only operator() remains synchronous.
// Each batch lane is an independent 3-D real FFT or circular convolution.
template <size_t batchSize>
class FftDistributed {
public:
    static_assert(batchSize > 0);
    using RealBatch = std::array<double, batchSize>;
    using ComplexBatch = std::array<std::complex<double>, batchSize>;

    FftDistributed(size_t sideLength, size_t localityCount, size_t locality, double weightPerByte, double weightPerMessage,
        FftCommunicationWorkload workload = FftCommunicationWorkload::prepositionedMultiplier)
      : sideLength_(sideLength)
      , localityCount_(localityCount)
      , locality_(locality)
      , plan_(findOptimalFftCommunicationPlan<double, batchSize>(totalValues(sideLength), localityCount, weightPerByte, weightPerMessage, workload)) {
        PROFILE();
        expectIndex(locality_, localityCount_);
        switch (plan_.strategy) {
        case FftCommunicationStrategy::pencilSlab:
            if (localityCount_ == sideLength_) {
                pencilBoundary_ = std::make_unique<FftPencilBoundary<batchSize>>(sideLength_, localityCount_, locality_, plan_);
            } else {
                pencil_ = std::make_unique<FftPencilSlab<batchSize>>(sideLength_, localityCount_, locality_, plan_);
            }
            break;
        case FftCommunicationStrategy::packedVectorRadix:
            packed_ = std::make_unique<FftPackedVectorRadix<batchSize>>(sideLength_, localityCount_, locality_);
            break;
        case FftCommunicationStrategy::fullSpectrumVectorRadix:
            vector_ = std::make_unique<FftVectorRadix<batchSize>>(sideLength_, localityCount_, locality_);
            break;
        }
    }

    FftCommunicationPlan const& plan() const noexcept {
        PROFILE();
        return plan_;
    }

    size_t spectralValueCount() const noexcept {
        PROFILE();
        if (pencilBoundary_) {
            return pencilBoundary_->spectralValueCount();
        }
        if (vector_) {
            return vector_->localValueCount();
        }
        if (packed_) {
            return packed_->spectralValueCount();
        }
        return (sideLength_ * sideLength_ * sideLength_) / (2 * localityCount_);
    }

    template <typename Exchange>
    std::vector<ComplexBatch> forward(std::span<RealBatch const> block, Exchange&& exchange) {
        PROFILE();
        if (pencil_) {
            return pencil_->forward(block, exchange);
        }
        if (pencilBoundary_) {
            return pencilBoundary_->forward(block, exchange);
        }
        if (packed_) {
            return packed_->forward(block, exchange);
        }
        return vector_->forward(block, exchange);
    }

    template <typename Exchange>
    std::vector<RealBatch> inverse(std::span<ComplexBatch const> spectrum, Exchange&& exchange) {
        PROFILE();
        if (pencil_) {
            return pencil_->inverse(spectrum, exchange);
        }
        if (pencilBoundary_) {
            return pencilBoundary_->inverse(spectrum, exchange);
        }
        if (packed_) {
            return packed_->inverse(spectrum, exchange);
        }
        return vector_->inverse(spectrum, exchange);
    }

    // Both arguments are outputs of forward() using the same plan. With the
    // packed vector strategy, both operands cross the conjugate-partner link;
    // that adds one packed payload beyond the planner's prepositioned-kernel
    // model while retaining the same number of messages.
    template <typename Exchange>
    std::vector<ComplexBatch> multiplySpectra(std::span<ComplexBatch const> left, std::span<ComplexBatch const> right, Exchange&& exchange) {
        PROFILE();
        if (pencil_) {
            return pencil_->multiply(left, right);
        }
        if (pencilBoundary_) {
            return pencilBoundary_->multiply(left, right);
        }
        if (packed_) {
            return packed_->multiply(left, right, exchange);
        }
        return multiplyFull(left, right, spectralValueCount());
    }

    // For packed vector radix, multiplier contains the complete prepositioned
    // complex spectrum in full-vector layout (N/P complex values). Other
    // strategies accept their ordinary forward() spectral layout. This call
    // follows the planner's one-forward, one-inverse communication model.
    template <typename Exchange>
    std::vector<ComplexBatch> multiplyWithPrepositionedMultiplier(
        std::span<ComplexBatch const> spectrum, std::span<ComplexBatch const> multiplier, Exchange&& exchange) {
        PROFILE();
        if (pencil_) {
            return pencil_->multiply(spectrum, multiplier);
        }
        if (pencilBoundary_) {
            return pencilBoundary_->multiply(spectrum, multiplier);
        }
        if (packed_) {
            return packed_->multiplyWithPrepositionedMultiplier(spectrum, multiplier, exchange);
        }
        return multiplyFull(spectrum, multiplier, spectralValueCount());
    }

    template <typename Exchange>
    std::vector<RealBatch> convolve(std::span<RealBatch const> left, std::span<RealBatch const> right, Exchange&& exchange) {
        PROFILE();
        expectEqual(plan_.workload, FftCommunicationWorkload::twoForwardInputs);
        std::vector<ComplexBatch> const leftSpectrum = forward(left, exchange);
        std::vector<ComplexBatch> const rightSpectrum = forward(right, exchange);
        std::vector<ComplexBatch> const product = multiplySpectra(leftSpectrum, rightSpectrum, exchange);
        return inverse(product, exchange);
    }

    template <typename Exchange>
    std::vector<RealBatch> convolveWithPrepositionedMultiplier(
        std::span<RealBatch const> input, std::span<ComplexBatch const> multiplier, Exchange&& exchange) {
        PROFILE();
        expectEqual(plan_.workload, FftCommunicationWorkload::prepositionedMultiplier);
        std::vector<ComplexBatch> const spectrum = forward(input, exchange);
        std::vector<ComplexBatch> const product = multiplyWithPrepositionedMultiplier(spectrum, multiplier, exchange);
        return inverse(product, exchange);
    }

private:
    static size_t totalValues(size_t sideLength) {
        PROFILE();
        expectPositive(sideLength);
        expect(sideLength <= std::numeric_limits<size_t>::max() / sideLength);
        expect(sideLength * sideLength <= std::numeric_limits<size_t>::max() / sideLength);
        return sideLength * sideLength * sideLength;
    }

    static std::vector<ComplexBatch> multiplyFull(
        std::span<ComplexBatch const> left, std::span<ComplexBatch const> right, [[maybe_unused]] size_t expectedSize) {
        PROFILE();
        expectEqual(left.size(), expectedSize);
        expectEqual(right.size(), expectedSize);
        std::vector<ComplexBatch> product(left.size());
        detail::parallelForFft(left.size(),
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    for (size_t lane = 0; lane != batchSize; lane++) {
                        product[index][lane] = left[index][lane] * right[index][lane];
                    }
                },
                "FftDistributed multiply full spectra"),
            batchSize);
        return product;
    }

    size_t sideLength_;
    size_t localityCount_;
    size_t locality_;
    FftCommunicationPlan plan_;
    std::unique_ptr<FftPencilSlab<batchSize>> pencil_;
    std::unique_ptr<FftPencilBoundary<batchSize>> pencilBoundary_;
    std::unique_ptr<FftPackedVectorRadix<batchSize>> packed_;
    std::unique_ptr<FftVectorRadix<batchSize>> vector_;
};

}    // namespace cosmictiger
