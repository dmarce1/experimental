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

#include <cosmictiger/fft/fft.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cosmictiger {

// One bank lives on each participating locality. Create filters with the same
// names and in the same order on all localities, then use each local bank for
// later convolutions. Calls on an individual bank are sequential. The bank
// retains its FFT, including the inverse path, after startup filter creation.
template <size_t batchSize>
class FftFilterBank {
public:
    using RealBatch = typename FftDistributed<batchSize>::RealBatch;
    using ComplexBatch = typename FftDistributed<batchSize>::ComplexBatch;

    FftFilterBank(size_t sideLength, size_t localityCount, size_t locality, double weightPerByte, double weightPerMessage)
      : sideLength_(sideLength)
      , localityCount_(localityCount)
      , locality_(locality)
      , weightPerByte_(weightPerByte)
      , weightPerMessage_(weightPerMessage) {
        PROFILE();
        expectNonNegative(weightPerByte_);
        expectNonNegative(weightPerMessage_);
        expectIndex(locality, localityCount);
    }

    // Tuning is allowed until the first filter has been created. The selected
    // plan and filter layout then remain fixed for this bank's lifetime.
    void setCommunicationWeights(double newWeightPerByte, double newWeightPerMessage) {
        PROFILE();
        if (newWeightPerByte == weightPerByte_ && newWeightPerMessage == weightPerMessage_) {
            return;
        }
        expect(filters_.empty());
        weightPerByte_ = expectNonNegative(newWeightPerByte);
        weightPerMessage_ = expectNonNegative(newWeightPerMessage);
        fft_.reset();
    }

    // The real kernel uses the same local cubic block layout as convolution
    // input. Only the packed convolution path needs a different startup FFT:
    // its prepositioned multiplier occupies the full complex vector layout,
    // whereas its ordinary forward() result has half as many packed values.
    template <typename Exchange>
    void createFilter(std::string name, std::span<RealBatch const> kernel, Exchange&& exchange) {
        PROFILE();
        expect(!name.empty());
        expect(!filters_.contains(name));
        if (!fft_) {
            fft_ = std::make_unique<FftDistributed<batchSize>>(
                sideLength_, localityCount_, locality_, weightPerByte_, weightPerMessage_, FftCommunicationWorkload::prepositionedMultiplier);
        }
        std::vector<ComplexBatch> spectrum;
        if (fft_->plan().strategy == FftCommunicationStrategy::packedVectorRadix) {
            FftVectorRadix<batchSize> fullSpectrumFft(sideLength_, localityCount_, locality_);
            spectrum = fullSpectrumFft.forward(kernel, exchange);
        } else {
            spectrum = fft_->forward(kernel, exchange);
        }
        filters_.emplace(std::move(name), std::move(spectrum));
    }

    template <typename Exchange>
    std::vector<RealBatch> convolve(std::span<RealBatch const> input, std::string_view filterName, Exchange&& exchange) {
        PROFILE();
        auto const filter = filters_.find(filterName);
        expect(filter != filters_.end());
        return fft_->convolveWithPrepositionedMultiplier(input, filter->second, exchange);
    }

    size_t filterValueCount(std::string_view filterName) const {
        PROFILE();
        auto const filter = filters_.find(filterName);
        expect(filter != filters_.end());
        return filter->second.size();
    }

    FftCommunicationPlan const& plan() const {
        PROFILE();
        expect(!filters_.empty());
        return fft_->plan();
    }

private:
    size_t sideLength_;
    size_t localityCount_;
    size_t locality_;
    double weightPerByte_;
    double weightPerMessage_;
    std::unique_ptr<FftDistributed<batchSize>> fft_;
    std::map<std::string, std::vector<ComplexBatch>, std::less<>> filters_;
};

}    // namespace cosmictiger
