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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/fft/fft_vector_radix.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_contraction.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_plummer_convolution.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_progress.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <optional>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger::gravity {

// One instance per grid locality. Every locality must call computeExpansions
// in the same sequence with a matching FFT exchange callback. The callback's
// partner IDs are FFT IDs (z fastest), while CartesianGrid IDs are x fastest.
// A full-complex vector-radix spectrum lets all source/Green tensor products
// share one distributed spectral layout without an exchange per contraction
// term. Consecutive coefficients share FFT batch lanes, including the constant
// Green field and the mass/dipole/trace background fields. Green spectra are
// cached after the first call; calls on one instance are sequential. The
// function replaces interior expansions and leaves ghosts untouched. Near-cell
// direct gravity is excluded by the Green filter.
template <int sourceOrder = kMultipoleOrder, int expansionOrder = kExpansionOrder, size_t batchSize = 4>
class PmmmPeriodicConvolution {
public:
    static_assert(sourceOrder >= 0 && sourceOrder <= kMultipoleOrder, "PMMM source order exceeds the cell moment order");
    static_assert(
        expansionOrder >= 2 && expansionOrder <= ::cosmictiger::gravity::kExpansionOrder, "PMMM periodic expansion requires the quadratic background");
    static_assert(batchSize > 0, "PMMM FFT batch must have at least one lane");
    static constexpr size_t sourceCoefficientCount() noexcept {
        PROFILE();
        return kSourceCoefficientCount_;
    }
    static constexpr size_t localCoefficientCount() noexcept {
        PROFILE();
        return kLocalCoefficientCount_;
    }
    static constexpr size_t greenCoefficientCount() noexcept {
        PROFILE();
        return kGreenCoefficientCount_;
    }
    using RealBatch = typename FftVectorRadix<batchSize>::RealBatch;
    using ComplexBatch = typename FftVectorRadix<batchSize>::ComplexBatch;
    using Spectrum = std::vector<ComplexBatch>;

    PmmmPeriodicConvolution(CartesianGridLayout const& layout, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5,
        int ghostWidth = kPmmmNeighborWidth, SofteningModel model = SofteningModel::ferrers, double epsilon = 0.0)
      : sideLength_(layout.globalSideLength())
      , localitySide_(layout.localitiesPerAxis())
      , gridLocality_(layout.localityIndex())
      , fftLocality_(fftIndex(layout))
      , blockSide_(layout.interiorSideLength())
      , spacing_(layout.spacing())
      , imageCutoff_(imageCutoff)
      , waveCutoff_(waveCutoff)
      , alphaBox_(alphaBox)
      , ghostWidth_(ghostWidth)
      , fft_(sideLength_, layout.localityCount(), fftLocality_)
      , terms_(pmmmContractionTerms<sourceOrder, expansionOrder>()) {
        PROFILE();
        if (ghostWidth_ < 1 || ghostWidth_ > kPmmmMaximumNeighborWidth) {
            throw std::invalid_argument("PMMM ghost width must be one or two cells");
        }
        if (sideLength_ < static_cast<size_t>(2 * ghostWidth_ + 1)) {
            throw std::invalid_argument("PMMM convolution requires distinct near cells within the periodic grid");
        }
        expect(!(imageCutoff < 0 || waveCutoff < 0 || !(alphaBox > 0.0)));
        if (model == SofteningModel::plummer) {
            plummer_.emplace(layout, epsilon, imageCutoff, waveCutoff, alphaBox, ghostWidth);
        }
    }

    size_t fftLocalityIndex() const noexcept {
        PROFILE();
        return fftLocality_;
    }

    template <typename Exchange>
    void computeExpansions(CartesianGrid& grid, Exchange&& exchange, PmmmProgressObserver const& observer = {}) {
        size_t constexpr kDipoleCount = NDIM + 1;
        size_t constexpr kFieldCount = std::max(kSourceCoefficientCount_, kDipoleCount);
        size_t constexpr kLocalBatchCount = (kLocalCoefficientCount_ + batchSize - 1) / batchSize;
        size_t constexpr kBackgroundBatchCount = (kDipoleCount + batchSize) / batchSize;
        PROFILE();
        [[maybe_unused]] auto const& layout = grid.layout();
        expect(!(layout.globalSideLength() != sideLength_ || layout.localitiesPerAxis() != localitySide_ || layout.localityIndex() != gridLocality_));
        if (plummer_) {
            plummer_->computeExpansions(grid, exchange, observer);
            return;
        }
        if (greenSpectra_.empty()) {
            createFilters(exchange, observer);
        }

        std::vector<Spectrum> sourceSpectra((kFieldCount + batchSize) / batchSize);
        std::vector<RealBatch> input(fft_.localValueCount());
        detail::reportPmmmProgress(observer, PmmmProgressStage::multipoleFft, 0, sourceSpectra.size());
        for (size_t batch = 0; batch < sourceSpectra.size(); batch++) {
            ::cosmictiger::detail::parallelForFft(input.size(),
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        auto const& cell = grid.cell(localCoordinates(index));
                        FpeGuard fpeGuard;
                        input[index] = {};
                        for (size_t lane = 0; lane < batchSize; lane++) {
                            size_t const coefficient = batch * batchSize + lane;
                            if (coefficient == kFieldCount) {
                                input[index][lane] = cell.secondMomentTrace;
                            } else {
                                if (coefficient < kFieldCount && coefficient < cell.multipole.size()) {
                                    input[index][lane] = static_cast<double>(cell.multipole[coefficient]);
                                }
                            }
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            sourceSpectra[batch] = fft_.forward(input, exchange);
            detail::reportPmmmProgress(observer, PmmmProgressStage::multipoleFft, batch + 1, sourceSpectra.size());
        }

        using LocalValues = std::array<double, kLocalCoefficientCount_ + 1>;
        std::vector<LocalValues> pending(input.size());
        Spectrum localSpectrum(fft_.localValueCount());
        size_t termCursor = 0;
        detail::reportPmmmProgress(observer, PmmmProgressStage::contraction, 0, kLocalBatchCount);
        for (size_t firstLocal = 0; firstLocal < kLocalCoefficientCount_; firstLocal += batchSize) {
            std::array<size_t, batchSize + 1> termOffsets{};
            termOffsets[0] = termCursor;
            for (size_t lane = 0; lane < batchSize; lane++) {
                size_t const local = firstLocal + lane;
                while (termCursor < terms_.size() && terms_[termCursor].localCoefficient == local) {
                    termCursor++;
                }
                termOffsets[lane + 1] = termCursor;
            }
            ::cosmictiger::detail::parallelForFft(localSpectrum.size(),
                hpx::annotated_function(
                    [&](size_t frequency) {
                        PROFILE();
                        FpeGuard fpeGuard;
                        for (size_t lane = 0; lane < batchSize; lane++) {
                            std::complex<double> sum{};
                            for (size_t termIndex = termOffsets[lane]; termIndex < termOffsets[lane + 1]; termIndex++) {
                                auto const& term = terms_[termIndex];
                                sum += term.weight * sourceSpectra[term.sourceCoefficient / batchSize][frequency][term.sourceCoefficient % batchSize] *
                                    greenSpectra_[term.greenCoefficient / batchSize][frequency][term.greenCoefficient % batchSize];
                            }
                            localSpectrum[frequency][lane] = sum;
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            detail::reportPmmmProgress(observer, PmmmProgressStage::contraction, firstLocal / batchSize + 1, kLocalBatchCount);
            if (firstLocal == 0) {
                detail::reportPmmmProgress(observer, PmmmProgressStage::expansionFft, 0, kLocalBatchCount);
            }
            auto const values = fft_.inverse(localSpectrum, exchange);
            ::cosmictiger::detail::parallelForFft(values.size(),
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        for (size_t lane = 0; lane < batchSize && firstLocal + lane < kLocalCoefficientCount_; lane++) {
                            pending[index][firstLocal + lane] = values[index][lane];
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            detail::reportPmmmProgress(observer, PmmmProgressStage::expansionFft, firstLocal / batchSize + 1, kLocalBatchCount);
        }

        using BackgroundValues = std::array<double, kDipoleCount + 1>;
        std::vector<BackgroundValues> backgroundFields(input.size());
        detail::reportPmmmProgress(observer, PmmmProgressStage::periodicBackground, 0, kBackgroundBatchCount + 1);
        for (size_t firstField = 0; firstField < kDipoleCount + 1; firstField += batchSize) {
            ::cosmictiger::detail::parallelForFft(localSpectrum.size(),
                hpx::annotated_function(
                    [&](size_t frequency) {
                        PROFILE();
                        FpeGuard fpeGuard;
                        localSpectrum[frequency] = {};
                        auto const constant = greenSpectra_[kGreenCoefficientCount_ / batchSize][frequency][kGreenCoefficientCount_ % batchSize];
                        for (size_t lane = 0; lane < batchSize && firstField + lane < kDipoleCount + 1; lane++) {
                            size_t const field = firstField + lane;
                            size_t const source = field == kDipoleCount ? kFieldCount : field;
                            localSpectrum[frequency][lane] = sourceSpectra[source / batchSize][frequency][source % batchSize] * constant;
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            auto const values = fft_.inverse(localSpectrum, exchange);
            ::cosmictiger::detail::parallelForFft(values.size(),
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        for (size_t lane = 0; lane < batchSize && firstField + lane < kDipoleCount + 1; lane++) {
                            backgroundFields[index][firstField + lane] = values[index][lane];
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            detail::reportPmmmProgress(observer, PmmmProgressStage::periodicBackground, firstField / batchSize + 1, kBackgroundBatchCount + 1);
        }

        ::cosmictiger::detail::parallelForFft(pending.size(),
            hpx::annotated_function(
                [&](size_t index) {
                    PROFILE();
                    FpeGuard fpeGuard;
                    auto const& background = backgroundFields[index];
                    std::array<double, NDIM> const totalDipole = {
                        background[stf::index(1, 0, 0)], background[stf::index(0, 1, 0)], background[stf::index(0, 0, 1)]};
                    pmmmAddPeriodicBackground<double, expansionOrder>(
                        pending[index], background[0], totalDipole, background[kDipoleCount], sideLength_, spacing_);
                    auto& expansion = grid.expansion(localCoordinates(index));
                    expansion = {};
                    for (size_t coefficient = 0; coefficient < kLocalCoefficientCount_; coefficient++) {
                        expansion[coefficient] = static_cast<float>(pending[index][coefficient]);
                    }
                    expansion.back() = static_cast<float>(pending[index].back());
                },
                "pmmmConvolution.parallelForFft"));
        detail::reportPmmmProgress(observer, PmmmProgressStage::periodicBackground, kBackgroundBatchCount + 1, kBackgroundBatchCount + 1);
    }

private:
    static constexpr size_t kSourceCoefficientCount_ = stf::coefficientCount(sourceOrder);
    static constexpr size_t kLocalCoefficientCount_ = stf::coefficientCount(expansionOrder);
    static constexpr size_t kGreenCoefficientCount_ = stf::coefficientCount(sourceOrder + expansionOrder);

    static size_t fftIndex(CartesianGridLayout const& layout) noexcept {
        PROFILE();
        auto const coordinates = layout.localityCoordinates();
        size_t const side = layout.localitiesPerAxis();
        return (static_cast<size_t>(coordinates[0]) * side + static_cast<size_t>(coordinates[1])) * side + static_cast<size_t>(coordinates[2]);
    }

    GridIndices localCoordinates(size_t index) const noexcept {
        PROFILE();
        size_t const x = index % blockSide_;
        size_t const y = (index / blockSide_) % blockSide_;
        size_t const z = index / (blockSide_ * blockSide_);
        return {static_cast<std::int64_t>(x), static_cast<std::int64_t>(y), static_cast<std::int64_t>(z)};
    }

    template <typename Exchange>
    void createFilters(Exchange&& exchange, PmmmProgressObserver const& observer) {
        size_t constexpr kGreenBatchCount = (kGreenCoefficientCount_ + batchSize) / batchSize;
        PROFILE();
        auto const blocks = ghostWidth_ == 1 ? pmmmPeriodicGreenFilterBlock<sourceOrder + expansionOrder, 1, kGreenCoefficientCount_>(
                                                   sideLength_, localitySide_, fftLocality_, spacing_, 0, imageCutoff_, waveCutoff_, alphaBox_, observer) :
                                               pmmmPeriodicGreenFilterBlock<sourceOrder + expansionOrder, 2, kGreenCoefficientCount_>(
                                                   sideLength_, localitySide_, fftLocality_, spacing_, 0, imageCutoff_, waveCutoff_, alphaBox_, observer);
        std::vector<RealBatch> input(fft_.localValueCount());
        std::vector<Spectrum> nextGreenSpectra;

        nextGreenSpectra.reserve(kGreenBatchCount);
        detail::reportPmmmProgress(observer, PmmmProgressStage::filterFft, 0, kGreenBatchCount);
        for (size_t batch = 0; batch < kGreenBatchCount; batch++) {
            ::cosmictiger::detail::parallelForFft(input.size(),
                hpx::annotated_function(
                    [&](size_t index) {
                        PROFILE();
                        input[index] = {};
                        for (size_t lane = 0; lane < batchSize; lane++) {
                            size_t const coefficient = batch * batchSize + lane;
                            if (coefficient < kGreenCoefficientCount_) {
                                input[index][lane] = blocks[index][coefficient];
                            } else {
                                if (coefficient == kGreenCoefficientCount_) {
                                    input[index][lane] = 1.0;
                                }
                            }
                        }
                    },
                    "pmmmConvolution.parallelForFft"));
            nextGreenSpectra.push_back(fft_.forward(input, exchange));
            detail::reportPmmmProgress(observer, PmmmProgressStage::filterFft, batch + 1, kGreenBatchCount);
        }
        greenSpectra_.swap(nextGreenSpectra);
    }

    size_t sideLength_;
    size_t localitySide_;
    size_t gridLocality_;
    size_t fftLocality_;
    size_t blockSide_;
    double spacing_;
    int imageCutoff_;
    int waveCutoff_;
    double alphaBox_;
    int ghostWidth_;
    FftVectorRadix<batchSize> fft_;
    std::vector<PmmmContractionTerm> terms_;
    std::vector<Spectrum> greenSpectra_;
    std::optional<PmmmPlummerConvolution<sourceOrder, expansionOrder, batchSize>> plummer_;
};

// One-shot form when Green spectra need not be reused on a later step.
template <typename Exchange>
void computePmmmExpansions(CartesianGrid& grid, Exchange&& exchange, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5,
    int ghostWidth = kPmmmNeighborWidth, SofteningModel model = SofteningModel::ferrers, double epsilon = 0.0) {
    PROFILE();
    PmmmPeriodicConvolution<> convolution(grid.layout(), imageCutoff, waveCutoff, alphaBox, ghostWidth, model, epsilon);
    convolution.computeExpansions(grid, exchange);
}

}    // namespace cosmictiger::gravity
