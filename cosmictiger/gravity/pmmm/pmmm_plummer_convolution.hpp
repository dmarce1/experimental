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

#include <array>
#include <complex>
#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/fft/fft_vector_radix.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_plummer_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_symmetric_contraction.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <stdexcept>
#include <vector>

namespace cosmictiger::gravity {

// A distinct full-tensor PMMM path: Plummer's traces at every rank are already
// present in the periodic Green derivatives. No STF background is appended.
template <int sourceOrder = kMultipoleOrder, int expansionOrder = kExpansionOrder, size_t batchSize = 4>
class PmmmPlummerConvolution {
public:
    using Contraction = PmmmSymmetricContraction<sourceOrder, expansionOrder>;
    using Transform = FftVectorRadix<batchSize>;
    using RealBatch = typename Transform::RealBatch;
    using Spectrum = std::vector<typename Transform::ComplexBatch>;

    PmmmPlummerConvolution(CartesianGridLayout const& layout, double epsilon, int imageCutoff, int waveCutoff, double alphaBox, int ghostWidth)
      : side_(layout.globalSideLength())
      , localitySide_(layout.localitiesPerAxis())
      , blockSide_(layout.interiorSideLength())
      , fftLocality_(fftIndex(layout))
      , spacing_(layout.spacing())
      , epsilon_(epsilon)
      , imageCutoff_(imageCutoff)
      , waveCutoff_(waveCutoff)
      , alphaBox_(alphaBox)
      , ghostWidth_(ghostWidth)
      , fft_(side_, layout.localityCount(), fftLocality_) {
        static_assert(sourceOrder <= kMultipoleOrder && expansionOrder <= kExpansionOrder);
        PROFILE();
        if (ghostWidth_ < 1 || ghostWidth_ > kPmmmMaximumNeighborWidth) {
            throw std::invalid_argument("Plummer PMMM ghost width must be one or two cells");
        }
        if (side_ < static_cast<size_t>(2 * ghostWidth_ + 1)) {
            throw std::invalid_argument("Plummer PMMM requires distinct near cells within the periodic grid");
        }
        expectPositive(epsilon_);
        expectNonNegative(imageCutoff_);
        expectNonNegative(waveCutoff_);
        expectPositive(alphaBox_);
    }

    template <typename Exchange>
    void computeExpansions(CartesianGrid& grid, Exchange&& exchange, PmmmProgressObserver const& observer = {}) {
        constexpr size_t kSourceBatches = (Contraction::sourceCount() + batchSize - 1) / batchSize;
        constexpr size_t kLocalBatches = (Contraction::localCount() + batchSize - 1) / batchSize;
        PROFILE();
        auto const& layout = grid.layout();
        expectEqual(layout.globalSideLength(), side_);
        expectEqual(layout.localitiesPerAxis(), localitySide_);
        expectEqual(fftIndex(layout), fftLocality_);
        if (greens_.empty()) {
            createFilters(exchange, observer);
        }
        std::vector<RealBatch> input(fft_.localValueCount());
        std::vector<Spectrum> sources;
        sources.reserve(kSourceBatches);
        detail::reportPmmmProgress(observer, PmmmProgressStage::multipoleFft, 0, kSourceBatches);
        for (size_t batch = 0; batch < kSourceBatches; batch++) {
            ::cosmictiger::detail::parallelForFft(input.size(),
                hpx::annotated_function(
                    [&](size_t i) {
                        PROFILE();
                        input[i] = {};
                        auto const& moments = grid.cell(coordinates(i)).fullMultipole;
                        for (size_t lane = 0; lane < batchSize && batch * batchSize + lane < Contraction::sourceCount(); lane++) {
                            input[i][lane] = moments[batch * batchSize + lane];
                        }
                    },
                    "plummerPmmm.packMoments"));
            sources.push_back(fft_.forward(input, exchange));
            detail::reportPmmmProgress(observer, PmmmProgressStage::multipoleFft, batch + 1, kSourceBatches);
        }
        std::vector<std::array<double, Contraction::localCount()>> pending(input.size());
        Spectrum local(fft_.localValueCount());
        detail::reportPmmmProgress(observer, PmmmProgressStage::contraction, 0, kLocalBatches);
        for (size_t batch = 0; batch < kLocalBatches; batch++) {
            ::cosmictiger::detail::parallelForFft(local.size(),
                hpx::annotated_function(
                    [&](size_t frequency) {
                        PROFILE();
                        FpeGuard guard;
                        local[frequency] = {};
                        for (size_t lane = 0; lane < batchSize && batch * batchSize + lane < Contraction::localCount(); lane++) {
                            size_t const alpha = batch * batchSize + lane;
                            std::complex<double> sum{};
                            for (size_t beta = 0; beta < Contraction::sourceCount(); beta++) {
                                size_t const gamma = Contraction::greenIndex(alpha, beta);
                                sum += Contraction::sourceWeight(beta) * sources[beta / batchSize][frequency][beta % batchSize] *
                                    greens_[gamma / batchSize][frequency][gamma % batchSize];
                            }
                            local[frequency][lane] = sum;
                        }
                    },
                    "plummerPmmm.contract"));
            detail::reportPmmmProgress(observer, PmmmProgressStage::contraction, batch + 1, kLocalBatches);
            if (batch == 0) {
                detail::reportPmmmProgress(observer, PmmmProgressStage::expansionFft, 0, kLocalBatches);
            }
            auto const values = fft_.inverse(local, exchange);
            ::cosmictiger::detail::parallelForFft(values.size(),
                hpx::annotated_function(
                    [&](size_t i) {
                        PROFILE();
                        for (size_t lane = 0; lane < batchSize && batch * batchSize + lane < Contraction::localCount(); lane++) {
                            pending[i][batch * batchSize + lane] = values[i][lane];
                        }
                    },
                    "plummerPmmm.unpackLocals"));
            detail::reportPmmmProgress(observer, PmmmProgressStage::expansionFft, batch + 1, kLocalBatches);
        }
        ::cosmictiger::detail::parallelForFft(pending.size(),
            hpx::annotated_function(
                [&](size_t i) {
                    PROFILE();
                    auto& expansion = grid.fullExpansion(coordinates(i));
                    expansion = {};
                    for (size_t coefficient = 0; coefficient < Contraction::localCount(); coefficient++) {
                        expansion[coefficient] = static_cast<float>(pending[i][coefficient]);
                    }
                },
                "plummerPmmm.publishLocals"));
    }

private:
    static size_t fftIndex(CartesianGridLayout const& layout) {
        PROFILE();
        auto const c = layout.localityCoordinates();
        size_t const side = layout.localitiesPerAxis();
        return (static_cast<size_t>(c[0]) * side + static_cast<size_t>(c[1])) * side + static_cast<size_t>(c[2]);
    }
    GridIndices coordinates(size_t i) const {
        PROFILE();
        return {static_cast<int64_t>(i % blockSide_), static_cast<int64_t>((i / blockSide_) % blockSide_), static_cast<int64_t>(i / (blockSide_ * blockSide_))};
    }
    template <typename Exchange>
    void createFilters(Exchange&& exchange, PmmmProgressObserver const& observer) {
        constexpr size_t kBatches = (Contraction::greenCount() + batchSize - 1) / batchSize;
        PROFILE();
        auto const block = ghostWidth_ == 1 ? pmmmPlummerPeriodicGreenFilterBlock<sourceOrder + expansionOrder, 1, Contraction::greenCount()>(
                                                  side_, localitySide_, fftLocality_, spacing_, epsilon_, 0, imageCutoff_, waveCutoff_, alphaBox_, observer) :
                                              pmmmPlummerPeriodicGreenFilterBlock<sourceOrder + expansionOrder, 2, Contraction::greenCount()>(
                                                  side_, localitySide_, fftLocality_, spacing_, epsilon_, 0, imageCutoff_, waveCutoff_, alphaBox_, observer);
        std::vector<RealBatch> input(fft_.localValueCount());
        std::vector<Spectrum> spectra;
        spectra.reserve(kBatches);
        detail::reportPmmmProgress(observer, PmmmProgressStage::filterFft, 0, kBatches);
        for (size_t batch = 0; batch < kBatches; batch++) {
            ::cosmictiger::detail::parallelForFft(input.size(),
                hpx::annotated_function(
                    [&](size_t i) {
                        PROFILE();
                        input[i] = {};
                        for (size_t lane = 0; lane < batchSize && batch * batchSize + lane < Contraction::greenCount(); lane++) {
                            input[i][lane] = block[i][batch * batchSize + lane];
                        }
                    },
                    "plummerPmmm.packGreen"));
            spectra.push_back(fft_.forward(input, exchange));
            detail::reportPmmmProgress(observer, PmmmProgressStage::filterFft, batch + 1, kBatches);
        }
        greens_.swap(spectra);
    }
    size_t side_;
    size_t localitySide_;
    size_t blockSide_;
    size_t fftLocality_;
    double spacing_;
    double epsilon_;
    int imageCutoff_;
    int waveCutoff_;
    double alphaBox_;
    int ghostWidth_;
    Transform fft_;
    std::vector<Spectrum> greens_;
};
}    // namespace cosmictiger::gravity
