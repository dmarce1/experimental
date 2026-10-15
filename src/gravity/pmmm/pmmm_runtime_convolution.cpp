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

#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_runtime_convolution.hpp>
#include <cosmictiger/profiler.hpp>

#include <stdexcept>

namespace cosmictiger::gravity {

class PmmmRuntimeConvolution::Convolution {
public:
    virtual ~Convolution() = default;
    virtual size_t fftLocalityIndex() const noexcept = 0;
    virtual void computeExpansions(CartesianGrid& grid, FftHpxExchange& exchange, PmmmProgressObserver const& observer) = 0;
};

template <int fmmOrder>
class PmmmRuntimeConvolution::Model final : public Convolution {
public:
    Model(CartesianGridLayout const& layout, int imageCutoff, int waveCutoff, double alphaBox, int ghostWidth, SofteningModel model, double epsilon)
      : convolution_(layout, imageCutoff, waveCutoff, alphaBox, ghostWidth, model, epsilon) {
        PROFILE();
    }

    size_t fftLocalityIndex() const noexcept override {
        PROFILE();
        return convolution_.fftLocalityIndex();
    }

    void computeExpansions(CartesianGrid& grid, FftHpxExchange& exchange, PmmmProgressObserver const& observer) override {
        PROFILE();
        convolution_.computeExpansions(grid, exchange, observer);
    }

private:
    PmmmPeriodicConvolution<fmmOrder - 1, fmmOrder> convolution_;
};

template <int candidateOrder>
std::unique_ptr<PmmmRuntimeConvolution::Convolution> PmmmRuntimeConvolution::makeConvolution(
    CartesianGridLayout const& layout, int fmmOrder, int imageCutoff, int waveCutoff, double alphaBox, int ghostWidth, SofteningModel model, double epsilon) {
    PROFILE();
    if constexpr (candidateOrder <= kMaximumFmmOrder) {
        if (fmmOrder == candidateOrder) {
            return std::make_unique<Model<candidateOrder>>(layout, imageCutoff, waveCutoff, alphaBox, ghostWidth, model, epsilon);
        }
        if constexpr (candidateOrder < kMaximumFmmOrder) {
            return makeConvolution<candidateOrder + 1>(layout, fmmOrder, imageCutoff, waveCutoff, alphaBox, ghostWidth, model, epsilon);
        }
    }
    throw std::invalid_argument("gravity.fmmOrder is outside the supported local/source rank range");
}

PmmmRuntimeConvolution::PmmmRuntimeConvolution(
    CartesianGridLayout const& layout, int fmmOrder, int imageCutoff, int waveCutoff, double alphaBox, int ghostWidth, SofteningModel model, double epsilon)
  : convolution_(makeConvolution<kMinimumFmmOrder>(layout, fmmOrder, imageCutoff, waveCutoff, alphaBox, ghostWidth, model, epsilon)) {
    PROFILE();
}

PmmmRuntimeConvolution::~PmmmRuntimeConvolution() = default;
PmmmRuntimeConvolution::PmmmRuntimeConvolution(PmmmRuntimeConvolution&&) noexcept = default;
PmmmRuntimeConvolution& PmmmRuntimeConvolution::operator=(PmmmRuntimeConvolution&&) noexcept = default;

size_t PmmmRuntimeConvolution::fftLocalityIndex() const noexcept {
    PROFILE();
    return convolution_->fftLocalityIndex();
}

void PmmmRuntimeConvolution::computeExpansions(CartesianGrid& grid, FftHpxExchange& exchange, PmmmProgressObserver const& observer) {
    PROFILE();
    convolution_->computeExpansions(grid, exchange, observer);
}

}    // namespace cosmictiger::gravity
