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

#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_progress.hpp>
#include <cosmictiger/gravity/softening.hpp>

#include <memory>
#include <stddef.h>

namespace cosmictiger {
class CartesianGrid;
class CartesianGridLayout;
class FftHpxExchange;
}    // namespace cosmictiger

namespace cosmictiger::gravity {

// Selects a compiled source/local specialization once for each grid locality.
// The active local rank is fmmOrder and the active source rank is fmmOrder - 1.
// Cell storage retains its compiled capacity, with inactive local coefficients
// cleared by the selected convolution before CPU or CUDA particle evaluation.
class PmmmRuntimeConvolution {
public:
    PmmmRuntimeConvolution(CartesianGridLayout const& layout, int fmmOrder, int imageCutoff = 4, int waveCutoff = 4, double alphaBox = 1.5,
        int ghostWidth = kPmmmNeighborWidth, SofteningModel model = SofteningModel::ferrers, double epsilon = 0.0);
    ~PmmmRuntimeConvolution();

    PmmmRuntimeConvolution(PmmmRuntimeConvolution&&) noexcept;
    PmmmRuntimeConvolution& operator=(PmmmRuntimeConvolution&&) noexcept;
    PmmmRuntimeConvolution(PmmmRuntimeConvolution const&) = delete;
    PmmmRuntimeConvolution& operator=(PmmmRuntimeConvolution const&) = delete;

    size_t fftLocalityIndex() const noexcept;
    void computeExpansions(CartesianGrid& grid, FftHpxExchange& exchange, PmmmProgressObserver const& observer = {});

private:
    class Convolution;
    template <int fmmOrder>
    class Model;

    template <int candidateOrder>
    static std::unique_ptr<Convolution> makeConvolution(CartesianGridLayout const& layout, int fmmOrder, int imageCutoff, int waveCutoff, double alphaBox,
        int ghostWidth, SofteningModel model, double epsilon);

    std::unique_ptr<Convolution> convolution_;
};

}    // namespace cosmictiger::gravity
