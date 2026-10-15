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

#include <stdexcept>

namespace cosmictiger::gravity {

enum class SofteningModel { ferrers, plummer };

// The length is Ferrers support radius q or Plummer epsilon, respectively.
// Build once at startup and pass by value to CPU or CUDA kernels.
struct SofteningLength {
    float qSquared;
    float inverseQ;
    float inverseQSquared;
    float inverseQCubed;
    SofteningModel model = SofteningModel::ferrers;
};

inline SofteningLength makeSoftening(float length, SofteningModel model) {
    PROFILE();
    if (model != SofteningModel::ferrers && model != SofteningModel::plummer) {
        throw std::invalid_argument("unknown gravity softening model");
    }
    expectPositive(length);
    FpeGuard fpeGuard;
    float const inverseLength = 1.0f / length;
    float const inverseLengthSquared = inverseLength * inverseLength;
    return {length * length, inverseLength, inverseLengthSquared, inverseLengthSquared * inverseLength, model};
}

inline SofteningLength makeFerrersSoftening(float q) {
    PROFILE();
    return makeSoftening(q, SofteningModel::ferrers);
}

inline SofteningLength makePlummerSoftening(float epsilon) {
    PROFILE();
    return makeSoftening(epsilon, SofteningModel::plummer);
}

}    // namespace cosmictiger::gravity
