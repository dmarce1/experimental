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
#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/profiler.hpp>
#include <stddef.h>

namespace cosmictiger::gravity {

// Full Cartesian contraction at one spatial Fourier frequency. Moments use
// raw, cell-normalized powers; Green entries are cell-normalized derivatives.
template <int sourceOrder, int expansionOrder>
class PmmmSymmetricContraction {
public:
    static constexpr size_t sourceCount() {
        PROFILE();
        return symmetric::coefficientCount(sourceOrder);
    }
    static constexpr size_t localCount() {
        PROFILE();
        return symmetric::coefficientCount(expansionOrder);
    }
    static constexpr size_t greenCount() {
        PROFILE();
        return symmetric::coefficientCount(sourceOrder + expansionOrder);
    }

    template <typename Value>
    static Value coefficient(size_t local, std::array<Value, sourceCount()> const& source, std::array<Value, greenCount()> const& green) {
        PROFILE();
        FpeGuard guard;
        expectIndex(local, localCount());
        Value result{};
        for (size_t beta = 0; beta < sourceCount(); beta++) {
            result += source[beta] * green[kGreenIndices_[local][beta]] * kWeights_[beta];
        }
        return result;
    }

    template <typename Value>
    static std::array<Value, localCount()> apply(std::array<Value, sourceCount()> const& source, std::array<Value, greenCount()> const& green) {
        PROFILE();
        std::array<Value, localCount()> result{};
        for (size_t alpha = 0; alpha < localCount(); alpha++) {
            result[alpha] = coefficient(alpha, source, green);
        }
        return result;
    }

    static constexpr double sourceWeight(size_t index) {
        PROFILE();
        return kWeights_[expectIndex(index, sourceCount())];
    }
    static constexpr size_t greenIndex(size_t local, size_t source) {
        PROFILE();
        return kGreenIndices_[expectIndex(local, localCount())][expectIndex(source, sourceCount())];
    }

private:
    static constexpr auto kWeights_ = [] {
        PROFILE();
        std::array<double, sourceOrder + 1> inverseFactorials{};
        inverseFactorials[0] = 1.0;
        for (int degree = 1; degree <= sourceOrder; degree++) {
            inverseFactorials[degree] = inverseFactorials[degree - 1] / degree;
        }
        std::array<double, symmetric::coefficientCount(sourceOrder)> weights{};
        for (int rank = 0; rank <= sourceOrder; rank++) {
            for (int z = 0; z <= rank; z++) {
                for (int x = 0; x <= rank - z; x++) {
                    int const y = rank - z - x;
                    weights[symmetric::index(x, y, z)] = (rank % 2 ? -1.0 : 1.0) * inverseFactorials[x] * inverseFactorials[y] * inverseFactorials[z];
                }
            }
        }
        return weights;
    }();

    static constexpr auto kGreenIndices_ = [] {
        PROFILE();
        std::array<std::array<size_t, symmetric::coefficientCount(sourceOrder)>, symmetric::coefficientCount(expansionOrder)> indices{};
        for (int localRank = 0; localRank <= expansionOrder; localRank++) {
            for (int lz = 0; lz <= localRank; lz++) {
                for (int lx = 0; lx <= localRank - lz; lx++) {
                    int const ly = localRank - lz - lx;
                    for (int sourceRank = 0; sourceRank <= sourceOrder; sourceRank++) {
                        for (int sz = 0; sz <= sourceRank; sz++) {
                            for (int sx = 0; sx <= sourceRank - sz; sx++) {
                                int const sy = sourceRank - sz - sx;
                                indices[symmetric::index(lx, ly, lz)][symmetric::index(sx, sy, sz)] = symmetric::index(lx + sx, ly + sy, lz + sz);
                            }
                        }
                    }
                }
            }
        }
        return indices;
    }();
};

}    // namespace cosmictiger::gravity
