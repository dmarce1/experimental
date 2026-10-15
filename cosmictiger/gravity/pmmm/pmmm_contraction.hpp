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
#include <cosmictiger/gravity/pmmm/pmmm_green.hpp>
#include <cosmictiger/profiler.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <stddef.h>
#include <tuple>
#include <vector>

namespace cosmictiger::gravity {

// One contribution to a PMMM local coefficient. The Green coefficient is a
// spatial convolution kernel, so equal (sourceCoefficient, greenCoefficient)
// pairs can be convolved once and accumulated into localCoefficient.
struct PmmmContractionTerm {
    size_t localCoefficient;
    size_t sourceCoefficient;
    size_t greenCoefficient;
    double weight;
};

namespace detail {
    // A compact STF tensor stores only nz=0,1. Recover full components with
    // T(nx,ny,nz) = -T(nx+2,ny,nz-2) - T(nx,ny+2,nz-2).
    inline void pmmmAccumulateReducedComponent(std::map<size_t, std::int64_t>& coefficients, int nx, int ny, int nz, std::int64_t factor) {
        PROFILE();
        if (nz < 2) {
            coefficients[pmmmIndex(nx, ny, nz)] += factor;
            return;
        }
        pmmmAccumulateReducedComponent(coefficients, nx + 2, ny, nz - 2, -factor);
        pmmmAccumulateReducedComponent(coefficients, nx, ny + 2, nz - 2, -factor);
    }

    struct PmmmRationalWeight {
        std::int64_t numerator = 0;
        std::int64_t denominator = 1;
    };
}    // namespace detail

// Build the sparse tensor contraction for all source and local ranks. For a
// local index alpha and full source index beta, the term is
// (-1)^|beta| M_beta G_(alpha+beta)/(beta_x! beta_y! beta_z!). Both M and G
// are STF, and the source moments and Green derivatives use the cell spacing
// normalization already. This includes n+k>expansionOrder, unlike FMM M2L.
// The returned terms are unique and sorted by local, source, Green index.
template <int sourceOrder, int expansionOrder>
std::vector<PmmmContractionTerm> pmmmContractionTerms() {
    PROFILE();
    FpeGuard fpeGuard;
    static_assert(sourceOrder >= stf::kMinOrder && sourceOrder <= stf::kMaxOrder, "PMMM source order is outside the supported STF range");
    static_assert(expansionOrder >= stf::kMinOrder && expansionOrder <= stf::kMaxOrder, "PMMM expansion order is outside the supported STF range");
    static_assert(sourceOrder + expansionOrder <= 18, "PMMM Green rank is outside the supported STF range");

    std::array<std::int64_t, sourceOrder + 1> factorials{};
    factorials[0] = 1;
    for (int rank = 1; rank <= sourceOrder; rank++) {
        factorials[rank] = factorials[rank - 1] * rank;
    }

    using TermKey = std::tuple<size_t, size_t, size_t>;
    std::map<TermKey, detail::PmmmRationalWeight> rationalTerms;
    for (int localRank = 0; localRank <= expansionOrder; localRank++) {
        for (int localZ = 0; localZ <= 1 && localZ <= localRank; localZ++) {
            for (int localX = 0; localX <= localRank - localZ; localX++) {
                int const localY = localRank - localX - localZ;
                size_t const localIndex = detail::pmmmIndex(localX, localY, localZ);
                for (int sourceRank = 0; sourceRank <= sourceOrder; sourceRank++) {
                    for (int sourceZ = 0; sourceZ <= sourceRank; sourceZ++) {
                        for (int sourceX = 0; sourceX <= sourceRank - sourceZ; sourceX++) {
                            int const sourceY = sourceRank - sourceX - sourceZ;
                            std::int64_t const multinomial = factorials[sourceRank] / (factorials[sourceX] * factorials[sourceY] * factorials[sourceZ]);
                            std::int64_t const signedFactor = sourceRank % 2 == 0 ? multinomial : -multinomial;
                            std::map<size_t, std::int64_t> sourceComponents;
                            std::map<size_t, std::int64_t> greenComponents;
                            detail::pmmmAccumulateReducedComponent(sourceComponents, sourceX, sourceY, sourceZ, 1);
                            detail::pmmmAccumulateReducedComponent(greenComponents, localX + sourceX, localY + sourceY, localZ + sourceZ, 1);
                            for (auto const& [sourceIndex, sourceFactor] : sourceComponents) {
                                for (auto const& [greenIndex, greenFactor] : greenComponents) {
                                    auto& weight = rationalTerms[TermKey{localIndex, sourceIndex, greenIndex}];
                                    weight.denominator = factorials[sourceRank];
                                    weight.numerator += signedFactor * sourceFactor * greenFactor;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<PmmmContractionTerm> terms;
    terms.reserve(rationalTerms.size());
    for (auto const& [key, rational] : rationalTerms) {
        if (rational.numerator != 0) {
            terms.push_back(
                {std::get<0>(key), std::get<1>(key), std::get<2>(key), static_cast<double>(rational.numerator) / static_cast<double>(rational.denominator)});
        }
    }
    return terms;
}

}    // namespace cosmictiger::gravity
