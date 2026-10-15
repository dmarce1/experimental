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

#include <cosmictiger/gravity/kick/kick_sums.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>

#include <hpx/collectives/all_reduce.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/string.hpp>

#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cosmictiger::gravity {

namespace detail {

    struct KickReduction {
        KickSums sums;
        size_t potentialSites;
        std::string error;
    };

    template <typename Archive>
    void serialize(Archive& archive, KickReduction& reduction, unsigned) {
        PROFILE();
        archive & reduction.sums & reduction.potentialSites & reduction.error;
    }

}    // namespace detail

KickSums addKickSums(KickSums const& first, KickSums const& second) {
    PROFILE();
    if (second.particleCount > std::numeric_limits<ParticleCount>::max() - first.particleCount) {
        throw std::overflow_error("kick diagnostic particle count overflow");
    }
    FpeGuard guard;
    KickSums result;
    for (size_t axis = 0; axis < NDIM; axis++) {
        result.acceleration[axis] = first.acceleration[axis] + second.acceleration[axis];
    }
    result.potential = first.potential + second.potential;
    result.particleCount = first.particleCount + second.particleCount;
    result.momentumSquared = first.momentumSquared + second.momentumSquared;
    result.maximumRung = first.maximumRung > second.maximumRung ? first.maximumRung : second.maximumRung;
    return result;
}

KickSums allReduceKickSums(KickSums local, bool includesPotential, std::uint64_t operationId, size_t siteCount, size_t siteIndex, std::string localError) {
    PROFILE();
    // Invalid participation metadata cannot safely join a collective.
    if (siteCount == 0 || siteIndex >= siteCount) {
        throw std::invalid_argument("invalid kick diagnostic collective site");
    }
    if (!includesPotential) {
        local.potential = 0.0;
    }
    if (!localError.empty()) {
        local = {};
    }
    std::string const name = "/cosmictiger/kick/sums/" + std::to_string(operationId);
    auto const result =
        hpx::collectives::all_reduce(name.c_str(), detail::KickReduction{local, includesPotential ? size_t(1) : size_t(0), std::move(localError)},
            hpx::annotated_function(
                [](detail::KickReduction const& first, detail::KickReduction const& second) {
                    PROFILE();
                    size_t const potentialSites = first.potentialSites + second.potentialSites;
                    if (!first.error.empty() || !second.error.empty()) {
                        return detail::KickReduction{{}, potentialSites, first.error.empty() ? second.error : first.error};
                    }
                    try {
                        return detail::KickReduction{addKickSums(first.sums, second.sums), potentialSites, {}};
                    } catch (std::exception const& error) {
                        return detail::KickReduction{{}, potentialSites, error.what()};
                    }
                },
                "kickSums.reduceDiagnostics"),
            hpx::collectives::num_sites_arg(siteCount), hpx::collectives::this_site_arg(siteIndex))
            .get();
    if (!result.error.empty()) {
        throw std::runtime_error("kick diagnostic collective failed: " + result.error);
    }
    if (result.potentialSites != 0 && result.potentialSites != siteCount) {
        throw std::invalid_argument("kick diagnostic potential selection differs between localities");
    }
    return result.sums;
}

KickSums allReduceKickSums(KickSums local, bool includesPotential, std::uint64_t operationId, std::string localError) {
    PROFILE();
    auto const localities = hpx::find_all_localities();
    auto const here = hpx::find_here();
    size_t siteIndex = 0;
    while (siteIndex < localities.size() && localities[siteIndex] != here) {
        siteIndex++;
    }
    return allReduceKickSums(local, includesPotential, operationId, localities.size(), siteIndex, std::move(localError));
}

}    // namespace cosmictiger::gravity
