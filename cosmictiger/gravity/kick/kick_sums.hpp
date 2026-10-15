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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/serialization/array.hpp>

#include <array>
#include <cstdint>
#include <stddef.h>
#include <string>

namespace cosmictiger::gravity {

// Unweighted sums of complete raw gravity fields, g0 and phi. Count each
// evaluated owned sink once, after combining every source contribution.
// A restricted sink evaluation sums only that set. Evaluate all owned sinks
// for global diagnostics, independently of the minimum rung being kicked.
// These are neither mass-weighted force nor gravitational potential energy.
// momentumSquared sums synchronized |p|^2 without mass or scale-factor weights;
// maximumRung is the largest post-kick rung among evaluated owned particles.
struct KickSums {
    std::array<double, NDIM> acceleration{};
    double potential = 0.0;
    ParticleCount particleCount = 0;
    double momentumSquared = 0.0;
    std::uint32_t maximumRung = 0;
};

template <typename Archive>
void serialize(Archive& archive, KickSums& sums, unsigned) {
    PROFILE();
    archive & sums.acceleration & sums.potential & sums.particleCount & sums.momentumSquared & sums.maximumRung;
}

KickSums addKickSums(KickSums const& first, KickSums const& second);

// Collective on all physical HPX localities, returning the global sums to
// every caller. operationId must be globally unique for this collective.
// Every locality must call with the same includesPotential. A disagreement
// throws on every locality after the reduction completes. When false,
// input potential values are ignored and the returned potential is zero.
// Momentum squared is always summed and maximumRung is always maximized.
// Call after all local kick tasks and GPU completion futures have finished.
// Pass a local error after failed local work instead of skipping the collective:
// every participant then throws with one of the reported errors. Partial sums
// are discarded; particle updates already completed are not rolled back.
KickSums allReduceKickSums(KickSums local, bool includesPotential, std::uint64_t operationId, std::string localError = {});

// Explicit logical-site variant: sites can share an HPX locality. Every
// participant supplies the same siteCount and a distinct index in [0,count).
KickSums allReduceKickSums(KickSums local, bool includesPotential, std::uint64_t operationId, size_t siteCount, size_t siteIndex, std::string localError = {});

}    // namespace cosmictiger::gravity
