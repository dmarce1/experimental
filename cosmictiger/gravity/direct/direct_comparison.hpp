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

#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <span>
#include <string>

namespace cosmictiger::gravity {

struct DirectErrorNorms {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double referenceL1 = 0.0;
    double referenceL2 = 0.0;
    double referenceLinf = 0.0;
    double relativeL1 = 0.0;
    double relativeL2 = 0.0;
    double relativeLinf = 0.0;
};

struct DirectComparison {
    size_t sourceCount = 0;
    size_t sinkCount = 0;
    DirectErrorNorms acceleration;
    DirectErrorNorms potential;
};

// Acceleration errors use vector magnitudes per sink. L1 is a mean, L2 is
// an RMS, and Linf is a maximum. Normalize by the corresponding reference
// norm, never by individual particle forces. A zero reference yields zero
// for zero error, otherwise infinity. Potential uses absolute scalar errors.
DirectComparison compareDirectFields(std::span<DirectSolveField const> reference, std::span<DirectSolveField const> approximation);

// Collective over all physical HPX localities with a fresh operationId.
// PMMM fields must be current and stored on particles. Only interior ranges
// participate, regardless of ghost flags. All owned sources are gathered;
// sampled sinks are split among their owning ranks. No momentum/rung changes.
// Sampling is reproducible for a fixed rank/cell/particle ordering.
DirectComparison compareDirectGravity(
    ParticleStorage const& particles, CartesianGrid const& grid, GlobalConstantsData const& constants, std::uint64_t operationId);

std::string formatDirectComparison(DirectComparison const& comparison);

}    // namespace cosmictiger::gravity
