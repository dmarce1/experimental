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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/lightcone/lightcone_options.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <cosmictiger/units.hpp>

#include <array>
#include <cstdint>
#include <stddef.h>
#include <vector>

namespace cosmictiger {

// Offsets and integrals share the original tick background, including when
// snapshots divide a tick. Keeping offsets separate avoids rounding a fine
// interval away when adding it to a much larger absolute conformal time.
struct LightconeStep {
    FriedmannState tickBackground;
    double tickConformalTime = 0.0;
    double beginOffset = 0.0;
    double endOffset = 0.0;
    double oldDrift = 0.0;
    double endDrift = 0.0;
    // Optional run-wide interpolant; standalone callers may use direct integration.
    SimulationBackgroundTable const* backgroundTable = nullptr;
    double tableTickOffset = 0.0;
};

struct LightconeCrossing {
    std::uint64_t pixel = 0;
    ParticleCount localParticleIndex = 0;
    std::array<std::int64_t, NDIM> image{};
    std::array<double, NDIM> position{};    // Observer-relative comoving box lengths.
    std::array<double, NDIM> velocity{};    // Peculiar velocity p/a in H0*L units.
    double conformalTime = 0.0;
    double tickOffset = 0.0;
    double scaleFactor = 0.0;
    double radius = 0.0;
    double inverseRadiusSquared = 0.0;
};

struct LightconeMapBin {
    std::uint64_t pixel = 0;
    std::uint64_t count = 0;
    double inverseRadiusSquared = 0.0;
    // Sum 1/(a_crossing * radius)^2; convert using the squared box length.
    double inverseProperRadiusSquared = 0.0;
};

struct LightconeSample {
    std::vector<LightconeCrossing> crossings;
    // Sorted, occupied pixels only; count and weight conserve the catalog.
    std::vector<LightconeMapBin> bins;
};

class Lightcone {
public:
    // lightSpeed=c/(H0 L); observationConformalTime is since original handoff.
    Lightcone(LightconeOptions options, double observationConformalTime, double lightSpeed);

    // The interval is (beginOffset,endOffset]. Positions are the still-undrifted
    // positions at beginOffset; momenta remain constant over this drift.
    // Only grid interior ownership ranges contribute. Ghost flags alone do not
    // establish ownership. Enumerates all periodic images inside maximumRadius.
    // Origin crossings have undefined sky direction and are deliberately omitted.
    // Requires subluminal peculiar motion and exclusive, synchronized host reads.
    LightconeSample sample(
        ParticleStorage const& particles, CartesianGrid const& grid, SimulationBackground const& background, LightconeStep const& step) const;

    // Per-run caller prefix plus step/rank make immutable output segments.
    // Writes .crossings.csv and .map.csv, then publishes .complete last.
    // Readers must require .complete; incomplete files after a process failure
    // are not committed output. Rejects all existing segment collisions.
    // Preserve native columns and append comoving Mpc and proper CGS views.
    // Both velocity views are peculiar; no Hubble-flow term is included.
    void write(
        LightconeSample const& sample, LightconeStep const& step, size_t localityIndex, std::uint64_t operationId, units::UnitScales const& scales) const;

private:
    LightconeOptions options_;
    double observationConformalTime_;
    double lightSpeed_;
};

}    // namespace cosmictiger
