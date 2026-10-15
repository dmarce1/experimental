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

#include <cosmictiger/profiler.hpp>
#include <cstdint>

namespace cosmictiger {

// W = integral (2K+U) d ln(a) is signed and includes contraction. The
// Layzer-Irvine invariant is K+U+W, referenced to the original first sample.
struct EnergyHistory {
    bool initialized = false;
    double initialEnergy = 0.0;
    double expansionWork = 0.0;
    double previousScaleFactor = 0.0;
    double previousKinetic = 0.0;
    double previousPotential = 0.0;
    double previousWorkRate = 0.0;
    double kinetic = 0.0;
    double potential = 0.0;
    double conservedEnergy = 0.0;
};

// The last processed event lies at (nominalEpoch+timeTick/2^63)*dtau0.
// An arbitrary stop can lie conformalOffset later without changing the
// event lattice or the staggered particle momenta retained by a restart.
struct EvolutionState {
    double scaleFactor = 0.0;
    double tickScaleFactor = 0.0;
    double conformalDerivative = 0.0;
    double tickConformalDerivative = 0.0;
    double physicalTime = 0.0;
    double tickPhysicalTime = 0.0;
    double conformalOffset = 0.0;
    std::uint64_t nominalEpoch = 0;
    std::uint64_t timeTick = 0;
    std::uint64_t operationId = 0;
    std::uint8_t maximumRung = 0;
    bool momentaStaggered = false;
    EnergyHistory energy;
};

struct EvolutionEvent {
    std::uint64_t nominalEpoch;
    std::uint64_t timeTick;
};

EvolutionEvent nextEvolutionEvent(EvolutionState const& state);
std::uint8_t activeEvolutionRung(std::uint64_t timeTick);

// Called before publishing the new run constants and before modifying
// particle rungs. Rejects an inexact or out-of-capacity remapping first.
void extendEvolutionClock(EvolutionState& state, unsigned rungOffset);
void updateEnergyHistory(EnergyHistory& history, double scaleFactor, double kinetic, double potential, double conformalDerivative, double elapsedConformalTime);

template <typename Archive>
void serialize(Archive& archive, EnergyHistory& history, unsigned) {
    PROFILE();
    archive & history.initialized & history.initialEnergy & history.expansionWork & history.previousScaleFactor & history.previousKinetic &
        history.previousPotential & history.previousWorkRate & history.kinetic & history.potential & history.conservedEnergy;
}

template <typename Archive>
void serialize(Archive& archive, EvolutionState& state, unsigned) {
    PROFILE();
    archive & state.scaleFactor & state.tickScaleFactor & state.conformalDerivative & state.tickConformalDerivative & state.physicalTime &
        state.tickPhysicalTime & state.conformalOffset & state.nominalEpoch & state.timeTick & state.operationId & state.maximumRung & state.momentaStaggered &
        state.energy;
}

}    // namespace cosmictiger
