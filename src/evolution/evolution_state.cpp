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

#include <cosmictiger/evolution/evolution_state.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/profiler.hpp>

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosmictiger {

EvolutionEvent nextEvolutionEvent(EvolutionState const& state) {
    PROFILE();
    auto const nominalTicks = gravity::rungTicks(0);
    if (state.timeTick >= nominalTicks || state.maximumRung > Rung::maximumRung()) {
        throw std::invalid_argument("invalid evolution clock state");
    }
    auto const step = gravity::rungTicks(state.maximumRung);
    if ((state.timeTick & (step - 1)) != 0) {
        throw std::invalid_argument("evolution clock is not synchronized with its finest rung");
    }
    auto const nextTick = state.timeTick + step;
    if (nextTick == nominalTicks) {
        if (state.nominalEpoch == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("evolution clock exhausted its nominal epoch range");
        }
        return {state.nominalEpoch + 1, 0};
    }
    return {state.nominalEpoch, nextTick};
}

std::uint8_t activeEvolutionRung(std::uint64_t timeTick) {
    PROFILE();
    if (timeTick >= gravity::rungTicks(0)) {
        throw std::invalid_argument("evolution tick lies outside its nominal epoch");
    }
    if (timeTick == 0) {
        return 0;
    }
    // HPX has no countr_zero equivalent; integer bits identify the exact
    // coarsest synchronized rung without floating-point logarithms.
    return static_cast<std::uint8_t>(Rung::maximumRung() - std::countr_zero(timeTick));
}

void extendEvolutionClock(EvolutionState& state, unsigned rungOffset) {
    PROFILE();
    if (state.timeTick >= gravity::rungTicks(0) || state.maximumRung > Rung::maximumRung() || rungOffset > Rung::maximumRung() ||
        rungOffset > unsigned(Rung::maximumRung() - state.maximumRung)) {
        throw std::range_error("restart timestep extension exceeds the available rung range");
    }
    if (rungOffset == 0) {
        return;
    }
    std::uint64_t const mask = (std::uint64_t(1) << rungOffset) - 1;
    if ((state.timeTick & mask) != 0) {
        throw std::range_error("restart timestep extension cannot preserve the current clock exactly");
    }
    state.timeTick = ((state.nominalEpoch & mask) << (Rung::maximumRung() - rungOffset)) | (state.timeTick >> rungOffset);
    state.nominalEpoch >>= rungOffset;
    state.maximumRung = static_cast<std::uint8_t>(state.maximumRung + rungOffset);
}

void updateEnergyHistory(
    EnergyHistory& history, double scaleFactor, double kinetic, double potential, double conformalDerivative, double elapsedConformalTime) {
    constexpr double kKineticWorkFactor = 2.0;
    constexpr double kTrapezoidWeight = 0.5;
    PROFILE();

    FpeGuard guard;
    expectPositive(scaleFactor);
    expectNonNegative(kinetic);
    expectNonNegative(elapsedConformalTime);
    double const workRate = conformalDerivative / scaleFactor * (kKineticWorkFactor * kinetic + potential);
    if (history.initialized) {
        // Integrate in monotone conformal time: a may turn around within
        // an interval, so an endpoint-only delta ln(a) loses its history.
        history.expansionWork += kTrapezoidWeight * (history.previousWorkRate + workRate) * elapsedConformalTime;
    } else {
        history.initialEnergy = kinetic + potential;
        history.expansionWork = 0.0;
        history.initialized = true;
    }
    history.previousScaleFactor = scaleFactor;
    history.previousKinetic = kinetic;
    history.previousPotential = potential;
    history.previousWorkRate = workRate;
    history.kinetic = kinetic;
    history.potential = potential;
    history.conservedEnergy = kinetic + potential + history.expansionWork;
}

}    // namespace cosmictiger
