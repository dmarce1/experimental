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
#include <cosmictiger/profiler.hpp>
#include <hpx/modules/serialization.hpp>

#include <gtest/gtest.h>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cosmictiger::activeEvolutionRung;
using cosmictiger::EnergyHistory;
using cosmictiger::EvolutionState;
using cosmictiger::extendEvolutionClock;
using cosmictiger::nextEvolutionEvent;
using cosmictiger::updateEnergyHistory;

TEST(EvolutionState, FinestEventCarriesIntoNextNominalEpoch) {
    PROFILE();
    EvolutionState state;
    state.nominalEpoch = 6;
    state.timeTick = std::uint64_t(7) << 60;
    state.maximumRung = 3;
    auto const next = nextEvolutionEvent(state);
    EXPECT_EQ(next.nominalEpoch, 7);
    EXPECT_EQ(next.timeTick, 0);
    state.timeTick = std::uint64_t(3) << 60;
    auto const middle = nextEvolutionEvent(state);
    EXPECT_EQ(middle.nominalEpoch, 6);
    EXPECT_EQ(middle.timeTick, std::uint64_t(1) << 62);
}

TEST(EvolutionState, ActiveRungUsesExactIntegerSynchronization) {
    PROFILE();
    EXPECT_EQ(activeEvolutionRung(0), 0);
    EXPECT_EQ(activeEvolutionRung(std::uint64_t(1) << 62), 1);
    EXPECT_EQ(activeEvolutionRung(std::uint64_t(3) << 60), 3);
    EXPECT_EQ(activeEvolutionRung(1), 63);
}

TEST(EvolutionState, RejectsClockOverflowAndMisalignedEvents) {
    PROFILE();
    EvolutionState state;
    state.nominalEpoch = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW(nextEvolutionEvent(state), std::overflow_error);
    state.nominalEpoch = 0;
    state.timeTick = 1;
    EXPECT_THROW(nextEvolutionEvent(state), std::invalid_argument);
    EXPECT_THROW(activeEvolutionRung(std::uint64_t(1) << 63), std::invalid_argument);
}

TEST(EvolutionState, LongerNominalIntervalPreservesTimeRungAndMomentumPhase) {
    PROFILE();
    EvolutionState state;
    state.nominalEpoch = 13;
    state.timeTick = std::uint64_t(1) << 61;
    state.maximumRung = 2;
    state.scaleFactor = 0.4;
    state.tickScaleFactor = 0.35;
    state.conformalOffset = 0.015;
    state.conformalDerivative = -0.3;
    state.tickConformalDerivative = -0.2;
    state.physicalTime = 1.7;
    state.tickPhysicalTime = 1.69;
    state.momentaStaggered = true;
    extendEvolutionClock(state, 3);
    EXPECT_EQ(state.nominalEpoch, 1);
    EXPECT_EQ(state.timeTick, std::uint64_t(21) << 58);
    EXPECT_EQ(state.maximumRung, 5);
    EXPECT_DOUBLE_EQ(8.0 * (double(state.nominalEpoch) + std::ldexp(double(state.timeTick), -63)), 13.25);
    EXPECT_DOUBLE_EQ(std::ldexp(8.0, -int(state.maximumRung)), 0.25);
    EXPECT_DOUBLE_EQ(state.scaleFactor, 0.4);
    EXPECT_DOUBLE_EQ(state.tickScaleFactor, 0.35);
    EXPECT_DOUBLE_EQ(state.conformalOffset, 0.015);
    EXPECT_DOUBLE_EQ(state.conformalDerivative, -0.3);
    EXPECT_DOUBLE_EQ(state.tickConformalDerivative, -0.2);
    EXPECT_DOUBLE_EQ(state.physicalTime, 1.7);
    EXPECT_DOUBLE_EQ(state.tickPhysicalTime, 1.69);
    EXPECT_TRUE(state.momentaStaggered);
    auto const next = nextEvolutionEvent(state);
    EXPECT_DOUBLE_EQ(8.0 * (double(next.nominalEpoch) + std::ldexp(double(next.timeTick), -63)), 13.5);
}

TEST(EvolutionState, MaximumExtensionDoesNotShiftByWordWidth) {
    PROFILE();
    EvolutionState state;
    state.nominalEpoch = std::numeric_limits<std::uint64_t>::max();
    extendEvolutionClock(state, 63);
    EXPECT_EQ(state.nominalEpoch, 1);
    EXPECT_EQ(state.timeTick, (std::uint64_t(1) << 63) - 1);
    EXPECT_EQ(state.maximumRung, 63);
}

TEST(EvolutionState, ExtensionFailureLeavesClockUnchanged) {
    PROFILE();
    EvolutionState state;
    state.nominalEpoch = 7;
    state.timeTick = 1;
    state.maximumRung = 2;
    EXPECT_THROW(extendEvolutionClock(state, 1), std::range_error);
    EXPECT_EQ(state.nominalEpoch, 7);
    EXPECT_EQ(state.timeTick, 1);
    EXPECT_EQ(state.maximumRung, 2);
    state.timeTick = 0;
    EXPECT_THROW(extendEvolutionClock(state, 62), std::range_error);
    EXPECT_EQ(state.nominalEpoch, 7);
    EXPECT_EQ(state.timeTick, 0);
    EXPECT_EQ(state.maximumRung, 2);
    extendEvolutionClock(state, 0);
    EXPECT_EQ(state.nominalEpoch, 7);
}

TEST(EvolutionEnergy, FreelyStreamingMatterIncludesExpansionWork) {
    constexpr int kIntervals = 2048;
    constexpr double kStep = 1.0 / kIntervals;
    PROFILE();

    for (double const sign : {-1.0, 1.0}) {
        EnergyHistory history;
        updateEnergyHistory(history, 1.0, 1.0, 0.0, sign, 0.0);
        for (int index = 1; index <= kIntervals; index++) {
            double const a = std::exp(sign * index * kStep);
            updateEnergyHistory(history, a, 1.0 / (a * a), 0.0, sign * a, kStep);
        }
        EXPECT_NEAR(history.conservedEnergy, 1.0, 6.0e-7);
        EXPECT_DOUBLE_EQ(history.initialEnergy, 1.0);
        EXPECT_GT(sign * history.expansionWork, 0.0);
    }
}

TEST(EvolutionEnergy, TimeQuadratureRetainsTurnaroundHistoryAtEqualScaleFactors) {
    PROFILE();
    EnergyHistory history;
    updateEnergyHistory(history, 1.0, 2.0, -1.0, 1.0, 0.0);
    updateEnergyHistory(history, 1.0, 3.0, -1.0, -1.0, 2.0);
    EXPECT_DOUBLE_EQ(history.expansionWork, -2.0);
    EXPECT_DOUBLE_EQ(history.conservedEnergy, 0.0);
    EXPECT_DOUBLE_EQ(history.initialEnergy, 1.0);
    EXPECT_DOUBLE_EQ(history.previousWorkRate, -5.0);
    updateEnergyHistory(history, 1.0, 3.0, -1.0, -1.0, 0.0);
    EXPECT_DOUBLE_EQ(history.expansionWork, -2.0);
    EXPECT_DOUBLE_EQ(history.initialEnergy, 1.0);
}

TEST(EvolutionState, SerializationPreservesClockAndEnergyContinuation) {
    PROFILE();
    EvolutionState state;
    state.scaleFactor = 0.8;
    state.tickScaleFactor = 0.81;
    state.conformalDerivative = -0.4;
    state.tickConformalDerivative = -0.39;
    state.physicalTime = 2.0;
    state.tickPhysicalTime = 1.99;
    state.conformalOffset = 0.0125;
    state.nominalEpoch = 8;
    state.timeTick = std::uint64_t(1) << 61;
    state.operationId = 123;
    state.maximumRung = 4;
    state.momentaStaggered = true;
    updateEnergyHistory(state.energy, 1.0, 2.0, -1.0, 1.0, 0.0);
    updateEnergyHistory(state.energy, state.scaleFactor, 3.0, -2.0, state.conformalDerivative, 0.1);
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << state;
    }
    EvolutionState restored;
    {
        hpx::serialization::input_archive archive(bytes, bytes.size());
        archive >> restored;
    }
    EXPECT_DOUBLE_EQ(restored.scaleFactor, state.scaleFactor);
    EXPECT_DOUBLE_EQ(restored.tickScaleFactor, state.tickScaleFactor);
    EXPECT_DOUBLE_EQ(restored.conformalDerivative, state.conformalDerivative);
    EXPECT_DOUBLE_EQ(restored.tickConformalDerivative, state.tickConformalDerivative);
    EXPECT_DOUBLE_EQ(restored.physicalTime, state.physicalTime);
    EXPECT_DOUBLE_EQ(restored.tickPhysicalTime, state.tickPhysicalTime);
    EXPECT_DOUBLE_EQ(restored.conformalOffset, state.conformalOffset);
    EXPECT_EQ(restored.nominalEpoch, state.nominalEpoch);
    EXPECT_EQ(restored.timeTick, state.timeTick);
    EXPECT_EQ(restored.operationId, state.operationId);
    EXPECT_EQ(restored.maximumRung, state.maximumRung);
    EXPECT_EQ(restored.momentaStaggered, state.momentaStaggered);
    updateEnergyHistory(state.energy, 0.7, 4.0, -3.0, -0.5, 0.2);
    updateEnergyHistory(restored.energy, 0.7, 4.0, -3.0, -0.5, 0.2);
    EXPECT_DOUBLE_EQ(restored.energy.initialEnergy, state.energy.initialEnergy);
    EXPECT_DOUBLE_EQ(restored.energy.conservedEnergy, state.energy.conservedEnergy);
    EXPECT_DOUBLE_EQ(restored.energy.expansionWork, state.energy.expansionWork);
}

}    // namespace
