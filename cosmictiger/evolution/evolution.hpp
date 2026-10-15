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

#include <cosmictiger/evolution/evolution_state.hpp>
#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/gravity/kick/kick_sums.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_progress.hpp>
#include <cosmictiger/particle/particle_silo.hpp>
#include <hpx/functional/function.hpp>

#include <cstdint>

namespace cosmictiger {

using EvolutionObserver = hpx::function<void(EvolutionState const&, gravity::KickSums const&)>;
// Snapshot fields are synchronized to this epoch and remain valid only for
// the callback. The particle storage retains its staggered restart momenta.
using EvolutionSnapshotObserver = hpx::function<void(EvolutionState const&, gravity::KickSums const&, ParticleSiloOutput const&)>;

enum class EvolutionProgressKind { initial, rungStep, snapshotSample };

enum class EvolutionInitializationStage {
    backgroundTable,
    snapshotCadence,
    gridAndRungs,
    lightcone,
    periodicConvolution,
    fftExchange,
    ghostExchange,
    cellMultipoles,
    gravityKick,
    diagnostics,
    firstSnapshot
};

// Exclusive durations for one reported step/sample, in local wall seconds.
struct EvolutionPhaseTimes {
    double drift = 0.0;
    double exchange = 0.0;
    double multipole = 0.0;
    double gravity = 0.0;
    double kick = 0.0;
    double output = 0.0;
};

struct EvolutionProgress {
    EvolutionProgressKind kind;
    bool wroteSnapshot;
    std::uint8_t completedRung;
    EvolutionPhaseTimes phaseTimes;
    // HPX high-resolution-clock nanoseconds at the timing snapshot. An output
    // observer may add its own work before formatting the current timing row.
    std::uint64_t timingEnd;
};

using EvolutionProgressObserver = hpx::function<void(EvolutionState const&, EvolutionProgress const&)>;
using EvolutionInitializationObserver = hpx::function<void(EvolutionInitializationStage)>;

// Collective on all physical localities. Repartition owned particles onto the
// requested evolution mesh, preserving particle values and clearing old ghosts.
// Initialization retains its particle-matched FFT before this handoff.
void prepareEvolutionGrid(ParticleStorage& particles, CartesianGrid& grid, size_t meshExponent, size_t ghostWidth, std::uint64_t operationId);

// Collective on all physical localities. Momenta remain staggered, including
// at an off-lattice stopping time, so a restart resumes the same next event.
// Each observer sees synchronized global kinetic/potential/force diagnostics.
// snapshotObserver runs outputSnapshotCount times per invocation: one means
// final only; larger counts include the starting/final states and equally
// spaced conformal times between them. Partial snapshots preserve the event
// lattice and staggered momenta. The final epoch is the exact physical target.
// progressObserver runs after each completed rung step and after an initial or
// off-lattice snapshot sample. wroteSnapshot is true only after the snapshot
// observer succeeds. completedRung identifies a completed rung step.
// initializationObserver announces each first-evaluation phase before it begins.
// gravityInitializationObserver reports local work within that first gravity
// evaluation, including concurrent near/far work. It must not enter collectives.
// Enabled lightcone output samples owned particle trajectories before every
// drift, including partial snapshot drifts, with the same cosmological drift
// integral as the particle update. Each locality writes independent segments.
// Local observers must not schedule unmatched collectives.
void evolveParticles(ParticleStorage& particles, CartesianGrid& grid, GlobalConstantsData const& constants, EvolutionState& state,
    EvolutionObserver const& observer = {}, EvolutionSnapshotObserver const& snapshotObserver = {}, EvolutionProgressObserver const& progressObserver = {},
    EvolutionInitializationObserver const& initializationObserver = {}, gravity::PmmmProgressObserver const& gravityInitializationObserver = {});

// Enlarge dtau0 by 2^n externally and relabel particles and clock together.
// Must be called on an owned-only checkpoint before rebuilding ghosts.
void extendEvolutionRungs(ParticleStorage& particles, CartesianGrid& grid, EvolutionState& state, unsigned offset);

}    // namespace cosmictiger
