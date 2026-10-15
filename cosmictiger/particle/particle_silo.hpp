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
#include <cosmictiger/einstein_boltzmann.hpp>
#include <cosmictiger/particle/particle_storage.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <stddef.h>
#include <string>
#include <vector>

namespace cosmictiger {

struct InitialConditionsSiloMetadata {
    CosmologyParameters cosmology;
    double scaleFactor;
    double boxLength;                    // Comoving Mpc/h
    double particleMass;                 // Solar masses/h
    double maximumDensityDeviation;      // Realized maximum |delta| on the FFT mesh
    double requestedDensityDeviation;    // Requested linear-density handoff threshold
    std::uint64_t seed;
    size_t meshExponent;    // Initialization FFT and original particle lattice
    MatterComponent component;
    bool canonicalMomentum = false;    // velocity stores a d(x/L)/d(H0 tau)
};

// Non-owning, snapshot-epoch canonical momenta in native particle order. These
// arrays must remain alive throughout the synchronous writer call. They are
// used only for derived peculiar velocities; native restart momenta are kept.
struct ParticleSiloOutput {
    std::array<double const*, NDIM> synchronizedMomenta{};
    ParticleCount particleCount = 0;
};

// Write the owned prefix described by grid. Ghost-cell copies are never saved,
// including periodic copies whose Rung::isGhost() flag is false. Call once per
// locality with a distinct filename after all particle and GPU work finishes.
// A completed checkpoint atomically replaces an existing file at filename.
// Opt-in particle acceleration and potential fields are saved when allocated.
// Save the rung value only; packed ghost and reserved flags are not output.
void writeParticleSilo(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid);

// Adds cosmology, initialization controls and physical units to the checkpoint.
// Particle positions are unit-box coordinates. Velocities are d(x/L)/d ln(a)
// unless metadata.canonicalMomentum selects a d(x/L)/d(H0 tau).
// The Cartesian grid may use a different resolution from the initialization
// FFT and original particle lattice described by metadata.meshExponent.
// Metadata-bearing files also contain comoving Mpc/Msun/km/s and physical CGS
// visualization views. Initial momenta are synchronized unless output supplies
// a separate snapshot-epoch array. Velocities always exclude Hubble flow.
void writeParticleSilo(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid, InitialConditionsSiloMetadata const& metadata,
    ParticleSiloOutput const* output = nullptr);

// Publish a visualization-only Silo multimesh after all locality files for one
// timestep are complete. The master and locality files must share a directory.
// Each referenced leaf point mesh is included once, including blocks within a
// locality file; empty locality files contribute no blocks.
void writeParticleSiloMaster(std::string const& filename, std::vector<std::string> const& localityFiles, std::optional<double> conformalTime = std::nullopt);

// Throws if the initial-condition metadata is absent, invalid or incompatible.
InitialConditionsSiloMetadata readInitialConditionsSiloMetadata(std::string const& filename);

// Restore owned particles and their interior cell ranges. The supplied grid
// must have the same decomposition as the checkpoint. On failure, both output
// objects retain their previous contents. Ghosts and derived grid fields must
// be regenerated after loading. Optional gravity output fields are restored
// according to checkpoint metadata, including their allocation settings.
// Rungs are restored with ghost and reserved bits cleared. Legacy checkpoints
// with packed flags remain readable and are checked for ghost particles.
void readParticleSilo(std::string const& filename, ParticleStorage& particles, CartesianGrid& grid);

namespace detail {

    struct EvolutionSiloScalars {
        double scaleFactor;
        double conformalDerivative;
        double yearsFromPresent;
        double conformalTime;
        // Authoritative native source mass; legacy direct callers may omit it.
        double gravityParticleMass = -1.0;
    };

    // Evolution metadata is committed in the same atomic file replacement as
    // the particles and original initial-condition metadata. These hooks keep
    // the particle format independent of the evolution archive's C++ types.
    // Native Silo animation time is conformal time since the original handoff,
    // in H0^-1 units; the payload retains the exact integer restart clock.
    // Without output, evolved derived views omit velocity because native
    // momenta may be staggered. All other available derived fields are written.
    void writeParticleSiloEvolutionPayload(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid,
        InitialConditionsSiloMetadata const& initial, std::vector<char> const& payload, EvolutionSiloScalars const& scalars,
        ParticleSiloOutput const* output = nullptr);
    std::vector<char> readParticleSiloEvolutionPayload(std::string const& filename);
    void validateParticleSiloEvolutionLayout(
        std::string const& filename, size_t gridSideLength, size_t localitiesPerAxis, bool outputAcceleration, bool outputPotential);

}    // namespace detail

}    // namespace cosmictiger
