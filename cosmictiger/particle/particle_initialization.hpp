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

#include <hpx/functional/function.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstdint>
#include <stddef.h>
#include <string>
#include <vector>

namespace cosmictiger {

struct ParticleInitializationOptions {
    size_t meshExponent = 5;     // Initialization FFT and particle lattice side = 2^meshExponent
    double boxLength = 100.0;    // Comoving Mpc/h
    std::uint64_t seed = 42;
    // Maximum absolute resolved linear density contrast |rho-rhoBar|/rhoBar.
    double maximumDensityDeviation = 0.1;
    double earliestScaleFactor = 0.001;
    double latestScaleFactor = 1.0;
    double maximumLogScaleFactorStep = 0.1;
    double logScaleFactorTolerance = 1.0e-4;
    double growthDerivativeStep = 1.0e-3;    // Step in ln(a)
    size_t maximumSpectrumEvaluations = 256;
    int samplesPerLogInterval = 50;
    double accuracyBoost = 1.0;
    MatterComponent component = MatterComponent::cdmAndBaryons;
};

struct ParticleInitializationReport {
    double scaleFactor;
    double redshift;
    double maximumDensityDeviation;
    double previousScaleFactor;    // Last below-threshold endpoint
    double previousDensityDeviation;
    double boxLength;       // Comoving Mpc/h
    double particleMass;    // Solar masses/h, equal mass for all particles
    ParticleCount globalParticleCount;
    size_t spectrumEvaluations;
};

struct ParticleInitialConditions {
    ParticleStorage particles;
    CartesianGrid grid;
    ParticleInitializationReport report;
};

using ParticleInitializationObserver = hpx::function<void(std::string const&)>;

// Collective: call from an HPX task on every site with identical parameters,
// options, ordered localities and a globally unique operationId. Only site 0
// invokes the EB solver. Logical sites may share a process. Requires P=8^p,
// P<=FFT side. The FFT mesh, Cartesian cells, and particle lattice share the
// same resolution. Each particle begins at its Cartesian cell center before
// displacement. The returned lists have been redistributed to their Cartesian
// owners and sorted into interior cell and rung ranges; displaced cell
// occupancy may differ from one.
//
// A seeded Gaussian realization uses the linear component spectrum at every
// epoch. DC and all Nyquist planes are zero. The first sampled crossing of
// max_mesh |delta| is bracketed and bisected in ln(a); the early endpoint must
// be below the threshold and the late endpoint must reach it. This criterion
// measures the mesh-bandlimited linear field, not a nonlinear density estimate.
//
// Positions are periodic unit-box comoving coordinates. Velocities stored in
// ParticleStorage are d(x/L)/d ln(a), obtained by finite-differencing the CAMB
// mode amplitudes. Physical peculiar velocities in km/s are
// a*H(a)*boxLength/h times the stored values, for H(a) in km/s/Mpc.
// The combined CDM+baryon option represents one cold collisionless population;
// massive neutrinos affect CAMB but are not initialized as cold particles.
// No gravity, ghost refresh, or N-body integration is performed here.
ParticleInitialConditions initializeParticles(CosmologyParameters const& parameters, ParticleInitializationOptions const& options,
    std::vector<hpx::id_type> const& localities, size_t localityIndex, std::uint64_t operationId, EinsteinBoltzmannSolver const& solver,
    ParticleInitializationObserver const& progressObserver = {});

// Uses CAMB and HPX's ordered list of all localities. Every locality must call.
// When requested, site zero reports each spectrum evaluation and preparation
// stage through progressObserver before long-running work begins.
ParticleInitialConditions initializeParticles(CosmologyParameters const& parameters, ParticleInitializationOptions const& options, std::uint64_t operationId,
    ParticleInitializationObserver const& progressObserver = {});

}    // namespace cosmictiger
