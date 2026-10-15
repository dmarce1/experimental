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

#include <vector>

namespace cosmictiger {

enum class MatterComponent { totalMatter, cdmAndBaryons, cdm };

// Physical densities are Omega_i * h^2. Massive neutrinos use a single
// degenerate mass group. Physical species share a common temperature, set
// by effectiveNeutrinos (total N_eff), independent of the solver package.
struct CosmologyParameters {
    double hubble = 67.4;    // km/s/Mpc
    double omegaBaryonH2 = 0.0224;
    double omegaCdmH2 = 0.120;
    double omegaNeutrinoH2 = 0.00064;
    double omegaCurvature = 0.0;
    int massiveNeutrinos = 1;
    int masslessNeutrinos = 2;
    double effectiveNeutrinos = 3.046;
    double scalarAmplitude = 2.1e-9;
    double spectralIndex = 0.965;
    double spectralRunning = 0.0;
    double pivotMpcInverse = 0.05;
    double darkEnergyW = -1.0;
    double darkEnergyWa = 0.0;
    double temperatureCmb = 2.7255;    // kelvin
    double heliumFraction = 0.245;
    double opticalDepth = 0.054;
};

struct MatterPowerRequest {
    double redshift = 49.0;
    double maximumWaveNumber = 10.0;    // h/Mpc
    int samplesPerLogInterval = 50;     // per natural logarithm; 0 uses solver sampling
    double accuracyBoost = 1.0;
    MatterComponent component = MatterComponent::totalMatter;
};

struct MatterPowerSample {
    double waveNumber;    // h/Mpc
    double power;         // (Mpc/h)^3
};

// Linear, adiabatic, synchronous-gauge density power. Each component's
// density contrast uses that component's mean density as its normalization.
class MatterPowerSpectrum {
public:
    // Samples must have increasing positive wave numbers and positive powers.
    // The constructor stores the table without a validation traversal.
    MatterPowerSpectrum(double redshift, double hubbleParameter, MatterComponent component, std::vector<MatterPowerSample> samples);

    double redshift() const noexcept;
    double hubbleParameter() const noexcept;
    MatterComponent component() const noexcept;
    std::vector<MatterPowerSample> const& samples() const noexcept;

    // Log-log interpolation of an increasing positive table. Validation builds
    // check queried ranges and the samples consumed by the search/interpolation.
    double power(double waveNumber) const;

private:
    double redshift_;
    double hubbleParameter_;
    MatterComponent component_;
    std::vector<MatterPowerSample> samples_;
};

// Alternate external solvers and in-process implementations use this contract.
class EinsteinBoltzmannSolver {
public:
    virtual ~EinsteinBoltzmannSolver() = default;
    virtual MatterPowerSpectrum compute(CosmologyParameters const& parameters, MatterPowerRequest const& request = {}) const = 0;
};

}    // namespace cosmictiger
