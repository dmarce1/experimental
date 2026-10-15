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

#include <cstdint>
#include <stddef.h>
#include <string>

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/modules/serialization.hpp>

#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/lightcone/lightcone_options.hpp>
#include <cosmictiger/particle/particle_initialization.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/units.hpp>

namespace cosmictiger {

struct GlobalConstantsData {
    CosmologyParameters cosmology;
    ParticleInitializationOptions initializationOptions;
    bool generateInitialConditions = false;
    bool evolve = true;
    bool cudaDrift = false;
    bool cudaKick = false;
    bool cudaMultipole = false;
    bool cudaDirect = false;
    LightconeOptions lightcone;
    // Conformal observer epoch measured from the original initial-condition handoff.
    double lightconeObservationConformalTime = 0.0;
    // Speed of light in box lengths per H0^-1.
    double lightconeSpeed = 0.0;
    std::string lightconeRunId;
    std::string restartManifest;
    std::string checkpointPrefix = "simulation";
    size_t outputSnapshotCount = 1;
    unsigned restartRungZeroExtension = 0;
    unsigned rungZeroExtension = 0;
    std::string configurationFile;
    std::string outputPrefix;
    double timestepEta = 0.025;
    // Resolved evolution mesh exponent; defaults to the initial-condition exponent.
    size_t gravityMeshExponent = initializationOptions.meshExponent;
    int ghostWidth = 1;
    // Local-expansion rank; cell source multipoles use one rank less.
    int fmmOrder = gravity::kDefaultFmmOrder;
    gravity::SofteningModel softeningModel = gravity::SofteningModel::ferrers;
    // Solver choices; a restart may select different values.
    gravity::NearFieldMethod nearFieldMethod = gravity::NearFieldMethod::direct;
    double openingAngle = gravity::kDefaultOpeningAngle;
    double softeningCellFraction = 0.05;
    int softeningKernelOrder = 2;
    // Fraction of owned particles sampled for a direct periodic gravity comparison.
    double directSinkFraction = 0.0;
    std::uint64_t directSeed = 42;
    // Ferrers support radius q or Plummer epsilon in unit-box coordinates.
    double softeningLength = 0.0;
    double initialScaleFactor = 0.01;
    double stopTimeYears = 0.0;
    bool stopAtEnd = false;
    double minimumScaleFactor = 1.0e-6;
    // Physical times are H0*(t-tPresent), with the expanding a=1 epoch as zero.
    double stopPhysicalTime = 0.0;
    double endPhysicalTime = 0.0;
    double initialPhysicalTime = 0.0;
    // The total conformal interval in H0^-1 units, computed at initialization.
    double rungZeroTimestep = 0.0;
    double originalRungZeroTimestep = 0.0;
    // G*m/(H0^2*L^3), for unit-box gravity kernels with G=1.
    double gravityParticleMass = 0.0;
    double initialConformalDerivative = 0.0;
    bool outputAcceleration = false;
    bool outputPotential = false;
    size_t localityCount = 0;
    size_t meshSideLength = 0;
    size_t gridSideLength = 0;
    ParticleCount globalParticleCount = 0;
};

template <typename Archive>
void serialize(Archive& archive, CosmologyParameters& parameters, unsigned) {
    PROFILE();
    archive & parameters.hubble & parameters.omegaBaryonH2 & parameters.omegaCdmH2 & parameters.omegaNeutrinoH2 & parameters.omegaCurvature &
        parameters.massiveNeutrinos & parameters.masslessNeutrinos & parameters.effectiveNeutrinos & parameters.scalarAmplitude & parameters.spectralIndex &
        parameters.spectralRunning & parameters.pivotMpcInverse & parameters.darkEnergyW & parameters.darkEnergyWa & parameters.temperatureCmb &
        parameters.heliumFraction & parameters.opticalDepth;
}

template <typename Archive>
void serialize(Archive& archive, ParticleInitializationOptions& options, unsigned) {
    PROFILE();
    archive & options.meshExponent & options.boxLength & options.seed & options.maximumDensityDeviation & options.earliestScaleFactor &
        options.latestScaleFactor & options.maximumLogScaleFactorStep & options.logScaleFactorTolerance & options.growthDerivativeStep &
        options.maximumSpectrumEvaluations & options.samplesPerLogInterval & options.accuracyBoost & options.component;
}

template <typename Archive>
void serializeGlobalConstantsVersion2(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    archive & data.cosmology & data.initializationOptions & data.generateInitialConditions & data.configurationFile & data.outputPrefix & data.localityCount &
        data.meshSideLength & data.gridSideLength & data.globalParticleCount & data.timestepEta & data.initialScaleFactor & data.stopTimeYears &
        data.rungZeroTimestep & data.gravityParticleMass & data.initialConformalDerivative & data.outputAcceleration & data.outputPotential &
        data.softeningCellFraction & data.softeningKernelOrder & data.softeningLength & data.evolve & data.restartManifest & data.checkpointPrefix &
        data.restartRungZeroExtension & data.rungZeroExtension & data.originalRungZeroTimestep & data.stopAtEnd & data.minimumScaleFactor &
        data.stopPhysicalTime & data.endPhysicalTime & data.initialPhysicalTime & data.outputSnapshotCount;
}

template <typename Archive>
void serializeGlobalConstantsVersion3(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion2(archive, data);
    archive & data.cudaDrift & data.cudaKick;
}

template <typename Archive>
void serializeGlobalConstantsVersion4(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion3(archive, data);
    archive & data.cudaMultipole;
}

template <typename Archive>
void serializeGlobalConstantsVersion5(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion4(archive, data);
    archive & data.lightcone.enabled & data.lightcone.nside & data.lightcone.maximumRadius & data.lightcone.observer & data.lightcone.outputPrefix &
        data.lightcone.observationTimeYears & data.lightconeObservationConformalTime & data.lightconeSpeed & data.lightconeRunId;
}

template <typename Archive>
void serializeGlobalConstantsVersion6(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion5(archive, data);
    archive & data.directSinkFraction & data.directSeed & data.cudaDirect;
}

template <typename Archive>
void serializeGlobalConstantsVersion7(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion6(archive, data);
    archive & data.gravityMeshExponent & data.ghostWidth;
}

template <typename Archive>
void serializeGlobalConstantsVersion8(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion7(archive, data);
    archive & data.softeningModel;
}

template <typename Archive>
void serializeGlobalConstantsVersion9(Archive& archive, GlobalConstantsData& data) {
    PROFILE();
    serializeGlobalConstantsVersion8(archive, data);
    archive & data.fmmOrder;
}

template <typename Archive>
void serialize(Archive& archive, GlobalConstantsData& data, unsigned) {
    PROFILE();
    serializeGlobalConstantsVersion9(archive, data);
    archive & data.nearFieldMethod & data.openingAngle;
}

namespace server {

    class GlobalConstants : public hpx::components::component_base<GlobalConstants> {
    public:
        explicit GlobalConstants(GlobalConstantsData data);

        GlobalConstantsData getData() const;

        HPX_DEFINE_COMPONENT_ACTION(GlobalConstants, getData, GetDataAction)

    private:
        GlobalConstantsData const data_;
    };

}    // namespace server

class GlobalConstants : public hpx::components::client_base<GlobalConstants, server::GlobalConstants> {
public:
    explicit GlobalConstants(hpx::id_type const& componentId);

    GlobalConstantsData const& data() const noexcept;
    units::UnitScales const& unitScales() const noexcept;

private:
    GlobalConstantsData const data_;
    // Derived solely from the root's immutable data after fetching it. No new
    // serialized state is needed, so all existing checkpoint versions retain
    // their physical scales when the local cache is reconstructed.
    units::UnitScales const unitScales_;
};

GlobalConstants const& globalConstants();

// Set the actual selected initial epoch and derive conformal duration/mass
// before the root publishes the immutable component.
void initializeSimulationConstants(GlobalConstantsData& data, double initialScaleFactor);

// Recompute only the requested endpoint; restarting never recomputes the saved rung-zero interval.
void initializeSimulationEndpoint(GlobalConstantsData& data);
void initializeLightconeConstants(GlobalConstantsData& data);
double physicalTimeInYears(double physicalTime, CosmologyParameters const& cosmology);

// The root locality publishes once and waits for every locality to cache a copy.
void publishGlobalConstants(GlobalConstantsData data);

}    // namespace cosmictiger

HPX_REGISTER_ACTION_DECLARATION(cosmictiger::server::GlobalConstants::GetDataAction, CosmictigerGlobalConstantsGetDataAction)
