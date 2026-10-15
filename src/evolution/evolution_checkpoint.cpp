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

#include <cosmictiger/evolution/evolution_checkpoint.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmictiger {
namespace detail {

    constexpr std::uint64_t kEvolutionArchiveMagic = 0x435445564F4C3031ULL;
    // Version nine retains the runtime FMM rank along with the physical gravity settings.
    constexpr std::uint32_t kEvolutionArchiveVersion = 9;
    constexpr auto kEvolutionArchiveFlags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
        hpx::serialization::archive_flags::disable_data_chunking;

    void requireEvolutionMetadata(bool condition, char const* message) {
        PROFILE();
        if (!condition) {
            throw std::runtime_error(std::string("invalid evolution checkpoint: ") + message);
        }
    }

    std::vector<char> archiveCosmology(CosmologyParameters const& cosmology) {
        PROFILE();
        std::vector<char> bytes;
        hpx::serialization::output_archive archive(bytes, kEvolutionArchiveFlags);
        archive << cosmology;
        return bytes;
    }

    size_t checkpointLocalitySide(size_t localityCount) {
        constexpr size_t kLocalityFactor = size_t(1) << NDIM;
        PROFILE();

        requireEvolutionMetadata(localityCount != 0, "locality count is zero");
        size_t side = 1;
        while (localityCount > 1 && localityCount % kLocalityFactor == 0) {
            localityCount /= kLocalityFactor;
            side *= 2;
        }
        requireEvolutionMetadata(localityCount == 1, "locality count is not a supported Cartesian decomposition");
        return side;
    }

    void validateEvolutionMetadata(GlobalConstantsData const& constants, EvolutionState const& state, InitialConditionsSiloMetadata const& initial) {
        constexpr size_t kMinimumMeshExponent = 2;
        constexpr size_t kMaximumMeshExponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
        constexpr double kRelativeTimeTolerance = 64.0 * std::numeric_limits<double>::epsilon();
        constexpr std::uint32_t kMaximumNside = std::uint32_t(1) << 29;
        constexpr double kLargestExactImage = 0x1p52;
        constexpr int kMinimumGhostWidth = 1;
        constexpr int kMaximumGhostWidth = 2;
        PROFILE();

        size_t const exponent = constants.initializationOptions.meshExponent;
        requireEvolutionMetadata(exponent >= kMinimumMeshExponent && exponent <= kMaximumMeshExponent, "mesh exponent is out of range");
        size_t const side = size_t(1) << exponent;
        requireEvolutionMetadata(constants.meshSideLength == side, "initialization mesh size disagrees with its exponent");
        requireEvolutionMetadata(constants.gravityMeshExponent >= kMinimumMeshExponent && constants.gravityMeshExponent <= kMaximumMeshExponent,
            "gravity mesh exponent is out of range");
        size_t const gridSide = size_t(1) << constants.gravityMeshExponent;
        requireEvolutionMetadata(constants.gridSideLength == gridSide, "Cartesian grid size disagrees with its exponent");
        size_t const localitySide = checkpointLocalitySide(constants.localityCount);
        requireEvolutionMetadata(localitySide <= side && localitySide <= gridSide, "locality decomposition exceeds an initialization or gravity mesh");
        requireEvolutionMetadata(
            constants.localityCount <= side && constants.localityCount <= gridSide, "locality count exceeds an initialization or gravity FFT side length");
        requireEvolutionMetadata(constants.ghostWidth >= kMinimumGhostWidth && constants.ghostWidth <= kMaximumGhostWidth, "ghost width is out of range");
        requireEvolutionMetadata(
            constants.fmmOrder >= gravity::kMinimumFmmOrder && constants.fmmOrder <= gravity::kMaximumFmmOrder, "FMM order is outside the supported range");
        ParticleCount const count = ParticleCount(1) << (NDIM * exponent);
        requireEvolutionMetadata(constants.globalParticleCount == count, "global particle count differs from the original lattice");
        requireEvolutionMetadata(constants.outputSnapshotCount > 0, "snapshot count is zero");
        requireEvolutionMetadata(constants.directSinkFraction >= 0.0 && constants.directSinkFraction <= 1.0, "direct sink fraction is outside [0,1]");
        if (constants.lightcone.enabled) {
            auto const& lightcone = constants.lightcone;
            requireEvolutionMetadata(lightcone.nside > 0 && lightcone.nside <= kMaximumNside && (lightcone.nside & (lightcone.nside - 1)) == 0 &&
                    lightcone.maximumRadius > 0.0 && lightcone.maximumRadius < kLargestExactImage && !lightcone.outputPrefix.empty(),
                "lightcone resolution, radius, or output prefix is invalid");
            requireEvolutionMetadata(constants.lightconeObservationConformalTime > 0.0 &&
                    constants.lightconeObservationConformalTime <= std::numeric_limits<double>::max() && constants.lightconeSpeed > 0.0 &&
                    constants.lightconeSpeed <= std::numeric_limits<double>::max(),
                "lightcone conformal observer epoch or light speed is invalid");
            requireEvolutionMetadata(
                lightcone.observationTimeYears >= -std::numeric_limits<double>::max() && lightcone.observationTimeYears <= std::numeric_limits<double>::max(),
                "lightcone observer physical epoch is outside the supported range");
            for (double coordinate : lightcone.observer) {
                requireEvolutionMetadata(coordinate >= 0.0 && coordinate < 1.0, "lightcone observer lies outside the periodic box");
            }
        }
        requireEvolutionMetadata(constants.rungZeroTimestep > 0.0, "nominal conformal timestep is not positive");
        requireEvolutionMetadata(constants.originalRungZeroTimestep > 0.0 && constants.rungZeroExtension <= Rung::maximumRung(),
            "original nominal timestep or rung extension is invalid");
        requireEvolutionMetadata(constants.originalRungZeroTimestep <= std::ldexp(std::numeric_limits<double>::max(), -int(constants.rungZeroExtension)),
            "extended nominal timestep exceeds the available range");
        requireEvolutionMetadata(constants.rungZeroTimestep == std::ldexp(constants.originalRungZeroTimestep, int(constants.rungZeroExtension)),
            "nominal timestep differs from its saved dyadic extension");
        requireEvolutionMetadata(constants.timestepEta > 0.0 && constants.gravityParticleMass > 0.0, "timestep accuracy or particle mass is not positive");
        requireEvolutionMetadata(constants.softeningModel == gravity::SofteningModel::ferrers || constants.softeningModel == gravity::SofteningModel::plummer,
            "softening model is invalid");
        requireEvolutionMetadata(constants.softeningLength > 0.0 && constants.softeningCellFraction > 0.0 && constants.softeningCellFraction <= 1.0 &&
                constants.softeningKernelOrder >= 0 && constants.softeningKernelOrder <= 3,
            "softening parameters are invalid");
        requireEvolutionMetadata(
            constants.softeningLength == constants.softeningCellFraction / double(gridSide), "softening length disagrees with the gravity cell spacing");
        requireEvolutionMetadata(initial.canonicalMomentum, "particle data does not use canonical momentum");
        requireEvolutionMetadata(initial.component == MatterComponent::cdm || initial.component == MatterComponent::cdmAndBaryons ||
                initial.component == MatterComponent::totalMatter,
            "initial matter component is invalid");
        requireEvolutionMetadata(initial.scaleFactor > 0.0 && initial.scaleFactor <= 1.0 && initial.scaleFactor == constants.initialScaleFactor,
            "original initial epoch disagrees with the run constants");
        requireEvolutionMetadata(initial.meshExponent == exponent && initial.component == constants.initializationOptions.component &&
                initial.seed == constants.initializationOptions.seed && initial.boxLength == constants.initializationOptions.boxLength,
            "initial-condition controls disagree with the run constants");
        requireEvolutionMetadata(initial.boxLength > 0.0 && initial.particleMass > 0.0 && initial.requestedDensityDeviation > 0.0 &&
                initial.maximumDensityDeviation >= initial.requestedDensityDeviation &&
                initial.requestedDensityDeviation == constants.initializationOptions.maximumDensityDeviation,
            "initial-condition units or handoff threshold are invalid");
        requireEvolutionMetadata(
            archiveCosmology(initial.cosmology) == archiveCosmology(constants.cosmology), "initial-condition cosmology disagrees with the run constants");
        requireEvolutionMetadata(state.scaleFactor > 0.0 && state.tickScaleFactor > 0.0, "current or last-event scale factor is not positive");
        double const timeMagnitude = std::max(1.0, std::max(std::abs(constants.initialPhysicalTime), std::abs(constants.stopPhysicalTime)));
        double const timeTolerance = kRelativeTimeTolerance * timeMagnitude;
        requireEvolutionMetadata(constants.stopPhysicalTime > constants.initialPhysicalTime &&
                state.tickPhysicalTime >= constants.initialPhysicalTime - timeTolerance && state.physicalTime >= state.tickPhysicalTime - timeTolerance &&
                state.physicalTime <= constants.stopPhysicalTime + timeTolerance,
            "physical clock lies outside the saved run interval");
        requireEvolutionMetadata(state.maximumRung <= Rung::maximumRung() && state.timeTick < gravity::rungTicks(0), "rung clock is out of range");
        requireEvolutionMetadata(gravity::rungSynchronized(state.timeTick, state.maximumRung), "event clock is not synchronized with its finest rung");
        requireEvolutionMetadata(state.conformalOffset >= 0.0 && state.conformalOffset < gravity::rungTimestep(constants.rungZeroTimestep, state.maximumRung),
            "partial conformal drift reaches or exceeds its next rung event");
        if (state.conformalOffset == 0.0) {
            requireEvolutionMetadata(state.scaleFactor == state.tickScaleFactor && state.conformalDerivative == state.tickConformalDerivative &&
                    state.physicalTime == state.tickPhysicalTime,
                "current background differs from the event with no conformal offset");
        }
        if (!state.momentaStaggered) {
            requireEvolutionMetadata(
                state.nominalEpoch == 0 && state.timeTick == 0 && state.conformalOffset == 0.0, "unopened momenta appear after the initial epoch");
        }
        auto const& energy = state.energy;
        if (energy.initialized) {
            requireEvolutionMetadata(
                energy.previousScaleFactor > 0.0 && energy.previousKinetic >= 0.0 && energy.kinetic >= 0.0, "energy history has an invalid sample");
        }
    }

}    // namespace detail

void writeEvolutionCheckpoint(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid, GlobalConstantsData const& constants,
    EvolutionState const& state, InitialConditionsSiloMetadata const& initial, ParticleSiloOutput const* output) {
    PROFILE();
    if (filename.empty()) {
        throw std::invalid_argument("evolution checkpoint filename is empty");
    }
    detail::validateEvolutionMetadata(constants, state, initial);
    detail::requireEvolutionMetadata(grid.layout().globalSideLength() == constants.gridSideLength &&
            grid.layout().localitiesPerAxis() == detail::checkpointLocalitySide(constants.localityCount) &&
            grid.layout().ghostWidth() == static_cast<size_t>(constants.ghostWidth),
        "particle grid differs from the run constants");
    detail::requireEvolutionMetadata(particles.storesAcceleration() == constants.outputAcceleration && particles.storesPotential() == constants.outputPotential,
        "particle gravity fields differ from the run constants");
    std::vector<char> payload;
    {
        hpx::serialization::output_archive archive(payload, detail::kEvolutionArchiveFlags);
        archive << detail::kEvolutionArchiveMagic << detail::kEvolutionArchiveVersion << constants << state;
    }
    detail::EvolutionSiloScalars scalars{};
    {
        FpeGuard guard;
        scalars.scaleFactor = state.scaleFactor;
        scalars.gravityParticleMass = constants.gravityParticleMass;
        scalars.conformalDerivative = state.conformalDerivative;
        scalars.yearsFromPresent = physicalTimeInYears(state.physicalTime, constants.cosmology);
        double const fractionalEpoch = std::ldexp(double(state.timeTick), -int(Rung::maximumRung()));
        scalars.conformalTime =
            std::fma(double(state.nominalEpoch), constants.rungZeroTimestep, std::fma(fractionalEpoch, constants.rungZeroTimestep, state.conformalOffset));
    }
    detail::writeParticleSiloEvolutionPayload(filename, particles, grid, initial, payload, scalars, output);
}

EvolutionCheckpointMetadata readEvolutionCheckpointMetadata(std::string const& filename) {
    constexpr int kLegacyGhostWidth = 2;
    PROFILE();

    auto payload = detail::readParticleSiloEvolutionPayload(filename);
    EvolutionCheckpointMetadata metadata{};
    {
        hpx::serialization::input_archive archive(payload, payload.size());
        std::uint64_t magic = 0;
        std::uint32_t version = 0;
        archive >> magic >> version;
        detail::requireEvolutionMetadata(magic == detail::kEvolutionArchiveMagic &&
                (version == 2 || version == 3 || version == 4 || version == 5 || version == 6 || version == 7 || version == 8 ||
                    version == detail::kEvolutionArchiveVersion),
            "serialized evolution state has an incompatible schema");
        if (version == 2) {
            serializeGlobalConstantsVersion2(archive, metadata.constants);
        } else {
            if (version == 3) {
                serializeGlobalConstantsVersion3(archive, metadata.constants);
            } else if (version == 4) {
                serializeGlobalConstantsVersion4(archive, metadata.constants);
            } else if (version == 5) {
                serializeGlobalConstantsVersion5(archive, metadata.constants);
            } else if (version == 6) {
                serializeGlobalConstantsVersion6(archive, metadata.constants);
            } else if (version == 7) {
                serializeGlobalConstantsVersion7(archive, metadata.constants);
            } else if (version == 8) {
                serializeGlobalConstantsVersion8(archive, metadata.constants);
            } else {
                archive >> metadata.constants;
            }
        }
        if (version < 7) {
            metadata.constants.gravityMeshExponent = metadata.constants.initializationOptions.meshExponent;
            metadata.constants.ghostWidth = kLegacyGhostWidth;
        }
        archive >> metadata.state;
        detail::requireEvolutionMetadata(archive.bytes_read() == payload.size(), "serialized evolution state has trailing data");
    }
    metadata.initial = readInitialConditionsSiloMetadata(filename);
    detail::validateEvolutionMetadata(metadata.constants, metadata.state, metadata.initial);
    detail::validateParticleSiloEvolutionLayout(filename, metadata.constants.gridSideLength, detail::checkpointLocalitySide(metadata.constants.localityCount),
        metadata.constants.outputAcceleration, metadata.constants.outputPotential);
    return metadata;
}

}    // namespace cosmictiger
