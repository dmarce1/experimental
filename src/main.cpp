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

#include <cosmictiger/command_line_options.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/evolution/evolution.hpp>
#include <cosmictiger/evolution/evolution_checkpoint.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/gravity/kick/pmmm_kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_runtime_convolution.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>
#include <cosmictiger/particle/particle_initialization.hpp>
#include <cosmictiger/particle/particle_silo.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <cosmictiger/startup.hpp>
#include <cosmictiger/units.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <silo.h>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

static_assert(__cplusplus >= 202002L, "CosmicTiger requires C++20.");

namespace cosmictiger {
namespace detail {

    static std::optional<ParticleInitialConditions> initialConditions;
    static std::optional<EvolutionCheckpointMetadata> restartMetadata;

    std::string evolutionSnapshotStem(std::string const& prefix, size_t snapshot) {
        PROFILE();
        std::ostringstream name;
        name << prefix << ".snapshot" << std::setw(4) << std::setfill('0') << snapshot;
        return name.str();
    }

    std::string evolutionSnapshotFilename(std::string const& prefix, size_t snapshot, size_t locality) {
        PROFILE();
        return evolutionSnapshotStem(prefix, snapshot) + ".locality" + std::to_string(locality) + ".silo";
    }

    std::string evolutionMasterFilename(std::string const& prefix, size_t snapshot) {
        PROFILE();
        return evolutionSnapshotStem(prefix, snapshot) + ".master.silo";
    }

    void publishVisitSeries(std::string const& filename, std::vector<std::string> const& files) {
        PROFILE();
        if (files.empty()) {
            throw std::invalid_argument("VisIt series requires a completed timestep");
        }
        std::string const temporary = filename + ".tmp";
        {
            std::ofstream output;
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output.open(temporary);
            for (auto const& file : files) {
                output << hpx::filesystem::path(file).filename().string() << '\n';
            }
            output.close();
        }
        hpx::filesystem::rename(temporary, filename);
    }

    void publishEvolutionVisitSeries(std::string const& generationPrefix, size_t completedSnapshots, size_t localityCount) {
        PROFILE();
        std::vector<std::string> files;
        files.reserve(completedSnapshots);
        for (size_t snapshot = 0; snapshot < completedSnapshots; snapshot++) {
            files.push_back(
                localityCount == 1 ? evolutionSnapshotFilename(generationPrefix, snapshot, 0) : evolutionMasterFilename(generationPrefix, snapshot));
        }
        publishVisitSeries(generationPrefix + ".visit", files);
    }

    char const* evolutionInitializationLabel(EvolutionInitializationStage stage) {
        PROFILE();
        switch (stage) {
        case EvolutionInitializationStage::backgroundTable:
            return "integrating cosmology and building interpolation table";
        case EvolutionInitializationStage::snapshotCadence:
            return "computing evolution snapshot times";
        case EvolutionInitializationStage::gridAndRungs:
            return "expanding evolution grid and sorting particle rungs";
        case EvolutionInitializationStage::lightcone:
            return "preparing lightcone observer and HEALPix output";
        case EvolutionInitializationStage::periodicConvolution:
            return "building periodic convolution and FFT locality layout";
        case EvolutionInitializationStage::fftExchange:
            return "registering FFT communication channels";
        case EvolutionInitializationStage::ghostExchange:
            return "exchanging initial ghost particles";
        case EvolutionInitializationStage::cellMultipoles:
            return "computing initial cell multipoles";
        case EvolutionInitializationStage::gravityKick:
            return "evaluating initial direct and periodic gravity and kicking particles";
        case EvolutionInitializationStage::diagnostics:
            return "recording initial energy and force diagnostics";
        case EvolutionInitializationStage::firstSnapshot:
            return "writing first evolution Silo snapshot";
        }
        throw std::logic_error("unknown evolution initialization stage");
    }

    char const* gravityInitializationLabel(gravity::PmmmProgressStage stage, bool initialKick) {
        using gravity::PmmmProgressStage;
        PROFILE();
        switch (stage) {
        case PmmmProgressStage::directGravity:
            return "direct neighbor acceleration and potential (overlaps periodic solve)";
        case PmmmProgressStage::cudaPreparation:
            return "preparing CUDA particle and neighbor ranges";
        case PmmmProgressStage::ewaldFilters:
            return "building periodic Ewald filters";
        case PmmmProgressStage::filterFft:
            return "transforming and caching Ewald filters";
        case PmmmProgressStage::multipoleFft:
            return "transforming cell multipoles and second-moment traces";
        case PmmmProgressStage::contraction:
            return "Fourier multipole-to-local contractions";
        case PmmmProgressStage::expansionFft:
            return "inverse transforms of cell expansions";
        case PmmmProgressStage::periodicBackground:
            return "computing and applying periodic background corrections";
        case PmmmProgressStage::particleKick:
            return initialKick ? "evaluating particle expansions, kinetic diagnostics, rungs, half-kicks and sums" :
                                 "evaluating particle expansions and synchronized diagnostic sums";
        case PmmmProgressStage::cudaKick:
            return initialKick ? "CUDA particle expansions, direct gravity, kinetic diagnostics, rungs, half-kicks and sums" :
                                 "CUDA particle expansions, direct gravity and synchronized diagnostic sums";
        case PmmmProgressStage::particleSort:
            return "sorting particles within cells by timestep rung";
        case PmmmProgressStage::diagnosticReduction:
            return "combining force, potential, kinetic, particle-count and maximum-rung diagnostics across localities";
        }
        throw std::logic_error("unknown gravity initialization progress stage");
    }

    void reportGravityInitialization(gravity::PmmmProgress const& progress, bool initialKick) {
        constexpr double kPercentScale = 100.0;
        PROFILE();

        std::ostringstream line;
        line << "Initializing: gravity: " << gravityInitializationLabel(progress.stage, initialKick);
        if (progress.completed == progress.total) {
            line << " - complete";
        } else if (progress.completed == 0) {
            line << " - starting";
        }
        if (progress.total > 1) {
            auto const stage = progress.stage;
            char const* unit = stage == gravity::PmmmProgressStage::ewaldFilters ? "cells" :
                stage == gravity::PmmmProgressStage::directGravity || stage == gravity::PmmmProgressStage::particleKick ||
                    stage == gravity::PmmmProgressStage::cudaKick ?
                                                                                   "particles" :
                stage == gravity::PmmmProgressStage::periodicBackground ? "steps" :
                                                                          "batches";
            line << " [" << progress.completed << '/' << progress.total << ' ' << unit << ", " << std::fixed << std::setprecision(1)
                 << kPercentScale * double(progress.completed) / double(progress.total) << "%]";
        }
        line << '\n';
        hpx::cout << line.str() << hpx::iostreams::flush_type{};
    }

}    // namespace detail

double generateInitialConditions(GlobalConstantsData data, std::uint64_t operationId) {
    PROFILE();
    hpx::register_pre_shutdown_function(hpx::annotated_function(
        [] {
            PROFILE();
            detail::initialConditions.reset();
        },
        "main.register_pre_shutdown_function"));
    ParticleInitializationObserver const progress = hpx::annotated_function(
        [](std::string const& message) {
            PROFILE();
            hpx::cout << "Initializing: " << message << '\n' << hpx::iostreams::flush_type{};
        },
        "main.initializationProgress");
    detail::initialConditions.emplace(initializeParticles(data.cosmology, data.initializationOptions, operationId, progress));
    return detail::initialConditions->report.scaleFactor;
}

void writeInitialConditions() {
    constexpr std::uint64_t kGhostOperation = 0x494347484F535453ULL;
    constexpr std::uint64_t kGridOperation = 0x4943475249445052ULL;
    constexpr int kImageCutoff = 4;
    constexpr int kWaveCutoff = 4;
    constexpr double kAlphaBox = 1.5;
    PROFILE();
    auto const& data = globalConstants().data();
    if (!detail::initialConditions) {
        throw std::logic_error("initial particles have not been generated on this locality");
    }
    auto& initial = *detail::initialConditions;
    initial.particles.configureGravityFields(data.outputAcceleration || data.directSinkFraction > 0.0, data.outputPotential || data.directSinkFraction > 0.0);
    for (size_t particle = 0; particle < initial.particles.size(); particle++) {
        FpeGuard guard;
        for (size_t axis = 0; axis < NDIM; axis++) {
            initial.particles.velocity(axis, particle) = static_cast<float>(initial.particles.velocity(axis, particle) * data.initialConformalDerivative);
        }
    }
    if (data.outputAcceleration || data.outputPotential || data.directSinkFraction > 0.0) {
        if (hpx::get_locality_id() == 0) {
            hpx::cout << "Initializing: computing requested gravity fields for initial output\n" << hpx::iostreams::flush_type{};
        }
        prepareEvolutionGrid(initial.particles, initial.grid, data.gravityMeshExponent, static_cast<size_t>(data.ghostWidth), kGridOperation);
        auto const localities = hpx::find_all_localities();

        exchangeGhostParticles(initial.particles, initial.grid, localities, kGhostOperation);
#ifdef COSMICTIGER_ENABLE_CUDA
        if (data.cudaMultipole) {
            initial.grid.computeMultipolesCuda(
                initial.particles, static_cast<float>(data.gravityParticleMass), 0, data.softeningModel == gravity::SofteningModel::plummer);
        } else
#endif
        {
            initial.grid.computeMultipoles(
                initial.particles, static_cast<float>(data.gravityParticleMass), data.softeningModel == gravity::SofteningModel::plummer);
        }
        gravity::PmmmRuntimeConvolution convolution(
            initial.grid.layout(), data.fmmOrder, kImageCutoff, kWaveCutoff, kAlphaBox, data.ghostWidth, data.softeningModel, data.softeningLength);
        std::vector<hpx::id_type> fftLocalities(localities.size());
        size_t const localitySide = initial.grid.layout().localitiesPerAxis();
        for (size_t gridIndex = 0; gridIndex < localities.size(); gridIndex++) {
            size_t remainder = gridIndex;
            size_t fftIndex = 0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                fftIndex = fftIndex * localitySide + remainder % localitySide;
                remainder /= localitySide;
            }
            fftLocalities[fftIndex] = localities[gridIndex];
        }
        FftHpxExchange exchange(fftLocalities, 0x4943504D4D4D4646ULL, convolution.fftLocalityIndex());
        gravity::KickParameters parameters{data.timestepEta, 0.0, data.initialScaleFactor, 0, gravity::KickPhase::diagnostics, data.outputPotential};
        parameters.ghostWidth = data.ghostWidth;
        auto computeFarField = hpx::annotated_function(
            [&] {
                PROFILE();
                convolution.computeExpansions(initial.grid, exchange);
            },
            "main.computeFarField");
        gravity::KickSums sums;
#ifdef COSMICTIGER_ENABLE_CUDA
        if (data.cudaKick) {
            sums = gravity::kickPmmmParticlesOverlappedCuda(initial.grid, initial.particles,
                gravity::makeSoftening(static_cast<float>(data.softeningLength), data.softeningModel), data.softeningKernelOrder,
                static_cast<float>(data.gravityParticleMass), data.rungZeroTimestep, 0, parameters, computeFarField, 0x49434B49434B5355ULL);
        } else
#endif
        {
            sums = gravity::kickPmmmParticlesOverlapped(initial.grid, initial.particles,
                gravity::makeSoftening(static_cast<float>(data.softeningLength), data.softeningModel), data.softeningKernelOrder,
                static_cast<float>(data.gravityParticleMass), data.rungZeroTimestep, 0, parameters, computeFarField, 0x49434B49434B5355ULL);
        }
        if (data.directSinkFraction > 0.0) {
            if (hpx::get_locality_id() == 0) {
                hpx::cout << "Initializing: direct periodic gravity comparison\n" << hpx::iostreams::flush_type{};
            }
            auto const comparison = gravity::compareDirectGravity(initial.particles, initial.grid, data, 0x4943444952454354ULL);
            if (hpx::get_locality_id() == 0) {
                hpx::cout << gravity::formatDirectComparison(comparison) << hpx::iostreams::flush_type{};
            }
            initial.particles.configureGravityFields(data.outputAcceleration, data.outputPotential);
        }
        if (hpx::get_locality_id() == 0) {
            hpx::cout << "  Global force sum (code units):";
            for (size_t axis = 0; axis < NDIM; axis++) {
                hpx::cout << ' ' << data.gravityParticleMass * sums.acceleration[axis];
            }
            hpx::cout << '\n';
            if (parameters.sumPotential) {
                hpx::cout << "  Global potential sum (code units): " << sums.potential << '\n';
            }
        }
    }
    auto const& report = initial.report;
    InitialConditionsSiloMetadata const metadata{data.cosmology, report.scaleFactor, report.boxLength, report.particleMass, report.maximumDensityDeviation,
        data.initializationOptions.maximumDensityDeviation, data.initializationOptions.seed, data.initializationOptions.meshExponent,
        data.initializationOptions.component, true};
    std::string const filename = data.outputPrefix + ".locality" + std::to_string(hpx::get_locality_id()) + ".silo";
    if (hpx::get_locality_id() == 0) {
        hpx::cout << "Initializing: writing initial-condition Silo files\n" << hpx::iostreams::flush_type{};
    }
    writeParticleSilo(filename, initial.particles, initial.grid, metadata);
    if (hpx::get_locality_id() == 0) {
        hpx::cout << "  Initial scale factor: " << report.scaleFactor << " (redshift " << report.redshift << ")\n"
                  << "  Particles: " << report.globalParticleCount << ", box: " << report.boxLength << " Mpc/h\n"
                  << "  Particle mass: " << report.particleMass << " Msun/h\n"
                  << "  Maximum linear density deviation: " << report.maximumDensityDeviation << '\n'
                  << "  Spectrum evaluations: " << report.spectrumEvaluations << '\n'
                  << "  Rung-zero conformal timestep: " << data.rungZeroTimestep << " H0^-1\n"
                  << hpx::iostreams::flush_type{};
    }
    detail::initialConditions.reset();
}

std::vector<char> restoreEvolution(std::string filename, size_t localityCount) {
    PROFILE();
    auto metadata = readEvolutionCheckpointMetadata(filename);
    if (metadata.constants.localityCount != localityCount) {
        throw std::invalid_argument("restart requires the checkpoint locality decomposition");
    }
    size_t localityExponent = 0;
    for (size_t count = localityCount; count > 1; count >>= NDIM) {
        localityExponent++;
    }
    auto const& source = metadata.initial;
    ParticleInitializationReport report{source.scaleFactor, 1.0 / source.scaleFactor - 1.0, source.maximumDensityDeviation, source.scaleFactor,
        source.maximumDensityDeviation, source.boxLength, source.particleMass, metadata.constants.globalParticleCount, 0};
    ParticleInitialConditions restored{ParticleStorage{},
        CartesianGrid(metadata.constants.gravityMeshExponent, localityExponent, hpx::get_locality_id(), metadata.constants.ghostWidth), report};
    readParticleSilo(filename, restored.particles, restored.grid);
    restored.grid.sortParticlesByRung(restored.particles);
    detail::initialConditions.emplace(std::move(restored));
    detail::restartMetadata.emplace(std::move(metadata));
    hpx::register_pre_shutdown_function(hpx::annotated_function(
        [] {
            PROFILE();
            detail::initialConditions.reset();
            detail::restartMetadata.reset();
        },
        "main.register_pre_shutdown_function"));
    std::vector<char> summary;
    hpx::serialization::output_archive archive(summary);
    archive << detail::restartMetadata->constants << detail::restartMetadata->state;
    return summary;
}

void runEvolution(std::string generationPrefix) {
    PROFILE();
    auto const& data = globalConstants().data();
    if (!detail::initialConditions) {
        throw std::logic_error("evolution particles have not been initialized");
    }
    auto& initial = *detail::initialConditions;
    auto& particles = initial.particles;
    auto& grid = initial.grid;
    EvolutionState state;
    InitialConditionsSiloMetadata metadata{};
    std::string error;
    std::ofstream diagnostics;
    try {
        particles.configureGravityFields(data.outputAcceleration, data.outputPotential);
        if (detail::restartMetadata) {
            metadata = detail::restartMetadata->initial;
            state = detail::restartMetadata->state;
            extendEvolutionRungs(particles, grid, state, data.restartRungZeroExtension);
        } else {
            auto const& report = initial.report;
            metadata = {data.cosmology, report.scaleFactor, report.boxLength, report.particleMass, report.maximumDensityDeviation,
                data.initializationOptions.maximumDensityDeviation, data.initializationOptions.seed, data.initializationOptions.meshExponent,
                data.initializationOptions.component, true};
            for (ParticleCount index = 0; index < particles.size(); index++) {
                FpeGuard guard;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    particles.velocity(axis, index) = static_cast<float>(particles.velocity(axis, index) * data.initialConformalDerivative);
                }
            }
            state.scaleFactor = state.tickScaleFactor = data.initialScaleFactor;
            state.conformalDerivative = state.tickConformalDerivative = data.initialConformalDerivative;
            state.physicalTime = state.tickPhysicalTime = data.initialPhysicalTime;
        }
        if (hpx::get_locality_id() == 0) {
            diagnostics.exceptions(std::ios::badbit | std::ios::failbit);
            diagnostics.open(generationPrefix + ".diagnostics.csv");
            diagnostics << "yearsFromPresent,scaleFactor,daDt,K,U,expansionWork,conservedEnergy,energyDrift,forceX,forceY,forceZ,maximumRung"
                        << ",physicalTimeSeconds,daDtPerSecond,kineticMsunKm2PerSecond2,potentialMsunKm2PerSecond2,expansionWorkMsunKm2PerSecond2"
                        << ",conservedEnergyMsunKm2PerSecond2,energyDriftMsunKm2PerSecond2"
                        << ",forceXMsunKm2PerSecond2PerMpc,forceYMsunKm2PerSecond2PerMpc,forceZMsunKm2PerSecond2PerMpc"
                        << ",kineticErg,potentialErg,expansionWorkErg,conservedEnergyErg,energyDriftErg,forceXDyne,forceYDyne,forceZDyne\n";
        }
    } catch (std::exception const& failure) {
        error = failure.what();
    }
    gravity::allReduceKickSums({}, false, 0x45564F5052455000ULL, std::move(error));
    if (hpx::get_locality_id() == 0) {
        hpx::cout << "Initializing: evolution grid and first gravity evaluation\n" << hpx::iostreams::flush_type{};
    }
    size_t snapshotIndex = 0;
    size_t completedSteps = 0;
    size_t progressLines = 0;
    gravity::PmmmProgressObserver gravityInitializationProgress;
    if (hpx::get_locality_id() == 0) {
        gravityInitializationProgress = hpx::annotated_function(
            [initialKick = !state.momentaStaggered](gravity::PmmmProgress const& progress) {
                PROFILE();
                detail::reportGravityInitialization(progress, initialKick);
            },
            "reportGravityInitialization");
    }
    evolveParticles(particles, grid, data, state,
        hpx::annotated_function(
            [&](EvolutionState const& current, gravity::KickSums const& sums) {
                constexpr size_t kEnergyQuantityCount = 5;
                PROFILE();
                if (hpx::get_locality_id() == 0) {
                    auto const& scales = globalConstants().unitScales();
                    FpeGuard guard;
                    auto const& energy = current.energy;
                    double const years = physicalTimeInYears(current.physicalTime, data.cosmology);
                    double const energyAstro = scales.massSolar() * scales.velocityKmPerSecond() * scales.velocityKmPerSecond();
                    double const forceAstro = energyAstro / scales.lengthMpc();
                    double const inverseScaleSquared = 1.0 / (current.scaleFactor * current.scaleFactor);
                    std::array<double, NDIM> forces{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        forces[axis] = data.gravityParticleMass * sums.acceleration[axis] * inverseScaleSquared;
                    }
                    std::array<double, kEnergyQuantityCount> const energies{
                        energy.kinetic, energy.potential, energy.expansionWork, energy.conservedEnergy, energy.conservedEnergy - energy.initialEnergy};
                    diagnostics << std::setprecision(17) << years << ',' << current.scaleFactor << ','
                                << current.conformalDerivative / current.scaleFactor / physicalTimeInYears(1.0, data.cosmology) << ',' << energy.kinetic << ','
                                << energy.potential << ',' << energy.expansionWork << ',' << energy.conservedEnergy << ','
                                << energy.conservedEnergy - energy.initialEnergy;
                    for (double const force : forces) {
                        diagnostics << ',' << force;
                    }
                    diagnostics << ',' << unsigned(current.maximumRung) << ',' << current.physicalTime * scales.timeSeconds() << ','
                                << current.conformalDerivative / current.scaleFactor / scales.timeSeconds();
                    for (double const value : energies) {
                        diagnostics << ',' << value * energyAstro;
                    }
                    for (double const force : forces) {
                        diagnostics << ',' << force * forceAstro;
                    }
                    for (double const value : energies) {
                        diagnostics << ',' << value * scales.energyCgs();
                    }
                    for (double const force : forces) {
                        diagnostics << ',' << force * scales.forceCgs();
                    }
                    diagnostics << '\n';
                    diagnostics.flush();
                }
            },
            "evolutionDiagnostics"),
        hpx::annotated_function(
            [&](EvolutionState const& current, gravity::KickSums const&, ParticleSiloOutput const& output) {
                PROFILE();
                writeEvolutionCheckpoint(detail::evolutionSnapshotFilename(generationPrefix, snapshotIndex, hpx::get_locality_id()), particles, grid, data,
                    current, metadata, &output);
                snapshotIndex++;
            },
            "writeEvolutionSnapshot"),
        hpx::annotated_function(
            [&](EvolutionState const& current, EvolutionProgress const& progress) {
                constexpr double kSecondsPerNanosecond = 1.0e-9;
                constexpr double kPercent = 100.0;
                constexpr size_t kHeaderInterval = 25;
                constexpr int kStepWidth = 6;
                constexpr int kRungWidth = 2;
                constexpr int kValueWidth = 10;
                constexpr int kPhaseWidth = 5;
                PROFILE();
                if (hpx::get_locality_id() == 0) {
                    if (progress.kind == EvolutionProgressKind::rungStep) {
                        completedSteps++;
                    }
                    double physicalYears;
                    double conformalTime;
                    double physicalDerivative;
                    double redshift;
                    {
                        FpeGuard guard;
                        double const yearsPerHubbleTime = physicalTimeInYears(1.0, data.cosmology);
                        physicalYears = physicalTimeInYears(current.physicalTime, data.cosmology);
                        double const fractionalEpoch = std::ldexp(double(current.timeTick), -int(Rung::maximumRung()));
                        conformalTime = std::fma(
                            double(current.nominalEpoch), data.rungZeroTimestep, std::fma(fractionalEpoch, data.rungZeroTimestep, current.conformalOffset));
                        physicalDerivative = current.conformalDerivative / current.scaleFactor / yearsPerHubbleTime;
                        redshift = 1.0 / current.scaleFactor - 1.0;
                    }
                    if (progress.wroteSnapshot) {
                        if (data.localityCount > 1) {
                            std::vector<std::string> localityFiles;
                            localityFiles.reserve(data.localityCount);
                            for (size_t locality = 0; locality < data.localityCount; locality++) {
                                localityFiles.push_back(detail::evolutionSnapshotFilename(generationPrefix, snapshotIndex - 1, locality));
                            }
                            writeParticleSiloMaster(detail::evolutionMasterFilename(generationPrefix, snapshotIndex - 1), localityFiles, conformalTime);
                        }
                        detail::publishEvolutionVisitSeries(generationPrefix, snapshotIndex, data.localityCount);
                    }
                    auto const timingEnd = hpx::chrono::high_resolution_clock::now();
                    auto phasePercent = progress.phaseTimes;
                    {
                        FpeGuard guard;
                        phasePercent.output += double(timingEnd - progress.timingEnd) * kSecondsPerNanosecond;
                        double const total = phasePercent.drift + phasePercent.exchange + phasePercent.multipole + phasePercent.gravity + phasePercent.kick +
                            phasePercent.output;
                        double const percentScale = total > 0.0 ? kPercent / total : 0.0;
                        phasePercent.drift *= percentScale;
                        phasePercent.exchange *= percentScale;
                        phasePercent.multipole *= percentScale;
                        phasePercent.gravity *= percentScale;
                        phasePercent.kick *= percentScale;
                        phasePercent.output *= percentScale;
                    }
                    char const kind = progress.kind == EvolutionProgressKind::initial ? 'I' : progress.kind == EvolutionProgressKind::rungStep ? 'R' : 'S';
                    std::ostringstream line;
                    if (progressLines == 0) {
                        line << "I=initial R=rung S=sample; *=completed snapshot; r=completed rung, rm=maximum rung\n"
                             << "t: years from expanding a=1; tau: H0^-1 since original handoff; ad: da/dt [yr^-1]\n"
                             << "Phase %: D=drift, X=redistribute+ghosts, M=P2M, G=gravity prep, K=force+kick, O=diagnostics+output\n"
                             << "Percentages use root wall time for this row; progress reporting itself is excluded.\n";
                    }
                    if (progressLines % kHeaderInterval == 0) {
                        line << '\n'
                             << "* " << std::right << std::setw(kStepWidth) << "step" << " k " << std::setw(kRungWidth) << "r" << ' ' << std::setw(kValueWidth)
                             << "t[yr]" << ' ' << std::setw(kValueWidth) << "tau" << ' ' << std::setw(kValueWidth) << "a" << ' ' << std::setw(kValueWidth)
                             << "z" << ' ' << std::setw(kValueWidth) << "ad" << ' ' << std::setw(kRungWidth) << "rm" << ' ' << std::setw(kPhaseWidth) << "D%"
                             << ' ' << std::setw(kPhaseWidth) << "X%" << ' ' << std::setw(kPhaseWidth) << "M%" << ' ' << std::setw(kPhaseWidth) << "G%" << ' '
                             << std::setw(kPhaseWidth) << "K%" << ' ' << std::setw(kPhaseWidth) << "O%" << '\n';
                    }
                    line << (progress.wroteSnapshot ? '*' : ' ') << ' ' << std::right << std::setw(kStepWidth) << completedSteps << ' ' << kind << ' '
                         << std::setw(kRungWidth) << (progress.kind == EvolutionProgressKind::rungStep ? std::to_string(unsigned(progress.completedRung)) : "-")
                         << std::scientific << std::setprecision(3) << ' ' << std::setw(kValueWidth) << physicalYears << ' ' << std::setw(kValueWidth)
                         << conformalTime << ' ' << std::setw(kValueWidth) << current.scaleFactor << ' ' << std::setw(kValueWidth) << redshift << ' '
                         << std::setw(kValueWidth) << physicalDerivative << ' ' << std::setw(kRungWidth) << unsigned(current.maximumRung) << std::fixed
                         << std::setprecision(1) << ' ' << std::setw(kPhaseWidth) << phasePercent.drift << ' ' << std::setw(kPhaseWidth)
                         << phasePercent.exchange << ' ' << std::setw(kPhaseWidth) << phasePercent.multipole << ' ' << std::setw(kPhaseWidth)
                         << phasePercent.gravity << ' ' << std::setw(kPhaseWidth) << phasePercent.kick << ' ' << std::setw(kPhaseWidth) << phasePercent.output;
                    hpx::cout << line.str() << '\n' << hpx::iostreams::flush_type{};
                    progressLines++;
                }
            },
            "reportEvolutionProgress"),
        hpx::annotated_function(
            [&](EvolutionInitializationStage stage) {
                PROFILE();
                if (hpx::get_locality_id() == 0) {
                    hpx::cout << "Initializing: " << detail::evolutionInitializationLabel(stage) << '\n' << hpx::iostreams::flush_type{};
                    if (stage == EvolutionInitializationStage::gravityKick && data.localityCount > 1) {
                        hpx::cout << "Initializing: gravity: work counts describe locality 0; final diagnostics combine all localities\n"
                                  << hpx::iostreams::flush_type{};
                    }
                }
            },
            "reportEvolutionInitialization"),
        gravityInitializationProgress);
    if (snapshotIndex != data.outputSnapshotCount) {
        throw std::runtime_error("evolution did not write the requested snapshot count");
    }
    detail::initialConditions.reset();
    detail::restartMetadata.reset();
}

namespace detail {
    std::vector<std::string> readEvolutionManifest(std::string const& filename, size_t localityCount) {
        PROFILE();
        std::ifstream input(filename);
        std::string magic;
        unsigned version = 0;
        size_t count = 0;
        if (!(input >> magic >> version >> count) || magic != "CosmicTigerEvolution" || version != 1 || count != localityCount) {
            throw std::runtime_error("restart manifest is incomplete, incompatible, or has a different locality count");
        }
        std::vector<std::string> files(count);
        for (auto& file : files) {
            if (!(input >> std::quoted(file)) || file.empty()) {
                throw std::runtime_error("restart manifest is missing a locality checkpoint");
            }
        }
        std::string trailing;
        if (input >> trailing) {
            throw std::runtime_error("restart manifest contains unexpected trailing data");
        }
        return files;
    }

    template <typename T>
    std::vector<T> awaitEvolutionTasks(std::vector<hpx::future<T>>& tasks) {
        PROFILE();
        std::vector<T> result;
        std::exception_ptr failure;
        for (auto& task : tasks) {
            try {
                result.push_back(task.get());
            } catch (...) {
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        return result;
    }

    void awaitEvolutionTasks(std::vector<hpx::future<void>>& tasks) {
        PROFILE();
        std::exception_ptr failure;
        for (auto& task : tasks) {
            try {
                task.get();
            } catch (...) {
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}    // namespace detail

}    // namespace cosmictiger

HPX_PLAIN_ACTION(cosmictiger::generateInitialConditions, CosmictigerGenerateInitialConditionsAction)
HPX_PLAIN_ACTION(cosmictiger::writeInitialConditions, CosmictigerWriteInitialConditionsAction)
HPX_PLAIN_ACTION(cosmictiger::restoreEvolution, CosmictigerRestoreEvolutionAction)
HPX_PLAIN_ACTION(cosmictiger::runEvolution, CosmictigerRunEvolutionAction)

namespace {

void printCommandLineHelp(hpx::program_options::options_description const& options) {
    PROFILE();
    std::cout << "Usage: cosmictiger [options] [HPX options]\n\n" << options << "\nFor HPX runtime options, use --hpx:help.\n" << std::flush;
}

}    // namespace

int hpx_main(hpx::program_options::variables_map& vm) {
    PROFILE();
    try {
        if (vm.count("help") != 0) {
            hpx::program_options::options_description options("CosmicTiger options");
            cosmictiger::addCommandLineOptions(options);
            printCommandLineHelp(options);
            return hpx::finalize();
        }
        auto const localities = hpx::find_all_localities();
        auto data = cosmictiger::readCommandLineOptions(vm, localities.size());
        hpx::cout << cosmictiger::kStartupScreen << "  Silo version: " << DBVersion() << '\n';
#ifdef COSMICTIGER_ENABLE_CUDA
        hpx::cout << "  CUDA toolkit: " << CUDART_VERSION / 1000 << '.' << (CUDART_VERSION % 1000) / 10 << '\n';
#endif
        hpx::cout << "  HPX runtime started with MPI support.\n\n" << hpx::iostreams::flush_type{};
        if (!data.restartManifest.empty()) {
            if (!data.evolve || data.generateInitialConditions) {
                throw std::invalid_argument("restart.manifest requires evolution and cannot regenerate initial conditions");
            }
            hpx::cout << "Initializing: reading restart manifest and checkpoints\n" << hpx::iostreams::flush_type{};
            auto const files = cosmictiger::detail::readEvolutionManifest(data.restartManifest, localities.size());
            std::vector<hpx::future<std::vector<char>>> tasks;
            for (size_t index = 0; index < localities.size(); index++) {
                tasks.push_back(hpx::async<CosmictigerRestoreEvolutionAction>(localities[index], files[index], localities.size()));
            }
            auto const summaries = cosmictiger::detail::awaitEvolutionTasks(tasks);
            for (auto const& summary : summaries) {
                if (summary != summaries.front()) {
                    throw std::runtime_error("restart files belong to different checkpoint generations");
                }
            }
            hpx::cout << "Initializing: validating restart endpoint and rung clock\n" << hpx::iostreams::flush_type{};
            auto requested = data;
            cosmictiger::EvolutionState state;
            hpx::serialization::input_archive archive(summaries.front(), summaries.front().size());
            archive >> data >> state;
            cosmictiger::validateGravityRestartOptions(vm, data);
            data.stopTimeYears = requested.stopTimeYears;
            data.stopAtEnd = requested.stopAtEnd;
            data.minimumScaleFactor = requested.minimumScaleFactor;
            data.restartManifest = requested.restartManifest;
            data.checkpointPrefix = requested.checkpointPrefix;
            data.outputSnapshotCount = requested.outputSnapshotCount;
            data.configurationFile = requested.configurationFile;
            data.restartRungZeroExtension = requested.restartRungZeroExtension;
            data.cudaDrift = requested.cudaDrift;
            data.cudaKick = requested.cudaKick;
            data.cudaMultipole = requested.cudaMultipole;
            data.cudaDirect = requested.cudaDirect;
            data.directSinkFraction = requested.directSinkFraction;
            data.directSeed = requested.directSeed;
            cosmictiger::applyLightconeRestartOptions(vm, data);
            cosmictiger::initializeSimulationEndpoint(data);
            if (!(data.stopPhysicalTime > state.physicalTime)) {
                throw std::invalid_argument("restart stopping time must follow the checkpoint physical time");
            }
            cosmictiger::extendEvolutionClock(state, data.restartRungZeroExtension);
            if (data.restartRungZeroExtension > cosmictiger::Rung::maximumRung() - data.rungZeroExtension ||
                data.rungZeroTimestep > std::ldexp(std::numeric_limits<double>::max(), -int(data.restartRungZeroExtension))) {
                throw std::range_error("restart rung-zero extension exceeds the supported clock range");
            }
            data.rungZeroExtension += data.restartRungZeroExtension;
            data.rungZeroTimestep = std::ldexp(data.rungZeroTimestep, int(data.restartRungZeroExtension));
        } else {
            if ((data.evolve || (data.generateInitialConditions && (data.outputAcceleration || data.outputPotential || data.directSinkFraction > 0.0))) &&
                data.gridSideLength < static_cast<size_t>(2 * data.ghostWidth + 1)) {
                throw std::invalid_argument("PMMM gravity requires 2^gravity.meshExponent >= 2*gravity.ghostWidth+1");
            }
            if (data.generateInitialConditions || data.evolve) {
                hpx::cout << "Initializing: generating particle initial conditions\n" << hpx::iostreams::flush_type{};
                std::vector<hpx::future<double>> tasks;
                for (auto const& locality : localities) {
                    tasks.push_back(hpx::async<CosmictigerGenerateInitialConditionsAction>(locality, data, 0x47434F4E5354414EULL));
                }
                auto const epochs = cosmictiger::detail::awaitEvolutionTasks(tasks);
                data.initialScaleFactor = epochs.front();
                for (double epoch : epochs) {
                    if (epoch != data.initialScaleFactor) {
                        throw std::runtime_error("initial-condition epochs differ between localities");
                    }
                }
                hpx::cout << "Initializing: particle initial conditions generated at a=" << data.initialScaleFactor << '\n' << hpx::iostreams::flush_type{};
            }
            hpx::cout << "Initializing: computing background and run constants\n" << hpx::iostreams::flush_type{};
            cosmictiger::initializeSimulationConstants(data, data.initialScaleFactor);
        }
        if (data.lightcone.enabled) {
            data.lightconeRunId = std::to_string(getpid()) + "." + std::to_string(hpx::chrono::high_resolution_clock::now());
            hpx::cout << "Lightcone output series: " << data.lightcone.outputPrefix << '.' << data.lightconeRunId << '\n' << hpx::iostreams::flush_type{};
        }
        hpx::cout << "Initializing: publishing run constants\n" << hpx::iostreams::flush_type{};
        cosmictiger::publishGlobalConstants(std::move(data));
        auto const& constants = cosmictiger::globalConstants().data();
        hpx::cout << "  Particle lattice and initialization FFT: " << constants.meshSideLength << " per axis (" << constants.globalParticleCount
                  << " particles)\n"
                  << "  Evolution mesh: " << constants.gridSideLength << " per axis, ghost width: " << constants.ghostWidth << " cells\n";
        hpx::cout << "  Execution: drift=" << (constants.cudaDrift ? "CUDA" : "CPU") << ", kick=" << (constants.cudaKick ? "CUDA" : "CPU")
                  << ", multipole=" << (constants.cudaMultipole ? "CUDA" : "CPU") << '\n'
                  << hpx::iostreams::flush_type{};
        if (constants.evolve) {
            auto const outputParent = hpx::filesystem::path(constants.checkpointPrefix).parent_path();
            if (!outputParent.empty()) {
                hpx::filesystem::create_directories(outputParent);
            }
            std::string const generation = hpx::filesystem::absolute(constants.checkpointPrefix).string() + "." + std::to_string(getpid()) + "." +
                std::to_string(hpx::chrono::high_resolution_clock::now());
            hpx::cout << "Rung-zero conformal timestep: " << constants.rungZeroTimestep << " H0^-1\n" << hpx::iostreams::flush_type{};
            if (constants.stopAtEnd) {
                hpx::cout << "Predicted Big Crunch: " << cosmictiger::physicalTimeInYears(constants.endPhysicalTime, constants.cosmology)
                          << " years from present; stopping before the singularity at a=" << constants.minimumScaleFactor << ", remaining time="
                          << cosmictiger::physicalTimeInYears(constants.endPhysicalTime - constants.stopPhysicalTime, constants.cosmology) << " years\n";
            }
            std::vector<hpx::future<void>> tasks;
            for (auto const& locality : localities) {
                tasks.push_back(hpx::async<CosmictigerRunEvolutionAction>(locality, generation));
            }
            cosmictiger::detail::awaitEvolutionTasks(tasks);
            std::string const manifest = constants.checkpointPrefix + ".restart";
            std::string const temporary = generation + ".restart.tmp";
            {
                std::ofstream output;
                output.exceptions(std::ios::failbit | std::ios::badbit);
                output.open(temporary);
                output << "CosmicTigerEvolution 1 " << localities.size() << '\n';
                for (size_t index = 0; index < localities.size(); index++) {
                    output << std::quoted(cosmictiger::detail::evolutionSnapshotFilename(generation, constants.outputSnapshotCount - 1, index)) << '\n';
                }
                output.close();
            }
            hpx::filesystem::rename(temporary, manifest);
            hpx::cout << "Evolution complete. Restart: " << manifest << "\nDiagnostics: " << generation << ".diagnostics.csv\nVisIt series: " << generation
                      << ".visit\n"
                      << hpx::iostreams::flush_type{};
        } else {
            if (constants.generateInitialConditions) {
                std::vector<hpx::future<void>> tasks;
                for (auto const& locality : localities) {
                    tasks.push_back(hpx::async<CosmictigerWriteInitialConditionsAction>(locality));
                }
                cosmictiger::detail::awaitEvolutionTasks(tasks);
                std::vector<std::string> files;
                files.reserve(localities.size());
                for (size_t locality = 0; locality < localities.size(); locality++) {
                    files.push_back(constants.outputPrefix + ".locality" + std::to_string(locality) + ".silo");
                }
                if (localities.size() > 1) {
                    std::string const master = constants.outputPrefix + ".master.silo";
                    cosmictiger::writeParticleSiloMaster(master, files);
                    cosmictiger::detail::publishVisitSeries(constants.outputPrefix + ".visit", {master});
                } else {
                    cosmictiger::detail::publishVisitSeries(constants.outputPrefix + ".visit", files);
                }
                hpx::cout << "Initial conditions written to " << constants.outputPrefix << ".locality*.silo\nVisIt series: " << constants.outputPrefix
                          << ".visit\n"
                          << hpx::iostreams::flush_type{};
            }
        }
        return hpx::finalize();
    } catch (std::exception const& error) {
        hpx::cerr << "CosmicTiger: " << error.what() << '\n' << hpx::iostreams::flush_type{};
        hpx::finalize();
        return 1;
    }
}

int main(int argc, char* argv[]) {
#ifndef PROFILE_DISABLE
    // Register the OS-worker hooks before HPX creates any runtime threads.
    if (!profile::profileStart()) {
        std::cerr << "CosmicTiger: unable to start statistical CPU sampling\n";
        return 1;
    }
#endif
    PROFILE();
    try {
        hpx::program_options::options_description options("CosmicTiger options");
        cosmictiger::addCommandLineOptions(options);
        for (int index = 1; index < argc; index++) {
            if (std::strcmp(argv[index], "--help") == 0) {
                printCommandLineHelp(options);
                return 0;
            }
        }
        hpx::init_params params;
        params.desc_cmdline = options;
        return hpx::init(argc, argv, params);
    } catch (std::exception const& error) {
        std::cerr << "CosmicTiger: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "CosmicTiger: startup failed with an unknown exception\n";
        return 1;
    }
}
