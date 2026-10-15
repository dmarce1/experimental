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
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/profiler.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cosmictiger {
namespace detail {

    std::string preciseDefault(double value) {
        PROFILE();
        std::ostringstream stream;
        stream << std::setprecision(std::numeric_limits<double>::digits10) << value;
        return stream.str();
    }

    template <typename T>
    T optionOrDefault(hpx::program_options::variables_map const& options, char const* name, T defaultValue) {
        PROFILE();
        auto const option = options.find(name);
        if (option == options.end()) {
            return defaultValue;
        }
        return option->second.as<T>();
    }

    bool parseOnOff(std::string const& value, char const* name) {
        PROFILE();
        if (value == "on") {
            return true;
        }
        if (value == "off") {
            return false;
        }
        throw std::invalid_argument(std::string(name) + " must be on or off");
    }

    gravity::SofteningModel parseSofteningModel(std::string const& name) {
        PROFILE();
        if (name == "ferrers") {
            return gravity::SofteningModel::ferrers;
        }
        if (name == "plummer") {
            return gravity::SofteningModel::plummer;
        }
        throw std::invalid_argument("gravity.softening.model must be ferrers or plummer");
    }

    gravity::NearFieldMethod parseNearFieldMethod(std::string const& name) {
        PROFILE();
        if (name == "direct") {
            return gravity::NearFieldMethod::direct;
        }
        if (name == "tree") {
            return gravity::NearFieldMethod::tree;
        }
        throw std::invalid_argument("gravity.nearField must be direct or tree");
    }

    MatterComponent parseMatterComponent(std::string const& name) {
        PROFILE();
        if (name == "totalMatter") {
            return MatterComponent::totalMatter;
        }
        if (name == "cdmAndBaryons") {
            return MatterComponent::cdmAndBaryons;
        }
        if (name == "cdm") {
            return MatterComponent::cdm;
        }
        throw std::invalid_argument("initialConditions.component must be totalMatter, cdmAndBaryons, or cdm");
    }

    std::string matterComponentName(MatterComponent component) {
        PROFILE();
        switch (component) {
        case MatterComponent::totalMatter:
            return "totalMatter";
        case MatterComponent::cdmAndBaryons:
            return "cdmAndBaryons";
        case MatterComponent::cdm:
            return "cdm";
        }
        throw std::invalid_argument("invalid default initialConditions.component");
    }

    size_t sideLength(size_t exponent, char const* optionName) {
        constexpr size_t kMinimumExponent = 2;
        constexpr size_t kMaximumExponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
        PROFILE();
        if (exponent < kMinimumExponent || exponent > kMaximumExponent) {
            throw std::invalid_argument(
                std::string(optionName) + " must be between " + std::to_string(kMinimumExponent) + " and " + std::to_string(kMaximumExponent));
        }
        return size_t(1) << exponent;
    }

    ParticleCount particleCount(size_t meshSideLength) {
        PROFILE();
        ParticleCount count = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            count *= expectRange(size_t(1), meshSideLength, std::numeric_limits<ParticleCount>::max() / count);
        }
        return count;
    }

    hpx::program_options::variables_map configurationOptions(hpx::program_options::variables_map const& commandLine) {
        PROFILE();
        namespace po = hpx::program_options;
        auto const filename = optionOrDefault(commandLine, "configurationFile", std::string{});
        if (filename.empty()) {
            return commandLine;
        }
        std::ifstream input(filename);
        if (!input) {
            throw std::runtime_error("cannot open configurationFile: " + filename);
        }
        po::options_description descriptions;
        addCommandLineOptions(descriptions);
        po::variables_map result;
        po::store(po::parse_config_file(input, descriptions), result);
        for (auto const& option : commandLine) {
            if (!option.second.defaulted() || result.find(option.first) == result.end()) {
                result.erase(option.first);
                result.insert(option);
            }
        }
        po::notify(result);
        return result;
    }

    void readLightconeOptions(hpx::program_options::variables_map const& options, LightconeOptions& result, bool explicitOnly) {
        constexpr std::uint32_t kMaximumNside = std::uint32_t(1) << 29;
        constexpr double kLargestExactImage = 0x1p52;
        PROFILE();
        auto assign = [&options, explicitOnly](char const* name, auto& value) {
            PROFILE();
            auto const option = options.find(name);
            if (option != options.end() && (!explicitOnly || !option->second.defaulted())) {
                using Value = std::decay_t<decltype(value)>;
                value = option->second.as<Value>();
            }
        };
        assign("lightcone.enabled", result.enabled);
        assign("lightcone.nside", result.nside);
        assign("lightcone.maximumRadius", result.maximumRadius);
        assign("lightcone.observerX", result.observer[0]);
        assign("lightcone.observerY", result.observer[1]);
        assign("lightcone.observerZ", result.observer[2]);
        assign("lightcone.outputPrefix", result.outputPrefix);
        assign("lightcone.observationTime", result.observationTimeYears);
        if (result.nside == 0 || result.nside > kMaximumNside || (result.nside & (result.nside - 1)) != 0 ||
            !(result.maximumRadius > 0.0 && result.maximumRadius < kLargestExactImage) || result.outputPrefix.empty() ||
            !(result.observationTimeYears >= -std::numeric_limits<double>::max() && result.observationTimeYears <= std::numeric_limits<double>::max())) {
            throw std::invalid_argument("invalid lightcone resolution, radius, output prefix, or observer epoch");
        }
        for (double coordinate : result.observer) {
            if (!(coordinate >= 0.0 && coordinate < 1.0)) {
                throw std::invalid_argument("lightcone observer coordinates must lie in [0,1)");
            }
        }
    }

}    // namespace detail

void addCommandLineOptions(hpx::program_options::options_description& options) {
    PROFILE();
    namespace po = hpx::program_options;
    CosmologyParameters const cosmology;
    ParticleInitializationOptions const initializationOptions;
    GlobalConstantsData const defaults;
    std::string const fmmOrderDescription = "FMM local-expansion rank, from " + std::to_string(gravity::kMinimumFmmOrder) + " through " +
        std::to_string(gravity::kMaximumFmmOrder) + "; source multipoles use one rank less";
    po::options_description generalGroup("General");
    po::options_description simulationGroup("Simulation");
    po::options_description cosmologyGroup("Cosmology");
    po::options_description initializationGroup("Initial conditions");
    po::options_description timestepGroup("Time stepping");
    po::options_description gravityGroup("Gravity");
    po::options_description executionGroup("Execution");
    po::options_description restartGroup("Restart");
    po::options_description outputGroup("Output");
    po::options_description lightconeGroup("Lightcone");
    lightconeGroup.add_options()("lightcone.enabled", po::value<bool>()->default_value(false)->implicit_value(true),
        "Record past-lightcone particle crossings and sparse NESTED HEALPix maps")("lightcone.nside",
        po::value<std::uint32_t>()->default_value(defaults.lightcone.nside),
        "HEALPix side resolution, a power of two up to 2^29")("lightcone.maximumRadius", po::value<double>()->default_value(defaults.lightcone.maximumRadius),
        "Maximum comoving radius in box lengths")("lightcone.observerX", po::value<double>()->default_value(0.0), "Observer x coordinate in [0,1)")(
        "lightcone.observerY", po::value<double>()->default_value(0.0), "Observer y coordinate in [0,1)")(
        "lightcone.observerZ", po::value<double>()->default_value(0.0), "Observer z coordinate in [0,1)")("lightcone.observationTime",
        po::value<double>()->default_value(defaults.lightcone.observationTimeYears), "Observer epoch in physical years from expanding a=1")(
        "lightcone.outputPrefix", po::value<std::string>()->default_value(defaults.lightcone.outputPrefix), "Per-run, per-locality lightcone CSV path prefix");
    generalGroup.add_options()("help", "Show CosmicTiger options and exit")(
        "configurationFile", po::value<std::string>(), "INI configuration file; explicit command-line options take precedence");
    simulationGroup.add_options()("simulation.initialScaleFactor", po::value<double>()->default_value(defaults.initialScaleFactor),
        "Nominal initial scale factor; generated initial conditions select their own epoch")("simulation.stopTime",
        po::value<std::string>()->default_value("0"),
        "Physical years from the present (a=1), or end for a recollapsing cosmology")("simulation.minimumScaleFactor",
        po::value<double>()->default_value(defaults.minimumScaleFactor), "Positive scale factor at which an end run stops before the Big Crunch singularity")(
        "simulation.evolve", po::value<bool>()->default_value(true)->implicit_value(true), "Advance initialized or restarted particles");
    timestepGroup.add_options()("timestep.eta", po::value<double>()->default_value(defaults.timestepEta), "Acceleration-softening timestep accuracy parameter");
    gravityGroup.add_options()("gravity.fmmOrder", po::value<int>()->default_value(defaults.fmmOrder), fmmOrderDescription.c_str())(
        "gravity.meshExponent", po::value<size_t>(), "Evolution FFT and Cartesian cell side is 2^meshExponent; defaults to initialConditions.meshExponent")(
        "gravity.ghostWidth", po::value<int>()->default_value(defaults.ghostWidth), "Ghost-cell layers and direct-neighbor radius: one or two")(
        "gravity.softening.model", po::value<std::string>()->default_value("ferrers"), "Gravity softening model: ferrers or plummer")("gravity.nearField",
        po::value<std::string>()->default_value("direct"),
        "Near-cell gravity: direct P2P, or tree for a local FMM walk over per-cell kd-trees (Ferrers softening, CPU kick only)")("gravity.openingAngle",
        po::value<double>()->default_value(defaults.openingAngle), "Tree near-field opening angle theta, greater than zero and less than one")(
        "gravity.softening.cellFraction", po::value<double>()->default_value(defaults.softeningCellFraction),
        "Ferrers support radius q or Plummer epsilon as a fraction of the cell width, greater than zero and at most one")("gravity.softening.kernelOrder",
        po::value<int>()->default_value(defaults.softeningKernelOrder), "Ferrers softening kernel order, from zero through three")(
        "gravity.direct.sinkFraction", po::value<double>()->default_value(defaults.directSinkFraction),
        "Fraction of particles sampled for a direct periodic gravity comparison, from zero (disabled) to one")(
        "gravity.direct.seed", po::value<std::uint64_t>()->default_value(defaults.directSeed), "Independent random seed for direct-gravity sink selection");
    executionGroup.add_options()("execution.cuda.drift", po::value<std::string>()->default_value("off"), "Use CUDA for particle drift: on or off")(
        "execution.cuda.kick", po::value<std::string>()->default_value("off"), "Use CUDA for particle gravity kick: on or off")(
        "execution.cuda.multipole", po::value<std::string>()->default_value("off"), "Use CUDA for particle-to-multipole cell construction: on or off")(
        "execution.cuda.direct", po::value<std::string>()->default_value("off"), "Use CUDA for the direct periodic gravity reference solve: on or off");
    restartGroup.add_options()("restart.manifest", po::value<std::string>(), "Continue a completed evolution checkpoint manifest")(
        "restart.rungZeroExtension", po::value<unsigned>()->default_value(0), "Enlarge saved rung-zero interval by 2^n and add n to all particle rungs");
    outputGroup.add_options()("output.checkpointPrefix", po::value<std::string>()->default_value(defaults.checkpointPrefix),
        "Evolution checkpoint and diagnostics path prefix")("output.snapshotCount", po::value<size_t>()->default_value(defaults.outputSnapshotCount),
        "Silo snapshots evenly spaced in conformal time; counts above one include the initial and final states")(
        "output.acceleration", po::value<bool>()->default_value(false)->implicit_value(true), "Retain particle accelerations for Silo output")(
        "output.potential", po::value<bool>()->default_value(false)->implicit_value(true), "Retain particle potentials for Silo output");
    cosmologyGroup.add_options()("cosmology.hubble", po::value<double>()->default_value(cosmology.hubble), "Hubble constant in km/s/Mpc")(
        "cosmology.omegaBaryonH2", po::value<double>()->default_value(cosmology.omegaBaryonH2), "Physical baryon density, Omega_b h^2")(
        "cosmology.omegaCdmH2", po::value<double>()->default_value(cosmology.omegaCdmH2), "Physical cold dark matter density, Omega_c h^2")(
        "cosmology.omegaNeutrinoH2", po::value<double>()->default_value(cosmology.omegaNeutrinoH2), "Physical massive neutrino density, Omega_nu h^2")(
        "cosmology.omegaCurvature", po::value<double>()->default_value(cosmology.omegaCurvature), "Curvature density fraction")("cosmology.massiveNeutrinos",
        po::value<int>()->default_value(cosmology.massiveNeutrinos), "Number of massive neutrino species")("cosmology.masslessNeutrinos",
        po::value<int>()->default_value(cosmology.masslessNeutrinos), "Number of massless neutrino species")("cosmology.effectiveNeutrinos",
        po::value<double>()->default_value(cosmology.effectiveNeutrinos), "Total effective neutrino count")("cosmology.scalarAmplitude",
        po::value<double>()->default_value(cosmology.scalarAmplitude, detail::preciseDefault(cosmology.scalarAmplitude)), "Primordial scalar amplitude")(
        "cosmology.spectralIndex", po::value<double>()->default_value(cosmology.spectralIndex), "Primordial scalar spectral index")("cosmology.spectralRunning",
        po::value<double>()->default_value(cosmology.spectralRunning),
        "Primordial scalar spectral running")("cosmology.pivotMpcInverse", po::value<double>()->default_value(cosmology.pivotMpcInverse),
        "Scalar pivot in inverse Mpc")("cosmology.darkEnergyW", po::value<double>()->default_value(cosmology.darkEnergyW), "Dark energy equation-of-state w")(
        "cosmology.darkEnergyWa", po::value<double>()->default_value(cosmology.darkEnergyWa), "Dark energy equation-of-state wa")("cosmology.temperatureCmb",
        po::value<double>()->default_value(cosmology.temperatureCmb),
        "CMB temperature in kelvin")("cosmology.heliumFraction", po::value<double>()->default_value(cosmology.heliumFraction), "Primordial helium fraction")(
        "cosmology.opticalDepth", po::value<double>()->default_value(cosmology.opticalDepth), "Reionization optical depth");
    initializationGroup.add_options()("initialConditions.generate", po::bool_switch(), "Generate initial particle conditions and write Silo output")(
        "initialConditions.outputPrefix", po::value<std::string>(), "Silo output path prefix, required when generation is enabled")(
        "initialConditions.meshExponent", po::value<size_t>()->default_value(initializationOptions.meshExponent),
        "Initial-condition FFT and particle lattice side is 2^meshExponent; particle count is 2^(NDIM*meshExponent)")("initialConditions.boxLength",
        po::value<double>()->default_value(initializationOptions.boxLength), "Comoving box length in Mpc/h")("initialConditions.seed",
        po::value<std::uint64_t>()->default_value(initializationOptions.seed), "Independent Gaussian realization seed for Zel'dovich initial conditions")(
        "initialConditions.maximumDensityDeviation", po::value<double>()->default_value(initializationOptions.maximumDensityDeviation),
        "Linear density threshold for initial epoch")("initialConditions.earliestScaleFactor",
        po::value<double>()->default_value(initializationOptions.earliestScaleFactor), "Earliest scale factor to search")(
        "initialConditions.latestScaleFactor", po::value<double>()->default_value(initializationOptions.latestScaleFactor), "Latest scale factor to search")(
        "initialConditions.maximumLogScaleFactorStep", po::value<double>()->default_value(initializationOptions.maximumLogScaleFactorStep),
        "Maximum search step in ln(a)")("initialConditions.logScaleFactorTolerance",
        po::value<double>()->default_value(initializationOptions.logScaleFactorTolerance),
        "Bracket tolerance in ln(a)")("initialConditions.growthDerivativeStep", po::value<double>()->default_value(initializationOptions.growthDerivativeStep),
        "Growth derivative step in ln(a)")("initialConditions.maximumSpectrumEvaluations",
        po::value<size_t>()->default_value(initializationOptions.maximumSpectrumEvaluations),
        "Maximum spectrum evaluations")("initialConditions.samplesPerLogInterval", po::value<int>()->default_value(initializationOptions.samplesPerLogInterval),
        "CAMB samples per natural log wave-number interval")("initialConditions.accuracyBoost",
        po::value<double>()->default_value(initializationOptions.accuracyBoost), "CAMB accuracy boost")("initialConditions.component",
        po::value<std::string>()->default_value(detail::matterComponentName(initializationOptions.component)),
        "Matter component: totalMatter, cdmAndBaryons, or cdm");
    options.add(generalGroup)
        .add(simulationGroup)
        .add(cosmologyGroup)
        .add(initializationGroup)
        .add(timestepGroup)
        .add(gravityGroup)
        .add(executionGroup)
        .add(restartGroup)
        .add(outputGroup)
        .add(lightconeGroup);
}

GlobalConstantsData readCommandLineOptions(hpx::program_options::variables_map const& commandLine, size_t localityCount) {
    PROFILE();
    auto const options = detail::configurationOptions(commandLine);
    FpeGuard const fpeGuard;
    GlobalConstantsData result;
    detail::readLightconeOptions(options, result.lightcone, false);
    result.configurationFile = detail::optionOrDefault(options, "configurationFile", std::string{});
    result.fmmOrder = detail::optionOrDefault(options, "gravity.fmmOrder", result.fmmOrder);
    if (result.fmmOrder < gravity::kMinimumFmmOrder || result.fmmOrder > gravity::kMaximumFmmOrder) {
        throw std::invalid_argument(
            "gravity.fmmOrder must be between " + std::to_string(gravity::kMinimumFmmOrder) + " and " + std::to_string(gravity::kMaximumFmmOrder));
    }
    result.ghostWidth = detail::optionOrDefault(options, "gravity.ghostWidth", result.ghostWidth);
    if (result.ghostWidth != 1 && result.ghostWidth != 2) {
        throw std::invalid_argument("gravity.ghostWidth must be one or two");
    }
    result.timestepEta = expectPositive(detail::optionOrDefault(options, "timestep.eta", result.timestepEta));
    result.softeningModel = detail::parseSofteningModel(detail::optionOrDefault(options, "gravity.softening.model", std::string("ferrers")));
    result.nearFieldMethod = detail::parseNearFieldMethod(detail::optionOrDefault(options, "gravity.nearField", std::string("direct")));
    result.openingAngle = detail::optionOrDefault(options, "gravity.openingAngle", result.openingAngle);
    result.softeningCellFraction =
        expectRange(0.0, expectPositive(detail::optionOrDefault(options, "gravity.softening.cellFraction", result.softeningCellFraction)), 1.0);
    result.softeningKernelOrder = expectRange(0, detail::optionOrDefault(options, "gravity.softening.kernelOrder", result.softeningKernelOrder), 3);
    result.directSinkFraction = detail::optionOrDefault(options, "gravity.direct.sinkFraction", result.directSinkFraction);
    result.directSeed = detail::optionOrDefault(options, "gravity.direct.seed", result.directSeed);
    if (!(result.directSinkFraction >= 0.0 && result.directSinkFraction <= 1.0)) {
        throw std::invalid_argument("gravity.direct.sinkFraction must lie in [0,1]");
    }
    result.initialScaleFactor = expectPositive(detail::optionOrDefault(options, "simulation.initialScaleFactor", result.initialScaleFactor));
    auto const stopTime = detail::optionOrDefault(options, "simulation.stopTime", std::string("0"));
    result.stopAtEnd = stopTime == "end";
    if (!result.stopAtEnd) {
        size_t consumed = 0;
        result.stopTimeYears = std::stod(stopTime, &consumed);
        if (consumed != stopTime.size() || !(result.stopTimeYears >= -std::numeric_limits<double>::max()) ||
            !(result.stopTimeYears <= std::numeric_limits<double>::max())) {
            throw std::invalid_argument("simulation.stopTime must be a number of years or end");
        }
    }
    result.minimumScaleFactor = detail::optionOrDefault(options, "simulation.minimumScaleFactor", result.minimumScaleFactor);
    result.evolve = detail::optionOrDefault(options, "simulation.evolve", result.evolve);
    result.cudaDrift = detail::parseOnOff(detail::optionOrDefault(options, "execution.cuda.drift", std::string("off")), "execution.cuda.drift");
    result.cudaKick = detail::parseOnOff(detail::optionOrDefault(options, "execution.cuda.kick", std::string("off")), "execution.cuda.kick");
    result.cudaMultipole = detail::parseOnOff(detail::optionOrDefault(options, "execution.cuda.multipole", std::string("off")), "execution.cuda.multipole");
    result.cudaDirect = detail::parseOnOff(detail::optionOrDefault(options, "execution.cuda.direct", std::string("off")), "execution.cuda.direct");
#ifndef COSMICTIGER_ENABLE_CUDA
    if (result.cudaDrift) {
        throw std::invalid_argument("execution.cuda.drift=on requires a CUDA-enabled build");
    }
    if (result.cudaKick) {
        throw std::invalid_argument("execution.cuda.kick=on requires a CUDA-enabled build");
    }
    if (result.cudaMultipole) {
        throw std::invalid_argument("execution.cuda.multipole=on requires a CUDA-enabled build");
    }
    if (result.cudaDirect) {
        throw std::invalid_argument("execution.cuda.direct=on requires a CUDA-enabled build");
    }
#endif
    validateNearFieldOptions(result);
    result.restartManifest = detail::optionOrDefault(options, "restart.manifest", std::string{});
    result.checkpointPrefix = detail::optionOrDefault(options, "output.checkpointPrefix", result.checkpointPrefix);
    result.outputSnapshotCount = detail::optionOrDefault(options, "output.snapshotCount", result.outputSnapshotCount);
    result.restartRungZeroExtension = detail::optionOrDefault(options, "restart.rungZeroExtension", 0U);
    if (!(result.minimumScaleFactor > 0.0 && result.minimumScaleFactor < 1.0) || result.restartRungZeroExtension > Rung::maximumRung() ||
        result.checkpointPrefix.empty() || result.outputSnapshotCount == 0) {
        throw std::invalid_argument("invalid evolution endpoint, checkpoint prefix, or restart rung-zero extension");
    }
    if (result.restartManifest.empty() && result.restartRungZeroExtension != 0) {
        throw std::invalid_argument("restart.rungZeroExtension requires restart.manifest");
    }
    result.outputAcceleration = detail::optionOrDefault(options, "output.acceleration", result.outputAcceleration);
    result.outputPotential = detail::optionOrDefault(options, "output.potential", result.outputPotential);
    auto& cosmology = result.cosmology;
    cosmology.hubble = expectPositive(detail::optionOrDefault(options, "cosmology.hubble", cosmology.hubble));
    cosmology.omegaBaryonH2 = expectPositive(detail::optionOrDefault(options, "cosmology.omegaBaryonH2", cosmology.omegaBaryonH2));
    cosmology.omegaCdmH2 = expectNonNegative(detail::optionOrDefault(options, "cosmology.omegaCdmH2", cosmology.omegaCdmH2));
    cosmology.omegaNeutrinoH2 = expectNonNegative(detail::optionOrDefault(options, "cosmology.omegaNeutrinoH2", cosmology.omegaNeutrinoH2));
    cosmology.omegaCurvature = detail::optionOrDefault(options, "cosmology.omegaCurvature", cosmology.omegaCurvature);
    cosmology.massiveNeutrinos = expectNonNegative(detail::optionOrDefault(options, "cosmology.massiveNeutrinos", cosmology.massiveNeutrinos));
    expectEqual(cosmology.omegaNeutrinoH2 > 0.0, cosmology.massiveNeutrinos > 0);
    cosmology.masslessNeutrinos = expectNonNegative(detail::optionOrDefault(options, "cosmology.masslessNeutrinos", cosmology.masslessNeutrinos));
    cosmology.effectiveNeutrinos = expectNonNegative(detail::optionOrDefault(options, "cosmology.effectiveNeutrinos", cosmology.effectiveNeutrinos));
    std::int64_t const neutrinoCount = static_cast<std::int64_t>(cosmology.massiveNeutrinos) + cosmology.masslessNeutrinos;
    expectEqual(neutrinoCount == 0, cosmology.effectiveNeutrinos == 0.0);
    cosmology.scalarAmplitude = expectPositive(detail::optionOrDefault(options, "cosmology.scalarAmplitude", cosmology.scalarAmplitude));
    cosmology.spectralIndex = detail::optionOrDefault(options, "cosmology.spectralIndex", cosmology.spectralIndex);
    cosmology.spectralRunning = detail::optionOrDefault(options, "cosmology.spectralRunning", cosmology.spectralRunning);
    cosmology.pivotMpcInverse = expectPositive(detail::optionOrDefault(options, "cosmology.pivotMpcInverse", cosmology.pivotMpcInverse));
    cosmology.darkEnergyW = detail::optionOrDefault(options, "cosmology.darkEnergyW", cosmology.darkEnergyW);
    cosmology.darkEnergyWa = detail::optionOrDefault(options, "cosmology.darkEnergyWa", cosmology.darkEnergyWa);
    expectNonPositive(cosmology.darkEnergyW + cosmology.darkEnergyWa);
    cosmology.temperatureCmb = expectPositive(detail::optionOrDefault(options, "cosmology.temperatureCmb", cosmology.temperatureCmb));
    cosmology.heliumFraction = expectNonNegative(detail::optionOrDefault(options, "cosmology.heliumFraction", cosmology.heliumFraction));
    expect(cosmology.heliumFraction < 1.0);
    cosmology.opticalDepth = expectNonNegative(detail::optionOrDefault(options, "cosmology.opticalDepth", cosmology.opticalDepth));

    auto& initializationOptions = result.initializationOptions;
    initializationOptions.meshExponent = detail::optionOrDefault(options, "initialConditions.meshExponent", initializationOptions.meshExponent);
    initializationOptions.boxLength = expectPositive(detail::optionOrDefault(options, "initialConditions.boxLength", initializationOptions.boxLength));
    initializationOptions.seed = detail::optionOrDefault(options, "initialConditions.seed", initializationOptions.seed);
    initializationOptions.maximumDensityDeviation =
        expectPositive(detail::optionOrDefault(options, "initialConditions.maximumDensityDeviation", initializationOptions.maximumDensityDeviation));
    initializationOptions.earliestScaleFactor =
        expectPositive(detail::optionOrDefault(options, "initialConditions.earliestScaleFactor", initializationOptions.earliestScaleFactor));
    initializationOptions.latestScaleFactor =
        expectPositive(detail::optionOrDefault(options, "initialConditions.latestScaleFactor", initializationOptions.latestScaleFactor));
    expect(initializationOptions.earliestScaleFactor < initializationOptions.latestScaleFactor);
    expect(initializationOptions.latestScaleFactor <= 1.0);
    initializationOptions.maximumLogScaleFactorStep =
        expectPositive(detail::optionOrDefault(options, "initialConditions.maximumLogScaleFactorStep", initializationOptions.maximumLogScaleFactorStep));
    initializationOptions.logScaleFactorTolerance =
        expectPositive(detail::optionOrDefault(options, "initialConditions.logScaleFactorTolerance", initializationOptions.logScaleFactorTolerance));
    expect(initializationOptions.logScaleFactorTolerance < initializationOptions.maximumLogScaleFactorStep);
    initializationOptions.growthDerivativeStep =
        expectPositive(detail::optionOrDefault(options, "initialConditions.growthDerivativeStep", initializationOptions.growthDerivativeStep));
    expectRange(1.0e-6, initializationOptions.growthDerivativeStep, 0.1);
    initializationOptions.maximumSpectrumEvaluations =
        detail::optionOrDefault(options, "initialConditions.maximumSpectrumEvaluations", initializationOptions.maximumSpectrumEvaluations);
    expect(initializationOptions.maximumSpectrumEvaluations >= 5);
    initializationOptions.samplesPerLogInterval =
        detail::optionOrDefault(options, "initialConditions.samplesPerLogInterval", initializationOptions.samplesPerLogInterval);
    initializationOptions.accuracyBoost =
        expectPositive(detail::optionOrDefault(options, "initialConditions.accuracyBoost", initializationOptions.accuracyBoost));
    initializationOptions.component = detail::parseMatterComponent(
        detail::optionOrDefault(options, "initialConditions.component", detail::matterComponentName(initializationOptions.component)));

    result.generateInitialConditions = detail::optionOrDefault(options, "initialConditions.generate", false);
    if (result.directSinkFraction > 0.0 && !result.evolve && !result.generateInitialConditions) {
        throw std::invalid_argument("gravity.direct.sinkFraction requires simulation.evolve or initialConditions.generate");
    }
    result.outputPrefix = detail::optionOrDefault(options, "initialConditions.outputPrefix", std::string{});
    result.localityCount = expectPositive(localityCount);

    result.gravityMeshExponent = detail::optionOrDefault(options, "gravity.meshExponent", initializationOptions.meshExponent);
    result.meshSideLength = detail::sideLength(initializationOptions.meshExponent, "initialConditions.meshExponent");
    result.gridSideLength = detail::sideLength(result.gravityMeshExponent, "gravity.meshExponent");
    result.globalParticleCount = detail::particleCount(result.meshSideLength);
    if (result.generateInitialConditions || (result.evolve && result.restartManifest.empty())) {
        double const selectedDensity =
            initializationOptions.component == MatterComponent::cdm ? cosmology.omegaCdmH2 : cosmology.omegaCdmH2 + cosmology.omegaBaryonH2;
        if (selectedDensity <= 0.0) {
            throw std::invalid_argument("initial particle population must have positive physical matter density");
        }
        if (localityCount == 0 || localityCount > result.meshSideLength) {
            throw std::invalid_argument("initialConditions.meshExponent must provide an FFT side at least as large as the locality count");
        }
        if (localityCount > result.gridSideLength) {
            throw std::invalid_argument("gravity.meshExponent must provide an FFT side at least as large as the locality count");
        }
        size_t localityExponent = 0;
        size_t remainingLocalities = localityCount;
        size_t const localityFactor = size_t(1) << NDIM;
        while (remainingLocalities > 1 && remainingLocalities % localityFactor == 0) {
            remainingLocalities /= localityFactor;
            localityExponent++;
        }
        expectEqual(remainingLocalities, size_t(1));
        expect(initializationOptions.meshExponent >= localityExponent);
        expect(result.gravityMeshExponent >= localityExponent);
        expect(initializationOptions.component != MatterComponent::totalMatter || cosmology.omegaNeutrinoH2 == 0.0);
    }
    if (result.generateInitialConditions && result.outputPrefix.empty()) {
        throw std::invalid_argument("initialConditions.outputPrefix is required when initialConditions.generate is set");
    }
    return result;
}

void validateNearFieldOptions(GlobalConstantsData const& data) {
    PROFILE();
    if (!(data.openingAngle > 0.0 && data.openingAngle < 1.0)) {
        throw std::invalid_argument("gravity.openingAngle must lie in (0,1)");
    }
    if (data.nearFieldMethod == gravity::NearFieldMethod::tree) {
        // The tree accepts multipole interactions only beyond the softening
        // support, which a Plummer kernel never reaches.
        if (data.softeningModel != gravity::SofteningModel::ferrers) {
            throw std::invalid_argument("gravity.nearField=tree requires gravity.softening.model=ferrers");
        }
        if (data.cudaKick) {
            throw std::invalid_argument("gravity.nearField=tree is not yet supported with execution.cuda.kick=on");
        }
    }
}

void validateGravityRestartOptions(hpx::program_options::variables_map const& commandLine, GlobalConstantsData const& data) {
    PROFILE();
    auto const options = detail::configurationOptions(commandLine);
    auto const order = options.find("gravity.fmmOrder");
    if (order != options.end() && !order->second.defaulted() && order->second.as<int>() != data.fmmOrder) {
        throw std::invalid_argument("gravity.fmmOrder cannot change the saved FMM order on restart");
    }
    auto const model = options.find("gravity.softening.model");
    if (model != options.end() && !model->second.defaulted() && detail::parseSofteningModel(model->second.as<std::string>()) != data.softeningModel) {
        throw std::invalid_argument("gravity.softening.model cannot change the saved softening model on restart");
    }
    auto const mesh = options.find("gravity.meshExponent");
    if (mesh != options.end() && !mesh->second.defaulted() && mesh->second.as<size_t>() != data.gravityMeshExponent) {
        throw std::invalid_argument("gravity.meshExponent cannot change the saved evolution mesh on restart");
    }
    auto const width = options.find("gravity.ghostWidth");
    if (width != options.end() && !width->second.defaulted() && width->second.as<int>() != data.ghostWidth) {
        throw std::invalid_argument("gravity.ghostWidth cannot change the saved ghost width on restart");
    }
}

void applyLightconeRestartOptions(hpx::program_options::variables_map const& commandLine, GlobalConstantsData& data) {
    PROFILE();
    auto const options = detail::configurationOptions(commandLine);
    bool const previouslyEnabled = data.lightcone.enabled;
    double const previousObservationTime = data.lightcone.observationTimeYears;
    detail::readLightconeOptions(options, data.lightcone, true);
    if (!data.lightcone.enabled || !previouslyEnabled || data.lightcone.observationTimeYears != previousObservationTime) {
        initializeLightconeConstants(data);
    }
}

}    // namespace cosmictiger
