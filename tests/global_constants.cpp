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

#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <numbers>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <unistd.h>

#include <cosmictiger/command_line_options.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::detail {

#ifdef COSMICTIGER_ENABLE_CUDA
constexpr bool kExpectedCudaKick = true;
constexpr bool kExpectedCudaMultipole = true;
constexpr bool kExpectedCudaDirect = true;
#else
constexpr bool kExpectedCudaKick = false;
constexpr bool kExpectedCudaMultipole = false;
constexpr bool kExpectedCudaDirect = false;
#endif

GlobalConstantsData parseOptions(hpx::program_options::options_description const& options, std::vector<std::string> const& arguments, size_t localityCount) {
    PROFILE();
    std::vector<char const*> argumentPointers;
    argumentPointers.reserve(arguments.size());
    for (auto const& argument : arguments) {
        argumentPointers.push_back(argument.c_str());
    }
    hpx::program_options::variables_map variables;
    hpx::program_options::store(
        hpx::program_options::parse_command_line(static_cast<int>(argumentPointers.size()), argumentPointers.data(), options), variables);
    hpx::program_options::notify(variables);
    return readCommandLineOptions(variables, localityCount);
}

void verifyGlobalConstants(size_t expectedLocalityCount) {
    constexpr double kReferenceHubble = 100.0;
    constexpr double kScaleTolerance = 1.0e-14;
    PROFILE();
    GlobalConstantsData const& first = globalConstants().data();
    GlobalConstantsData const& second = globalConstants().data();
    if (&first != &second) {
        throw std::runtime_error("GlobalConstants did not retain a stable local data reference");
    }
    auto const& scales = globalConstants().unitScales();
    double const expectedLengthMpc = first.initializationOptions.boxLength / (first.cosmology.hubble / kReferenceHubble);
    double const expectedTimeSeconds = physicalTimeInYears(1.0, first.cosmology) * units::kYear.value();
    if (&scales != &globalConstants().unitScales() || std::abs(scales.lengthMpc() / expectedLengthMpc - 1.0) > kScaleTolerance ||
        std::abs(scales.timeSeconds() / expectedTimeSeconds - 1.0) > kScaleTolerance) {
        throw std::runtime_error("GlobalConstants did not retain the root's immutable physical unit scales");
    }
    if (first.cosmology.scalarAmplitude != 3.25e-9 || first.cosmology.hubble != 67.4 || first.initializationOptions.meshExponent != 4 ||
        first.initializationOptions.seed != std::uint64_t(987654321) || first.initializationOptions.component != MatterComponent::cdm ||
        first.initializationOptions.boxLength != 100.0 || first.generateInitialConditions || !first.outputPrefix.empty() ||
        first.localityCount != expectedLocalityCount || first.meshSideLength != (size_t(1) << 4) || first.gridSideLength != (size_t(1) << 3) ||
        first.gravityMeshExponent != 3 || first.ghostWidth != 2 || first.fmmOrder != gravity::kMinimumFmmOrder ||
        first.globalParticleCount != (size_t(1) << (NDIM * 4)) || first.timestepEta != 0.125 || !first.outputAcceleration || first.outputPotential ||
        first.initialScaleFactor != 0.03 || first.stopTimeYears != 1000000 || !(first.rungZeroTimestep > 0.0) || !(first.initialConformalDerivative > 0.0) ||
        !(first.gravityParticleMass > 0.0) || first.softeningModel != gravity::SofteningModel::plummer || first.softeningCellFraction != 0.125 ||
        first.softeningKernelOrder != 3 || first.softeningLength != 0.125 / 8.0 || first.cudaDrift || first.cudaKick != kExpectedCudaKick ||
        first.cudaMultipole != kExpectedCudaMultipole || first.cudaDirect != kExpectedCudaDirect || first.directSinkFraction != 0.03125 ||
        first.directSeed != std::uint64_t(123456789) || !first.lightcone.enabled || first.lightcone.nside != 128 || first.lightcone.observer[1] != 0.5 ||
        !(first.lightconeSpeed > 0.0)) {
        throw std::runtime_error("GlobalConstants local copy differs from parsed and derived values");
    }
    if (hpx::get_locality_id() != 0) {
        GlobalConstantsData attempt = first;
        attempt.fmmOrder = gravity::kMinimumFmmOrder - 1;
        bool rejected = false;
        try {
            publishGlobalConstants(std::move(attempt));
        } catch (std::exception const&) {
            rejected = true;
        }
        if (!rejected || &globalConstants().data() != &first || globalConstants().data().cosmology.scalarAmplitude != 3.25e-9 ||
            globalConstants().data().fmmOrder != gravity::kMinimumFmmOrder) {
            throw std::runtime_error("a remote locality republished GlobalConstants");
        }
    }
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::verifyGlobalConstants, CosmictigerVerifyGlobalConstantsAction)

int hpx_main() {
    PROFILE();
    try {
        namespace po = hpx::program_options;
        auto const localities = hpx::find_all_localities();
        po::options_description options("GlobalConstants test options");
        cosmictiger::addCommandLineOptions(options);
        auto const configurationPath = hpx::filesystem::temp_directory_path() / ("cosmictiger-global-constants-" + std::to_string(getpid()) + ".ini");
        {
            std::ofstream configuration(configurationPath);
            configuration.exceptions(std::ios::failbit | std::ios::badbit);
            configuration << "[timestep]\neta=0.05\n[simulation]\ninitialScaleFactor=0.03\nstopTime=1000000\n[output]\nacceleration=true\npotential=false\n";
            configuration << "[gravity.softening]\nmodel=plummer\ncellFraction=0.125\nkernelOrder=3\n";
            configuration << "[gravity]\nmeshExponent=4\nghostWidth=1\nfmmOrder=" << cosmictiger::gravity::kMinimumFmmOrder << '\n';
            configuration << "[gravity.direct]\nsinkFraction=0.0625\nseed=17\n";
            configuration << "[execution.cuda]\ndrift=on\nkick=off\nmultipole=on\n";
            configuration << "[lightcone]\nenabled=true\nnside=32\nobserverY=0.5\n";
        }
        std::string const configurationArgument = "--configurationFile=" + configurationPath.string();
        auto const defaults = cosmictiger::detail::parseOptions(options, {"global_constants_test"}, localities.size());
        if (defaults.cudaDrift || defaults.cudaKick || defaults.cudaMultipole || defaults.cudaDirect || defaults.lightcone.enabled) {
            throw std::runtime_error("CUDA execution options must default to off independently");
        }
        if (defaults.directSinkFraction != 0.0 || defaults.directSeed != 42 || defaults.initializationOptions.seed != 42) {
            throw std::runtime_error("direct sampling must default to disabled and independent random seeds must default to 42");
        }
        if (defaults.ghostWidth != 1 || defaults.gravityMeshExponent != defaults.initializationOptions.meshExponent ||
            defaults.gridSideLength != defaults.meshSideLength) {
            throw std::runtime_error("default gravity mesh must follow the particle lattice with one ghost layer");
        }
        if (defaults.fmmOrder != cosmictiger::gravity::kDefaultFmmOrder) {
            throw std::runtime_error("the default FMM order must preserve the configured default rank independently of compiled capacities");
        }
        if (defaults.softeningModel != cosmictiger::gravity::SofteningModel::ferrers) {
            throw std::runtime_error("the default softening model must preserve Ferrers");
        }
        if (defaults.nearFieldMethod != cosmictiger::gravity::NearFieldMethod::direct || defaults.openingAngle != cosmictiger::gravity::kDefaultOpeningAngle) {
            throw std::runtime_error("the near field must default to direct P2P with the default opening angle");
        }
        auto treeNear =
            cosmictiger::detail::parseOptions(options, {"global_constants_test", "--gravity.nearField=tree", "--gravity.openingAngle=0.7"}, localities.size());
        if (treeNear.nearFieldMethod != cosmictiger::gravity::NearFieldMethod::tree || treeNear.openingAngle != 0.7) {
            throw std::runtime_error("command-line tree near field and opening angle were not retained");
        }
        {
            std::vector<char> bytes;
            {
                hpx::serialization::output_archive archive(bytes);
                archive << treeNear;
            }
            cosmictiger::GlobalConstantsData restored;
            hpx::serialization::input_archive archive(bytes, bytes.size());
            archive >> restored;
            if (restored.nearFieldMethod != treeNear.nearFieldMethod || restored.openingAngle != treeNear.openingAngle) {
                throw std::runtime_error("serialization did not retain the near-field method and opening angle");
            }
        }
        for (std::vector<std::string> const arguments :
            {std::vector<std::string>{"--gravity.nearField=fmm"}, std::vector<std::string>{"--gravity.openingAngle=0"},
                std::vector<std::string>{"--gravity.openingAngle=1"}, std::vector<std::string>{"--gravity.openingAngle=-0.5"},
                std::vector<std::string>{"--gravity.nearField=tree", "--gravity.softening.model=plummer"}}) {
            std::vector<std::string> commandLine{"global_constants_test"};
            commandLine.insert(commandLine.end(), arguments.begin(), arguments.end());
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, commandLine, localities.size());
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("invalid near-field option was accepted: " + arguments.back());
            }
        }
        auto plummer = cosmictiger::detail::parseOptions(options,
            {"global_constants_test", "--gravity.softening.model=plummer", "--gravity.meshExponent=3", "--gravity.softening.cellFraction=0.125"},
            localities.size());
        cosmictiger::initializeSimulationConstants(plummer, plummer.initialScaleFactor);
        if (plummer.softeningModel != cosmictiger::gravity::SofteningModel::plummer || plummer.softeningLength != 0.125 / 8.0) {
            throw std::runtime_error("Plummer epsilon must be the configured fraction of the evolution cell width");
        }
        std::vector<char> plummerBytes;
        {
            hpx::serialization::output_archive archive(plummerBytes);
            archive << plummer;
        }
        cosmictiger::GlobalConstantsData restoredPlummer;
        {
            hpx::serialization::input_archive archive(plummerBytes, plummerBytes.size());
            archive >> restoredPlummer;
        }
        if (restoredPlummer.softeningModel != plummer.softeningModel || restoredPlummer.softeningLength != plummer.softeningLength) {
            throw std::runtime_error("serialization did not retain the Plummer softening model and epsilon");
        }
        auto coarse = cosmictiger::detail::parseOptions(options,
            {"global_constants_test", "--initialConditions.meshExponent=4", "--gravity.meshExponent=3", "--gravity.ghostWidth=1",
                "--gravity.fmmOrder=" + std::to_string(cosmictiger::gravity::kMinimumFmmOrder)},
            localities.size());
        auto fine = cosmictiger::detail::parseOptions(
            options, {"global_constants_test", "--initialConditions.meshExponent=4", "--gravity.meshExponent=5"}, localities.size());
        auto const matched = cosmictiger::detail::parseOptions(options, {"global_constants_test", "--initialConditions.meshExponent=4"}, localities.size());
        cosmictiger::initializeSimulationConstants(coarse, coarse.initialScaleFactor);
        cosmictiger::initializeSimulationConstants(fine, fine.initialScaleFactor);
        if (coarse.meshSideLength != 16 || fine.meshSideLength != 16 || coarse.globalParticleCount != (size_t(1) << (NDIM * 4)) ||
            fine.globalParticleCount != coarse.globalParticleCount || coarse.gridSideLength != 8 || fine.gridSideLength != 32 ||
            coarse.gravityParticleMass != fine.gravityParticleMass || coarse.softeningLength != coarse.softeningCellFraction / 8.0 ||
            fine.softeningLength != fine.softeningCellFraction / 32.0 || coarse.ghostWidth != 1 || fine.ghostWidth != 1 || matched.gravityMeshExponent != 4 ||
            matched.gridSideLength != matched.meshSideLength) {
            throw std::runtime_error("gravity mesh changed the particle lattice or mass, or softening did not follow the evolution cells");
        }
        for (std::string const argument :
            {"--gravity.ghostWidth=0", "--gravity.ghostWidth=3", "--gravity.ghostWidth=-1", "--gravity.meshExponent=1", "--gravity.meshExponent=64",
                "--initialConditions.meshExponent=1", "--initialConditions.meshExponent=64", "--gravity.softening.model=unknown"}) {
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", argument}, localities.size());
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("invalid mesh or ghost width was accepted: " + argument);
            }
        }
        for (int order = cosmictiger::gravity::kMinimumFmmOrder; order <= cosmictiger::gravity::kMaximumFmmOrder; order++) {
            auto const selected =
                cosmictiger::detail::parseOptions(options, {"global_constants_test", "--gravity.fmmOrder=" + std::to_string(order)}, localities.size());
            if (selected.fmmOrder != order) {
                throw std::runtime_error("command-line FMM order was not retained: " + std::to_string(order));
            }
        }
        for (int const order : {0, 1, -1, cosmictiger::gravity::kMaximumFmmOrder + 1}) {
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--gravity.fmmOrder=" + std::to_string(order)}, localities.size());
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("invalid FMM order was accepted: " + std::to_string(order));
            }
        }
        for (auto const& arguments :
            {std::vector<std::string>{"global_constants_test", "--initialConditions.meshExponent=3", "--gravity.meshExponent=2", "--gravity.ghostWidth=1"},
                std::vector<std::string>{"global_constants_test", "--initialConditions.meshExponent=2", "--gravity.meshExponent=3"}}) {
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, arguments, size_t(1) << NDIM);
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("an FFT side smaller than the locality count was accepted");
            }
        }
        for (auto const& arguments : {std::vector<std::string>{}, std::vector<std::string>{"--gravity.meshExponent=3", "--gravity.ghostWidth=1"},
                 std::vector<std::string>{"--gravity.fmmOrder=" + std::to_string(coarse.fmmOrder)},
                 std::vector<std::string>{configurationArgument, "--gravity.meshExponent=3", "--gravity.ghostWidth=1", "--gravity.softening.model=ferrers"}}) {
            po::variables_map restartGeometry;
            po::store(po::command_line_parser(arguments).options(options).run(), restartGeometry);
            po::notify(restartGeometry);
            cosmictiger::validateGravityRestartOptions(restartGeometry, coarse);
        }
        for (auto const& arguments : {std::vector<std::string>{"--gravity.meshExponent=4"}, std::vector<std::string>{"--gravity.ghostWidth=2"},
                 std::vector<std::string>{"--gravity.softening.model=plummer"},
                 std::vector<std::string>{"--gravity.fmmOrder=" + std::to_string(coarse.fmmOrder + 1)}, std::vector<std::string>{configurationArgument}}) {
            po::variables_map restartGeometry;
            po::store(po::command_line_parser(arguments).options(options).run(), restartGeometry);
            po::notify(restartGeometry);
            bool rejected = false;
            try {
                cosmictiger::validateGravityRestartOptions(restartGeometry, coarse);
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("restart accepted an explicit change to saved gravity settings");
            }
        }
        auto const configuredComparison = cosmictiger::detail::parseOptions(
            options, {"global_constants_test", configurationArgument, "--execution.cuda.drift=off", "--execution.cuda.multipole=off"}, localities.size());
        if (configuredComparison.directSinkFraction != 0.0625 || configuredComparison.directSeed != 17 ||
            configuredComparison.initializationOptions.seed != 42 || configuredComparison.gravityMeshExponent != 4 ||
            configuredComparison.gridSideLength != 16 || configuredComparison.meshSideLength != defaults.meshSideLength ||
            configuredComparison.ghostWidth != 1 || configuredComparison.fmmOrder != cosmictiger::gravity::kMinimumFmmOrder) {
            throw std::runtime_error("direct-gravity INI settings were not parsed independently of initial-condition defaults");
        }
        auto const overriddenOrder = cosmictiger::detail::parseOptions(options,
            {"global_constants_test", configurationArgument, "--execution.cuda.drift=off", "--execution.cuda.multipole=off",
                "--gravity.fmmOrder=" + std::to_string(cosmictiger::gravity::kMaximumFmmOrder)},
            localities.size());
        if (overriddenOrder.fmmOrder != cosmictiger::gravity::kMaximumFmmOrder) {
            throw std::runtime_error("command-line FMM order did not override the INI order");
        }
        auto const changedInitialSeed = cosmictiger::detail::parseOptions(options, {"global_constants_test", "--initialConditions.seed=73"}, localities.size());
        auto const changedDirectSeed = cosmictiger::detail::parseOptions(
            options, {"global_constants_test", "--gravity.direct.seed=91", "--gravity.direct.sinkFraction=1"}, localities.size());
        if (changedInitialSeed.initializationOptions.seed != 73 || changedInitialSeed.directSeed != 42 || changedDirectSeed.initializationOptions.seed != 42 ||
            changedDirectSeed.directSeed != 91 || changedDirectSeed.directSinkFraction != 1.0) {
            throw std::runtime_error("initial-condition and direct-gravity random seeds are not independent");
        }
        for (std::string const argument : {"--gravity.direct.sinkFraction=-0.01", "--gravity.direct.sinkFraction=1.01", "--execution.cuda.direct=true"}) {
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", argument}, localities.size());
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("invalid direct-gravity option was accepted: " + argument);
            }
        }
        bool rejectedUnusedComparison = false;
        try {
            (void) cosmictiger::detail::parseOptions(
                options, {"global_constants_test", "--simulation.evolve=false", "--gravity.direct.sinkFraction=0.1"}, localities.size());
        } catch (std::invalid_argument const&) {
            rejectedUnusedComparison = true;
        }
        if (!rejectedUnusedComparison) {
            throw std::runtime_error("direct-gravity comparison accepted a configuration without particles");
        }
        auto const initialOnlyComparison = cosmictiger::detail::parseOptions(options,
            {"global_constants_test", "--simulation.evolve=false", "--initialConditions.generate", "--initialConditions.outputPrefix=direct_test",
                "--gravity.direct.sinkFraction=0.1"},
            localities.size());
        if (initialOnlyComparison.evolve || !initialOnlyComparison.generateInitialConditions || initialOnlyComparison.directSinkFraction != 0.1) {
            throw std::runtime_error("direct-gravity comparison was not enabled for initial-condition output");
        }
        auto lightcone = cosmictiger::detail::parseOptions(options,
            {"global_constants_test", "--lightcone.enabled", "--lightcone.nside=128", "--lightcone.observerX=0.25", "--lightcone.maximumRadius=2.5",
                "--lightcone.outputPrefix=test_cone"},
            localities.size());
        cosmictiger::initializeSimulationConstants(lightcone, lightcone.initialScaleFactor);
        if (!lightcone.lightcone.enabled || lightcone.lightcone.nside != 128 || lightcone.lightcone.observer[0] != 0.25 ||
            lightcone.lightcone.maximumRadius != 2.5 || lightcone.lightcone.outputPrefix != "test_cone" ||
            !(lightcone.lightconeObservationConformalTime > 0.0) ||
            std::abs(lightcone.lightconeSpeed - 299792.458 / (100.0 * lightcone.initializationOptions.boxLength)) > 1.0e-14) {
            throw std::runtime_error("lightcone options or units were not initialized correctly");
        }
        for (std::string const argument : {"--lightcone.nside=3", "--lightcone.maximumRadius=0", "--lightcone.observerY=1"}) {
            bool rejected = false;
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", argument}, localities.size());
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("invalid lightcone option was accepted: " + argument);
            }
        }
        po::variables_map restartOptions;
        po::store(po::command_line_parser(std::vector<std::string>{"--lightcone.nside=32"}).options(options).run(), restartOptions);
        po::notify(restartOptions);
        double const savedObserverTime = lightcone.lightconeObservationConformalTime;
        cosmictiger::applyLightconeRestartOptions(restartOptions, lightcone);
        if (!lightcone.lightcone.enabled || lightcone.lightcone.nside != 32 || lightcone.lightcone.observer[0] != 0.25 ||
            lightcone.lightconeObservationConformalTime != savedObserverTime) {
            throw std::runtime_error("lightcone restart did not preserve unspecified settings and the saved observer epoch");
        }
        auto const invalidDrift = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.drift=yes"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.drift must be on or off") != std::string::npos;
            }
            return false;
        }();
        auto const invalidKick = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.kick=true"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.kick must be on or off") != std::string::npos;
            }
            return false;
        }();
        auto const invalidMultipole = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.multipole=true"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.multipole must be on or off") != std::string::npos;
            }
            return false;
        }();
        if (!invalidDrift || !invalidKick || !invalidMultipole) {
            throw std::runtime_error("CUDA execution options accepted a value other than on or off");
        }
#ifdef COSMICTIGER_ENABLE_CUDA
        auto const driftOnly =
            cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.drift=on", "--execution.cuda.kick=off"}, localities.size());
        auto const kickOnly =
            cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.drift=off", "--execution.cuda.kick=on"}, localities.size());
        auto const multipoleOnly = cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.multipole=on"}, localities.size());
        auto const directOnly = cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.direct=on"}, localities.size());
        auto const configurationOnly = cosmictiger::detail::parseOptions(options, {"global_constants_test", configurationArgument}, localities.size());
        auto const multipoleOverridden =
            cosmictiger::detail::parseOptions(options, {"global_constants_test", configurationArgument, "--execution.cuda.multipole=off"}, localities.size());
        if (!driftOnly.cudaDrift || driftOnly.cudaKick || driftOnly.cudaMultipole || kickOnly.cudaDrift || !kickOnly.cudaKick || kickOnly.cudaMultipole ||
            multipoleOnly.cudaDrift || multipoleOnly.cudaKick || !multipoleOnly.cudaMultipole || !configurationOnly.cudaDrift || configurationOnly.cudaKick ||
            !configurationOnly.cudaMultipole || !multipoleOverridden.cudaDrift || multipoleOverridden.cudaKick || multipoleOverridden.cudaMultipole ||
            directOnly.cudaDrift || directOnly.cudaKick || directOnly.cudaMultipole || !directOnly.cudaDirect ||
            configurationOnly.directSinkFraction != 0.0625 || configurationOnly.directSeed != 17) {
            throw std::runtime_error("independent CUDA selections or INI values were parsed incorrectly");
        }
#else
        auto const rejectedDrift = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.drift=on"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.drift=on requires a CUDA-enabled build") != std::string::npos;
            }
            return false;
        }();
        auto const rejectedKick = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.kick=on"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.kick=on requires a CUDA-enabled build") != std::string::npos;
            }
            return false;
        }();
        auto const rejectedMultipole = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.multipole=on"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.multipole=on requires a CUDA-enabled build") != std::string::npos;
            }
            return false;
        }();
        auto const rejectedIniMultipole = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(
                    options, {"global_constants_test", configurationArgument, "--execution.cuda.drift=off"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.multipole=on requires a CUDA-enabled build") != std::string::npos;
            }
            return false;
        }();
        auto const rejectedDirect = [&] {
            PROFILE();
            try {
                (void) cosmictiger::detail::parseOptions(options, {"global_constants_test", "--execution.cuda.direct=on"}, localities.size());
            } catch (std::invalid_argument const& error) {
                return std::string(error.what()).find("execution.cuda.direct=on requires a CUDA-enabled build") != std::string::npos;
            }
            return false;
        }();
        if (!rejectedDrift || !rejectedKick || !rejectedMultipole || !rejectedIniMultipole || !rejectedDirect) {
            throw std::runtime_error("CPU-only build accepted a requested CUDA execution path");
        }
#endif
        std::vector<std::string> arguments{"global_constants_test", "--cosmology.scalarAmplitude=3.25e-9", "--initialConditions.meshExponent=4",
            "--initialConditions.seed=987654321", "--initialConditions.component=cdm", "--timestep.eta=0.125", "--execution.cuda.drift=off",
            "--lightcone.nside=128", "--gravity.direct.sinkFraction=0.03125", "--gravity.direct.seed=123456789", "--gravity.meshExponent=3",
            "--gravity.ghostWidth=2", configurationArgument};
#ifdef COSMICTIGER_ENABLE_CUDA
        arguments.push_back("--execution.cuda.kick=on");
        arguments.push_back("--execution.cuda.direct=on");
#else
        arguments.push_back("--execution.cuda.multipole=off");
#endif
        auto data = cosmictiger::detail::parseOptions(options, arguments, localities.size());
        hpx::filesystem::remove(configurationPath);
        if (data.cosmology.scalarAmplitude != 3.25e-9 || data.initializationOptions.meshExponent != 4 ||
            data.initializationOptions.seed != std::uint64_t(987654321) || data.initializationOptions.component != cosmictiger::MatterComponent::cdm ||
            data.meshSideLength != (size_t(1) << 4) || data.gridSideLength != (size_t(1) << 3) || data.gravityMeshExponent != 3 || data.ghostWidth != 2 ||
            data.fmmOrder != cosmictiger::gravity::kMinimumFmmOrder || data.globalParticleCount != (size_t(1) << (NDIM * 4)) ||
            data.localityCount != localities.size() || data.timestepEta != 0.125 || !data.outputAcceleration || data.outputPotential ||
            data.initialScaleFactor != 0.03 || data.stopTimeYears != 1000000 || data.cudaDrift || data.cudaKick != cosmictiger::detail::kExpectedCudaKick ||
            data.cudaMultipole != cosmictiger::detail::kExpectedCudaMultipole || data.cudaDirect != cosmictiger::detail::kExpectedCudaDirect ||
            data.directSinkFraction != 0.03125 || data.directSeed != std::uint64_t(123456789)) {
            throw std::runtime_error("dotted command-line options were not parsed or derived correctly");
        }
        std::vector<char> serializedData;
        {
            hpx::serialization::output_archive archive(serializedData);
            archive << data;
        }
        cosmictiger::GlobalConstantsData restoredData;
        {
            hpx::serialization::input_archive archive(serializedData);
            archive >> restoredData;
        }
        cosmictiger::units::UnitScales const originalScales(data.cosmology.hubble, data.initializationOptions.boxLength);
        cosmictiger::units::UnitScales const restoredScales(restoredData.cosmology.hubble, restoredData.initializationOptions.boxLength);
        if (restoredScales.lengthCgs() != originalScales.lengthCgs() || restoredScales.massCgs() != originalScales.massCgs() ||
            restoredScales.timeSeconds() != originalScales.timeSeconds()) {
            throw std::runtime_error("serialized cosmology did not reconstruct the original physical unit scales");
        }
        if (restoredData.directSinkFraction != data.directSinkFraction || restoredData.directSeed != data.directSeed ||
            restoredData.cudaDirect != data.cudaDirect || restoredData.initializationOptions.seed != data.initializationOptions.seed ||
            restoredData.lightcone.nside != data.lightcone.nside || restoredData.gravityMeshExponent != data.gravityMeshExponent ||
            restoredData.ghostWidth != data.ghostWidth || restoredData.gridSideLength != data.gridSideLength ||
            restoredData.meshSideLength != data.meshSideLength || restoredData.fmmOrder != data.fmmOrder) {
            throw std::runtime_error("GlobalConstants serialization lost gravity options or independent seeds");
        }
        cosmictiger::initializeSimulationConstants(data, data.initialScaleFactor);
        double const h = data.cosmology.hubble / 100.0;
        double const expectedMass = 3.0 * data.cosmology.omegaCdmH2 / (8.0 * std::numbers::pi * h * h * data.globalParticleCount);
        if (std::abs(data.gravityParticleMass / expectedMass - 1.0) > 1.0e-14) {
            throw std::runtime_error("gravity particle mass does not use Hubble-time/unit-box units");
        }
        cosmictiger::publishGlobalConstants(std::move(data));

        std::vector<hpx::future<void>> checks;
        checks.reserve(localities.size());
        for (auto const& locality : localities) {
            checks.push_back(hpx::async<CosmictigerVerifyGlobalConstantsAction>(locality, localities.size()));
        }
        for (auto& check : checks) {
            check.get();
        }

        cosmictiger::GlobalConstantsData attempt = cosmictiger::globalConstants().data();
        attempt.fmmOrder = cosmictiger::gravity::kMinimumFmmOrder - 1;
        bool rejected = false;
        try {
            cosmictiger::publishGlobalConstants(std::move(attempt));
        } catch (std::exception const&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("root locality republished GlobalConstants");
        }
        cosmictiger::detail::verifyGlobalConstants(localities.size());
        return hpx::finalize();
    } catch (std::exception const& error) {
        hpx::cerr << "GlobalConstants test: " << error.what() << '\n' << hpx::iostreams::flush_type{};
        hpx::finalize();
        return 1;
    }
}

int main(int argc, char* argv[]) {
    PROFILE();
    return hpx::init(argc, argv);
}
