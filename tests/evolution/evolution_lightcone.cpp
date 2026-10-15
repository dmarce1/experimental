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

#include <cosmictiger/evolution/evolution.hpp>
#include <cosmictiger/evolution/evolution_checkpoint.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr size_t kLatticeExponent = 3;
constexpr size_t kLatticeSide = size_t(1) << kLatticeExponent;
constexpr double kInitialScaleFactor = 0.5;
constexpr double kObservationConformalTime = 0.35;
constexpr double kFinalConformalTime = 0.30;
constexpr double kSplitConformalTime = 0.15;
constexpr double kSpeedOfLightKmPerSecond = 299792.458;
constexpr double kHubbleLengthNormalization = 100.0;

cosmictiger::GlobalConstantsData lightconeConstants(std::string const& prefix) {
    constexpr std::uint32_t kNside = 4;
    constexpr double kMaximumRadius = 0.3;
    PROFILE();

    cosmictiger::GlobalConstantsData data;
    data.cosmology.hubble = 50.0;
    data.cosmology.omegaBaryonH2 = 0.05;
    data.cosmology.omegaCdmH2 = 0.20;
    data.cosmology.omegaNeutrinoH2 = 0.0;
    data.cosmology.massiveNeutrinos = 0;
    data.cosmology.masslessNeutrinos = 0;
    data.cosmology.effectiveNeutrinos = 0.0;
    data.cosmology.temperatureCmb = 0.0;
    data.initializationOptions.meshExponent = kLatticeExponent;
    data.gravityMeshExponent = data.initializationOptions.meshExponent;
    data.initializationOptions.boxLength = kSpeedOfLightKmPerSecond / kHubbleLengthNormalization;
    data.meshSideLength = data.gridSideLength = kLatticeSide;
    data.globalParticleCount = size_t(1) << (NDIM * kLatticeExponent);
    data.localityCount = 1;
    data.timestepEta = 1.0e6;
    cosmictiger::initializeSimulationConstants(data, kInitialScaleFactor);
    cosmictiger::SimulationBackground const background(data.cosmology);
    double const observationPhysicalTime =
        data.initialPhysicalTime + background.advance(background.state(kInitialScaleFactor), kObservationConformalTime).physicalInterval;
    data.lightcone.enabled = true;
    data.lightcone.nside = kNside;
    data.lightcone.maximumRadius = kMaximumRadius;
    data.lightcone.observer = {};
    data.lightcone.outputPrefix = prefix;
    data.lightcone.observationTimeYears = cosmictiger::physicalTimeInYears(observationPhysicalTime, data.cosmology);
    cosmictiger::initializeSimulationConstants(data, kInitialScaleFactor);
    return data;
}

void setLightconeStop(cosmictiger::GlobalConstantsData& data, double conformalTime) {
    PROFILE();
    cosmictiger::SimulationBackground const background(data.cosmology);
    data.stopPhysicalTime = data.initialPhysicalTime + background.advance(background.state(kInitialScaleFactor), conformalTime).physicalInterval;
}

cosmictiger::EvolutionState lightconeInitialState(cosmictiger::GlobalConstantsData const& data, std::uint64_t operation) {
    PROFILE();
    cosmictiger::EvolutionState state;
    state.scaleFactor = state.tickScaleFactor = data.initialScaleFactor;
    state.conformalDerivative = state.tickConformalDerivative = data.initialConformalDerivative;
    state.physicalTime = state.tickPhysicalTime = data.initialPhysicalTime;
    state.operationId = operation;
    // Start with an already opened kick and stationary particles. The test
    // stops before any rung event, so it isolates lightcone sampling from
    // forces and leaves every position exactly unchanged.
    state.momentaStaggered = true;
    return state;
}

void lightconeLattice(cosmictiger::ParticleStorage& particles, cosmictiger::CartesianGrid& grid) {
    constexpr double kCellCenter = 0.5;
    PROFILE();

    for (size_t z = 0; z < kLatticeSide; z++) {
        for (size_t y = 0; y < kLatticeSide; y++) {
            for (size_t x = 0; x < kLatticeSide; x++) {
                auto const begin = particles.size();
                particles.append({cosmictiger::Coordinate((x + kCellCenter) / kLatticeSide), cosmictiger::Coordinate((y + kCellCenter) / kLatticeSide),
                                     cosmictiger::Coordinate((z + kCellCenter) / kLatticeSide)},
                    {}, 0);
                grid.setInteriorRange({std::int64_t(x), std::int64_t(y), std::int64_t(z)}, {begin, begin + 1});
            }
        }
    }
}

struct LightconeCsv {
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rows;
    size_t files = 0;
};

std::vector<std::string> splitCsvLine(std::string const& line) {
    PROFILE();
    std::vector<std::string> columns;
    std::istringstream stream(line);
    std::string column;
    while (std::getline(stream, column, ',')) {
        columns.push_back(column);
    }
    return columns;
}

LightconeCsv readLightconeCsv(std::string const& prefix, std::string const& suffix) {
    PROFILE();
    LightconeCsv result;
    hpx::filesystem::path const path(prefix);
    std::string const basename = path.filename().string() + ".step";
    for (auto const& entry : hpx::filesystem::directory_iterator(path.parent_path())) {
        std::string const name = entry.path().filename().string();
        if (!name.starts_with(basename) || !name.ends_with(suffix)) {
            continue;
        }
        auto const marker = entry.path().parent_path() / (name.substr(0, name.size() - suffix.size()) + ".complete");
        if (!hpx::filesystem::exists(marker)) {
            continue;
        }
        result.files++;
        std::ifstream input(entry.path().string());
        if (!input) {
            throw std::runtime_error("cannot read lightcone regression output");
        }
        std::string line;
        bool header = true;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') {
                continue;
            }
            auto columns = splitCsvLine(line);
            if (header) {
                if (result.names.empty()) {
                    result.names = columns;
                } else {
                    EXPECT_EQ(columns, result.names);
                }
                header = false;
            } else {
                result.rows.push_back(std::move(columns));
            }
        }
    }
    return result;
}

size_t lightconeColumn(LightconeCsv const& csv, std::string const& name) {
    PROFILE();
    for (size_t index = 0; index < csv.names.size(); index++) {
        if (csv.names[index] == name) {
            return index;
        }
    }
    throw std::runtime_error("lightcone output lacks column " + name);
}

cosmictiger::InitialConditionsSiloMetadata lightconeMetadata(cosmictiger::GlobalConstantsData const& data) {
    constexpr double kParticleMass = 1.0;
    PROFILE();

    return {data.cosmology, data.initialScaleFactor, data.initializationOptions.boxLength, kParticleMass, data.initializationOptions.maximumDensityDeviation,
        data.initializationOptions.maximumDensityDeviation, data.initializationOptions.seed, kLatticeExponent, data.initializationOptions.component, true};
}

struct LightconeTotals {
    std::map<std::uint64_t, std::uint64_t> counts;
    std::map<std::uint64_t, double> weights;
};

LightconeTotals checkLightconeOutput(std::string const& prefix) {
    constexpr size_t kExpectedCrossings = 56;
    constexpr double kHalf = 0.5;
    constexpr double kMaximumRadius = 0.3;
    constexpr double kTolerance = 1.0e-10;
    PROFILE();

    auto const crossings = readLightconeCsv(prefix, ".crossings.csv");
    auto const map = readLightconeCsv(prefix, ".map.csv");
    EXPECT_GT(crossings.files, 0);
    EXPECT_EQ(crossings.files, map.files);
    EXPECT_EQ(crossings.rows.size(), kExpectedCrossings);
    size_t const pixelColumn = lightconeColumn(crossings, "pixel");
    size_t const timeColumn = lightconeColumn(crossings, "conformalTime");
    size_t const scaleColumn = lightconeColumn(crossings, "scaleFactor");
    size_t const radiusColumn = lightconeColumn(crossings, "radius");
    size_t const weightColumn = lightconeColumn(crossings, "inverseRadiusSquared");
    std::array<size_t, NDIM> const coordinateColumns{lightconeColumn(crossings, "x"), lightconeColumn(crossings, "y"), lightconeColumn(crossings, "z")};
    std::array<size_t, NDIM> const velocityColumns{lightconeColumn(crossings, "vx"), lightconeColumn(crossings, "vy"), lightconeColumn(crossings, "vz")};
    LightconeTotals catalogTotals;
    for (auto const& row : crossings.rows) {
        EXPECT_EQ(row.size(), crossings.names.size());
        auto const pixel = std::stoull(row.at(pixelColumn));
        double const time = std::stod(row.at(timeColumn));
        double const radius = std::stod(row.at(radiusColumn));
        double const weight = std::stod(row.at(weightColumn));
        double squaredRadius = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const coordinate = std::stod(row.at(coordinateColumns[axis]));
            squaredRadius += coordinate * coordinate;
            EXPECT_DOUBLE_EQ(std::stod(row.at(velocityColumns[axis])), 0.0);
        }
        EXPECT_NEAR(radius, std::sqrt(squaredRadius), kTolerance);
        EXPECT_GT(radius, kObservationConformalTime - kFinalConformalTime);
        EXPECT_LT(radius, kMaximumRadius);
        EXPECT_NEAR(time, kObservationConformalTime - radius, kTolerance);
        double const expectedRootA = std::sqrt(kInitialScaleFactor) + kHalf * time;
        EXPECT_NEAR(std::stod(row.at(scaleColumn)), expectedRootA * expectedRootA, kTolerance);
        EXPECT_NEAR(weight, 1.0 / squaredRadius, kTolerance);
        catalogTotals.counts[pixel]++;
        catalogTotals.weights[pixel] += weight;
    }
    size_t const mapPixelColumn = lightconeColumn(map, "pixel");
    size_t const mapCountColumn = lightconeColumn(map, "count");
    size_t const mapWeightColumn = lightconeColumn(map, "inverseRadiusSquared");
    LightconeTotals mapTotals;
    for (auto const& row : map.rows) {
        EXPECT_EQ(row.size(), map.names.size());
        auto const pixel = std::stoull(row.at(mapPixelColumn));
        mapTotals.counts[pixel] += std::stoull(row.at(mapCountColumn));
        mapTotals.weights[pixel] += std::stod(row.at(mapWeightColumn));
    }
    EXPECT_EQ(mapTotals.counts, catalogTotals.counts);
    EXPECT_EQ(mapTotals.weights.size(), catalogTotals.weights.size());
    for (auto const& [pixel, weight] : catalogTotals.weights) {
        EXPECT_NEAR(mapTotals.weights[pixel], weight, kTolerance);
    }
    return catalogTotals;
}

TEST(EvolutionLightcone, SnapshotSegmentsUseOwnedParticlesAndPreserveTheTrajectory) {
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
    auto const directory = hpx::filesystem::temp_directory_path() / ("cosmictiger_evolution_lightcone_segments_" + std::to_string(getpid()));
    hpx::filesystem::remove_all(directory);
    hpx::filesystem::create_directories(directory);
    auto data = lightconeConstants((directory / "enabled").string());
    data.lightconeRunId = "invocation";
    data.outputSnapshotCount = 4;
    setLightconeStop(data, kFinalConformalTime);
    EXPECT_NEAR(data.lightconeObservationConformalTime, kObservationConformalTime, 1.0e-11);
    EXPECT_DOUBLE_EQ(data.lightconeSpeed, 1.0);
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(kLatticeExponent, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    lightconeLattice(particles, grid);
    auto state = lightconeInitialState(data, 700000);
    size_t lightconeInitializations = 0;
    cosmictiger::evolveParticles(particles, grid, data, state, {}, {}, {},
        hpx::annotated_function(
            [&](cosmictiger::EvolutionInitializationStage stage) {
                PROFILE();
                if (stage == cosmictiger::EvolutionInitializationStage::lightcone) {
                    lightconeInitializations++;
                }
            },
            "test.evolution.observeLightconeInitialization"));
    EXPECT_EQ(lightconeInitializations, 1);
    std::string const actualPrefix = data.lightcone.outputPrefix + "." + data.lightconeRunId;
    auto const crossings = readLightconeCsv(actualPrefix, ".crossings.csv");
    EXPECT_EQ(crossings.files, data.outputSnapshotCount - 1);
    checkLightconeOutput(actualPrefix);

    auto disabled = data;
    disabled.lightcone.enabled = false;
    disabled.lightcone.outputPrefix = (directory / "disabled").string();
    cosmictiger::ParticleStorage disabledParticles;
    cosmictiger::CartesianGrid disabledGrid(kLatticeExponent, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    lightconeLattice(disabledParticles, disabledGrid);
    auto disabledState = lightconeInitialState(disabled, 710000);
    cosmictiger::evolveParticles(disabledParticles, disabledGrid, disabled, disabledState);
    EXPECT_EQ(readLightconeCsv(disabled.lightcone.outputPrefix + "." + disabled.lightconeRunId, ".crossings.csv").files, 0);
    ASSERT_EQ(particles.size(), disabledParticles.size());
    for (size_t particle = 0; particle < particles.size(); particle++) {
        EXPECT_EQ(particles.flags(particle).rung(), disabledParticles.flags(particle).rung());
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(particles.position(axis, particle).raw(), disabledParticles.position(axis, particle).raw());
            EXPECT_EQ(particles.velocity(axis, particle), disabledParticles.velocity(axis, particle));
        }
    }
    EXPECT_DOUBLE_EQ(state.scaleFactor, disabledState.scaleFactor);
    EXPECT_DOUBLE_EQ(state.physicalTime, disabledState.physicalTime);
    EXPECT_DOUBLE_EQ(state.conformalOffset, disabledState.conformalOffset);
    hpx::filesystem::remove_all(directory);
}

TEST(EvolutionLightcone, CheckpointAndRungExtensionPreserveObserverEpochAndMapCounts) {
    constexpr unsigned kRungExtension = 2;
    constexpr double kTolerance = 1.0e-10;
    PROFILE();

#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU evolution regression uses the CPU-only build";
#endif
    auto const directory = hpx::filesystem::temp_directory_path() / ("cosmictiger_evolution_lightcone_restart_" + std::to_string(getpid()));
    hpx::filesystem::remove_all(directory);
    hpx::filesystem::create_directories(directory);
    auto fullData = lightconeConstants((directory / "full").string());
    setLightconeStop(fullData, kFinalConformalTime);
    cosmictiger::ParticleStorage fullParticles;
    cosmictiger::CartesianGrid fullGrid(kLatticeExponent, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    lightconeLattice(fullParticles, fullGrid);
    auto fullState = lightconeInitialState(fullData, 720000);
    cosmictiger::evolveParticles(fullParticles, fullGrid, fullData, fullState);
    auto const fullTotals = checkLightconeOutput(fullData.lightcone.outputPrefix);

    auto splitData = fullData;
    splitData.lightcone.outputPrefix = (directory / "split").string();
    setLightconeStop(splitData, kSplitConformalTime);
    cosmictiger::ParticleStorage splitParticles;
    cosmictiger::CartesianGrid splitGrid(kLatticeExponent, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    lightconeLattice(splitParticles, splitGrid);
    auto splitState = lightconeInitialState(splitData, 730000);
    cosmictiger::evolveParticles(splitParticles, splitGrid, splitData, splitState);
    ASSERT_GT(splitState.conformalOffset, 0.0);
    auto const filename = (directory / "restart.silo").string();
    cosmictiger::writeEvolutionCheckpoint(filename, splitParticles, splitGrid, splitData, splitState, lightconeMetadata(splitData));
    auto saved = cosmictiger::readEvolutionCheckpointMetadata(filename);
    cosmictiger::ParticleStorage resumedParticles;
    cosmictiger::CartesianGrid resumedGrid(kLatticeExponent, 0, 0, cosmictiger::gravity::kPmmmNeighborWidth);
    cosmictiger::readParticleSilo(filename, resumedParticles, resumedGrid);
    auto resumed = saved.state;
    auto resumedData = saved.constants;
    EXPECT_TRUE(resumedData.lightcone.enabled);
    EXPECT_EQ(resumedData.lightcone.outputPrefix, splitData.lightcone.outputPrefix);
    EXPECT_DOUBLE_EQ(resumedData.lightconeObservationConformalTime, fullData.lightconeObservationConformalTime);
    EXPECT_DOUBLE_EQ(resumedData.lightconeSpeed, fullData.lightconeSpeed);
    cosmictiger::extendEvolutionRungs(resumedParticles, resumedGrid, resumed, kRungExtension);
    resumedData.rungZeroTimestep = std::ldexp(resumedData.rungZeroTimestep, int(kRungExtension));
    resumedData.rungZeroExtension += kRungExtension;
    setLightconeStop(resumedData, kFinalConformalTime);
    cosmictiger::evolveParticles(resumedParticles, resumedGrid, resumedData, resumed);
    EXPECT_DOUBLE_EQ(resumedData.lightconeObservationConformalTime, fullData.lightconeObservationConformalTime);
    auto const resumedTotals = checkLightconeOutput(resumedData.lightcone.outputPrefix);
    EXPECT_EQ(resumedTotals.counts, fullTotals.counts);
    ASSERT_EQ(resumedTotals.weights.size(), fullTotals.weights.size());
    for (auto const& [pixel, weight] : fullTotals.weights) {
        EXPECT_NEAR(resumedTotals.weights.at(pixel), weight, kTolerance);
    }
    EXPECT_DOUBLE_EQ(resumed.physicalTime, fullState.physicalTime);
    EXPECT_NEAR(resumed.scaleFactor, fullState.scaleFactor, kTolerance);
    hpx::filesystem::remove_all(directory);
}

}    // namespace
