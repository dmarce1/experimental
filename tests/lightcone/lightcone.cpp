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

#include <cosmictiger/lightcone/lightcone.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/modules/filesystem.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using CsvRow = std::map<std::string, double>;

std::vector<CsvRow> readCsvRows(std::string const& filename) {
    PROFILE();
    std::ifstream input(filename);
    std::string line;
    while (std::getline(input, line) && line.starts_with('#')) {
    }
    std::vector<std::string> columns;
    std::istringstream header(line);
    std::string column;
    while (std::getline(header, column, ',')) {
        columns.push_back(column);
    }
    std::vector<CsvRow> rows;
    while (std::getline(input, line)) {
        std::istringstream values(line);
        CsvRow row;
        for (auto const& name : columns) {
            std::string value;
            if (!std::getline(values, value, ',')) {
                throw std::runtime_error("missing lightcone CSV value");
            }
            row.emplace(name, std::stod(value));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

cosmictiger::CosmologyParameters matterCosmology() {
    PROFILE();
    cosmictiger::CosmologyParameters parameters;
    parameters.hubble = 50.0;
    parameters.omegaBaryonH2 = 0.05;
    parameters.omegaCdmH2 = 0.20;
    parameters.omegaNeutrinoH2 = 0.0;
    parameters.massiveNeutrinos = 0;
    parameters.masslessNeutrinos = 0;
    parameters.effectiveNeutrinos = 0.0;
    parameters.temperatureCmb = 0.0;
    return parameters;
}

class LightconeTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        GTEST_SKIP() << "Lightcone host tests run in the CPU-only build to respect the shared GPU budget";
#endif
    }

    cosmictiger::LightconeOptions options() const {
        PROFILE();
        cosmictiger::LightconeOptions result;
        result.enabled = true;
        result.nside = 4;
        result.maximumRadius = 0.4;
        return result;
    }

    cosmictiger::LightconeStep step(double begin, double end) const {
        PROFILE();
        auto const initial = background_.state(0.5);
        return {initial, 0.0, begin, end, background_.advance(initial, begin).driftFactor, background_.advance(initial, end).driftFactor};
    }

    void append(std::array<double, NDIM> const& position, std::array<float, NDIM> const& momentum = {}) {
        PROFILE();
        std::array<cosmictiger::Coordinate, NDIM> fixed{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            fixed[axis] = cosmictiger::Coordinate(position[axis]);
        }
        particles_.append(fixed, momentum, 0);
        grid_.setInteriorRange({}, {0, particles_.size()});
    }

    cosmictiger::SimulationBackground const& background() const {
        PROFILE();
        return background_;
    }

    cosmictiger::ParticleStorage& particles() {
        PROFILE();
        return particles_;
    }

    cosmictiger::CartesianGrid const& grid() const {
        PROFILE();
        return grid_;
    }

private:
    cosmictiger::SimulationBackground background_{matterCosmology()};
    cosmictiger::ParticleStorage particles_;
    cosmictiger::CartesianGrid grid_{0, 0, 0};
};

TEST_F(LightconeTest, StationaryCrossingUsesItsOwnEpochAndConservesSparseMap) {
    PROFILE();
    append({0.25, 0.0, 0.0});
    cosmictiger::Lightcone cone(options(), 0.75, 1.0);
    auto const sample = cone.sample(particles(), grid(), background(), step(0.0, 0.6));
    ASSERT_EQ(sample.crossings.size(), 1u);
    EXPECT_DOUBLE_EQ(sample.crossings.front().tickOffset, 0.5);
    EXPECT_DOUBLE_EQ(sample.crossings.front().radius, 0.25);
    double const expectedRootScale = std::sqrt(0.5) + 0.25;
    EXPECT_NEAR(sample.crossings.front().scaleFactor, expectedRootScale * expectedRootScale, 1.0e-11);
    ASSERT_EQ(sample.bins.size(), 1u);
    EXPECT_EQ(sample.bins.front().count, 1u);
    EXPECT_DOUBLE_EQ(sample.bins.front().inverseRadiusSquared, 16.0);
    EXPECT_DOUBLE_EQ(particles().position(0, 0).toDouble(), 0.25);
}

TEST_F(LightconeTest, MovingCrossingFollowsIntegratedBackgroundInsteadOfEndpointChord) {
    constexpr double kHit = 0.25;
    constexpr float kMomentum = 0.125f;
    constexpr double kInitialPosition = 0.125;
    PROFILE();
    append({kInitialPosition, 0.0, 0.0}, {kMomentum, 0.0f, 0.0f});
    double const rootScale = std::sqrt(0.5);
    double const analyticDrift = 2.0 * (1.0 / rootScale - 1.0 / (rootScale + kHit / 2.0));
    double const hitPosition = kInitialPosition + kMomentum * analyticDrift;
    cosmictiger::Lightcone cone(options(), kHit + hitPosition, 1.0);
    auto const sample = cone.sample(particles(), grid(), background(), step(0.0, 0.5));
    ASSERT_EQ(sample.crossings.size(), 1u);
    auto const& crossing = sample.crossings.front();
    EXPECT_NEAR(crossing.tickOffset, kHit, 2.0e-11);
    EXPECT_NEAR(crossing.position[0], hitPosition, 2.0e-11);
    EXPECT_NEAR(crossing.velocity[0], kMomentum / ((rootScale + kHit / 2.0) * (rootScale + kHit / 2.0)), 2.0e-11);
}

TEST_F(LightconeTest, TimestepBoundaryBelongsOnlyToEndingInterval) {
    PROFILE();
    append({0.25, 0.0, 0.0});
    cosmictiger::Lightcone cone(options(), 0.75, 1.0);
    auto const first = cone.sample(particles(), grid(), background(), step(0.0, 0.5));
    auto const second = cone.sample(particles(), grid(), background(), step(0.5, 0.7));
    ASSERT_EQ(first.crossings.size(), 1u);
    EXPECT_TRUE(second.crossings.empty());
}

TEST_F(LightconeTest, ContractingBackgroundUsesSignedEvolutionAtCrossing) {
    PROFILE();
    append({0.25, 0.0, 0.0});
    cosmictiger::Lightcone cone(options(), 0.5, 1.0);
    auto const initial = background().state(0.5, cosmictiger::ExpansionBranch::contracting);
    cosmictiger::LightconeStep const interval{initial, 0.0, 0.0, 0.4, 0.0, background().advance(initial, 0.4).driftFactor};
    auto const sample = cone.sample(particles(), grid(), background(), interval);
    ASSERT_EQ(sample.crossings.size(), 1u);
    double const expectedRootScale = std::sqrt(0.5) - 0.125;
    EXPECT_NEAR(sample.crossings.front().scaleFactor, expectedRootScale * expectedRootScale, 1.0e-11);
    EXPECT_DOUBLE_EQ(sample.crossings.front().tickOffset, 0.25);
}

TEST_F(LightconeTest, TranslatedObserverFindsPeriodicImageAcrossBoxBoundary) {
    PROFILE();
    append({0.875, 0.0, 0.0});
    auto config = options();
    config.observer = {0.125, 0.0, 0.0};
    cosmictiger::Lightcone cone(config, 0.5, 1.0);
    auto const sample = cone.sample(particles(), grid(), background(), step(0.0, 0.4));
    ASSERT_EQ(sample.crossings.size(), 1u);
    auto const& crossing = sample.crossings.front();
    EXPECT_EQ(crossing.image, (std::array<std::int64_t, NDIM>{-1, 0, 0}));
    EXPECT_DOUBLE_EQ(crossing.position[0], -0.25);
    EXPECT_DOUBLE_EQ(crossing.tickOffset, 0.25);
}

TEST_F(LightconeTest, EnumeratesMultipleImagesAndOmitsUndefinedOrigin) {
    PROFILE();
    append({0.0, 0.0, 0.0});
    auto config = options();
    config.maximumRadius = 1.2;
    cosmictiger::Lightcone cone(config, 2.0, 1.0);
    auto const sample = cone.sample(particles(), grid(), background(), step(0.0, 2.0));
    ASSERT_EQ(sample.crossings.size(), 2u * NDIM);
    std::uint64_t count = 0;
    double weight = 0.0;
    for (auto const& bin : sample.bins) {
        count += bin.count;
        weight += bin.inverseRadiusSquared;
    }
    EXPECT_EQ(count, 2u * NDIM);
    EXPECT_DOUBLE_EQ(weight, 2.0 * NDIM);
}

TEST_F(LightconeTest, OwnershipComesFromInteriorRangesIncludingUnsetGhostFlags) {
    PROFILE();
    append({0.25, 0.0, 0.0});
    particles().append({cosmictiger::Coordinate(0.25), cosmictiger::Coordinate(0.0), cosmictiger::Coordinate(0.0)}, {}, 0, false);
    particles().append({cosmictiger::Coordinate(0.25), cosmictiger::Coordinate(0.0), cosmictiger::Coordinate(0.0)}, {}, 0, true);
    cosmictiger::Lightcone cone(options(), 0.75, 1.0);
    auto const sample = cone.sample(particles(), grid(), background(), step(0.0, 0.6));
    ASSERT_EQ(sample.crossings.size(), 1u);
    EXPECT_EQ(sample.crossings.front().localParticleIndex, 0u);
}

TEST_F(LightconeTest, SplitDriftPreservesCrossingWithinCoordinateQuantization) {
    constexpr double kSplit = 0.125;
    constexpr double kInitialPosition = 0.125;
    constexpr float kMomentum = 0.125f;
    PROFILE();
    append({kInitialPosition, 0.0, 0.0}, {kMomentum, 0.0f, 0.0f});
    cosmictiger::Lightcone cone(options(), 0.5, 1.0);
    auto const full = cone.sample(particles(), grid(), background(), step(0.0, 0.5));
    auto const first = cone.sample(particles(), grid(), background(), step(0.0, kSplit));
    auto const secondStep = step(kSplit, 0.5);
    particles().position(0, 0) = cosmictiger::Coordinate(kInitialPosition + kMomentum * secondStep.oldDrift);
    auto const second = cone.sample(particles(), grid(), background(), secondStep);
    ASSERT_EQ(full.crossings.size(), 1u);
    EXPECT_TRUE(first.crossings.empty());
    ASSERT_EQ(second.crossings.size(), 1u);
    EXPECT_NEAR(full.crossings.front().tickOffset, second.crossings.front().tickOffset, 1.0e-8);
    EXPECT_EQ(full.crossings.front().pixel, second.crossings.front().pixel);
}

TEST_F(LightconeTest, FineOffsetSurvivesLargeTickTime) {
    constexpr double kInterval = 0x1p-20;
    constexpr double kLargeTime = 0x1p40;
    PROFILE();
    append({0.25, 0.0, 0.0});
    cosmictiger::Lightcone cone(options(), kLargeTime + 0.5, 1.0);
    auto interval = step(0.25 - kInterval, 0.25);
    interval.tickConformalTime = kLargeTime;
    auto const sample = cone.sample(particles(), grid(), background(), interval);
    ASSERT_EQ(sample.crossings.size(), 1u);
    EXPECT_DOUBLE_EQ(sample.crossings.front().tickOffset, 0.25);
}

TEST_F(LightconeTest, RejectsSuperluminalTrajectory) {
    PROFILE();
    append({0.25, 0.0, 0.0}, {1.0f, 0.0f, 0.0f});
    cosmictiger::Lightcone cone(options(), 0.75, 1.0);
    EXPECT_THROW(cone.sample(particles(), grid(), background(), step(0.0, 0.6)), std::runtime_error);
}

TEST_F(LightconeTest, WritesReadableSparseSegmentsAndRefusesOverwrite) {
    constexpr double kLightSpeedKmPerSecond = 299792.458;
    constexpr double kReferenceHubble = 100.0;
    PROFILE();
    cosmictiger::units::UnitScales const scales(50.0, kLightSpeedKmPerSecond / kReferenceHubble);
    append({0.25, 0.0, 0.0});
    auto config = options();
    auto const directory = hpx::filesystem::temp_directory_path() / ("cosmictiger-lightcone-" + std::to_string(::getpid()));
    config.outputPrefix = (directory / "test").string();
    cosmictiger::Lightcone cone(config, 0.75, 1.0);
    auto const interval = step(0.0, 0.6);
    auto const sample = cone.sample(particles(), grid(), background(), interval);
    cone.write(sample, interval, 0, 1, scales);
    ASSERT_TRUE(hpx::filesystem::exists(config.outputPrefix + ".step1.rank0.complete"));
    std::ifstream input(config.outputPrefix + ".step1.rank0.map.csv");
    std::string line;
    while (std::getline(input, line) && line.starts_with('#')) {
    }
    EXPECT_EQ(line, "pixel,count,inverseRadiusSquared,inverseComovingRadiusSquaredPerMpc2,inverseProperRadiusSquaredPerCm2");
    ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
    std::istringstream row(line);
    std::uint64_t pixel = 0;
    std::uint64_t count = 0;
    double weight = 0.0;
    char separator = 0;
    row >> pixel >> separator >> count >> separator >> weight;
    EXPECT_EQ(pixel, sample.crossings.front().pixel);
    EXPECT_EQ(count, 1u);
    EXPECT_DOUBLE_EQ(weight, 16.0);
    EXPECT_THROW(cone.write(sample, interval, 0, 1, scales), std::runtime_error);
    hpx::filesystem::remove_all(directory);
}

TEST_F(LightconeTest, OutputUsesEachCrossingEpochAndDoesNotAddHubbleFlow) {
    constexpr double kLightSpeedKmPerSecond = 299792.458;
    constexpr double kReferenceHubble = 100.0;
    constexpr double kHubble = 50.0;
    constexpr double kMegaparsecCm = 3.085677581e24;
    constexpr double kTolerance = 1.0e-12;
    constexpr double kEpochTolerance = 1.0e-11;
    PROFILE();
    cosmictiger::units::UnitScales const scales(kHubble, kLightSpeedKmPerSecond / kReferenceHubble);
    append({0.25, 0.0, 0.0});
    append({0.375, 0.0, 0.0});
    auto config = options();
    auto const directory = hpx::filesystem::temp_directory_path() / ("cosmictiger-lightcone-units-" + std::to_string(::getpid()));
    config.outputPrefix = (directory / "test").string();
    cosmictiger::Lightcone cone(config, 0.75, 1.0);
    auto const interval = step(0.0, 0.6);
    auto const sample = cone.sample(particles(), grid(), background(), interval);
    ASSERT_EQ(sample.crossings.size(), 2u);
    ASSERT_EQ(sample.bins.size(), 1u);
    cone.write(sample, interval, 0, 1, scales);
    auto const rows = readCsvRows(config.outputPrefix + ".step1.rank0.crossings.csv");
    ASSERT_EQ(rows.size(), 2u);
    double properWeight = 0.0;
    double comovingWeight = 0.0;
    double const expectedLengthMpc = kLightSpeedKmPerSecond / kHubble;
    for (auto const& row : rows) {
        double const expectedRootScale = std::sqrt(0.5) + row.at("tickOffset") / 2.0;
        double const expectedScale = expectedRootScale * expectedRootScale;
        EXPECT_NEAR(row.at("scaleFactor"), expectedScale, kEpochTolerance);
        EXPECT_NEAR(row.at("xComovingMpc") / (row.at("x") * expectedLengthMpc), 1.0, kTolerance);
        EXPECT_NEAR(row.at("xProperCm") / (row.at("xComovingMpc") * kMegaparsecCm), row.at("scaleFactor"), kTolerance);
        EXPECT_DOUBLE_EQ(row.at("peculiarVxKmPerSecond"), 0.0);
        EXPECT_DOUBLE_EQ(row.at("peculiarVyKmPerSecond"), 0.0);
        EXPECT_DOUBLE_EQ(row.at("peculiarVzKmPerSecond"), 0.0);
        EXPECT_DOUBLE_EQ(row.at("peculiarVxCmPerSecond"), 0.0);
        EXPECT_DOUBLE_EQ(row.at("peculiarVyCmPerSecond"), 0.0);
        EXPECT_DOUBLE_EQ(row.at("peculiarVzCmPerSecond"), 0.0);
        double const radiusMpc = row.at("radiusComovingMpc");
        double const radiusCm = row.at("radiusProperCm");
        EXPECT_NEAR(row.at("inverseProperRadiusSquaredPerCm2") * radiusCm * radiusCm, 1.0, kTolerance);
        comovingWeight += 1.0 / (radiusMpc * radiusMpc);
        properWeight += 1.0 / (radiusCm * radiusCm);
    }
    auto const bins = readCsvRows(config.outputPrefix + ".step1.rank0.map.csv");
    ASSERT_EQ(bins.size(), 1u);
    EXPECT_DOUBLE_EQ(bins.front().at("count"), 2.0);
    EXPECT_NEAR(bins.front().at("inverseComovingRadiusSquaredPerMpc2") / comovingWeight, 1.0, kTolerance);
    EXPECT_NEAR(bins.front().at("inverseProperRadiusSquaredPerCm2") / properWeight, 1.0, kTolerance);
    hpx::filesystem::remove_all(directory);
}

TEST_F(LightconeTest, OutputConvertsPeculiarVelocityWithoutAnotherScaleFactor) {
    constexpr double kLightSpeedKmPerSecond = 299792.458;
    constexpr double kReferenceHubble = 100.0;
    constexpr double kCentimetersPerKilometer = 1.0e5;
    constexpr float kMomentum = 0.125f;
    constexpr double kTolerance = 1.0e-12;
    PROFILE();
    cosmictiger::units::UnitScales const scales(50.0, kLightSpeedKmPerSecond / kReferenceHubble);
    append({0.125, 0.0, 0.0}, {kMomentum, 0.0f, 0.0f});
    auto config = options();
    auto const directory = hpx::filesystem::temp_directory_path() / ("cosmictiger-lightcone-velocity-" + std::to_string(::getpid()));
    config.outputPrefix = (directory / "test").string();
    cosmictiger::Lightcone cone(config, 0.5, 1.0);
    auto const interval = step(0.0, 0.5);
    auto const sample = cone.sample(particles(), grid(), background(), interval);
    ASSERT_EQ(sample.crossings.size(), 1u);
    cone.write(sample, interval, 0, 1, scales);
    auto const rows = readCsvRows(config.outputPrefix + ".step1.rank0.crossings.csv");
    ASSERT_EQ(rows.size(), 1u);
    auto const& row = rows.front();
    EXPECT_NEAR(row.at("peculiarVxKmPerSecond") * row.at("scaleFactor") / (kMomentum * kLightSpeedKmPerSecond), 1.0, kTolerance);
    EXPECT_NEAR(row.at("peculiarVxCmPerSecond") / (row.at("peculiarVxKmPerSecond") * kCentimetersPerKilometer), 1.0, kTolerance);
    EXPECT_DOUBLE_EQ(row.at("peculiarVyCmPerSecond"), 0.0);
    EXPECT_DOUBLE_EQ(row.at("peculiarVzCmPerSecond"), 0.0);
    hpx::filesystem::remove_all(directory);
}

}    // namespace
