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

#include <gtest/gtest.h>
#include <hpx/modules/filesystem.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {

using SiloFile = std::unique_ptr<DBfile, std::decay_t<decltype(hpx::annotated_function(&DBClose, "SiloFile::close"))>>;
using PointMesh = std::unique_ptr<DBpointmesh, std::decay_t<decltype(hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"))>>;
using SiloComponent = std::unique_ptr<void, std::decay_t<decltype(hpx::annotated_function(&std::free, "SiloComponent::free"))>>;

class EvolutionCheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        int count = 0;
        try {
            count = cosmictiger::cuda::deviceCount().get();
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA device unavailable: " << error.what();
            }
            throw;
        }
        if (count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        int const device = cosmictiger::cuda::device().get();
        if (!cosmictiger::cuda::deviceProperties(device).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
#endif
    }
};

class CheckpointDirectory {
public:
    CheckpointDirectory() {
        PROFILE();
        auto directory = (hpx::filesystem::temp_directory_path() / "cosmictiger_evolution_checkpoint_XXXXXX").string();
        if (::mkdtemp(directory.data()) == nullptr) {
            throw std::runtime_error("cannot create evolution checkpoint test directory");
        }
        path_ = directory;
    }

    ~CheckpointDirectory() {
        PROFILE();
        std::error_code ignored;
        hpx::filesystem::remove_all(path_, ignored);
    }

    std::string file(char const* name) const {
        PROFILE();
        return (path_ / name).string();
    }

private:
    hpx::filesystem::path path_;
};

cosmictiger::EvolutionCheckpointMetadata sampleMetadata() {
    PROFILE();
    cosmictiger::EvolutionCheckpointMetadata result{};
    auto& constants = result.constants;
    constants.initializationOptions.meshExponent = 2;
    constants.gravityMeshExponent = 2;
    constants.localityCount = 1;
    constants.meshSideLength = 4;
    constants.gridSideLength = 4;
    constants.globalParticleCount = 64;
    constants.softeningLength = constants.softeningCellFraction / 4.0;
    constants.gravityParticleMass = 0.003;
    constants.originalRungZeroTimestep = 0.25;
    constants.rungZeroTimestep = 0.5;
    constants.rungZeroExtension = 1;
    constants.initialPhysicalTime = -0.9;
    constants.stopPhysicalTime = 0.4;
    constants.stopTimeYears = 5.8e9;
    constants.initialConformalDerivative = 0.2;
    constants.outputAcceleration = true;
    constants.outputPotential = true;
    constants.cudaDrift = true;
    constants.cudaKick = true;
    constants.cudaMultipole = true;
    constants.cudaDirect = true;
    constants.directSinkFraction = 0.125;
    constants.directSeed = 271828;
    result.initial = {constants.cosmology, constants.initialScaleFactor, constants.initializationOptions.boxLength, 1.4e11, 0.101,
        constants.initializationOptions.maximumDensityDeviation, constants.initializationOptions.seed, constants.initializationOptions.meshExponent,
        constants.initializationOptions.component, true};
    auto& state = result.state;
    state.scaleFactor = 1.4;
    state.tickScaleFactor = 1.41;
    state.conformalDerivative = -0.2;
    state.tickConformalDerivative = -0.19;
    state.physicalTime = 0.35;
    state.tickPhysicalTime = 0.34;
    state.conformalOffset = 0.007;
    state.nominalEpoch = 2;
    state.timeTick = cosmictiger::gravity::rungTicks(4);
    state.operationId = 93;
    state.maximumRung = 4;
    state.momentaStaggered = true;
    state.energy.initialized = true;
    state.energy.initialEnergy = -0.2;
    state.energy.expansionWork = 0.013;
    state.energy.previousScaleFactor = 1.4;
    state.energy.previousKinetic = 0.07;
    state.energy.previousPotential = -0.28;
    state.energy.previousWorkRate = 0.02;
    state.energy.kinetic = 0.07;
    state.energy.potential = -0.28;
    state.energy.conservedEnergy = -0.197;
    return result;
}

void populateParticles(cosmictiger::ParticleStorage& particles, cosmictiger::CartesianGrid& grid, cosmictiger::ParticleCount particlesPerCell = 1) {
    PROFILE();
    particles.configureGravityFields(true, true);
    cosmictiger::ParticleCount particle = 0;
    for (size_t cell = 0; cell < grid.layout().storedCellCount(); cell++) {
        auto const coordinates = grid.layout().cellCoordinates(cell);
        if (!grid.layout().isInterior(coordinates)) {
            continue;
        }
        std::array<cosmictiger::Coordinate, NDIM> position{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            position[axis] = cosmictiger::Coordinate((double(coordinates[axis]) + 0.5) / double(grid.layout().globalSideLength()));
        }
        cosmictiger::ParticleCount const begin = particle;
        for (cosmictiger::ParticleCount copy = 0; copy < particlesPerCell; copy++) {
            particles.append(position, {0.1f, -0.25f, 0.5f}, particle % 5);
            particles.potential(particle) = -0.5f;
            for (size_t axis = 0; axis < NDIM; axis++) {
                particles.acceleration(axis, particle) = float(axis) - 0.75f;
            }
            particle++;
        }
        grid.setInteriorRange(coordinates, {begin, particle});
    }
    particles.append({cosmictiger::Coordinate(0.125), cosmictiger::Coordinate(0.125), cosmictiger::Coordinate(0.125)}, {9.0f, 8.0f, 7.0f}, 2);
    grid.setGhostRanges({-1, 0, 0}, {particle, particle + 1}, {0, 1});
}

void writeSample(std::string const& filename, cosmictiger::EvolutionCheckpointMetadata const& metadata) {
    PROFILE();
    cosmictiger::CartesianGrid grid(metadata.constants.gravityMeshExponent, 0, 0, static_cast<size_t>(metadata.constants.ghostWidth));
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::writeEvolutionCheckpoint(filename, particles, grid, metadata.constants, metadata.state, metadata.initial);
}

TEST_F(EvolutionCheckpointTest, RoundTripsContractingFutureClockEnergyAndOwnedParticles) {
    PROFILE();
    CheckpointDirectory directory;
    auto const filename = directory.file("restart.silo");
    auto const original = sampleMetadata();
    writeSample(filename, original);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_DOUBLE_EQ(restored.state.scaleFactor, 1.4);
    EXPECT_DOUBLE_EQ(restored.state.conformalDerivative, -0.2);
    EXPECT_DOUBLE_EQ(restored.state.tickConformalDerivative, -0.19);
    EXPECT_DOUBLE_EQ(restored.state.physicalTime, 0.35);
    EXPECT_DOUBLE_EQ(restored.state.tickPhysicalTime, 0.34);
    EXPECT_DOUBLE_EQ(restored.state.conformalOffset, 0.007);
    EXPECT_EQ(restored.state.nominalEpoch, 2);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
    EXPECT_EQ(restored.state.operationId, 93);
    EXPECT_EQ(restored.state.maximumRung, 4);
    EXPECT_TRUE(restored.state.momentaStaggered);
    EXPECT_TRUE(restored.state.energy.initialized);
    EXPECT_DOUBLE_EQ(restored.state.energy.initialEnergy, -0.2);
    EXPECT_DOUBLE_EQ(restored.state.energy.expansionWork, 0.013);
    EXPECT_DOUBLE_EQ(restored.state.energy.previousWorkRate, 0.02);
    EXPECT_DOUBLE_EQ(restored.state.energy.conservedEnergy, -0.197);
    EXPECT_DOUBLE_EQ(restored.constants.stopTimeYears, 5.8e9);
    EXPECT_DOUBLE_EQ(restored.constants.originalRungZeroTimestep, 0.25);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, 0.5);
    EXPECT_EQ(restored.constants.rungZeroExtension, 1);
    EXPECT_TRUE(restored.constants.cudaDrift);
    EXPECT_TRUE(restored.constants.cudaKick);
    EXPECT_TRUE(restored.constants.cudaMultipole);
    EXPECT_TRUE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, 0.125);
    EXPECT_EQ(restored.constants.directSeed, 271828);
    EXPECT_EQ(restored.constants.gravityMeshExponent, original.constants.gravityMeshExponent);
    EXPECT_EQ(restored.constants.ghostWidth, original.constants.ghostWidth);
    EXPECT_EQ(restored.constants.fmmOrder, original.constants.fmmOrder);
    EXPECT_EQ(restored.constants.initializationOptions.seed, original.constants.initializationOptions.seed);
    EXPECT_DOUBLE_EQ(restored.initial.scaleFactor, original.constants.initialScaleFactor);
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    cosmictiger::readParticleSilo(filename, particles, grid);
    ASSERT_EQ(particles.size(), 64);
    EXPECT_TRUE(particles.storesAcceleration());
    EXPECT_TRUE(particles.storesPotential());
    for (size_t particle = 0; particle < particles.size(); particle++) {
        EXPECT_EQ(particles.flags(particle).rung(), particle % 5);
        EXPECT_FLOAT_EQ(particles.velocity(0, particle), 0.1f);
        EXPECT_FLOAT_EQ(particles.velocity(1, particle), -0.25f);
        EXPECT_FLOAT_EQ(particles.potential(particle), -0.5f);
    }
}

TEST_F(EvolutionCheckpointTest, PreservesIndependentInitializationMeshGravityMeshAndGhostWidth) {
    constexpr size_t kInitialExponent = 3;
    constexpr size_t kGravityExponent = 2;
    constexpr cosmictiger::ParticleCount kParticlesPerCell = cosmictiger::ParticleCount(1) << NDIM;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("independent_meshes.silo");
    auto original = sampleMetadata();
    original.constants.initializationOptions.meshExponent = kInitialExponent;
    original.constants.meshSideLength = size_t(1) << kInitialExponent;
    original.constants.globalParticleCount = cosmictiger::ParticleCount(1) << (NDIM * kInitialExponent);
    original.constants.gravityMeshExponent = kGravityExponent;
    original.constants.gridSideLength = size_t(1) << kGravityExponent;
    original.constants.ghostWidth = 1;
    original.constants.softeningLength = original.constants.softeningCellFraction / double(original.constants.gridSideLength);
    original.initial.meshExponent = kInitialExponent;
    cosmictiger::CartesianGrid grid(kGravityExponent, 0, 0, original.constants.ghostWidth);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid, kParticlesPerCell);
    cosmictiger::writeEvolutionCheckpoint(filename, particles, grid, original.constants, original.state, original.initial);

    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.meshSideLength, size_t(1) << kInitialExponent);
    EXPECT_EQ(restored.constants.gridSideLength, size_t(1) << kGravityExponent);
    EXPECT_EQ(restored.constants.globalParticleCount, original.constants.globalParticleCount);
    EXPECT_EQ(restored.constants.gravityMeshExponent, kGravityExponent);
    EXPECT_EQ(restored.constants.ghostWidth, 1);
    EXPECT_EQ(restored.initial.meshExponent, kInitialExponent);
    EXPECT_DOUBLE_EQ(restored.constants.softeningLength, original.constants.softeningCellFraction / double(original.constants.gridSideLength));

    cosmictiger::CartesianGrid restoredGrid(kGravityExponent, 0, 0, restored.constants.ghostWidth);
    cosmictiger::ParticleStorage restoredParticles;
    cosmictiger::readParticleSilo(filename, restoredParticles, restoredGrid);
    EXPECT_EQ(restoredParticles.size(), original.constants.globalParticleCount);
    for (size_t cell = 0; cell < restoredGrid.layout().storedCellCount(); cell++) {
        auto const coordinates = restoredGrid.layout().cellCoordinates(cell);
        if (restoredGrid.layout().isInterior(coordinates)) {
            auto const range = restoredGrid.cell(coordinates).localParticles;
            EXPECT_EQ(range.end - range.begin, kParticlesPerCell);
        }
    }
}

TEST_F(EvolutionCheckpointTest, ChangedPhysicalEndpointRetainsNominalTimestepAndEnergyReference) {
    PROFILE();
    CheckpointDirectory directory;
    auto const originalPath = directory.file("original.silo");
    auto const extendedPath = directory.file("extended.silo");
    writeSample(originalPath, sampleMetadata());
    auto metadata = cosmictiger::readEvolutionCheckpointMetadata(originalPath);
    metadata.constants.stopTimeYears = 13.8e9;
    metadata.constants.stopPhysicalTime = 0.95;
    writeSample(extendedPath, metadata);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(extendedPath);
    EXPECT_DOUBLE_EQ(restored.constants.stopTimeYears, 13.8e9);
    EXPECT_DOUBLE_EQ(restored.constants.stopPhysicalTime, 0.95);
    EXPECT_DOUBLE_EQ(restored.constants.originalRungZeroTimestep, 0.25);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, 0.5);
    EXPECT_DOUBLE_EQ(restored.state.energy.initialEnergy, -0.2);
    EXPECT_DOUBLE_EQ(restored.state.energy.expansionWork, 0.013);
}

TEST_F(EvolutionCheckpointTest, PreservesPlummerSofteningModelAndLength) {
    PROFILE();
    CheckpointDirectory directory;
    auto const filename = directory.file("plummer.silo");
    auto original = sampleMetadata();
    original.constants.softeningModel = cosmictiger::gravity::SofteningModel::plummer;
    writeSample(filename, original);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.softeningModel, cosmictiger::gravity::SofteningModel::plummer);
    EXPECT_DOUBLE_EQ(restored.constants.softeningLength, original.constants.softeningLength);
    EXPECT_DOUBLE_EQ(restored.constants.softeningCellFraction, original.constants.softeningCellFraction);
}

TEST_F(EvolutionCheckpointTest, PreservesNondefaultFmmOrder) {
    PROFILE();
    CheckpointDirectory directory;
    auto const filename = directory.file("fmm_order.silo");
    auto original = sampleMetadata();
    original.constants.fmmOrder = cosmictiger::gravity::kMinimumFmmOrder;
    writeSample(filename, original);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.fmmOrder, cosmictiger::gravity::kMinimumFmmOrder);
    EXPECT_EQ(restored.constants.gravityMeshExponent, original.constants.gravityMeshExponent);
    EXPECT_EQ(restored.constants.ghostWidth, original.constants.ghostWidth);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionEightCheckpointWithDefaultFmmOrderAndSavedSoftening) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 8;
    constexpr auto kFlags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
        hpx::serialization::archive_flags::disable_data_chunking;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_eight.silo");
    auto original = sampleMetadata();
    original.constants.ghostWidth = 1;
    original.constants.softeningModel = cosmictiger::gravity::SofteningModel::plummer;
    original.constants.fmmOrder = cosmictiger::gravity::kMinimumFmmOrder;
    std::vector<char> payload;
    {
        hpx::serialization::output_archive archive(payload, kFlags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion8(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0, 1);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.fmmOrder, cosmictiger::gravity::kDefaultFmmOrder);
    EXPECT_EQ(restored.constants.softeningModel, cosmictiger::gravity::SofteningModel::plummer);
    EXPECT_EQ(restored.constants.ghostWidth, 1);
    EXPECT_EQ(restored.constants.gravityMeshExponent, original.constants.gravityMeshExponent);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionSevenCheckpointWithFerrersAndSavedMeshGeometry) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 7;
    constexpr auto kFlags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
        hpx::serialization::archive_flags::disable_data_chunking;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_seven.silo");
    auto original = sampleMetadata();
    original.constants.ghostWidth = 1;
    // The v7 format omits the model even when the in-memory value differs.
    original.constants.softeningModel = cosmictiger::gravity::SofteningModel::plummer;
    std::vector<char> payload;
    {
        hpx::serialization::output_archive archive(payload, kFlags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion7(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0, 1);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.fmmOrder, cosmictiger::gravity::kDefaultFmmOrder);
    EXPECT_EQ(restored.constants.softeningModel, cosmictiger::gravity::SofteningModel::ferrers);
    EXPECT_EQ(restored.constants.ghostWidth, 1);
    EXPECT_EQ(restored.constants.gravityMeshExponent, original.constants.gravityMeshExponent);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionTwoCheckpointWithCpuExecutionDefaults) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 2;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_two.silo");
    auto original = sampleMetadata();
    std::vector<char> payload;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
            hpx::serialization::archive_flags::disable_data_chunking;
        hpx::serialization::output_archive archive(payload, flags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion2(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_FALSE(restored.constants.cudaDrift);
    EXPECT_FALSE(restored.constants.cudaKick);
    EXPECT_FALSE(restored.constants.cudaMultipole);
    EXPECT_FALSE(restored.constants.lightcone.enabled);
    EXPECT_FALSE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, 0.0);
    EXPECT_EQ(restored.constants.directSeed, 42);
    EXPECT_EQ(restored.constants.gravityMeshExponent, original.constants.initializationOptions.meshExponent);
    EXPECT_EQ(restored.constants.ghostWidth, 2);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, original.constants.rungZeroTimestep);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionThreeCheckpointWithMultipoleDefaultOff) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 3;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_three.silo");
    auto original = sampleMetadata();
    std::vector<char> payload;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
            hpx::serialization::archive_flags::disable_data_chunking;
        hpx::serialization::output_archive archive(payload, flags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion3(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_TRUE(restored.constants.cudaDrift);
    EXPECT_TRUE(restored.constants.cudaKick);
    EXPECT_FALSE(restored.constants.cudaMultipole);
    EXPECT_FALSE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, 0.0);
    EXPECT_EQ(restored.constants.directSeed, 42);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, original.constants.rungZeroTimestep);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionFourCheckpointWithLightconeDefaultOff) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 4;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_four.silo");
    auto original = sampleMetadata();
    std::vector<char> payload;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
            hpx::serialization::archive_flags::disable_data_chunking;
        hpx::serialization::output_archive archive(payload, flags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion4(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_TRUE(restored.constants.cudaDrift);
    EXPECT_TRUE(restored.constants.cudaKick);
    EXPECT_TRUE(restored.constants.cudaMultipole);
    EXPECT_FALSE(restored.constants.lightcone.enabled);
    EXPECT_FALSE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, 0.0);
    EXPECT_EQ(restored.constants.directSeed, 42);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, original.constants.rungZeroTimestep);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, ReadsVersionFiveCheckpointWithDirectComparisonDefaultOff) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 5;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_five.silo");
    auto original = sampleMetadata();
    original.constants.lightcone.enabled = true;
    original.constants.lightcone.nside = 128;
    original.constants.lightconeObservationConformalTime = 2.75;
    original.constants.lightconeSpeed = 29.9792458;
    original.constants.lightconeRunId = "legacy-run";
    original.constants.initializationOptions.seed = 17;
    original.initial.seed = 17;
    std::vector<char> payload;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
            hpx::serialization::archive_flags::disable_data_chunking;
        hpx::serialization::output_archive archive(payload, flags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion5(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_TRUE(restored.constants.cudaDrift);
    EXPECT_TRUE(restored.constants.cudaKick);
    EXPECT_TRUE(restored.constants.cudaMultipole);
    EXPECT_TRUE(restored.constants.lightcone.enabled);
    EXPECT_EQ(restored.constants.lightcone.nside, 128);
    EXPECT_DOUBLE_EQ(restored.constants.lightconeObservationConformalTime, 2.75);
    EXPECT_EQ(restored.constants.lightconeRunId, "legacy-run");
    EXPECT_FALSE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, 0.0);
    EXPECT_EQ(restored.constants.directSeed, 42);
    EXPECT_EQ(restored.constants.initializationOptions.seed, 17);
    EXPECT_EQ(restored.initial.seed, 17);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, original.constants.rungZeroTimestep);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, PreservesLightconeObserverAndOutputSettings) {
    PROFILE();
    CheckpointDirectory directory;
    auto const filename = directory.file("lightcone.silo");
    auto original = sampleMetadata();
    original.constants.lightcone.enabled = true;
    original.constants.lightcone.nside = 128;
    original.constants.lightcone.maximumRadius = 2.5;
    original.constants.lightcone.observer = {0.25, 0.5, 0.75};
    original.constants.lightcone.observationTimeYears = 1.0e9;
    original.constants.lightcone.outputPrefix = "observer";
    original.constants.lightconeObservationConformalTime = 2.75;
    original.constants.lightconeSpeed = 29.9792458;
    original.constants.lightconeRunId = "test-run";
    writeSample(filename, original);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_TRUE(restored.constants.lightcone.enabled);
    EXPECT_EQ(restored.constants.lightcone.nside, 128);
    EXPECT_DOUBLE_EQ(restored.constants.lightcone.maximumRadius, 2.5);
    EXPECT_EQ(restored.constants.lightcone.observer, original.constants.lightcone.observer);
    EXPECT_DOUBLE_EQ(restored.constants.lightcone.observationTimeYears, 1.0e9);
    EXPECT_EQ(restored.constants.lightcone.outputPrefix, "observer");
    EXPECT_DOUBLE_EQ(restored.constants.lightconeObservationConformalTime, 2.75);
    EXPECT_DOUBLE_EQ(restored.constants.lightconeSpeed, 29.9792458);
    EXPECT_EQ(restored.constants.lightconeRunId, "test-run");
}

TEST_F(EvolutionCheckpointTest, ReadsVersionSixCheckpointWithOriginalMeshAndDefaultGhostWidth) {
    constexpr std::uint64_t kMagic = 0x435445564F4C3031ULL;
    constexpr std::uint32_t kVersion = 6;
    constexpr size_t kInitialExponent = 3;
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("version_six.silo");
    auto original = sampleMetadata();
    original.constants.initializationOptions.meshExponent = kInitialExponent;
    original.constants.meshSideLength = size_t(1) << kInitialExponent;
    original.constants.gridSideLength = original.constants.meshSideLength;
    original.constants.globalParticleCount = cosmictiger::ParticleCount(1) << (NDIM * kInitialExponent);
    original.constants.softeningLength = original.constants.softeningCellFraction / double(original.constants.gridSideLength);
    original.initial.meshExponent = kInitialExponent;
    std::vector<char> payload;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization |
            hpx::serialization::archive_flags::disable_data_chunking;
        hpx::serialization::output_archive archive(payload, flags);
        archive << kMagic << kVersion;
        cosmictiger::serializeGlobalConstantsVersion6(archive, original.constants);
        archive << original.state;
    }
    cosmictiger::CartesianGrid grid(kInitialExponent, 0, 0, original.constants.ghostWidth);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(filename, particles, grid, original.initial, payload, scalars);

    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_EQ(restored.constants.gravityMeshExponent, kInitialExponent);
    EXPECT_EQ(restored.constants.gridSideLength, restored.constants.meshSideLength);
    EXPECT_EQ(restored.constants.ghostWidth, 2);
    EXPECT_TRUE(restored.constants.cudaDirect);
    EXPECT_DOUBLE_EQ(restored.constants.directSinkFraction, original.constants.directSinkFraction);
    EXPECT_EQ(restored.constants.directSeed, original.constants.directSeed);
    EXPECT_EQ(restored.state.timeTick, original.state.timeTick);
}

TEST_F(EvolutionCheckpointTest, NativeSiloTimeDescribesCurrentEpochAndOriginalConformalOrigin) {
    constexpr double kConformalTime = 1.03825;
    constexpr std::array<char const*, 4> kNames{
        "evolution/scale_factor", "evolution/conformal_derivative", "evolution/years_from_present", "evolution/conformal_time"};
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("native_time.silo");
    auto const original = sampleMetadata();
    writeSample(filename, original);
    SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_TRUE(file);
    // Two extended nominal intervals, one sixteenth of an interval, and
    // a partial drift: 2*0.5 + (1/16)*0.5 + 0.007 = 1.03825 H0^-1.
    std::array<double, 4> const expected{
        1.4, -0.2, cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), kConformalTime};
    for (size_t index = 0; index < kNames.size(); index++) {
        ASSERT_EQ(DBGetVarType(file.get(), kNames[index]), DB_DOUBLE);
        ASSERT_EQ(DBGetVarLength(file.get(), kNames[index]), 1);
        double actual = 0.0;
        ASSERT_EQ(DBReadVar(file.get(), kNames[index], &actual), 0);
        EXPECT_DOUBLE_EQ(actual, expected[index]);
    }
    PointMesh mesh(DBGetPointmesh(file.get(), "block_0/mesh"), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
    ASSERT_TRUE(mesh);
    EXPECT_DOUBLE_EQ(mesh->dtime, kConformalTime);
    ASSERT_EQ(DBGetComponentType(file.get(), "particles", "dtime"), DB_DOUBLE);
    SiloComponent multiTime(DBGetComponent(file.get(), "particles", "dtime"), hpx::annotated_function(&std::free, "SiloComponent::free"));
    ASSERT_TRUE(multiTime);
    EXPECT_DOUBLE_EQ(*static_cast<double const*>(multiTime.get()), kConformalTime);
}

TEST_F(EvolutionCheckpointTest, InvalidReplacementPreservesPreviousCheckpoint) {
    constexpr std::array kInvalidFmmOrders{cosmictiger::gravity::kMinimumFmmOrder - 1, cosmictiger::gravity::kMaximumFmmOrder + 1};
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("preserved.silo");
    auto original = sampleMetadata();
    writeSample(filename, original);
    original.state.timeTick = cosmictiger::gravity::rungTicks(0);
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    original = sampleMetadata();
    original.constants.rungZeroTimestep *= 1.5;
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    original = sampleMetadata();
    original.constants.globalParticleCount *= 2;
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    original = sampleMetadata();
    original.constants.softeningModel = static_cast<cosmictiger::gravity::SofteningModel>(2);
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    original = sampleMetadata();
    original.constants.softeningLength *= 2.0;
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    for (int const order : kInvalidFmmOrders) {
        original = sampleMetadata();
        original.constants.fmmOrder = order;
        EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    }
    original = sampleMetadata();
    original.state.physicalTime = original.constants.stopPhysicalTime + 1.0;
    EXPECT_THROW(writeSample(filename, original), std::runtime_error);
    auto const restored = cosmictiger::readEvolutionCheckpointMetadata(filename);
    EXPECT_DOUBLE_EQ(restored.state.physicalTime, 0.35);
    EXPECT_DOUBLE_EQ(restored.constants.rungZeroTimestep, 0.5);
}

TEST_F(EvolutionCheckpointTest, RejectsMalformedStoredStateAndGravityOrderAndTruncatedArchive) {
    constexpr std::array kInvalidFmmOrders{cosmictiger::gravity::kMinimumFmmOrder - 1, cosmictiger::gravity::kMaximumFmmOrder + 1};
    PROFILE();

    CheckpointDirectory directory;
    auto const validPath = directory.file("valid.silo");
    auto const invalidPath = directory.file("invalid.silo");
    auto const truncatedPath = directory.file("truncated.silo");
    auto const original = sampleMetadata();
    writeSample(validPath, original);
    auto bytes = cosmictiger::detail::readParticleSiloEvolutionPayload(validPath);
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    cosmictiger::GlobalConstantsData constants;
    cosmictiger::EvolutionState state;
    {
        hpx::serialization::input_archive archive(bytes, bytes.size());
        archive >> magic >> version >> constants >> state;
    }
    state.timeTick = cosmictiger::gravity::rungTicks(0);
    std::vector<char> invalid;
    {
        auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization;
        hpx::serialization::output_archive archive(invalid, flags);
        archive << magic << version << constants << state;
    }
    cosmictiger::CartesianGrid grid(2, 0, 0);
    cosmictiger::ParticleStorage particles;
    populateParticles(particles, grid);
    cosmictiger::detail::EvolutionSiloScalars const scalars{original.state.scaleFactor, original.state.conformalDerivative,
        cosmictiger::physicalTimeInYears(original.state.physicalTime, original.constants.cosmology), 1.03825};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(invalidPath, particles, grid, original.initial, invalid, scalars);
    EXPECT_THROW(cosmictiger::readEvolutionCheckpointMetadata(invalidPath), std::runtime_error);
    state = original.state;
    for (int const order : kInvalidFmmOrders) {
        constants.fmmOrder = order;
        invalid.clear();
        {
            auto const flags = hpx::serialization::archive_flags::endian_little | hpx::serialization::archive_flags::disable_array_optimization;
            hpx::serialization::output_archive archive(invalid, flags);
            archive << magic << version << constants << state;
        }
        cosmictiger::detail::writeParticleSiloEvolutionPayload(invalidPath, particles, grid, original.initial, invalid, scalars);
        EXPECT_THROW(cosmictiger::readEvolutionCheckpointMetadata(invalidPath), std::runtime_error);
    }
    bytes.resize(bytes.size() / 2);
    cosmictiger::detail::writeParticleSiloEvolutionPayload(truncatedPath, particles, grid, original.initial, bytes, scalars);
    EXPECT_THROW(cosmictiger::readEvolutionCheckpointMetadata(truncatedPath), std::exception);
}

TEST_F(EvolutionCheckpointTest, RejectsLocalityCountExceedingEitherFftSideLength) {
    constexpr size_t kSmallerExponent = 2;
    constexpr size_t kLargerExponent = 3;
    constexpr size_t kLocalityExponent = 1;
    constexpr size_t kLocalityCount = size_t(1) << (NDIM * kLocalityExponent);
    constexpr std::array kMeshExponents{std::array{kSmallerExponent, kLargerExponent}, std::array{kLargerExponent, kSmallerExponent}};
    PROFILE();

    CheckpointDirectory directory;
    auto const filename = directory.file("invalid_fft_decomposition.silo");
    for (auto const& [initialExponent, gravityExponent] : kMeshExponents) {
        auto metadata = sampleMetadata();
        metadata.constants.initializationOptions.meshExponent = initialExponent;
        metadata.initial.meshExponent = initialExponent;
        metadata.constants.meshSideLength = size_t(1) << initialExponent;
        metadata.constants.globalParticleCount = cosmictiger::ParticleCount(1) << (NDIM * initialExponent);
        metadata.constants.gravityMeshExponent = gravityExponent;
        metadata.constants.gridSideLength = size_t(1) << gravityExponent;
        metadata.constants.localityCount = kLocalityCount;
        metadata.constants.softeningLength = metadata.constants.softeningCellFraction / double(metadata.constants.gridSideLength);
        cosmictiger::CartesianGrid grid(gravityExponent, kLocalityExponent, 0, metadata.constants.ghostWidth);
        cosmictiger::ParticleStorage particles;
        populateParticles(particles, grid);
        EXPECT_THROW(
            cosmictiger::writeEvolutionCheckpoint(filename, particles, grid, metadata.constants, metadata.state, metadata.initial), std::runtime_error);
    }
}

}    // namespace
