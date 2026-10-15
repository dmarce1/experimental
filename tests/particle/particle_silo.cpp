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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_silo.hpp>
#include <cosmictiger/simulation_time.hpp>

#include <gtest/gtest.h>
#include <hpx/bit.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <unistd.h>

namespace {

using SiloFile = std::unique_ptr<DBfile, std::decay_t<decltype(hpx::annotated_function(&DBClose, "SiloFile::close"))>>;
using SiloMultimesh = std::unique_ptr<DBmultimesh, std::decay_t<decltype(hpx::annotated_function(&DBFreeMultimesh, "SiloMultimesh::free"))>>;
using SiloMultivar = std::unique_ptr<DBmultivar, std::decay_t<decltype(hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"))>>;
using PointMesh = std::unique_ptr<DBpointmesh, std::decay_t<decltype(hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"))>>;
using PointVar = std::unique_ptr<DBmeshvar, std::decay_t<decltype(hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"))>>;

using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleStorage;

class ParticleSiloTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        int deviceCount = 0;
        try {
            deviceCount = cosmictiger::cuda::deviceCount().get();
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA device unavailable: " << error.what();
            }
            throw;
        }
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        int const device = cosmictiger::cuda::device().get();
        auto const properties = cosmictiger::cuda::deviceProperties(device).get();
        if (!properties.managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
#endif
    }
};

class TestDirectory {
public:
    TestDirectory() {
        PROFILE();
        char const* const root = std::getenv("TMPDIR");
        std::string pathTemplate = root != nullptr && root[0] != '\0' ? root : "/tmp";
        if (pathTemplate.back() != '/') {
            pathTemplate += '/';
        }
        pathTemplate += "cosmictiger_particle_silo_XXXXXX";
        if (::mkdtemp(pathTemplate.data()) == nullptr) {
            throw std::runtime_error("could not create particle Silo test directory_");
        }
        directory_ = pathTemplate;
    }

    TestDirectory(TestDirectory const&) = delete;
    TestDirectory& operator=(TestDirectory const&) = delete;

    ~TestDirectory() {
        PROFILE();
        for (auto const& path : paths_) {
            (void) std::remove(path.c_str());
        }
        (void) ::rmdir(directory_.c_str());
    }

    std::string file(std::string const& name) {
        PROFILE();
        paths_.push_back(directory_ + '/' + name);
        return paths_.back();
    }

private:
    std::string directory_;
    std::vector<std::string> paths_;
};

void seedOwnedAndGhostParticles(ParticleStorage& particles, CartesianGrid& grid) {
    PROFILE();
    particles.append({Coordinate::fromRaw(1), Coordinate::fromRaw(0x20000000), Coordinate::fromRaw(0x3fffffff)}, {-0.0f, 1.25f, -2.5f}, 0);
    particles.append({Coordinate::fromRaw(0x40000001), Coordinate::fromRaw(1), Coordinate::fromRaw(0x20000001)}, {3.5f, -4.25f, 0.0f}, 63);
    // The reserved bit is internal state and must not survive a Silo restart.
    particles.flags(1) = hpx::bit_cast<cosmictiger::Rung>(std::uint8_t(0x80 | 63));
    particles.append({Coordinate::fromRaw(0x80000001), Coordinate::fromRaw(0x10000000), Coordinate::fromRaw(0x40000000)}, {-5.75f, 6.5f, 7.25f}, 17);

    // This periodic copy belongs to the same locality, so isGhost() is false.
    // Its ghost cell placement, rather than its flag, must exclude it.
    particles.append({Coordinate::fromRaw(0xf0000000), Coordinate::fromRaw(0x10000000), Coordinate::fromRaw(0x10000000)}, {101.0f, 102.0f, 103.0f}, 29);
    particles.append({Coordinate::fromRaw(0x10000000), Coordinate::fromRaw(0x10000000), Coordinate::fromRaw(0x10000000)}, {201.0f, 202.0f, 203.0f}, 31, true);

    std::int64_t const side = static_cast<std::int64_t>(grid.layout().interiorSideLength());
    ParticleCount cursor = 0;
    for (std::int64_t z = 0; z < side; z++) {
        for (std::int64_t y = 0; y < side; y++) {
            for (std::int64_t x = 0; x < side; x++) {
                ParticleCount count = 0;
                for (ParticleCount particle = 0; particle < 3; particle++) {
                    std::array<Coordinate, NDIM> position{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        position[axis] = particles.position(axis, particle);
                    }
                    if (grid.layout().particleOwner(position).coordinates == GridIndices{x, y, z}) {
                        count++;
                    }
                }
                grid.setInteriorRange({x, y, z}, {cursor, cursor + count});
                cursor += count;
            }
        }
    }
    grid.setGhostRanges({-1, 0, 0}, {3, 4}, {2, 3});
    grid.setGhostRanges({side, 0, 0}, {4, 5}, {0, 1});
}

void writeLegacyFlags(std::string const& path, int version, std::array<std::uint8_t, 3> const& packedFlags) {
    constexpr int kExtent = 1;
    PROFILE();

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_APPEND), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(DBWrite(file.get(), "particle_format_version", &version, &kExtent, 1, DB_INT), 0);
    ASSERT_EQ(DBSetDir(file.get(), "block_0"), 0);
    std::array<char, 3> flags{};
    for (size_t index = 0; index < flags.size(); index++) {
        flags[index] = hpx::bit_cast<char>(packedFlags[index]);
    }
    ASSERT_EQ(DBPutPointvar1(file.get(), "flags", "mesh", flags.data(), static_cast<int>(flags.size()), DB_CHAR, nullptr), 0);
}

void expectAllRangesEmpty(CartesianGrid const& grid) {
    PROFILE();
    for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
        auto const coordinates = grid.layout().cellCoordinates(cellIndex);
        auto const& cell = grid.cell(coordinates);
        EXPECT_EQ(cell.localParticles.begin, 0);
        EXPECT_EQ(cell.localParticles.end, 0);
        EXPECT_EQ(cell.ownerParticles.begin, 0);
        EXPECT_EQ(cell.ownerParticles.end, 0);
    }
}

cosmictiger::InitialConditionsSiloMetadata initialConditionsMetadata() {
    PROFILE();
    cosmictiger::CosmologyParameters cosmology;
    cosmology.hubble = 70.1;
    cosmology.omegaBaryonH2 = 0.023;
    cosmology.omegaCdmH2 = 0.111;
    cosmology.omegaNeutrinoH2 = 0.0013;
    cosmology.omegaCurvature = 0.001;
    cosmology.massiveNeutrinos = 2;
    cosmology.masslessNeutrinos = 1;
    cosmology.effectiveNeutrinos = 3.12;
    cosmology.scalarAmplitude = 2.17e-9;
    cosmology.spectralIndex = 0.971;
    cosmology.spectralRunning = 0.004;
    cosmology.pivotMpcInverse = 0.06;
    cosmology.darkEnergyW = -0.97;
    cosmology.darkEnergyWa = 0.02;
    cosmology.temperatureCmb = 2.731;
    cosmology.heliumFraction = 0.246;
    cosmology.opticalDepth = 0.058;
    return {cosmology, 0.021, 85.0, 2.34e10, 0.10008, 0.1, 0xfedcba9876543210ULL, 2, cosmictiger::MatterComponent::cdmAndBaryons};
}

TEST_F(ParticleSiloTest, InitialConditionsMetadataRoundTripsCosmologyUnitsAndOwnedParticles) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("initial_conditions.silo");
    CartesianGrid sourceGrid(2, 0, 0);
    ParticleStorage source;
    seedOwnedAndGhostParticles(source, sourceGrid);
    auto const expected = initialConditionsMetadata();
    cosmictiger::writeParticleSilo(path, source, sourceGrid, expected);

    auto const restored = cosmictiger::readInitialConditionsSiloMetadata(path);
    EXPECT_DOUBLE_EQ(restored.scaleFactor, expected.scaleFactor);
    EXPECT_DOUBLE_EQ(restored.boxLength, expected.boxLength);
    EXPECT_DOUBLE_EQ(restored.particleMass, expected.particleMass);
    EXPECT_DOUBLE_EQ(restored.maximumDensityDeviation, expected.maximumDensityDeviation);
    EXPECT_DOUBLE_EQ(restored.requestedDensityDeviation, expected.requestedDensityDeviation);
    EXPECT_EQ(restored.seed, expected.seed);
    EXPECT_EQ(restored.meshExponent, expected.meshExponent);
    EXPECT_EQ(restored.component, expected.component);
    EXPECT_DOUBLE_EQ(restored.cosmology.hubble, expected.cosmology.hubble);
    EXPECT_DOUBLE_EQ(restored.cosmology.omegaBaryonH2, expected.cosmology.omegaBaryonH2);
    EXPECT_DOUBLE_EQ(restored.cosmology.omegaCdmH2, expected.cosmology.omegaCdmH2);
    EXPECT_DOUBLE_EQ(restored.cosmology.omegaNeutrinoH2, expected.cosmology.omegaNeutrinoH2);
    EXPECT_DOUBLE_EQ(restored.cosmology.omegaCurvature, expected.cosmology.omegaCurvature);
    EXPECT_EQ(restored.cosmology.massiveNeutrinos, expected.cosmology.massiveNeutrinos);
    EXPECT_EQ(restored.cosmology.masslessNeutrinos, expected.cosmology.masslessNeutrinos);
    EXPECT_DOUBLE_EQ(restored.cosmology.effectiveNeutrinos, expected.cosmology.effectiveNeutrinos);
    EXPECT_DOUBLE_EQ(restored.cosmology.scalarAmplitude, expected.cosmology.scalarAmplitude);
    EXPECT_DOUBLE_EQ(restored.cosmology.spectralIndex, expected.cosmology.spectralIndex);
    EXPECT_DOUBLE_EQ(restored.cosmology.spectralRunning, expected.cosmology.spectralRunning);
    EXPECT_DOUBLE_EQ(restored.cosmology.pivotMpcInverse, expected.cosmology.pivotMpcInverse);
    EXPECT_DOUBLE_EQ(restored.cosmology.darkEnergyW, expected.cosmology.darkEnergyW);
    EXPECT_DOUBLE_EQ(restored.cosmology.darkEnergyWa, expected.cosmology.darkEnergyWa);
    EXPECT_DOUBLE_EQ(restored.cosmology.temperatureCmb, expected.cosmology.temperatureCmb);
    EXPECT_DOUBLE_EQ(restored.cosmology.heliumFraction, expected.cosmology.heliumFraction);
    EXPECT_DOUBLE_EQ(restored.cosmology.opticalDepth, expected.cosmology.opticalDepth);

    CartesianGrid restoredGrid(2, 0, 0);
    ParticleStorage restoredParticles;
    cosmictiger::readParticleSilo(path, restoredParticles, restoredGrid);
    ASSERT_EQ(restoredParticles.size(), 3);
    for (ParticleCount particle = 0; particle < restoredParticles.size(); particle++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(restoredParticles.position(axis, particle).raw(), source.position(axis, particle).raw());
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(restoredParticles.velocity(axis, particle)), hpx::bit_cast<std::uint32_t>(source.velocity(axis, particle)));
        }
        EXPECT_EQ(hpx::bit_cast<std::uint8_t>(restoredParticles.flags(particle)), source.flags(particle).rung());
        EXPECT_FALSE(restoredParticles.flags(particle).isGhost());
    }

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(DBSetDir(file.get(), "initial_conditions"), 0);
    auto expectText = [&](char const* name, std::string const& text) {
        PROFILE();
        ASSERT_EQ(DBGetVarType(file.get(), name), DB_CHAR);
        ASSERT_EQ(DBGetVarLength(file.get(), name), text.size() + 1);
        std::vector<char> actual(text.size() + 1);
        ASSERT_EQ(DBReadVar(file.get(), name, actual.data()), 0);
        EXPECT_EQ(actual.back(), '\0');
        EXPECT_EQ(std::string(actual.data(), actual.size() - 1), text);
    };
    expectText("position_units", "unit_box");
    expectText("velocity_convention", "d(x/L)/d ln(a)");
    expectText("box_length_units", "Mpc/h");
    expectText("particle_mass_units", "Msun/h");
}

TEST_F(ParticleSiloTest, InitializationMetadataCanDescribeALargerParticleLatticeThanTheStoredGrid) {
    constexpr size_t kInitialExponent = 3;
    constexpr size_t kGravityExponent = 2;
    PROFILE();

    TestDirectory temporary;
    std::string const path = temporary.file("independent_initial_mesh.silo");
    CartesianGrid sourceGrid(kGravityExponent, 0, 0);
    ParticleStorage source;
    seedOwnedAndGhostParticles(source, sourceGrid);
    auto metadata = initialConditionsMetadata();
    metadata.meshExponent = kInitialExponent;
    cosmictiger::writeParticleSilo(path, source, sourceGrid, metadata);

    auto const restoredMetadata = cosmictiger::readInitialConditionsSiloMetadata(path);
    EXPECT_EQ(restoredMetadata.meshExponent, kInitialExponent);
    CartesianGrid restoredGrid(kGravityExponent, 0, 0);
    ParticleStorage restoredParticles;
    cosmictiger::readParticleSilo(path, restoredParticles, restoredGrid);
    ASSERT_EQ(restoredParticles.size(), 3);
    EXPECT_EQ(restoredGrid.layout().globalSideLength(), size_t(1) << kGravityExponent);
    for (ParticleCount particle = 0; particle < restoredParticles.size(); particle++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(restoredParticles.position(axis, particle).raw(), source.position(axis, particle).raw());
        }
    }
}

TEST_F(ParticleSiloTest, OrdinaryCheckpointHasNoInitialConditionsMetadata) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("ordinary_checkpoint.silo");
    CartesianGrid grid(1, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    cosmictiger::writeParticleSilo(path, particles, grid);
    EXPECT_THROW(cosmictiger::readInitialConditionsSiloMetadata(path), std::runtime_error);

    CartesianGrid restoredGrid(1, 0, 0);
    ParticleStorage restored;
    EXPECT_NO_THROW(cosmictiger::readParticleSilo(path, restored, restoredGrid));
    EXPECT_EQ(restored.size(), 3);
}

TEST_F(ParticleSiloTest, GravityOutputOptionsIndependentlyControlFieldsAndRoundTripOwnedValues) {
    PROFILE();
    TestDirectory temporary;
    for (int configuration = 0; configuration < 4; configuration++) {
        bool const acceleration = (configuration & 1) != 0;
        bool const potential = (configuration & 2) != 0;
        std::string const path = temporary.file("gravity_" + std::to_string(configuration) + ".silo");
        CartesianGrid sourceGrid(2, 0, 0);
        ParticleStorage source;
        source.configureGravityFields(acceleration, potential);
        seedOwnedAndGhostParticles(source, sourceGrid);
        for (ParticleCount index = 0; index < source.size(); index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                if (acceleration) {
                    source.acceleration(axis, index) = float(index * NDIM + axis) + 0.25f;
                }
            }
            if (potential) {
                source.potential(index) = -float(index) - 0.5f;
            }
        }
        cosmictiger::writeParticleSilo(path, source, sourceGrid);
        DBfile* const file = DBOpen(path.c_str(), DB_UNKNOWN, DB_READ);
        ASSERT_NE(file, nullptr);
        EXPECT_EQ(DBInqVarExists(file, "acceleration"), acceleration ? 1 : 0);
        EXPECT_EQ(DBInqVarExists(file, "potential"), potential ? 1 : 0);
        EXPECT_EQ(DBClose(file), 0);

        CartesianGrid restoredGrid(2, 0, 0);
        ParticleStorage restored;
        restored.configureGravityFields(!acceleration, !potential);
        cosmictiger::readParticleSilo(path, restored, restoredGrid);
        ASSERT_EQ(restored.size(), 3);
        EXPECT_EQ(restored.storesAcceleration(), acceleration);
        EXPECT_EQ(restored.storesPotential(), potential);
        for (ParticleCount index = 0; index < restored.size(); index++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                if (acceleration) {
                    EXPECT_EQ(restored.acceleration(axis, index), source.acceleration(axis, index));
                } else {
                    EXPECT_EQ(restored.accelerationData(axis), nullptr);
                }
            }
            if (potential) {
                EXPECT_EQ(restored.potential(index), source.potential(index));
            } else {
                EXPECT_EQ(restored.potentialData(), nullptr);
            }
        }
    }
}

TEST_F(ParticleSiloTest, EmptyCheckpointPreservesGravityFieldConfiguration) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("empty_gravity.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage source;
    source.configureGravityFields(true, true);
    cosmictiger::writeParticleSilo(path, source, grid);
    ParticleStorage restored;
    cosmictiger::readParticleSilo(path, restored, grid);
    EXPECT_EQ(restored.size(), 0);
    EXPECT_TRUE(restored.storesAcceleration());
    EXPECT_TRUE(restored.storesPotential());
    restored.resize(1);
    EXPECT_EQ(restored.acceleration(0, 0), 0.0f);
    EXPECT_EQ(restored.potential(0), 0.0f);
}

TEST_F(ParticleSiloTest, CanonicalMomentumConventionRoundTripsWithConformalTimeUnits) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("canonical_momentum.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    auto metadata = initialConditionsMetadata();
    metadata.canonicalMomentum = true;
    cosmictiger::writeParticleSilo(path, particles, grid, metadata);
    auto const restored = cosmictiger::readInitialConditionsSiloMetadata(path);
    EXPECT_TRUE(restored.canonicalMomentum);
    EXPECT_EQ(restored.scaleFactor, metadata.scaleFactor);
}

TEST_F(ParticleSiloTest, UnitViewsUseCurrentEpochSynchronizedMomentaAndDoubleCgsWithoutChangingRestart) {
    constexpr ParticleCount kOwnedCount = 3;
    constexpr double kScaleFactor = 0.25;
    constexpr double kLengthMpc = 200.0;
    constexpr double kMegaparsecCm = 3.085677581e24;
    constexpr double kSolarMassGrams = 1.98847e33;
    constexpr double kVelocityKmPerSecond = 10000.0;
    constexpr double kVelocityCmPerSecond = 1.0e9;
    constexpr double kParticleMassSolar = 4.0e10;
    constexpr double kGravitationalConstantCgs = 6.67430e-8;
    constexpr double kGravityParticleMass =
        kParticleMassSolar * kSolarMassGrams * kGravitationalConstantCgs / (kLengthMpc * kMegaparsecCm * kVelocityCmPerSecond * kVelocityCmPerSecond);
    constexpr double kSecondsPerYear = 31557600.0;
    constexpr double kHubblePerSecond = kVelocityCmPerSecond / (kLengthMpc * kMegaparsecCm);
    constexpr double kRelativeTolerance = 2.0e-14;
    PROFILE();

    TestDirectory temporary;
    auto const path = temporary.file("unit_views.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    seedOwnedAndGhostParticles(particles, grid);
    for (ParticleCount index = 0; index < particles.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.acceleration(axis, index) = float(index + axis) + 0.5f;
        }
        particles.potential(index) = -float(index) - 1.5f;
    }
    auto metadata = initialConditionsMetadata();
    metadata.cosmology.hubble = 50.0;
    metadata.boxLength = 100.0;
    // Evolution output must use its authoritative gravitational mass, even
    // when historical initialization mass metadata uses a different scale.
    metadata.particleMass = 7.0e9;
    metadata.canonicalMomentum = true;
    std::array<std::array<double, kOwnedCount>, NDIM> const momenta{{{0.0, 0.25, -0.5}, {1.0, -2.0, 3.0}, {0.5, 0.0, -0.75}}};
    cosmictiger::ParticleSiloOutput output{};
    output.particleCount = kOwnedCount;
    for (size_t axis = 0; axis < NDIM; axis++) {
        output.synchronizedMomenta[axis] = momenta[axis].data();
    }
    cosmictiger::detail::EvolutionSiloScalars const epoch{kScaleFactor, 0.3, -1.0e9, 0.125, kGravityParticleMass};
    std::vector<char> const payload{'x'};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(path, particles, grid, metadata, payload, epoch, &output);

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    double savedScale = 0.0;
    ASSERT_EQ(DBReadVar(file.get(), "output_units/scale_factor", &savedScale), 0);
    EXPECT_EQ(savedScale, kScaleFactor);
    double physicalTime = 0.0;
    double conformalTime = 0.0;
    ASSERT_EQ(DBReadVar(file.get(), "output_units/physical_time_s", &physicalTime), 0);
    ASSERT_EQ(DBReadVar(file.get(), "output_units/conformal_time_s", &conformalTime), 0);
    EXPECT_DOUBLE_EQ(physicalTime, epoch.yearsFromPresent * kSecondsPerYear);
    EXPECT_NEAR(conformalTime, epoch.conformalTime / kHubblePerSecond, kRelativeTolerance * std::abs(conformalTime));
    for (bool cgs : {false, true}) {
        auto const directory = std::string(cgs ? "cgs" : "comoving") + "/block_0/";
        PointMesh mesh(DBGetPointmesh(file.get(), (directory + "mesh").c_str()), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
        PointVar velocity(DBGetPointvar(file.get(), (directory + "velocity").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar acceleration(DBGetPointvar(file.get(), (directory + "acceleration").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar potential(DBGetPointvar(file.get(), (directory + "potential").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar mass(DBGetPointvar(file.get(), (directory + "mass").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar rung(DBGetPointvar(file.get(), (directory + "rung").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        ASSERT_NE(mesh, nullptr);
        ASSERT_NE(velocity, nullptr);
        ASSERT_NE(acceleration, nullptr);
        ASSERT_NE(potential, nullptr);
        ASSERT_NE(mass, nullptr);
        ASSERT_NE(rung, nullptr);
        ASSERT_EQ(mesh->nels, kOwnedCount);
        EXPECT_EQ(mesh->datatype, DB_DOUBLE);
        EXPECT_EQ(mesh->dtime, epoch.conformalTime);
        for (auto const* field : {velocity.get(), acceleration.get(), potential.get(), mass.get()}) {
            EXPECT_EQ(field->datatype, DB_DOUBLE);
            EXPECT_EQ(field->nels, kOwnedCount);
        }
        EXPECT_STREQ(velocity->units, cgs ? "cm/s" : "km/s");
        EXPECT_STREQ(acceleration->units, cgs ? "cm/s^2" : "(km/s)^2/Mpc");
        EXPECT_STREQ(potential->units, cgs ? "erg/g" : "(km/s)^2");
        EXPECT_STREQ(mass->units, cgs ? "g" : "Msun");
        double const lengthFactor = kLengthMpc * (cgs ? kScaleFactor * kMegaparsecCm : 1.0);
        double const speedFactor = (cgs ? kVelocityCmPerSecond : kVelocityKmPerSecond) / kScaleFactor;
        double const accelerationFactor = cgs ? kVelocityCmPerSecond * kVelocityCmPerSecond / (kLengthMpc * kMegaparsecCm * kScaleFactor * kScaleFactor) :
                                                kVelocityKmPerSecond * kVelocityKmPerSecond / kLengthMpc;
        double const potentialFactor = cgs ? kVelocityCmPerSecond * kVelocityCmPerSecond / kScaleFactor : kVelocityKmPerSecond * kVelocityKmPerSecond;
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_STREQ(mesh->units[axis], cgs ? "cm" : "Mpc");
            for (ParticleCount index = 0; index < kOwnedCount; index++) {
                double const expectedPosition = double(particles.position(axis, index)) * lengthFactor;
                EXPECT_NEAR(static_cast<double const*>(mesh->coords[axis])[index], expectedPosition, kRelativeTolerance * std::abs(expectedPosition));
                double const expectedVelocity = momenta[axis][index] * speedFactor;
                EXPECT_NEAR(static_cast<double const*>(velocity->vals[axis])[index], expectedVelocity, kRelativeTolerance * std::abs(expectedVelocity));
                double const expectedAcceleration = particles.acceleration(axis, index) * accelerationFactor;
                EXPECT_NEAR(
                    static_cast<double const*>(acceleration->vals[axis])[index], expectedAcceleration, kRelativeTolerance * std::abs(expectedAcceleration));
            }
        }
        for (ParticleCount index = 0; index < kOwnedCount; index++) {
            double const expectedPotential = particles.potential(index) * potentialFactor;
            EXPECT_NEAR(static_cast<double const*>(potential->vals[0])[index], expectedPotential, kRelativeTolerance * std::abs(expectedPotential));
            double const expectedMass = kParticleMassSolar * (cgs ? kSolarMassGrams : 1.0);
            EXPECT_NEAR(static_cast<double const*>(mass->vals[0])[index], expectedMass, kRelativeTolerance * expectedMass);
            EXPECT_EQ(static_cast<char const*>(rung->vals[0])[index], particles.flags(index).rung());
        }
        EXPECT_EQ(DBInqVarExists(file.get(), (directory + "flags").c_str()), 0);
    }
    file.reset();
    EXPECT_EQ(cosmictiger::readInitialConditionsSiloMetadata(path).particleMass, metadata.particleMass);
    ParticleStorage restored;
    CartesianGrid restoredGrid(2, 0, 0);
    cosmictiger::readParticleSilo(path, restored, restoredGrid);
    ASSERT_EQ(restored.size(), kOwnedCount);
    for (ParticleCount index = 0; index < restored.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(restored.position(axis, index).raw(), particles.position(axis, index).raw());
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(restored.velocity(axis, index)), hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index)));
            EXPECT_EQ(restored.acceleration(axis, index), particles.acceleration(axis, index));
        }
        EXPECT_EQ(restored.potential(index), particles.potential(index));
        EXPECT_EQ(restored.flags(index).rung(), particles.flags(index).rung());
    }
}

TEST_F(ParticleSiloTest, InitialLogarithmicDisplacementDerivativeBecomesPeculiarVelocity) {
    constexpr ParticleCount kOwnedCount = 3;
    constexpr double kReferenceHubble = 100.0;
    constexpr double kRelativeTolerance = 2.0e-14;
    PROFILE();

    TestDirectory temporary;
    auto const path = temporary.file("initial_unit_velocity.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    auto const metadata = initialConditionsMetadata();
    cosmictiger::writeParticleSilo(path, particles, grid, metadata);
    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    PointVar velocity(DBGetPointvar(file.get(), "comoving/block_0/velocity"), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
    ASSERT_NE(velocity, nullptr);
    double const factor = kReferenceHubble * metadata.boxLength *
        cosmictiger::SimulationBackground(metadata.cosmology).conformalDerivative(metadata.scaleFactor) / metadata.scaleFactor;
    for (size_t axis = 0; axis < NDIM; axis++) {
        for (ParticleCount index = 0; index < kOwnedCount; index++) {
            double const expected = particles.velocity(axis, index) * factor;
            EXPECT_NEAR(static_cast<double const*>(velocity->vals[axis])[index], expected, kRelativeTolerance * std::abs(expected));
        }
    }
}

TEST_F(ParticleSiloTest, EvolvedViewsOmitUnsynchronizedVelocityAndRejectIncompleteMomentumArrays) {
    PROFILE();
    TestDirectory temporary;
    auto const path = temporary.file("staggered_unit_velocity.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    auto metadata = initialConditionsMetadata();
    metadata.canonicalMomentum = true;
    cosmictiger::detail::EvolutionSiloScalars const epoch{0.5, 0.3, -1.0e9, 0.125};
    std::vector<char> const payload{'x'};
    cosmictiger::detail::writeParticleSiloEvolutionPayload(path, particles, grid, metadata, payload, epoch);
    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(DBInqVarExists(file.get(), "cgs/block_0/velocity"), 0);
    EXPECT_EQ(DBInqVarExists(file.get(), "comoving/block_0/velocity"), 0);
    EXPECT_EQ(DBInqVarExists(file.get(), "cgs/block_0/mass"), 1);
    EXPECT_EQ(DBInqVarExists(file.get(), "block_0/velocity"), 1);
    file.reset();
    cosmictiger::ParticleSiloOutput output{};
    EXPECT_THROW(cosmictiger::detail::writeParticleSiloEvolutionPayload(path, particles, grid, metadata, payload, epoch, &output), std::invalid_argument);
    output.particleCount = 3;
    EXPECT_THROW(cosmictiger::detail::writeParticleSiloEvolutionPayload(path, particles, grid, metadata, payload, epoch, &output), std::invalid_argument);
    EXPECT_EQ(cosmictiger::detail::readParticleSiloEvolutionPayload(path), payload);
}

TEST_F(ParticleSiloTest, LegacyVersionsRestoreRungsWithoutGhostOrReservedFlags) {
    PROFILE();
    TestDirectory temporary;
    for (int version : {1, 2}) {
        std::string const path = temporary.file("legacy_" + std::to_string(version) + ".silo");
        CartesianGrid grid(2, 0, 0);
        ParticleStorage source;
        source.configureGravityFields(version == 2, version == 2);
        seedOwnedAndGhostParticles(source, grid);
        if (version == 2) {
            for (ParticleCount index = 0; index < source.size(); index++) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    source.acceleration(axis, index) = float(index * NDIM + axis) + 0.25f;
                }
                source.potential(index) = -float(index) - 0.5f;
            }
        }
        cosmictiger::writeParticleSilo(path, source, grid);
        writeLegacyFlags(path, version, {0, std::uint8_t(0x80 | 63), 17});
        if (version == 2) {
            EXPECT_NO_THROW(cosmictiger::detail::validateParticleSiloEvolutionLayout(
                path, grid.layout().globalSideLength(), grid.layout().localitiesPerAxis(), true, true));
        }
        ParticleStorage restored;
        cosmictiger::readParticleSilo(path, restored, grid);
        ASSERT_EQ(restored.size(), 3);
        EXPECT_EQ(restored.storesAcceleration(), version == 2);
        EXPECT_EQ(restored.storesPotential(), version == 2);
        for (ParticleCount index = 0; index < restored.size(); index++) {
            EXPECT_EQ(hpx::bit_cast<std::uint8_t>(restored.flags(index)), source.flags(index).rung());
            EXPECT_FALSE(restored.flags(index).isGhost());
            if (version == 2) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    EXPECT_EQ(restored.acceleration(axis, index), source.acceleration(axis, index));
                }
                EXPECT_EQ(restored.potential(index), source.potential(index));
            }
        }
    }
}

TEST_F(ParticleSiloTest, InvalidRungsLeaveDestinationUnchanged) {
    PROFILE();
    TestDirectory temporary;
    for (int invalidRung : {-1, 64}) {
        std::string const path = temporary.file("invalid_rung_" + std::to_string(invalidRung) + ".silo");
        CartesianGrid sourceGrid(1, 0, 0);
        ParticleStorage source;
        seedOwnedAndGhostParticles(source, sourceGrid);
        cosmictiger::writeParticleSilo(path, source, sourceGrid);
        {
            SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_APPEND), hpx::annotated_function(&DBClose, "SiloFile::close"));
            ASSERT_NE(file, nullptr);
            ASSERT_EQ(DBSetDir(file.get(), "block_0"), 0);
            std::array<char, 3> const rungs{0, static_cast<char>(invalidRung), 17};
            ASSERT_EQ(DBPutPointvar1(file.get(), "rung", "mesh", rungs.data(), static_cast<int>(rungs.size()), DB_CHAR, nullptr), 0);
        }

        CartesianGrid destinationGrid(1, 0, 0);
        ParticleStorage destination;
        destination.append({Coordinate::fromRaw(9), Coordinate::fromRaw(10), Coordinate::fromRaw(11)}, {-0.0f, 12.0f, 13.0f}, 27);
        destinationGrid.setInteriorRange({0, 0, 0}, {0, 1});
        ParticleCount const capacity = destination.capacity();
        EXPECT_THROW(cosmictiger::readParticleSilo(path, destination, destinationGrid), std::runtime_error);
        ASSERT_EQ(destination.size(), 1);
        EXPECT_EQ(destination.capacity(), capacity);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(destination.position(axis, 0).raw(), 9 + axis);
        }
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(destination.velocity(0, 0)), hpx::bit_cast<std::uint32_t>(-0.0f));
        EXPECT_EQ(destination.velocity(1, 0), 12.0f);
        EXPECT_EQ(destination.velocity(2, 0), 13.0f);
        EXPECT_EQ(hpx::bit_cast<std::uint8_t>(destination.flags(0)), 27);
        EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.begin, 0);
        EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.end, 1);
    }
}

TEST_F(ParticleSiloTest, LegacyGhostAndRungMismatchLeaveDestinationUnchanged) {
    PROFILE();
    TestDirectory temporary;
    for (int version : {1, 2}) {
        for (bool ghost : {false, true}) {
            std::string const path = temporary.file("invalid_legacy_" + std::to_string(version) + (ghost ? "_ghost.silo" : "_rung.silo"));
            CartesianGrid sourceGrid(1, 0, 0);
            ParticleStorage source;
            seedOwnedAndGhostParticles(source, sourceGrid);
            cosmictiger::writeParticleSilo(path, source, sourceGrid);
            cosmictiger::Rung const invalidFlags(ghost ? 63 : 62, ghost);
            writeLegacyFlags(path, version, {0, hpx::bit_cast<std::uint8_t>(invalidFlags), 17});

            CartesianGrid destinationGrid(1, 0, 0);
            ParticleStorage destination;
            destination.append({Coordinate::fromRaw(9), Coordinate::fromRaw(10), Coordinate::fromRaw(11)}, {-0.0f, 12.0f, 13.0f}, 27);
            destinationGrid.setInteriorRange({0, 0, 0}, {0, 1});
            ParticleCount const capacity = destination.capacity();
            EXPECT_THROW(cosmictiger::readParticleSilo(path, destination, destinationGrid), std::runtime_error);
            ASSERT_EQ(destination.size(), 1);
            EXPECT_EQ(destination.capacity(), capacity);
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_EQ(destination.position(axis, 0).raw(), 9 + axis);
            }
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(destination.velocity(0, 0)), hpx::bit_cast<std::uint32_t>(-0.0f));
            EXPECT_EQ(destination.velocity(1, 0), 12.0f);
            EXPECT_EQ(destination.velocity(2, 0), 13.0f);
            EXPECT_EQ(hpx::bit_cast<std::uint8_t>(destination.flags(0)), 27);
            EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.begin, 0);
            EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.end, 1);
        }
    }
}

TEST_F(ParticleSiloTest, InvalidGravityFieldMetadataLeavesDestinationUnchanged) {
    constexpr int kInvalid = 2;
    constexpr int kCount = 1;
    PROFILE();

    TestDirectory temporary;
    std::string const path = temporary.file("invalid_gravity.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage source;
    cosmictiger::writeParticleSilo(path, source, grid);
    DBfile* const file = DBOpen(path.c_str(), DB_UNKNOWN, DB_APPEND);
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(DBWrite(file, "particle_has_acceleration", &kInvalid, &kCount, 1, DB_INT), 0);
    EXPECT_EQ(DBClose(file), 0);
    ParticleStorage destination(1);
    destination.configureGravityFields(true, true);
    destination.acceleration(0, 0) = 7.25f;
    destination.potential(0) = -12.0f;
    EXPECT_THROW(cosmictiger::readParticleSilo(path, destination, grid), std::runtime_error);
    EXPECT_EQ(destination.size(), 1);
    EXPECT_TRUE(destination.storesAcceleration());
    EXPECT_TRUE(destination.storesPotential());
    EXPECT_EQ(destination.acceleration(0, 0), 7.25f);
    EXPECT_EQ(destination.potential(0), -12.0f);
}

TEST_F(ParticleSiloTest, InvalidInitialConditionsMetadataPreservesExistingCheckpoint) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    TestDirectory temporary;
    std::string const path = temporary.file("preserved_initial_conditions.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    auto const original = initialConditionsMetadata();
    cosmictiger::writeParticleSilo(path, particles, grid, original);

    auto check = [&](auto change) {
        PROFILE();
        auto invalid = original;
        change(invalid);
        EXPECT_THROW(cosmictiger::writeParticleSilo(path, particles, grid, invalid), std::runtime_error);
        auto const preserved = cosmictiger::readInitialConditionsSiloMetadata(path);
        EXPECT_DOUBLE_EQ(preserved.scaleFactor, original.scaleFactor);
        EXPECT_DOUBLE_EQ(preserved.boxLength, original.boxLength);
        EXPECT_EQ(preserved.seed, original.seed);
    };
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.scaleFactor = 0.0;
        },
        "ParticleSiloTest::zeroScaleFactor"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.scaleFactor = 1.01;
        },
        "ParticleSiloTest::excessScaleFactor"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.maximumDensityDeviation = metadata.requestedDensityDeviation / 2.0;
        },
        "ParticleSiloTest::insufficientDensityDeviation"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.meshExponent = 1;
        },
        "ParticleSiloTest::invalidMeshExponent"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.cosmology.heliumFraction = 1.0;
        },
        "ParticleSiloTest::invalidHeliumFraction"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.cosmology.scalarAmplitude = -1.0;
        },
        "ParticleSiloTest::invalidScalarAmplitude"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.component = cosmictiger::MatterComponent::totalMatter;
        },
        "ParticleSiloTest::invalidTotalMatter"));
    check(hpx::annotated_function(
        [](auto& metadata) {
            PROFILE();
            metadata.component = static_cast<cosmictiger::MatterComponent>(-1);
        },
        "ParticleSiloTest::invalidComponent"));

    CartesianGrid restoredGrid(2, 0, 0);
    ParticleStorage restored;
    cosmictiger::readParticleSilo(path, restored, restoredGrid);
    ASSERT_EQ(restored.size(), 3);
    EXPECT_EQ(restored.position(0, 2).raw(), particles.position(0, 2).raw());
    EXPECT_DOUBLE_EQ(restored.velocity(1, 2), particles.velocity(1, 2));
}

TEST_F(ParticleSiloTest, RejectsIncompatibleInitialConditionsVelocityConvention) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("incompatible_initial_conditions.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    cosmictiger::writeParticleSilo(path, particles, grid, initialConditionsMetadata());

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_APPEND), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(DBSetDir(file.get(), "initial_conditions"), 0);
    // Preserve the existing text extent while changing the derivative convention.
    char const incompatible[] = "d(x/L)/d ln(t)";
    int const extent = sizeof(incompatible);
    ASSERT_EQ(DBWrite(file.get(), "velocity_convention", incompatible, &extent, 1, DB_CHAR), 0);
    file.reset();
    EXPECT_THROW(cosmictiger::readInitialConditionsSiloMetadata(path), std::runtime_error);
}

TEST_F(ParticleSiloTest, RejectsInitialConditionsMetadataWithInvalidMeshExponent) {
    constexpr int kInvalidMeshExponent = 1;
    constexpr int kExtent = 1;
    PROFILE();

    TestDirectory temporary;
    std::string const path = temporary.file("inconsistent_initial_conditions.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);
    cosmictiger::writeParticleSilo(path, particles, grid, initialConditionsMetadata());

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_APPEND), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(DBSetDir(file.get(), "initial_conditions"), 0);
    ASSERT_EQ(DBWrite(file.get(), "mesh_exponent", &kInvalidMeshExponent, &kExtent, 1, DB_INT), 0);
    file.reset();
    EXPECT_THROW(cosmictiger::readInitialConditionsSiloMetadata(path), std::runtime_error);

    // General particle checkpoints retain their independent grid contract.
    CartesianGrid restoredGrid(2, 0, 0);
    ParticleStorage restored;
    EXPECT_NO_THROW(cosmictiger::readParticleSilo(path, restored, restoredGrid));
    EXPECT_EQ(restored.size(), 3);
}

TEST_F(ParticleSiloTest, WritesOnlyOwnedParticlesWithRequestedSiloTypes) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("particles.silo");
    CartesianGrid grid(1, 0, 0);
    ParticleStorage particles;
    seedOwnedAndGhostParticles(particles, grid);

    cosmictiger::writeParticleSilo(path, particles, grid);

    SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    SiloMultimesh multimesh(DBGetMultimesh(file.get(), "particles"), hpx::annotated_function(&DBFreeMultimesh, "SiloMultimesh::free"));
    ASSERT_NE(multimesh, nullptr);
    int version = 0;
    ASSERT_EQ(DBReadVar(file.get(), "particle_format_version", &version), 0);
    EXPECT_EQ(version, 3);
    EXPECT_EQ(DBInqVarExists(file.get(), "flags"), 0);
    for (char const* name : {"velocity", "rung"}) {
        SiloMultivar multivar(DBGetMultivar(file.get(), name), hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"));
        EXPECT_NE(multivar, nullptr) << name;
    }

    ASSERT_EQ(DBSetDir(file.get(), "block_0"), 0);
    EXPECT_EQ(DBInqVarExists(file.get(), "flags"), 0);
    PointMesh mesh(DBGetPointmesh(file.get(), "mesh"), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
    ASSERT_NE(mesh, nullptr);
    EXPECT_EQ(mesh->ndims, NDIM);
    EXPECT_EQ(mesh->nels, 3);
    EXPECT_EQ(mesh->datatype, DB_DOUBLE);
    for (size_t axis = 0; axis < NDIM; axis++) {
        ASSERT_NE(mesh->coords[axis], nullptr);
        auto const* positions = static_cast<double const*>(mesh->coords[axis]);
        for (ParticleCount index = 0; index < 3; index++) {
            double const expected = std::ldexp(static_cast<double>(particles.position(axis, index).raw()), -Coordinate::fractionalBits());
            EXPECT_EQ(positions[index], expected) << "axis=" << axis << " index=" << index;
        }
    }

    PointVar velocity(DBGetPointvar(file.get(), "velocity"), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
    ASSERT_NE(velocity, nullptr);
    EXPECT_EQ(velocity->datatype, DB_FLOAT);
    EXPECT_EQ(velocity->nvals, NDIM);
    EXPECT_EQ(velocity->nels, 3);
    for (size_t axis = 0; axis < NDIM; axis++) {
        ASSERT_NE(velocity->vals[axis], nullptr);
        auto const* values = static_cast<float const*>(velocity->vals[axis]);
        for (ParticleCount index = 0; index < 3; index++) {
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(values[index]), hpx::bit_cast<std::uint32_t>(particles.velocity(axis, index)))
                << "axis=" << axis << " index=" << index;
        }
    }

    PointVar rung(DBGetPointvar(file.get(), "rung"), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
    ASSERT_NE(rung, nullptr);
    EXPECT_EQ(rung->datatype, DB_CHAR);
    EXPECT_EQ(rung->nvals, 1);
    EXPECT_EQ(rung->nels, 3);
    ASSERT_NE(rung->vals[0], nullptr);
    auto const* rungs = static_cast<unsigned char const*>(rung->vals[0]);
    for (ParticleCount index = 0; index < 3; index++) {
        EXPECT_EQ(rungs[index], particles.flags(index).rung());
    }
}

TEST_F(ParticleSiloTest, VisualizationMasterCombinesLeafMeshesAndSkipsEmptyLocalities) {
    PROFILE();
    TestDirectory temporary;
    std::string const first = temporary.file("locality0.silo");
    std::string const second = temporary.file("locality1.silo");
    std::string const empty = temporary.file("locality2.silo");
    std::string const masterPath = temporary.file("snapshot.master.silo");

    CartesianGrid occupiedGrid(1, 0, 0);
    ParticleStorage occupied;
    seedOwnedAndGhostParticles(occupied, occupiedGrid);
    occupied.configureGravityFields(true, true);
    cosmictiger::writeParticleSilo(first, occupied, occupiedGrid);
    cosmictiger::writeParticleSilo(second, occupied, occupiedGrid);

    CartesianGrid emptyGrid(1, 0, 0);
    ParticleStorage emptyParticles;
    emptyParticles.configureGravityFields(true, true);
    cosmictiger::writeParticleSilo(empty, emptyParticles, emptyGrid);
    cosmictiger::writeParticleSiloMaster(masterPath, {first, second, empty}, 0.25);

    SiloFile master(DBOpen(masterPath.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(master, nullptr);
    SiloMultimesh mesh(DBGetMultimesh(master.get(), "particles"), hpx::annotated_function(&DBFreeMultimesh, "SiloMultimesh::free"));
    ASSERT_NE(mesh, nullptr);
    ASSERT_EQ(mesh->nblocks, 2);
    EXPECT_STREQ(mesh->meshnames[0], "locality0.silo:block_0/mesh");
    EXPECT_STREQ(mesh->meshnames[1], "locality1.silo:block_0/mesh");
    EXPECT_EQ(DBInqVarExists(master.get(), "flags"), 0);
    for (char const* name : {"velocity", "acceleration", "potential", "rung"}) {
        SiloMultivar variable(DBGetMultivar(master.get(), name), hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"));
        ASSERT_NE(variable, nullptr) << name;
        EXPECT_EQ(variable->nvars, 2) << name;
        EXPECT_STREQ(variable->mmesh_name, "particles") << name;
    }
}

TEST_F(ParticleSiloTest, VisualizationMasterIncludesBothUnitViewsAndEmptyLocalities) {
    PROFILE();
    TestDirectory temporary;
    auto const first = temporary.file("units0.silo");
    auto const second = temporary.file("units1.silo");
    auto const empty = temporary.file("units2.silo");
    auto const masterPath = temporary.file("units.master.silo");
    CartesianGrid grid(2, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    seedOwnedAndGhostParticles(particles, grid);
    auto metadata = initialConditionsMetadata();
    metadata.canonicalMomentum = true;
    cosmictiger::writeParticleSilo(first, particles, grid, metadata);
    cosmictiger::writeParticleSilo(second, particles, grid, metadata);
    CartesianGrid emptyGrid(2, 0, 0);
    ParticleStorage emptyParticles;
    emptyParticles.configureGravityFields(true, true);
    cosmictiger::writeParticleSilo(empty, emptyParticles, emptyGrid, metadata);
    cosmictiger::writeParticleSiloMaster(masterPath, {first, empty, second}, 0.25);

    SiloFile master(DBOpen(masterPath.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(master, nullptr);
    for (std::string const view : {"comoving", "cgs"}) {
        SiloMultimesh mesh(DBGetMultimesh(master.get(), (view + "/particles").c_str()), hpx::annotated_function(&DBFreeMultimesh, "SiloMultimesh::free"));
        ASSERT_NE(mesh, nullptr);
        ASSERT_EQ(mesh->nblocks, 2);
        EXPECT_STREQ(mesh->meshnames[0], ("units0.silo:/" + view + "/block_0/mesh").c_str());
        EXPECT_STREQ(mesh->meshnames[1], ("units1.silo:/" + view + "/block_0/mesh").c_str());
        for (std::string const name : {"velocity", "acceleration", "potential", "mass", "rung"}) {
            SiloMultivar variable(DBGetMultivar(master.get(), (view + "/" + name).c_str()), hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"));
            ASSERT_NE(variable, nullptr);
            ASSERT_EQ(variable->nvars, 2);
            EXPECT_STREQ(variable->mmesh_name, "particles");
            EXPECT_STREQ(variable->varnames[0], ("units0.silo:/" + view + "/block_0/" + name).c_str());
            EXPECT_STREQ(variable->varnames[1], ("units1.silo:/" + view + "/block_0/" + name).c_str());
        }
    }
    master.reset();
    metadata.boxLength *= 2.0;
    cosmictiger::writeParticleSilo(second, particles, grid, metadata);
    EXPECT_THROW(cosmictiger::writeParticleSiloMaster(masterPath, {first, second}), std::runtime_error);
}

TEST_F(ParticleSiloTest, RestartsOwnedParticlesAndRebuildsInteriorRanges) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("restart.silo");
    CartesianGrid sourceGrid(1, 0, 0);
    ParticleStorage source;
    seedOwnedAndGhostParticles(source, sourceGrid);
    cosmictiger::writeParticleSilo(path, source, sourceGrid);

    CartesianGrid restoredGrid(1, 0, 0);
    ParticleStorage restored;
    restored.append({Coordinate::fromRaw(9), Coordinate::fromRaw(9), Coordinate::fromRaw(9)}, {9.0f, 9.0f, 9.0f}, 9);
    restoredGrid.setInteriorRange({0, 0, 0}, {0, 1});
    restoredGrid.setGhostRanges({-1, 0, 0}, {0, 1}, {0, 1});
    cosmictiger::readParticleSilo(path, restored, restoredGrid);

    ASSERT_EQ(restored.size(), 3);
    for (ParticleCount index = 0; index < restored.size(); index++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_EQ(restored.position(axis, index).raw(), source.position(axis, index).raw());
            EXPECT_EQ(hpx::bit_cast<std::uint32_t>(restored.velocity(axis, index)), hpx::bit_cast<std::uint32_t>(source.velocity(axis, index)));
        }
        EXPECT_EQ(restored.flags(index).rung(), source.flags(index).rung());
        EXPECT_EQ(hpx::bit_cast<std::uint8_t>(restored.flags(index)), source.flags(index).rung());
        EXPECT_FALSE(restored.flags(index).isGhost());
    }

    ParticleCount cursor = 0;
    for (std::int64_t z = 0; z < 2; z++) {
        for (std::int64_t y = 0; y < 2; y++) {
            for (std::int64_t x = 0; x < 2; x++) {
                ParticleCount const count = x == 0 && y == 0 && z == 0 ? 2 : (x == 1 && y == 0 && z == 0 ? 1 : 0);
                auto const& cell = restoredGrid.cell({x, y, z});
                EXPECT_EQ(cell.localParticles.begin, cursor);
                EXPECT_EQ(cell.localParticles.end, cursor + count);
                EXPECT_EQ(cell.ownerParticles.begin, cursor);
                EXPECT_EQ(cell.ownerParticles.end, cursor + count);
                cursor += count;
            }
        }
    }
    for (size_t cellIndex = 0; cellIndex < restoredGrid.layout().storedCellCount(); cellIndex++) {
        GridIndices const coordinates = restoredGrid.layout().cellCoordinates(cellIndex);
        if (restoredGrid.layout().isInterior(coordinates)) {
            continue;
        }
        auto const& cell = restoredGrid.cell(coordinates);
        EXPECT_EQ(cell.localParticles.begin, cell.localParticles.end);
        EXPECT_EQ(cell.ownerParticles.begin, cell.ownerParticles.end);
    }
}

TEST_F(ParticleSiloTest, EmptyOwnedCheckpointOmitsGhostSuffix) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("empty.silo");
    CartesianGrid sourceGrid(1, 0, 0);
    ParticleStorage source;
    source.append({Coordinate::fromRaw(0xf0000000), Coordinate::fromRaw(0x10000000), Coordinate::fromRaw(0x10000000)}, {1.0f, 2.0f, 3.0f}, 4);
    sourceGrid.setGhostRanges({-1, 0, 0}, {0, 1}, {0, 1});
    cosmictiger::writeParticleSilo(path, source, sourceGrid);

    CartesianGrid restoredGrid(1, 0, 0);
    ParticleStorage restored;
    restored.append({Coordinate::fromRaw(9), Coordinate::fromRaw(9), Coordinate::fromRaw(9)}, {9.0f, 9.0f, 9.0f}, 9);
    restoredGrid.setInteriorRange({0, 0, 0}, {0, 1});
    cosmictiger::readParticleSilo(path, restored, restoredGrid);
    EXPECT_EQ(restored.size(), 0);
    expectAllRangesEmpty(restoredGrid);
}

TEST_F(ParticleSiloTest, SplitsLargeParticleCollectionsIntoPointMeshBlocks) {
    ParticleCount constexpr kCount = (ParticleCount(1) << 20) + 1;
    PROFILE();

    TestDirectory temporary;
    std::string const path = temporary.file("multiple_blocks.silo");
    {
        CartesianGrid sourceGrid(0, 0, 0);
        ParticleStorage source(kCount);
        source.position(0, kCount - 1) = Coordinate::fromRaw(0xffffffffU);
        source.velocity(2, kCount - 1) = -3.5f;
        source.flags(kCount - 1).setRung(63);
        sourceGrid.setInteriorRange({0, 0, 0}, {0, kCount});
        cosmictiger::writeParticleSilo(path, source, sourceGrid);
    }

    {
        SiloFile file(DBOpen(path.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        ASSERT_NE(file, nullptr);
        SiloMultimesh multimesh(DBGetMultimesh(file.get(), "particles"), hpx::annotated_function(&DBFreeMultimesh, "SiloMultimesh::free"));
        ASSERT_NE(multimesh, nullptr);
        EXPECT_EQ(multimesh->nblocks, 2);
        EXPECT_EQ(DBInqVarExists(file.get(), "flags"), 0);
        ASSERT_EQ(DBSetDir(file.get(), "block_0"), 0);
        EXPECT_EQ(DBInqVarExists(file.get(), "flags"), 0);
        ASSERT_EQ(DBSetDir(file.get(), "/"), 0);
        ASSERT_EQ(DBSetDir(file.get(), "block_1"), 0);
        EXPECT_EQ(DBInqVarExists(file.get(), "flags"), 0);
        PointMesh secondBlock(DBGetPointmesh(file.get(), "mesh"), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
        ASSERT_NE(secondBlock, nullptr);
        EXPECT_EQ(secondBlock->nels, 1);
    }

    CartesianGrid restoredGrid(0, 0, 0);
    ParticleStorage restored;
    cosmictiger::readParticleSilo(path, restored, restoredGrid);
    ASSERT_EQ(restored.size(), kCount);
    EXPECT_EQ(restored.position(0, kCount - 1).raw(), 0xffffffffU);
    EXPECT_EQ(restored.velocity(2, kCount - 1), -3.5f);
    EXPECT_EQ(restored.flags(kCount - 1).rung(), 63);
    EXPECT_EQ(restoredGrid.cell({0, 0, 0}).localParticles.end, kCount);
}

TEST_F(ParticleSiloTest, MismatchedGridLeavesDestinationUnchanged) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("wrong_grid.silo");
    CartesianGrid sourceGrid(1, 0, 0);
    ParticleStorage source;
    seedOwnedAndGhostParticles(source, sourceGrid);
    cosmictiger::writeParticleSilo(path, source, sourceGrid);

    CartesianGrid destinationGrid(0, 0, 0);
    ParticleStorage destination;
    destination.append({Coordinate::fromRaw(0x80000000), Coordinate::fromRaw(0x40000000), Coordinate::fromRaw(0x20000000)}, {-0.0f, 12.0f, 13.0f}, 27);
    destinationGrid.setInteriorRange({0, 0, 0}, {0, 1});
    destinationGrid.setGhostRanges({-1, 0, 0}, {0, 1}, {0, 1});
    ParticleCount const previousCapacity = destination.capacity();

    EXPECT_THROW(cosmictiger::readParticleSilo(path, destination, destinationGrid), std::exception);
    ASSERT_EQ(destination.size(), 1);
    EXPECT_EQ(destination.capacity(), previousCapacity);
    EXPECT_EQ(destination.position(0, 0).raw(), 0x80000000);
    EXPECT_EQ(destination.position(1, 0).raw(), 0x40000000);
    EXPECT_EQ(destination.position(2, 0).raw(), 0x20000000);
    EXPECT_EQ(hpx::bit_cast<std::uint32_t>(destination.velocity(0, 0)), hpx::bit_cast<std::uint32_t>(-0.0f));
    EXPECT_EQ(destination.velocity(1, 0), 12.0f);
    EXPECT_EQ(destination.velocity(2, 0), 13.0f);
    EXPECT_EQ(destination.flags(0).rung(), 27);
    EXPECT_FALSE(destination.flags(0).isGhost());
    EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.begin, 0);
    EXPECT_EQ(destinationGrid.cell({0, 0, 0}).localParticles.end, 1);
    EXPECT_EQ(destinationGrid.cell({-1, 0, 0}).localParticles.begin, 0);
    EXPECT_EQ(destinationGrid.cell({-1, 0, 0}).localParticles.end, 1);
    EXPECT_EQ(destinationGrid.cell({-1, 0, 0}).ownerParticles.begin, 0);
    EXPECT_EQ(destinationGrid.cell({-1, 0, 0}).ownerParticles.end, 1);
}

TEST_F(ParticleSiloTest, MissingCheckpointMetadataLeavesDestinationUnchanged) {
    PROFILE();
    TestDirectory temporary;
    std::string const path = temporary.file("incomplete.silo");
    SiloFile file(DBCreate(path.c_str(), DB_CLOBBER, DB_LOCAL, "incomplete checkpoint", DB_HDF5), hpx::annotated_function(&DBClose, "SiloFile::close"));
    ASSERT_NE(file, nullptr);
    file.reset();

    CartesianGrid grid(0, 0, 0);
    ParticleStorage particles;
    particles.append({Coordinate::fromRaw(0x80000000), Coordinate::fromRaw(0x40000000), Coordinate::fromRaw(0x20000000)}, {3.0f, 4.0f, 5.0f}, 7);
    grid.setInteriorRange({0, 0, 0}, {0, 1});
    ParticleCount const previousCapacity = particles.capacity();

    EXPECT_THROW(cosmictiger::readParticleSilo(path, particles, grid), std::exception);
    ASSERT_EQ(particles.size(), 1);
    EXPECT_EQ(particles.capacity(), previousCapacity);
    EXPECT_EQ(particles.position(0, 0).raw(), 0x80000000);
    EXPECT_EQ(particles.position(1, 0).raw(), 0x40000000);
    EXPECT_EQ(particles.position(2, 0).raw(), 0x20000000);
    EXPECT_EQ(particles.velocity(0, 0), 3.0f);
    EXPECT_EQ(particles.velocity(1, 0), 4.0f);
    EXPECT_EQ(particles.velocity(2, 0), 5.0f);
    EXPECT_EQ(particles.flags(0).rung(), 7);
    EXPECT_EQ(grid.cell({0, 0, 0}).localParticles.begin, 0);
    EXPECT_EQ(grid.cell({0, 0, 0}).localParticles.end, 1);
    EXPECT_EQ(grid.cell({0, 0, 0}).ownerParticles.begin, 0);
    EXPECT_EQ(grid.cell({0, 0, 0}).ownerParticles.end, 1);
}

}    // namespace
