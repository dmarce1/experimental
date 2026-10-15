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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/particle/particle_silo.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/units.hpp>

#include <hpx/bit.hpp>
#include <hpx/hpx_init.hpp>
#include <silo.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <numbers>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using SiloFile = std::unique_ptr<DBfile, std::decay_t<decltype(hpx::annotated_function(&DBClose, "SiloFile::close"))>>;
using SiloMultivar = std::unique_ptr<DBmultivar, std::decay_t<decltype(hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"))>>;
using PointVar = std::unique_ptr<DBmeshvar, std::decay_t<decltype(hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"))>>;
using PointMesh = std::unique_ptr<DBpointmesh, std::decay_t<decltype(hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"))>>;

struct VerificationSettings {
    size_t meshExponent;
    std::uint64_t seed;
    double requestedDensityDeviation;
    double earliestScaleFactor;
    bool gravityFields = false;
    size_t gravityMeshExponent = 0;
};

constexpr VerificationSettings kSmokeSettings{2, 1234, 0.05, 0.002};
constexpr VerificationSettings kDefaultRealizationSettings{5, 1, 0.1, 0.001};
constexpr VerificationSettings kGravityFieldSettings{3, 1234, 0.05, 0.002, true};
constexpr VerificationSettings kIndependentGridSettings{3, 1234, 0.05, 0.002, true, 2};

cosmictiger::ParticleCount expectedParticleCount(VerificationSettings const& settings) {
    PROFILE();
    return cosmictiger::ParticleCount{1} << (NDIM * settings.meshExponent);
}

void require(bool condition, std::string const& message) {
    PROFILE();
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireText(DBfile* file, char const* name, char const* expected) {
    PROFILE();
    size_t const length = std::strlen(expected) + 1;
    require(DBGetVarType(file, name) == DB_CHAR && DBGetVarLength(file, name) == static_cast<int>(length),
        std::string("incorrect initial-condition convention schema: ") + name);
    std::vector<char> value(length);
    require(DBReadVar(file, name, value.data()) == 0, std::string("could not read initial-condition convention: ") + name);
    require(value.back() == '\0' && std::string(value.data()) == expected, std::string("incorrect initial-condition convention: ") + name);
}

void verifyMetadata(std::string const& filename, VerificationSettings const& settings) {
    PROFILE();
    auto const metadata = cosmictiger::readInitialConditionsSiloMetadata(filename);
    require(metadata.canonicalMomentum, "initial particle velocities are not canonical conformal momenta");
    require(metadata.scaleFactor >= settings.earliestScaleFactor && metadata.scaleFactor < 1.0, "initial scale factor is outside the tested scan interval");
    require(metadata.boxLength == 100.0, "incorrect comoving box length");
    require(metadata.seed == settings.seed, "incorrect realization seed");
    require(metadata.meshExponent == settings.meshExponent, "incorrect FFT mesh exponent");
    require(metadata.component == cosmictiger::MatterComponent::cdmAndBaryons, "incorrect cold-particle matter component");
    require(metadata.requestedDensityDeviation == settings.requestedDensityDeviation, "incorrect requested density threshold");
    require(metadata.maximumDensityDeviation >= metadata.requestedDensityDeviation, "realization has not reached the requested density threshold");
    auto const& cosmology = metadata.cosmology;
    require(cosmology.hubble == 67.4 && cosmology.omegaBaryonH2 == 0.0224 && cosmology.omegaCdmH2 == 0.120 && cosmology.omegaNeutrinoH2 == 0.00064,
        "incorrect fiducial expansion or physical matter densities");
    require(cosmology.scalarAmplitude == 2.1e-9 && cosmology.spectralIndex == 0.965 && cosmology.spectralRunning == 0.0 && cosmology.pivotMpcInverse == 0.05,
        "incorrect fiducial primordial spectrum");
    require(cosmology.omegaCurvature == 0.0 && cosmology.darkEnergyW == -1.0 && cosmology.darkEnergyWa == 0.0, "incorrect fiducial geometry or dark energy");
    require(cosmology.massiveNeutrinos == 1 && cosmology.masslessNeutrinos == 2 && cosmology.effectiveNeutrinos == 3.046,
        "incorrect fiducial neutrino prescription");
    require(cosmology.temperatureCmb == 2.7255 && cosmology.heliumFraction == 0.245 && cosmology.opticalDepth == 0.054,
        "incorrect fiducial thermal or ionization parameters");

    double const relativeMassError = [&] {
        PROFILE();
        cosmictiger::FpeGuard guard;
        double const hubbleParameter = cosmology.hubble / 100.0;
        cosmictiger::units::UnitScales const scales(cosmology.hubble, metadata.boxLength);
        double const normalizedMass = 3.0 * (cosmology.omegaBaryonH2 + cosmology.omegaCdmH2) /
            (8.0 * std::numbers::pi * hubbleParameter * hubbleParameter * static_cast<double>(expectedParticleCount(settings)));
        double const expectedMass = normalizedMass * scales.massSolar() * hubbleParameter;
        return std::abs(metadata.particleMass / expectedMass - 1.0);
    }();
    require(relativeMassError < 1.0e-12, "particle mass does not represent the fiducial cold density in the periodic box");

    SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    require(file != nullptr, "could not open generated initial-condition Silo file");
    require(DBSetDir(file.get(), "initial_conditions") == 0, "initial-condition metadata directory is absent");
    requireText(file.get(), "position_units", "unit_box");
    requireText(file.get(), "velocity_convention", "a d(x/L)/d(H0 tau)");
    requireText(file.get(), "conformal_time_units", "H0^-1");
    requireText(file.get(), "box_length_units", "Mpc/h");
    requireText(file.get(), "particle_mass_units", "Msun/h");
}

void verifyParticles(std::string const& filename, VerificationSettings const& settings) {
    PROFILE();
    cosmictiger::ParticleStorage particles;
    cosmictiger::CartesianGrid grid(settings.gravityMeshExponent == 0 ? settings.meshExponent : settings.gravityMeshExponent, 0, 0);
    cosmictiger::readParticleSilo(filename, particles, grid);
    require(particles.size() == expectedParticleCount(settings), "generated Silo does not contain the expected number of owned initial particles");
    require(particles.storesAcceleration() == settings.gravityFields, "incorrect acceleration field allocation in initial checkpoint");
    require(particles.storesPotential() == settings.gravityFields, "incorrect potential field allocation in initial checkpoint");
    bool hasNonzeroVelocity = false;
    bool hasNonzeroAcceleration = false;
    bool hasNonzeroPotential = false;
    for (cosmictiger::ParticleCount index = 0; index < particles.size(); index++) {
        require(hpx::bit_cast<std::uint8_t>(particles.flags(index)) == 0, "initial particle is a ghost or has a nonzero rung or reserved flag");
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const position = static_cast<double>(particles.position(axis, index));
            double const velocity = particles.velocity(axis, index);
            require(position >= 0.0 && position < 1.0, "initial particle position is outside the periodic unit box");
            hasNonzeroVelocity = hasNonzeroVelocity || velocity != 0.0;
            if (settings.gravityFields) {
                hasNonzeroAcceleration = hasNonzeroAcceleration || particles.acceleration(axis, index) != 0.0f;
            }
        }
        if (settings.gravityFields) {
            hasNonzeroPotential = hasNonzeroPotential || particles.potential(index) != 0.0f;
        }
    }
    require(hasNonzeroVelocity, "the perturbed initial realization has zero velocity everywhere");
    if (settings.gravityFields) {
        require(hasNonzeroAcceleration, "the perturbed initial realization has zero acceleration everywhere");
        require(hasNonzeroPotential, "the perturbed initial realization has zero potential everywhere");
        SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        require(file != nullptr, "could not open initial-condition gravity fields");
        for (auto const name : {"acceleration", "potential"}) {
            SiloMultivar multivar(DBGetMultivar(file.get(), name), hpx::annotated_function(&DBFreeMultivar, "SiloMultivar::free"));
            require(multivar != nullptr && multivar->nvars == 1 && multivar->vartypes != nullptr && multivar->vartypes[0] == DB_POINTVAR,
                std::string("incorrect initial gravity multivariable schema: ") + name);
            std::string const blockName = std::string("block_0/") + name;
            PointVar variable(DBGetPointvar(file.get(), blockName.c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
            int const components = std::strcmp(name, "acceleration") == 0 ? NDIM : 1;
            require(variable != nullptr && variable->datatype == DB_FLOAT && variable->nvals == components &&
                    variable->nels == static_cast<int>(expectedParticleCount(settings)) && variable->meshname != nullptr &&
                    std::strcmp(variable->meshname, "mesh") == 0 && variable->vals != nullptr,
                std::string("incorrect initial gravity point-variable schema: ") + name);
            for (int component = 0; component < components; component++) {
                require(variable->vals[component] != nullptr, std::string("missing initial gravity component: ") + name);
            }
        }
    }
}

void verifyUnitViews(std::string const& filename, VerificationSettings const& settings) {
    constexpr double kRelativeTolerance = 2.0e-13;
    constexpr double kHubbleUnit = 100.0;
    PROFILE();

    auto const metadata = cosmictiger::readInitialConditionsSiloMetadata(filename);
    cosmictiger::units::UnitScales const units(metadata.cosmology.hubble, metadata.boxLength);
    SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    require(file != nullptr, "could not open initial-condition unit views");
    double scaleFactor = 0.0;
    require(DBReadVar(file.get(), "output_units/scale_factor", &scaleFactor) == 0 && scaleFactor == metadata.scaleFactor,
        "initial-condition unit views have the wrong epoch");
    requireText(file.get(), "output_units/velocity_convention", "physical peculiar velocity; Hubble flow excluded");
    PointMesh nativeMesh(DBGetPointmesh(file.get(), "block_0/mesh"), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
    PointVar nativeMomentum(DBGetPointvar(file.get(), "block_0/velocity"), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
    require(nativeMesh != nullptr && nativeMomentum != nullptr, "initial-condition native particle fields are absent");
    for (bool cgs : {false, true}) {
        auto const directory = std::string(cgs ? "cgs" : "comoving") + "/block_0/";
        PointMesh mesh(DBGetPointmesh(file.get(), (directory + "mesh").c_str()), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
        PointVar velocity(DBGetPointvar(file.get(), (directory + "velocity").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar mass(DBGetPointvar(file.get(), (directory + "mass").c_str()), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        require(mesh != nullptr && mesh->datatype == DB_DOUBLE && mesh->nels == static_cast<int>(expectedParticleCount(settings)),
            "initial-condition derived coordinate schema is incorrect");
        require(velocity != nullptr && velocity->datatype == DB_DOUBLE && velocity->nvals == NDIM && velocity->nels == mesh->nels,
            "initial-condition peculiar velocity schema is incorrect");
        require(mass != nullptr && mass->datatype == DB_DOUBLE && mass->nels == mesh->nels, "initial-condition derived particle mass schema is incorrect");
        require(velocity->units != nullptr && std::string(velocity->units) == (cgs ? "cm/s" : "km/s"), "peculiar velocity units are absent or incorrect");
        require(mass->units != nullptr && std::string(mass->units) == (cgs ? "g" : "Msun"), "particle mass units are absent or incorrect");
        double const lengthFactor = cgs ? scaleFactor * units.lengthCgs() : units.lengthMpc();
        double const velocityFactor = (cgs ? units.velocityCgs() : units.velocityKmPerSecond()) / scaleFactor;
        double const expectedMass = metadata.particleMass / (metadata.cosmology.hubble / kHubbleUnit) * (cgs ? cosmictiger::units::kSolarMass.value() : 1.0);
        for (size_t axis = 0; axis < NDIM; axis++) {
            require(mesh->units[axis] != nullptr && std::string(mesh->units[axis]) == (cgs ? "cm" : "Mpc"), "coordinate units are absent or incorrect");
            for (int index = 0; index < mesh->nels; index++) {
                cosmictiger::FpeGuard guard;
                double const expectedPosition = static_cast<double const*>(nativeMesh->coords[axis])[index] * lengthFactor;
                double const expectedVelocity = static_cast<float const*>(nativeMomentum->vals[axis])[index] * velocityFactor;
                require(std::abs(static_cast<double const*>(mesh->coords[axis])[index] - expectedPosition) <= kRelativeTolerance * std::abs(expectedPosition),
                    "initial-condition coordinates were not converted to the requested frame");
                require(std::abs(static_cast<double const*>(velocity->vals[axis])[index] - expectedVelocity) <= kRelativeTolerance * std::abs(expectedVelocity),
                    "initial-condition canonical momentum was not converted to peculiar velocity");
            }
        }
        for (int index = 0; index < mass->nels; index++) {
            require(std::abs(static_cast<double const*>(mass->vals[0])[index] - expectedMass) <= kRelativeTolerance * expectedMass,
                "initial-condition particle mass has the wrong units");
        }
    }
}

}    // namespace

int main(int argc, char* argv[]) {
    PROFILE();
    if ((argc != 2 && argc != 3) ||
        (argc == 3 && std::strcmp(argv[2], "--default-realization") != 0 && std::strcmp(argv[2], "--gravity-fields") != 0 &&
            std::strcmp(argv[2], "--independent-grid-fields") != 0)) {
        std::fprintf(stderr, "Usage: %s initial_conditions.silo [--default-realization|--gravity-fields|--independent-grid-fields]\n", argv[0]);
        return 2;
    }
    auto const settings = argc == 2 ?
        kSmokeSettings :
        (std::strcmp(argv[2], "--gravity-fields") == 0 ?
                kGravityFieldSettings :
                (std::strcmp(argv[2], "--independent-grid-fields") == 0 ? kIndependentGridSettings : kDefaultRealizationSettings));
    std::string const filename = argv[1];
    hpx::init_params params;
    params.cfg = {"hpx.os_threads=1", "hpx.threadpools.io_pool_size=1", "hpx.threadpools.timer_pool_size=1", "hpx.threadpools.parcel_pool_size=1",
        "hpx.parcel.mpi.enable=0"};
    // CUDA-backed ParticleStorage allocations use HPX's I/O executor too.
    // Pass only the executable name to HPX; the checker consumes its own args.
    return hpx::init(hpx::annotated_function(
                         [filename, settings](int, char**) {
                             PROFILE();
                             int result = 0;
                             try {
                                 verifyMetadata(filename, settings);
                                 verifyParticles(filename, settings);
                                 verifyUnitViews(filename, settings);
                                 std::printf("Verified %llu initial-condition particles and fiducial cosmology in %s\n",
                                     static_cast<unsigned long long>(expectedParticleCount(settings)), filename.c_str());
                             } catch (std::exception const& error) {
                                 std::fprintf(stderr, "Initial-condition verification failed: %s\n", error.what());
                                 result = 1;
                             }
                             hpx::finalize();
                             return result;
                         },
                         "test.initialCondition.verifySilo"),
        1, argv, params);
}
