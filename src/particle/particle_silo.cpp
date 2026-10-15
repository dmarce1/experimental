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
#include <cosmictiger/units.hpp>

#include <silo.h>

#include <hpx/bit.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace cosmictiger::detail {

constexpr int kParticleSiloVersion = 3;
constexpr int kParticleSiloBlockSize = 1 << 20;
constexpr int kInitialConditionsSiloVersion = 2;
constexpr int kEvolutionSiloVersion = 1;
constexpr int kMaximumEvolutionPayloadSize = 1 << 20;
constexpr int kUnitViewVersion = 1;
constexpr size_t kViewMesh = 0;
constexpr size_t kViewVelocity = 1;
constexpr size_t kViewAcceleration = 2;
constexpr size_t kViewPotential = 3;
constexpr std::array<char const*, 6> kViewLeafNames{"mesh", "velocity", "acceleration", "potential", "mass", "rung"};
constexpr std::array<char const*, 6> kViewMasterNames{"particles", "velocity", "acceleration", "potential", "mass", "rung"};
constexpr std::array<char const*, 2> kUnitViewNames{"comoving", "cgs"};

using CosmologyScalar = std::pair<char const*, double CosmologyParameters::*>;
constexpr std::array<CosmologyScalar, 15> kCosmologyScalars{{
    {"hubble", &CosmologyParameters::hubble},
    {"omega_baryon_h2", &CosmologyParameters::omegaBaryonH2},
    {"omega_cdm_h2", &CosmologyParameters::omegaCdmH2},
    {"omega_neutrino_h2", &CosmologyParameters::omegaNeutrinoH2},
    {"omega_curvature", &CosmologyParameters::omegaCurvature},
    {"effective_neutrinos", &CosmologyParameters::effectiveNeutrinos},
    {"scalar_amplitude", &CosmologyParameters::scalarAmplitude},
    {"spectral_index", &CosmologyParameters::spectralIndex},
    {"spectral_running", &CosmologyParameters::spectralRunning},
    {"pivot_mpc_inverse", &CosmologyParameters::pivotMpcInverse},
    {"dark_energy_w", &CosmologyParameters::darkEnergyW},
    {"dark_energy_wa", &CosmologyParameters::darkEnergyWa},
    {"temperature_cmb", &CosmologyParameters::temperatureCmb},
    {"helium_fraction", &CosmologyParameters::heliumFraction},
    {"optical_depth", &CosmologyParameters::opticalDepth},
}};

static_assert(sizeof(int) == sizeof(std::uint32_t) && CHAR_BIT == 8);
static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
static_assert(std::is_nothrow_move_assignable_v<CartesianGrid>);

using SiloFile = std::unique_ptr<DBfile, std::decay_t<decltype(hpx::annotated_function(&DBClose, "SiloFile::close"))>>;
using SiloOptions = std::unique_ptr<DBoptlist, std::decay_t<decltype(hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"))>>;
using PointMesh = std::unique_ptr<DBpointmesh, std::decay_t<decltype(hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"))>>;
using PointVar = std::unique_ptr<DBmeshvar, std::decay_t<decltype(hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"))>>;

void checkSilo(int status, std::string const& operation) {
    PROFILE();
    if (status < 0) {
        char const* const error = DBErrString();
        throw std::runtime_error("Silo " + operation + " failed" + (error == nullptr ? std::string{} : ": " + std::string(error)));
    }
}

class TemporarySiloFile {
public:
    explicit TemporarySiloFile(std::string const& destination)
      : target_(destination) {
        PROFILE();
        expect(!(destination.empty()));
        std::string const pattern = destination + ".tmp.XXXXXX";
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        int const descriptor = ::mkstemp(buffer.data());
        if (descriptor < 0) {
            throw std::system_error(errno, std::generic_category(), "create temporary Silo file");
        }
        temporary_ = buffer.data();
        if (::close(descriptor) != 0) {
            int const error = errno;
            std::error_code ignored;
            hpx::filesystem::remove(temporary_, ignored);
            throw std::system_error(error, std::generic_category(), "close temporary Silo file");
        }
    }

    TemporarySiloFile(TemporarySiloFile const&) = delete;
    TemporarySiloFile& operator=(TemporarySiloFile const&) = delete;

    ~TemporarySiloFile() {
        PROFILE();
        if (!committed_) {
            std::error_code ignored;
            hpx::filesystem::remove(temporary_, ignored);
        }
    }

    std::string const& path() const noexcept {
        PROFILE();
        return temporary_;
    }

    void commit() {
        PROFILE();
        hpx::filesystem::rename(temporary_, target_);
        committed_ = true;
    }

private:
    hpx::filesystem::path target_;
    std::string temporary_;
    bool committed_ = false;
};

void closeSilo(SiloFile& file) {
    PROFILE();
    DBfile* const handle = file.release();
    checkSilo(DBClose(handle), "file close");
}

void writeInt(DBfile* file, char const* name, int value) {
    int constexpr kExtent = 1;
    PROFILE();
    checkSilo(DBWrite(file, name, &value, &kExtent, 1, DB_INT), std::string("metadata ") + name);
}

int readInt(DBfile* file, char const* name) {
    PROFILE();
    if (DBGetVarType(file, name) != DB_INT || DBGetVarLength(file, name) != 1) {
        throw std::runtime_error(std::string("Silo metadata ") + name + " has the wrong type or length");
    }
    int value = 0;
    checkSilo(DBReadVar(file, name, &value), std::string("read metadata ") + name);
    return value;
}

void writeCount(DBfile* file, char const* name, std::uint64_t value) {
    PROFILE();
    std::array<int, 2> const words{hpx::bit_cast<int>(static_cast<std::uint32_t>(value)), hpx::bit_cast<int>(static_cast<std::uint32_t>(value >> 32))};
    int const extent = static_cast<int>(words.size());
    checkSilo(DBWrite(file, name, words.data(), &extent, 1, DB_INT), std::string("metadata ") + name);
}

std::uint64_t readCount(DBfile* file, char const* name) {
    PROFILE();
    if (DBGetVarType(file, name) != DB_INT || DBGetVarLength(file, name) != 2) {
        throw std::runtime_error(std::string("Silo metadata ") + name + " has the wrong type or length");
    }
    std::array<int, 2> words{};
    checkSilo(DBReadVar(file, name, words.data()), std::string("read metadata ") + name);
    return std::uint64_t(hpx::bit_cast<std::uint32_t>(words[0])) | (std::uint64_t(hpx::bit_cast<std::uint32_t>(words[1])) << 32);
}

void writeDouble(DBfile* file, char const* name, double value) {
    int constexpr kExtent = 1;
    PROFILE();
    checkSilo(DBWrite(file, name, &value, &kExtent, 1, DB_DOUBLE), std::string("metadata ") + name);
}

double readDouble(DBfile* file, char const* name) {
    PROFILE();
    if (DBGetVarType(file, name) != DB_DOUBLE || DBGetVarLength(file, name) != 1) {
        throw std::runtime_error(std::string("Silo metadata ") + name + " has the wrong type or length");
    }
    double value = 0.0;
    checkSilo(DBReadVar(file, name, &value), std::string("read metadata ") + name);
    return value;
}

void writeText(DBfile* file, char const* name, std::string const& value) {
    PROFILE();
    int const extent = static_cast<int>(value.size() + 1);
    checkSilo(DBWrite(file, name, value.c_str(), &extent, 1, DB_CHAR), std::string("metadata ") + name);
}

void requireText(DBfile* file, char const* name, std::string const& expected) {
    PROFILE();
    if (DBGetVarType(file, name) != DB_CHAR || DBGetVarLength(file, name) != static_cast<int>(expected.size() + 1)) {
        throw std::runtime_error(std::string("Silo metadata ") + name + " has the wrong type or length");
    }
    std::vector<char> value(expected.size() + 1);
    checkSilo(DBReadVar(file, name, value.data()), std::string("read metadata ") + name);
    if (value.back() != '\0' || std::string(value.data()) != expected) {
        throw std::runtime_error(std::string("Silo metadata ") + name + " has an incompatible convention");
    }
}

double checkedCosmologyScalar(double value, double CosmologyParameters::* member) {
    PROFILE();
    if (member == &CosmologyParameters::hubble || member == &CosmologyParameters::scalarAmplitude || member == &CosmologyParameters::pivotMpcInverse) {
        return expectPositive(value);
    }
    if (member == &CosmologyParameters::omegaBaryonH2 || member == &CosmologyParameters::omegaCdmH2 || member == &CosmologyParameters::omegaNeutrinoH2 ||
        member == &CosmologyParameters::effectiveNeutrinos || member == &CosmologyParameters::opticalDepth || member == &CosmologyParameters::temperatureCmb) {
        return expectNonNegative(value);
    }
    if (member == &CosmologyParameters::heliumFraction) {
        expect(value < 1.0);
        return expectNonNegative(value);
    }
    return value;
}

MatterComponent checkedMatterComponent(MatterComponent component, CosmologyParameters const& cosmology) {
    PROFILE();
    FpeGuard guard;
    expect(component == MatterComponent::cdm || component == MatterComponent::cdmAndBaryons || component == MatterComponent::totalMatter);
    expect(component != MatterComponent::totalMatter || cosmology.omegaNeutrinoH2 == 0.0);
    expectPositive(component == MatterComponent::cdm ? cosmology.omegaCdmH2 : cosmology.omegaCdmH2 + cosmology.omegaBaryonH2);
    return component;
}

void writeInitialConditionsMetadata(DBfile* file, InitialConditionsSiloMetadata const& metadata) {
    PROFILE();
    checkSilo(DBMkDir(file, "initial_conditions"), "create initial-condition metadata directory");
    checkSilo(DBSetDir(file, "initial_conditions"), "enter initial-condition metadata directory");
    writeInt(file, "format_version", metadata.canonicalMomentum ? kInitialConditionsSiloVersion : 1);
    writeText(file, "position_units", "unit_box");
    writeText(file, "velocity_convention", metadata.canonicalMomentum ? "a d(x/L)/d(H0 tau)" : "d(x/L)/d ln(a)");
    if (metadata.canonicalMomentum) {
        writeText(file, "conformal_time_units", "H0^-1");
    }
    writeText(file, "box_length_units", "Mpc/h");
    writeText(file, "particle_mass_units", "Msun/h");
    writeDouble(file, "scale_factor", expectRange(0, expectPositive(metadata.scaleFactor), 1));
    writeDouble(file, "box_length", expectPositive(metadata.boxLength));
    writeDouble(file, "particle_mass", expectPositive(metadata.particleMass));
    expect(metadata.maximumDensityDeviation >= metadata.requestedDensityDeviation);
    writeDouble(file, "maximum_density_deviation", expectPositive(metadata.maximumDensityDeviation));
    writeDouble(file, "requested_density_deviation", expectPositive(metadata.requestedDensityDeviation));
    writeCount(file, "random_seed", metadata.seed);
    writeInt(file, "mesh_exponent", static_cast<int>(expectRange(2, metadata.meshExponent, (std::numeric_limits<size_t>::digits - 1) / NDIM)));
    writeInt(file, "matter_component", static_cast<int>(checkedMatterComponent(metadata.component, metadata.cosmology)));
    for (auto const& [name, member] : kCosmologyScalars) {
        writeDouble(file, name, checkedCosmologyScalar(metadata.cosmology.*member, member));
    }
    expect(metadata.cosmology.omegaNeutrinoH2 == 0.0 || metadata.cosmology.massiveNeutrinos > 0);
    writeInt(file, "massive_neutrinos", expectNonNegative(metadata.cosmology.massiveNeutrinos));
    writeInt(file, "massless_neutrinos", expectNonNegative(metadata.cosmology.masslessNeutrinos));
    checkSilo(DBSetDir(file, "/"), "leave initial-condition metadata directory");
}

std::uint64_t blockCount(std::uint64_t count) noexcept {
    PROFILE();
    return count / kParticleSiloBlockSize + (count % kParticleSiloBlockSize != 0 ? 1 : 0);
}

std::string blockName(std::uint64_t block) {
    PROFILE();
    return "block_" + std::to_string(block);
}

ParticleCount ownedParticleCount([[maybe_unused]] ParticleStorage const& particles, CartesianGrid const& grid) {
    PROFILE();
    CartesianGridLayout const& layout = grid.layout();
    ParticleCount owned = 0;
    for (size_t cellIndex = 0; cellIndex < layout.storedCellCount(); cellIndex++) {
        GridIndices const coordinates = layout.cellCoordinates(cellIndex);
        if (!layout.isInterior(coordinates)) {
            continue;
        }
        Cell const& cell = grid.cell(coordinates);
        ParticleRange const range = cell.localParticles;
        expectRange(range.begin, range.end, particles.size());
        expectEqual(range.begin, owned);
        expectEqual(cell.ownerParticles.begin, range.begin);
        expectEqual(cell.ownerParticles.end, range.end);
        owned = range.end;
    }
    return owned;
}

SiloOptions unitMeshOptions(char const* lengthUnits, EvolutionSiloScalars const* evolution) {
    constexpr int kOptionCount = NDIM + 1;
    constexpr std::array<int, NDIM> kAxisUnitOptions{DBOPT_XUNITS, DBOPT_YUNITS, DBOPT_ZUNITS};
    PROFILE();

    SiloOptions options(DBMakeOptlist(kOptionCount), hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    if (!options) {
        checkSilo(-1, "allocate unit mesh options");
    }
    for (int option : kAxisUnitOptions) {
        checkSilo(DBAddOption(options.get(), option, const_cast<char*>(lengthUnits)), "set mesh axis units");
    }
    if (evolution != nullptr) {
        checkSilo(DBAddOption(options.get(), DBOPT_DTIME, const_cast<double*>(&evolution->conformalTime)), "set derived mesh conformal time");
    }
    return options;
}

SiloOptions unitVariableOptions(char const* unitName) {
    constexpr int kOptionCount = 1;
    PROFILE();

    SiloOptions options(DBMakeOptlist(kOptionCount), hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    if (!options) {
        checkSilo(-1, "allocate unit variable options");
    }
    checkSilo(DBAddOption(options.get(), DBOPT_UNITS, const_cast<char*>(unitName)), "set variable units");
    return options;
}

void writeUnitMultis(DBfile* file, std::array<std::vector<std::string>, kViewLeafNames.size()> const& references, EvolutionSiloScalars const* evolution) {
    constexpr int kOptionCount = 1;
    PROFILE();

    SiloOptions meshOptions(DBMakeOptlist(kOptionCount), hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    SiloOptions variableOptions(DBMakeOptlist(kOptionCount), hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    if (!meshOptions || !variableOptions) {
        checkSilo(-1, "allocate unit multiblock options");
    }
    if (evolution != nullptr) {
        checkSilo(DBAddOption(meshOptions.get(), DBOPT_DTIME, const_cast<double*>(&evolution->conformalTime)), "set unit multimesh time");
    }
    char meshName[] = "particles";
    checkSilo(DBAddOption(variableOptions.get(), DBOPT_MMESH_NAME, meshName), "set unit multivariable mesh association");
    for (size_t field = 0; field < references.size(); field++) {
        if (references[field].empty()) {
            continue;
        }
        std::vector<char const*> names;
        names.reserve(references[field].size());
        for (auto const& name : references[field]) {
            names.push_back(name.c_str());
        }
        bool const isMesh = field == kViewMesh;
        std::vector<int> types(names.size(), isMesh ? DB_POINTMESH : DB_POINTVAR);
        if (isMesh) {
            checkSilo(DBPutMultimesh(file, kViewMasterNames[field], static_cast<int>(names.size()), names.data(), types.data(), meshOptions.get()),
                "unit multimesh output");
        } else {
            checkSilo(DBPutMultivar(file, kViewMasterNames[field], static_cast<int>(names.size()), names.data(), types.data(), variableOptions.get()),
                "unit multivariable output");
        }
    }
}

void writeUnitViews(DBfile* file, ParticleStorage const& particles, ParticleCount owned, InitialConditionsSiloMetadata const& metadata,
    EvolutionSiloScalars const* evolution, ParticleSiloOutput const* output) {
    constexpr double kHubbleUnit = 100.0;
    PROFILE();

    if (output != nullptr) {
        if (output->particleCount < owned) {
            throw std::invalid_argument("Silo synchronized momentum does not cover all owned particles");
        }
        for (auto const* component : output->synchronizedMomenta) {
            if (owned != 0 && component == nullptr) {
                throw std::invalid_argument("Silo synchronized momentum has a missing component");
            }
        }
    }
    units::UnitScales const scales(metadata.cosmology.hubble, metadata.boxLength);
    double const scaleFactor = evolution != nullptr ? evolution->scaleFactor : metadata.scaleFactor;
    bool const hasVelocity = evolution == nullptr || output != nullptr;
    double velocityFactor;
    double particleMassSolar;
    {
        FpeGuard guard;
        expectPositive(scaleFactor);
        velocityFactor = metadata.canonicalMomentum || output != nullptr ?
            1.0 / scaleFactor :
            SimulationBackground(metadata.cosmology).conformalDerivative(scaleFactor) / scaleFactor;
        particleMassSolar = evolution != nullptr && evolution->gravityParticleMass >= 0.0 ? evolution->gravityParticleMass * scales.massSolar() :
                                                                                            metadata.particleMass / (metadata.cosmology.hubble / kHubbleUnit);
    }
    writeInt(file, "particle_unit_views_version", kUnitViewVersion);
    writeInt(file, "particle_unit_views_have_velocity", hasVelocity ? 1 : 0);
    checkSilo(DBMkDir(file, "output_units"), "create output unit metadata");
    checkSilo(DBSetDir(file, "output_units"), "enter output unit metadata");
    writeInt(file, "format_version", kUnitViewVersion);
    writeDouble(file, "scale_factor", scaleFactor);
    writeDouble(file, "hubble_parameter", metadata.cosmology.hubble / kHubbleUnit);
    writeDouble(file, "code_length_mpc", scales.lengthMpc());
    writeDouble(file, "code_length_cm", scales.lengthCgs());
    writeDouble(file, "code_time_s", scales.timeSeconds());
    writeDouble(file, "code_mass_g", scales.massCgs());
    writeDouble(file, "code_velocity_cm_s", scales.velocityCgs());
    writeDouble(file, "code_acceleration_cm_s2", scales.accelerationCgs());
    writeDouble(file, "code_potential_erg_g", scales.potentialCgs());
    writeDouble(file, "code_energy_erg", scales.energyCgs());
    writeText(file, "velocity_convention", "physical peculiar velocity; Hubble flow excluded");
    writeText(file, "velocity_epoch", hasVelocity ? "synchronized at scale_factor" : "omitted; synchronized momenta unavailable");
    writeText(file, "native_velocity_convention", metadata.canonicalMomentum ? "canonical momentum a d(x/L)/d(H0 tau)" : "d(x/L)/d ln(a)");
    writeText(file, "position_origin", "periodic box origin; coordinates in the primary box");
    writeText(file, "animation_time", "conformal time since original handoff in H0^-1");
    if (evolution != nullptr) {
        double physicalTime;
        double conformalTime;
        {
            FpeGuard guard;
            physicalTime = evolution->yearsFromPresent * units::kYear.value();
            conformalTime = evolution->conformalTime * scales.timeSeconds();
        }
        writeDouble(file, "physical_time_s", physicalTime);
        writeText(file, "physical_time_origin", "expanding a=1");
        writeDouble(file, "conformal_time_s", conformalTime);
        writeText(file, "conformal_time_origin", "original_initial_conditions");
    }
    checkSilo(DBSetDir(file, "/"), "leave output unit metadata");

    auto const numberOfBlocks = blockCount(owned);
    for (size_t view = 0; view < kUnitViewNames.size(); view++) {
        bool const cgs = view != 0;
        char const* const viewName = kUnitViewNames[view];
        char const* const lengthUnits = cgs ? "cm" : "Mpc";
        char const* const velocityUnits = cgs ? "cm/s" : "km/s";
        char const* const massUnits = cgs ? "g" : "Msun";
        char const* const accelerationUnits = cgs ? "cm/s^2" : "(km/s)^2/Mpc";
        char const* const potentialUnits = cgs ? "erg/g" : "(km/s)^2";
        double lengthFactor;
        double velocityUnitFactor;
        double mass;
        double accelerationFactor;
        double potentialFactor;
        {
            FpeGuard guard;
            lengthFactor = cgs ? scaleFactor * scales.lengthCgs() : scales.lengthMpc();
            velocityUnitFactor = velocityFactor * (cgs ? scales.velocityCgs() : scales.velocityKmPerSecond());
            mass = particleMassSolar * (cgs ? units::kSolarMass.value() : 1.0);
            double const velocityKmSquared = scales.velocityKmPerSecond() * scales.velocityKmPerSecond();
            accelerationFactor = cgs ? scales.accelerationCgs() / (scaleFactor * scaleFactor) : velocityKmSquared / scales.lengthMpc();
            potentialFactor = cgs ? scales.potentialCgs() / scaleFactor : velocityKmSquared;
        }
        checkSilo(DBMkDir(file, viewName), "create derived unit view");
        checkSilo(DBSetDir(file, viewName), "enter derived unit view");
        writeText(file, "position_convention", cgs ? "proper position" : "comoving position");
        writeText(
            file, "acceleration_convention", cgs ? "physical peculiar gravitational acceleration" : "a^2 times physical peculiar gravitational acceleration");
        writeText(file, "potential_convention",
            cgs ? "physical peculiar gravitational potential per unit mass" : "a times physical peculiar gravitational potential per unit mass");
        writeText(file, "velocity_convention", "physical peculiar velocity; Hubble flow excluded");
        std::array<std::vector<std::string>, kViewLeafNames.size()> references;
        auto meshOptions = unitMeshOptions(lengthUnits, evolution);
        auto velocityOptions = unitVariableOptions(velocityUnits);
        auto massOptions = unitVariableOptions(massUnits);
        auto accelerationOptions = unitVariableOptions(accelerationUnits);
        auto potentialOptions = unitVariableOptions(potentialUnits);
        for (std::uint64_t block = 0; block < numberOfBlocks; block++) {
            auto const directory = blockName(block);
            checkSilo(DBMkDir(file, directory.c_str()), "create derived block");
            checkSilo(DBSetDir(file, directory.c_str()), "enter derived block");
            ParticleCount const first = block * kParticleSiloBlockSize;
            int const count = static_cast<int>(std::min<ParticleCount>(owned - first, kParticleSiloBlockSize));
            std::array<std::vector<double>, NDIM> positions;
            std::array<std::vector<double>, NDIM> velocities;
            std::array<std::vector<double>, NDIM> accelerations;
            std::vector<double> masses(static_cast<size_t>(count), mass);
            std::vector<double> potentials(particles.storesPotential() ? static_cast<size_t>(count) : 0);
            std::vector<char> rungs(static_cast<size_t>(count));
            std::array<void const*, NDIM> positionPointers{};
            std::array<void const*, NDIM> velocityPointers{};
            std::array<void const*, NDIM> accelerationPointers{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                positions[axis].resize(static_cast<size_t>(count));
                positionPointers[axis] = positions[axis].data();
                if (hasVelocity) {
                    velocities[axis].resize(static_cast<size_t>(count));
                    velocityPointers[axis] = velocities[axis].data();
                }
                if (particles.storesAcceleration()) {
                    accelerations[axis].resize(static_cast<size_t>(count));
                    accelerationPointers[axis] = accelerations[axis].data();
                }
            }
            for (int offset = 0; offset < count; offset++) {
                FpeGuard guard;
                ParticleCount const index = first + static_cast<ParticleCount>(offset);
                for (size_t axis = 0; axis < NDIM; axis++) {
                    positions[axis][static_cast<size_t>(offset)] =
                        std::ldexp(static_cast<double>(particles.position(axis, index).raw()), -Coordinate::fractionalBits()) * lengthFactor;
                    if (hasVelocity) {
                        double const momentum = output == nullptr ? double(particles.velocity(axis, index)) : output->synchronizedMomenta[axis][index];
                        velocities[axis][static_cast<size_t>(offset)] = momentum * velocityUnitFactor;
                    }
                    if (particles.storesAcceleration()) {
                        accelerations[axis][static_cast<size_t>(offset)] = double(particles.acceleration(axis, index)) * accelerationFactor;
                    }
                }
                if (particles.storesPotential()) {
                    potentials[static_cast<size_t>(offset)] = double(particles.potential(index)) * potentialFactor;
                }
                rungs[static_cast<size_t>(offset)] = static_cast<char>(particles.flags(index).rung());
            }
            checkSilo(DBPutPointmesh(file, "mesh", NDIM, positionPointers.data(), count, DB_DOUBLE, meshOptions.get()), "derived point mesh");
            checkSilo(DBPutPointvar1(file, "mass", "mesh", masses.data(), count, DB_DOUBLE, massOptions.get()), "derived particle mass");
            checkSilo(DBPutPointvar1(file, "rung", "mesh", rungs.data(), count, DB_CHAR, nullptr), "derived particle rung");
            if (hasVelocity) {
                checkSilo(DBPutPointvar(file, "velocity", "mesh", NDIM, velocityPointers.data(), count, DB_DOUBLE, velocityOptions.get()),
                    "derived peculiar velocity");
            }
            if (particles.storesAcceleration()) {
                checkSilo(DBPutPointvar(file, "acceleration", "mesh", NDIM, accelerationPointers.data(), count, DB_DOUBLE, accelerationOptions.get()),
                    "derived acceleration");
            }
            if (particles.storesPotential()) {
                checkSilo(DBPutPointvar1(file, "potential", "mesh", potentials.data(), count, DB_DOUBLE, potentialOptions.get()), "derived potential");
            }
            for (size_t field = 0; field < references.size(); field++) {
                if ((field == kViewVelocity && !hasVelocity) || (field == kViewAcceleration && !particles.storesAcceleration()) ||
                    (field == kViewPotential && !particles.storesPotential())) {
                    continue;
                }
                references[field].push_back(directory + "/" + kViewLeafNames[field]);
            }
            checkSilo(DBSetDir(file, ".."), "leave derived block");
        }
        writeUnitMultis(file, references, evolution);
        checkSilo(DBSetDir(file, "/"), "leave derived unit view");
    }
}

void writeUnitMasters(DBfile* master, std::vector<std::string> const& localityFiles, std::optional<double> conformalTime) {
    PROFILE();
    std::array<std::array<std::vector<std::string>, kViewLeafNames.size()>, kUnitViewNames.size()> references;
    int version = -1;
    int hasVelocity = -1;
    double scaleFactor = 0.0;
    double codeLength = 0.0;
    double codeTime = 0.0;
    for (auto const& localityFile : localityFiles) {
        SiloFile input(DBOpen(localityFile.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        if (!input) {
            checkSilo(-1, "unit visualization locality open");
        }
        int const localVersion = DBInqVarExists(input.get(), "particle_unit_views_version") ? readInt(input.get(), "particle_unit_views_version") : 0;
        if (version < 0) {
            version = localVersion;
        } else if (version != localVersion) {
            throw std::runtime_error("Silo visualization locality unit views differ");
        }
        if (version == 0) {
            closeSilo(input);
            continue;
        }
        if (version != kUnitViewVersion) {
            throw std::runtime_error("Silo visualization unit-view version is incompatible");
        }
        int const velocity = readInt(input.get(), "particle_unit_views_have_velocity");
        double const localScale = readDouble(input.get(), "output_units/scale_factor");
        double const localLength = readDouble(input.get(), "output_units/code_length_cm");
        double const localTime = readDouble(input.get(), "output_units/code_time_s");
        if (hasVelocity < 0) {
            hasVelocity = velocity;
            scaleFactor = localScale;
            codeLength = localLength;
            codeTime = localTime;
        } else if (hasVelocity != velocity || scaleFactor != localScale || codeLength != localLength || codeTime != localTime) {
            throw std::runtime_error("Silo visualization locality unit scales or velocity availability differ");
        }
        if (velocity != 0 && velocity != 1) {
            throw std::runtime_error("Silo visualization velocity flag is invalid");
        }
        int const acceleration = readInt(input.get(), "particle_has_acceleration");
        int const potential = readInt(input.get(), "particle_has_potential");
        auto const blocks = blockCount(readCount(input.get(), "particle_count"));
        auto const prefix = hpx::filesystem::path(localityFile).filename().string() + ":/";
        for (size_t view = 0; view < kUnitViewNames.size(); view++) {
            for (std::uint64_t block = 0; block < blocks; block++) {
                auto const directory = prefix + kUnitViewNames[view] + "/" + blockName(block) + "/";
                for (size_t field = 0; field < kViewLeafNames.size(); field++) {
                    if ((field == kViewVelocity && !velocity) || (field == kViewAcceleration && !acceleration) || (field == kViewPotential && !potential)) {
                        continue;
                    }
                    references[view][field].push_back(directory + kViewLeafNames[field]);
                }
            }
        }
        closeSilo(input);
    }
    if (version == 0) {
        return;
    }
    EvolutionSiloScalars evolution{};
    evolution.conformalTime = conformalTime.value_or(0.0);
    writeInt(master, "particle_unit_views_version", version);
    writeInt(master, "particle_unit_views_have_velocity", hasVelocity);
    checkSilo(DBMkDir(master, "output_units"), "create master unit metadata");
    checkSilo(DBSetDir(master, "output_units"), "enter master unit metadata");
    writeDouble(master, "scale_factor", scaleFactor);
    writeDouble(master, "code_length_cm", codeLength);
    writeDouble(master, "code_time_s", codeTime);
    writeText(master, "velocity_convention", "physical peculiar velocity; Hubble flow excluded");
    checkSilo(DBSetDir(master, "/"), "leave master unit metadata");
    for (size_t view = 0; view < kUnitViewNames.size(); view++) {
        checkSilo(DBMkDir(master, kUnitViewNames[view]), "create master unit view");
        checkSilo(DBSetDir(master, kUnitViewNames[view]), "enter master unit view");
        writeUnitMultis(master, references[view], conformalTime ? &evolution : nullptr);
        checkSilo(DBSetDir(master, "/"), "leave master unit view");
    }
}

void writeBlocks(DBfile* file, ParticleStorage const& particles, [[maybe_unused]] CartesianGrid const& grid, ParticleCount count,
    EvolutionSiloScalars const* evolutionScalars) {
    PROFILE();
    SiloOptions options(nullptr, hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    double animationTime = 0.0;
    if (evolutionScalars != nullptr) {
        animationTime = evolutionScalars->conformalTime;
        options.reset(DBMakeOptlist(1));
        if (!options) {
            checkSilo(-1, "allocate mesh time options");
        }
        checkSilo(DBAddOption(options.get(), DBOPT_DTIME, &animationTime), "set conformal mesh time");
    }
    std::uint64_t const numberOfBlocks = blockCount(count);
    std::vector<std::string> meshNames;
    std::vector<std::string> velocityNames;
    std::vector<std::string> accelerationNames;
    std::vector<std::string> potentialNames;
    std::vector<std::string> rungNames;
    meshNames.reserve(static_cast<size_t>(numberOfBlocks));
    velocityNames.reserve(static_cast<size_t>(numberOfBlocks));
    if (particles.storesAcceleration()) {
        accelerationNames.reserve(static_cast<size_t>(numberOfBlocks));
    }
    if (particles.storesPotential()) {
        potentialNames.reserve(static_cast<size_t>(numberOfBlocks));
    }
    rungNames.reserve(static_cast<size_t>(numberOfBlocks));

    for (std::uint64_t block = 0; block < numberOfBlocks; block++) {
        std::string const directory = blockName(block);
        checkSilo(DBMkDir(file, directory.c_str()), "create block directory");
        checkSilo(DBSetDir(file, directory.c_str()), "enter block directory");

        ParticleCount const first = block * kParticleSiloBlockSize;
        int const countInBlock = static_cast<int>(std::min<ParticleCount>(count - first, kParticleSiloBlockSize));
        std::array<std::vector<double>, NDIM> positions;
        std::vector<char> rungs(static_cast<size_t>(countInBlock));
        std::array<void const*, NDIM> positionPointers{};
        std::array<void const*, NDIM> velocityPointers{};
        std::array<void const*, NDIM> accelerationPointers{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions[axis].resize(static_cast<size_t>(countInBlock));
            positionPointers[axis] = positions[axis].data();
            velocityPointers[axis] = particles.velocityData(axis) + static_cast<size_t>(first);
            if (particles.storesAcceleration()) {
                accelerationPointers[axis] = particles.accelerationData(axis) + static_cast<size_t>(first);
            }
        }
        for (int offset = 0; offset < countInBlock; offset++) {
            ParticleCount const index = first + static_cast<ParticleCount>(offset);
            FpeGuard guard;
            if constexpr (validationChecksEnabled()) {
                auto const& layout = grid.layout();
                auto const owner = layout.particleOwner({particles.position(0, index), particles.position(1, index), particles.position(2, index)});
                expectEqual(owner.localityIndex, layout.localityIndex());
                [[maybe_unused]] auto const range = grid.cell(owner.coordinates).localParticles;
                expect(index >= range.begin);
                expectIndex(index, range.end);
                expect(!particles.flags(index).isGhost());
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                positions[axis][static_cast<size_t>(offset)] =
                    std::ldexp(static_cast<double>(particles.position(axis, index).raw()), -Coordinate::fractionalBits());
            }
            rungs[static_cast<size_t>(offset)] = static_cast<char>(particles.flags(index).rung());
        }
        checkSilo(DBPutPointmesh(file, "mesh", NDIM, positionPointers.data(), countInBlock, DB_DOUBLE, options.get()), "point mesh output");
        checkSilo(DBPutPointvar(file, "velocity", "mesh", NDIM, velocityPointers.data(), countInBlock, DB_FLOAT, nullptr), "velocity output");
        if (particles.storesAcceleration()) {
            checkSilo(DBPutPointvar(file, "acceleration", "mesh", NDIM, accelerationPointers.data(), countInBlock, DB_FLOAT, nullptr), "acceleration output");
            accelerationNames.push_back(directory + "/acceleration");
        }
        if (particles.storesPotential()) {
            checkSilo(DBPutPointvar1(file, "potential", "mesh", particles.potentialData() + static_cast<size_t>(first), countInBlock, DB_FLOAT, nullptr),
                "potential output");
            potentialNames.push_back(directory + "/potential");
        }
        checkSilo(DBPutPointvar1(file, "rung", "mesh", rungs.data(), countInBlock, DB_CHAR, nullptr), "rung output");
        checkSilo(DBSetDir(file, "/"), "leave block directory");

        meshNames.push_back(directory + "/mesh");
        velocityNames.push_back(directory + "/velocity");
        rungNames.push_back(directory + "/rung");
    }

    if (numberOfBlocks != 0) {
        std::vector<char const*> names(static_cast<size_t>(numberOfBlocks));
        std::vector<int> types(static_cast<size_t>(numberOfBlocks));
        auto putMulti = [&](char const* name, std::vector<std::string> const& blockNames, int type) {
            PROFILE();
            for (size_t index = 0; index < names.size(); index++) {
                names[index] = blockNames[index].c_str();
                types[index] = type;
            }
            if (type == DB_POINTMESH) {
                checkSilo(DBPutMultimesh(file, name, static_cast<int>(numberOfBlocks), names.data(), types.data(), options.get()), "multiblock point mesh");
            } else {
                checkSilo(DBPutMultivar(file, name, static_cast<int>(numberOfBlocks), names.data(), types.data(), nullptr), "multiblock point variable");
            }
        };
        putMulti("particles", meshNames, DB_POINTMESH);
        putMulti("velocity", velocityNames, DB_POINTVAR);
        if (particles.storesAcceleration()) {
            putMulti("acceleration", accelerationNames, DB_POINTVAR);
        }
        if (particles.storesPotential()) {
            putMulti("potential", potentialNames, DB_POINTVAR);
        }
        putMulti("rung", rungNames, DB_POINTVAR);
    }
}

Coordinate readCoordinate(double value) {
    PROFILE();
    FpeGuard guard;
    expectNonNegative(value);
    expect(value < 1.0);
    double const scaled = std::ldexp(value, Coordinate::fractionalBits());
    std::uint32_t const raw = static_cast<std::uint32_t>(scaled);
    expectEqual(std::ldexp(static_cast<double>(raw), -Coordinate::fractionalBits()), value);
    return Coordinate::fromRaw(raw);
}

PointVar getPointVar(DBfile* file, char const* name, int countInBlock, int datatype, int components) {
    PROFILE();
    PointVar result(DBGetPointvar(file, name), hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
    if (!result || result->datatype != datatype || result->nels != countInBlock || result->nvals != components || result->vals == nullptr ||
        result->meshname == nullptr || std::string(result->meshname) != "mesh") {
        throw std::runtime_error(std::string("Silo point variable ") + name + " has the wrong schema");
    }
    for (int component = 0; component < components; component++) {
        if (result->vals[component] == nullptr) {
            throw std::runtime_error(std::string("Silo point variable ") + name + " has a missing component");
        }
    }
    return result;
}

void readBlocks(DBfile* file, ParticleStorage& particles, int formatVersion) {
    PROFILE();
    ParticleCount const count = particles.size();
    std::uint64_t const numberOfBlocks = blockCount(count);
    for (std::uint64_t block = 0; block < numberOfBlocks; block++) {
        std::string const directory = blockName(block);
        checkSilo(DBSetDir(file, directory.c_str()), "enter checkpoint block");
        ParticleCount const first = block * kParticleSiloBlockSize;
        int const countInBlock = static_cast<int>(std::min<ParticleCount>(count - first, kParticleSiloBlockSize));

        PointMesh mesh(DBGetPointmesh(file, "mesh"), hpx::annotated_function(&DBFreePointmesh, "PointMesh::free"));
        if (!mesh || mesh->datatype != DB_DOUBLE || mesh->ndims != NDIM || mesh->nels != countInBlock) {
            throw std::runtime_error("Silo point mesh has the wrong schema");
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (mesh->coords[axis] == nullptr) {
                throw std::runtime_error("Silo point mesh has a missing coordinate array");
            }
        }
        PointVar velocity = getPointVar(file, "velocity", countInBlock, DB_FLOAT, NDIM);
        PointVar acceleration(nullptr, hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        PointVar potential(nullptr, hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        if (particles.storesAcceleration()) {
            acceleration = getPointVar(file, "acceleration", countInBlock, DB_FLOAT, NDIM);
        }
        if (particles.storesPotential()) {
            potential = getPointVar(file, "potential", countInBlock, DB_FLOAT, 1);
        }
        PointVar rung = getPointVar(file, "rung", countInBlock, DB_CHAR, 1);
        PointVar flags(nullptr, hpx::annotated_function(&DBFreeMeshvar, "PointVar::free"));
        if (formatVersion < kParticleSiloVersion) {
            flags = getPointVar(file, "flags", countInBlock, DB_CHAR, 1);
        }
        for (int offset = 0; offset < countInBlock; offset++) {
            ParticleCount const index = first + static_cast<ParticleCount>(offset);
            for (size_t axis = 0; axis < NDIM; axis++) {
                particles.position(axis, index) = readCoordinate(static_cast<double const*>(mesh->coords[axis])[offset]);
                particles.velocity(axis, index) = static_cast<float const*>(velocity->vals[axis])[offset];
                if (particles.storesAcceleration()) {
                    particles.acceleration(axis, index) = static_cast<float const*>(acceleration->vals[axis])[offset];
                }
            }
            if (particles.storesPotential()) {
                particles.potential(index) = static_cast<float const*>(potential->vals[0])[offset];
            }
            char const rungValue = static_cast<char const*>(rung->vals[0])[offset];
            if (rungValue < 0 || rungValue > Rung::maximumRung()) {
                throw std::runtime_error("Silo checkpoint has an invalid rung");
            }
            if (flags) {
                std::uint8_t const packedFlags = hpx::bit_cast<std::uint8_t>(static_cast<char const*>(flags->vals[0])[offset]);
                Rung const particleFlags = hpx::bit_cast<Rung>(packedFlags);
                if (particleFlags.rung() != static_cast<std::uint8_t>(rungValue) || particleFlags.isGhost()) {
                    throw std::runtime_error("Silo checkpoint has inconsistent flags or a ghost particle");
                }
            }
            particles.flags(index) = Rung(static_cast<std::uint8_t>(rungValue), false);
        }
        checkSilo(DBSetDir(file, "/"), "leave checkpoint block");
    }
}

void rebuildInteriorRanges(ParticleStorage const& particles, CartesianGrid& grid) {
    PROFILE();
    CartesianGridLayout const& layout = grid.layout();
    ParticleCount cursor = 0;
    for (size_t cellIndex = 0; cellIndex < layout.storedCellCount(); cellIndex++) {
        GridIndices const coordinates = layout.cellCoordinates(cellIndex);
        if (!layout.isInterior(coordinates)) {
            continue;
        }
        ParticleCount const first = cursor;
        while (cursor < particles.size()) {
            std::array<Coordinate, NDIM> position{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                position[axis] = particles.position(axis, cursor);
            }
            GridCellOwner const owner = layout.particleOwner(position);
            if (owner.localityIndex != layout.localityIndex() || owner.coordinates != coordinates) {
                break;
            }
            cursor++;
        }
        grid.setInteriorRange(coordinates, {first, cursor});
    }
    if (cursor != particles.size()) {
        throw std::runtime_error("Silo checkpoint particles are not ordered by interior grid cell");
    }
}

}    // namespace cosmictiger::detail

namespace cosmictiger {

namespace detail {

    void writeParticleSiloImpl(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid,
        InitialConditionsSiloMetadata const* metadata, std::vector<char> const* evolutionPayload = nullptr,
        EvolutionSiloScalars const* evolutionScalars = nullptr, ParticleSiloOutput const* output = nullptr) {
        PROFILE();
        if ((evolutionPayload == nullptr) != (evolutionScalars == nullptr)) {
            throw std::invalid_argument("Silo evolution payload and readable clock metadata must be provided together");
        }
        if (evolutionPayload != nullptr && (evolutionPayload->empty() || evolutionPayload->size() > kMaximumEvolutionPayloadSize)) {
            throw std::length_error("Silo evolution metadata has an invalid payload length");
        }
        ParticleCount const owned = detail::ownedParticleCount(particles, grid);
        std::uint64_t const numberOfBlocks = detail::blockCount(owned);
        if (numberOfBlocks > INT_MAX) {
            throw std::length_error("Silo particle checkpoint has too many point-mesh blocks");
        }
        detail::TemporarySiloFile temporary(filename);
        detail::SiloFile file(DBCreate(temporary.path().c_str(), DB_CLOBBER, DB_LOCAL, "CosmicTiger particle checkpoint", DB_HDF5),
            hpx::annotated_function(&DBClose, "SiloFile::close"));
        if (!file) {
            detail::checkSilo(-1, "file creation");
        }
        detail::writeInt(file.get(), "particle_format_version", detail::kParticleSiloVersion);
        detail::writeInt(file.get(), "particle_dimensions", NDIM);
        detail::writeInt(file.get(), "particle_fractional_bits", Coordinate::fractionalBits());
        detail::writeInt(file.get(), "particle_block_size", detail::kParticleSiloBlockSize);
        detail::writeInt(file.get(), "particle_has_acceleration", particles.storesAcceleration() ? 1 : 0);
        detail::writeInt(file.get(), "particle_has_potential", particles.storesPotential() ? 1 : 0);
        detail::writeCount(file.get(), "particle_count", owned);
        detail::writeCount(file.get(), "grid_global_side", grid.layout().globalSideLength());
        detail::writeCount(file.get(), "grid_locality_side", grid.layout().localitiesPerAxis());
        detail::writeCount(file.get(), "grid_locality_index", grid.layout().localityIndex());
        detail::writeBlocks(file.get(), particles, grid, owned, evolutionScalars);
        if (metadata != nullptr) {
            writeInitialConditionsMetadata(file.get(), *metadata);
            writeUnitViews(file.get(), particles, owned, *metadata, evolutionScalars, output);
        }
        if (evolutionPayload != nullptr) {
            detail::checkSilo(DBMkDir(file.get(), "evolution"), "create evolution metadata directory");
            detail::checkSilo(DBSetDir(file.get(), "evolution"), "enter evolution metadata directory");
            detail::writeInt(file.get(), "format_version", kEvolutionSiloVersion);
            detail::writeText(file.get(), "momentum_convention", "a d(x/L)/d(H0 tau)");
            detail::writeDouble(file.get(), "scale_factor", evolutionScalars->scaleFactor);
            detail::writeDouble(file.get(), "conformal_derivative", evolutionScalars->conformalDerivative);
            detail::writeText(file.get(), "conformal_derivative_units", "d a/d(H0 integral(dt/a))");
            detail::writeDouble(file.get(), "years_from_present", evolutionScalars->yearsFromPresent);
            detail::writeText(file.get(), "physical_time_origin", "expanding a=1");
            detail::writeDouble(file.get(), "conformal_time", evolutionScalars->conformalTime);
            detail::writeText(file.get(), "conformal_time_units", "H0^-1");
            detail::writeText(file.get(), "conformal_time_origin", "original_initial_conditions");
            int const extent = static_cast<int>(evolutionPayload->size());
            detail::checkSilo(DBWrite(file.get(), "payload", evolutionPayload->data(), &extent, 1, DB_CHAR), "write evolution metadata");
            detail::checkSilo(DBSetDir(file.get(), "/"), "leave evolution metadata directory");
        }
        detail::closeSilo(file);
        temporary.commit();
#ifndef PROFILE_DISABLE
        // Report cumulative samples after each local snapshot is published.
        // Keep sampling active and retain the counters for later snapshots.
        profile::writeProfileReport();
#endif
    }

    void writeParticleSiloEvolutionPayload(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid,
        InitialConditionsSiloMetadata const& initial, std::vector<char> const& payload, EvolutionSiloScalars const& scalars, ParticleSiloOutput const* output) {
        PROFILE();
        writeParticleSiloImpl(filename, particles, grid, &initial, &payload, &scalars, output);
    }

    std::vector<char> readParticleSiloEvolutionPayload(std::string const& filename) {
        PROFILE();
        if (filename.empty()) {
            throw std::invalid_argument("Silo evolution checkpoint filename is empty");
        }
        SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        if (!file) {
            checkSilo(-1, "evolution checkpoint open");
        }
        if (DBInqVarExists(file.get(), "evolution/format_version") != 1) {
            throw std::runtime_error("Silo checkpoint has no evolution metadata");
        }
        checkSilo(DBSetDir(file.get(), "evolution"), "enter evolution metadata directory");
        if (readInt(file.get(), "format_version") != kEvolutionSiloVersion) {
            throw std::runtime_error("Silo evolution checkpoint format is incompatible");
        }
        requireText(file.get(), "momentum_convention", "a d(x/L)/d(H0 tau)");
        int const length = DBGetVarLength(file.get(), "payload");
        if (DBGetVarType(file.get(), "payload") != DB_CHAR || length <= 0 || length > kMaximumEvolutionPayloadSize) {
            throw std::runtime_error("Silo evolution metadata has an invalid payload type or length");
        }
        std::vector<char> payload(static_cast<size_t>(length));
        checkSilo(DBReadVar(file.get(), "payload", payload.data()), "read evolution metadata");
        closeSilo(file);
        return payload;
    }

    void validateParticleSiloEvolutionLayout(
        std::string const& filename, size_t gridSideLength, size_t localitiesPerAxis, bool outputAcceleration, bool outputPotential) {
        PROFILE();
        SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        if (!file) {
            checkSilo(-1, "evolution checkpoint layout open");
        }
        int const formatVersion = readInt(file.get(), "particle_format_version");
        if ((formatVersion != 2 && formatVersion != kParticleSiloVersion) || readCount(file.get(), "grid_global_side") != gridSideLength ||
            readCount(file.get(), "grid_locality_side") != localitiesPerAxis || readInt(file.get(), "particle_has_acceleration") != int(outputAcceleration) ||
            readInt(file.get(), "particle_has_potential") != int(outputPotential)) {
            throw std::runtime_error("Silo particle layout or gravity fields disagree with the evolution metadata");
        }
        closeSilo(file);
    }

}    // namespace detail

void writeParticleSilo(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid) {
    PROFILE();
    detail::writeParticleSiloImpl(filename, particles, grid, nullptr);
}

void writeParticleSilo(std::string const& filename, ParticleStorage const& particles, CartesianGrid const& grid, InitialConditionsSiloMetadata const& metadata,
    ParticleSiloOutput const* output) {
    PROFILE();
    detail::writeParticleSiloImpl(filename, particles, grid, &metadata, nullptr, nullptr, output);
}

void writeParticleSiloMaster(std::string const& filename, std::vector<std::string> const& localityFiles, std::optional<double> conformalTime) {
    constexpr std::array<char const*, 5> kLeafNames{"mesh", "velocity", "acceleration", "potential", "rung"};
    constexpr std::array<char const*, 5> kMasterNames{"particles", "velocity", "acceleration", "potential", "rung"};
    PROFILE();
    if (filename.empty() || localityFiles.empty()) {
        throw std::invalid_argument("Silo visualization master requires a filename and locality files");
    }
    auto const masterDirectory = hpx::filesystem::absolute(hpx::filesystem::path(filename)).parent_path();

    std::array<std::vector<std::string>, kLeafNames.size()> references;
    int hasAcceleration = -1;
    int hasPotential = -1;
    for (auto const& localityFile : localityFiles) {
        if (hpx::filesystem::absolute(hpx::filesystem::path(localityFile)).parent_path() != masterDirectory) {
            throw std::invalid_argument("Silo visualization master and locality files must share a directory");
        }
        detail::SiloFile input(DBOpen(localityFile.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
        if (!input) {
            detail::checkSilo(-1, "visualization locality file open");
        }
        int const acceleration = detail::readInt(input.get(), "particle_has_acceleration");
        int const potential = detail::readInt(input.get(), "particle_has_potential");
        if ((acceleration != 0 && acceleration != 1) || (potential != 0 && potential != 1)) {
            throw std::runtime_error("Silo visualization locality field flags are invalid");
        }
        if (hasAcceleration < 0) {
            hasAcceleration = acceleration;
            hasPotential = potential;
        } else {
            if (acceleration != hasAcceleration || potential != hasPotential) {
                throw std::runtime_error("Silo visualization locality fields differ");
            }
        }
        std::uint64_t const blocks = detail::blockCount(detail::readCount(input.get(), "particle_count"));
        if (blocks > static_cast<std::uint64_t>(INT_MAX - references[0].size())) {
            throw std::length_error("Silo visualization master has too many point-mesh blocks");
        }
        std::string const prefix = hpx::filesystem::path(localityFile).filename().string() + ":";
        for (std::uint64_t block = 0; block < blocks; block++) {
            std::string const directory = prefix + detail::blockName(block) + "/";
            for (size_t field = 0; field < kLeafNames.size(); field++) {
                if ((field == 2 && !hasAcceleration) || (field == 3 && !hasPotential)) {
                    continue;
                }
                references[field].push_back(directory + kLeafNames[field]);
            }
        }
        detail::closeSilo(input);
    }
    if (references[0].empty()) {
        throw std::runtime_error("Silo visualization master has no particle meshes");
    }

    detail::TemporarySiloFile temporary(filename);
    detail::SiloFile master(DBCreate(temporary.path().c_str(), DB_CLOBBER, DB_LOCAL, "CosmicTiger particle visualization", DB_HDF5),
        hpx::annotated_function(&DBClose, "SiloFile::close"));
    if (!master) {
        detail::checkSilo(-1, "visualization master file creation");
    }
    detail::SiloOptions meshOptions(nullptr, hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    if (conformalTime) {
        meshOptions.reset(DBMakeOptlist(1));
        if (!meshOptions) {
            detail::checkSilo(-1, "visualization master mesh options allocation");
        }
        detail::checkSilo(DBAddOption(meshOptions.get(), DBOPT_DTIME, &*conformalTime), "visualization master mesh time");
    }
    detail::SiloOptions variableOptions(DBMakeOptlist(1), hpx::annotated_function(&DBFreeOptlist, "SiloOptions::free"));
    if (!variableOptions) {
        detail::checkSilo(-1, "visualization master variable options allocation");
    }
    char meshName[] = "particles";
    detail::checkSilo(DBAddOption(variableOptions.get(), DBOPT_MMESH_NAME, meshName), "visualization master variable mesh association");
    for (size_t field = 0; field < references.size(); field++) {
        if (references[field].empty()) {
            continue;
        }
        std::vector<char const*> names;
        names.reserve(references[field].size());
        for (auto const& reference : references[field]) {
            names.push_back(reference.c_str());
        }
        int const type = field == 0 ? DB_POINTMESH : DB_POINTVAR;
        std::vector<int> types(names.size(), type);
        if (field == 0) {
            detail::checkSilo(DBPutMultimesh(master.get(), kMasterNames[field], static_cast<int>(names.size()), names.data(), types.data(), meshOptions.get()),
                "visualization master point mesh");
        } else {
            detail::checkSilo(
                DBPutMultivar(master.get(), kMasterNames[field], static_cast<int>(names.size()), names.data(), types.data(), variableOptions.get()),
                "visualization master point variable");
        }
    }
    detail::writeUnitMasters(master.get(), localityFiles, conformalTime);
    detail::closeSilo(master);
    temporary.commit();
}

InitialConditionsSiloMetadata readInitialConditionsSiloMetadata(std::string const& filename) {
    PROFILE();
    expect(!(filename.empty()));
    detail::SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    if (!file) {
        detail::checkSilo(-1, "initial-condition file open");
    }
    if (DBInqVarExists(file.get(), "initial_conditions/format_version") != 1) {
        throw std::runtime_error("Silo checkpoint has no initial-condition metadata");
    }
    detail::checkSilo(DBSetDir(file.get(), "initial_conditions"), "enter initial-condition metadata directory");
    int const formatVersion = detail::readInt(file.get(), "format_version");
    if (formatVersion != 1 && formatVersion != detail::kInitialConditionsSiloVersion) {
        throw std::runtime_error("Silo initial-condition metadata format is incompatible");
    }
    detail::requireText(file.get(), "position_units", "unit_box");
    bool const canonicalMomentum = formatVersion == detail::kInitialConditionsSiloVersion;
    detail::requireText(file.get(), "velocity_convention", canonicalMomentum ? "a d(x/L)/d(H0 tau)" : "d(x/L)/d ln(a)");
    if (canonicalMomentum) {
        detail::requireText(file.get(), "conformal_time_units", "H0^-1");
    }
    detail::requireText(file.get(), "box_length_units", "Mpc/h");
    detail::requireText(file.get(), "particle_mass_units", "Msun/h");
    InitialConditionsSiloMetadata metadata{};
    metadata.canonicalMomentum = canonicalMomentum;
    metadata.scaleFactor = expectRange(0, expectPositive(detail::readDouble(file.get(), "scale_factor")), 1);
    metadata.boxLength = expectPositive(detail::readDouble(file.get(), "box_length"));
    metadata.particleMass = expectPositive(detail::readDouble(file.get(), "particle_mass"));
    metadata.maximumDensityDeviation = expectPositive(detail::readDouble(file.get(), "maximum_density_deviation"));
    metadata.requestedDensityDeviation = expectPositive(detail::readDouble(file.get(), "requested_density_deviation"));
    metadata.seed = detail::readCount(file.get(), "random_seed");
    int const meshExponent = detail::readInt(file.get(), "mesh_exponent");
    if (meshExponent < 2 || meshExponent > (std::numeric_limits<size_t>::digits - 1) / NDIM) {
        throw std::runtime_error("Silo initial-condition mesh exponent is out of range");
    }
    metadata.meshExponent = static_cast<size_t>(meshExponent);
    metadata.component = static_cast<MatterComponent>(detail::readInt(file.get(), "matter_component"));
    for (auto const& [name, member] : detail::kCosmologyScalars) {
        metadata.cosmology.*member = detail::checkedCosmologyScalar(detail::readDouble(file.get(), name), member);
    }
    metadata.cosmology.massiveNeutrinos = expectNonNegative(detail::readInt(file.get(), "massive_neutrinos"));
    expect(metadata.cosmology.omegaNeutrinoH2 == 0.0 || metadata.cosmology.massiveNeutrinos > 0);
    metadata.cosmology.masslessNeutrinos = expectNonNegative(detail::readInt(file.get(), "massless_neutrinos"));
    metadata.component = detail::checkedMatterComponent(metadata.component, metadata.cosmology);
    expect(metadata.maximumDensityDeviation >= metadata.requestedDensityDeviation);
    detail::checkSilo(DBSetDir(file.get(), "/"), "leave initial-condition metadata directory");
    detail::closeSilo(file);
    return metadata;
}

void readParticleSilo(std::string const& filename, ParticleStorage& particles, CartesianGrid& grid) {
    PROFILE();
    expect(!(filename.empty()));
    detail::SiloFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), hpx::annotated_function(&DBClose, "SiloFile::close"));
    if (!file) {
        detail::checkSilo(-1, "file open");
    }
    int const formatVersion = detail::readInt(file.get(), "particle_format_version");
    if ((formatVersion < 1 || formatVersion > detail::kParticleSiloVersion) || detail::readInt(file.get(), "particle_dimensions") != NDIM ||
        detail::readInt(file.get(), "particle_fractional_bits") != Coordinate::fractionalBits() ||
        detail::readInt(file.get(), "particle_block_size") != detail::kParticleSiloBlockSize) {
        throw std::runtime_error("Silo particle checkpoint format is incompatible");
    }
    ParticleCount const count = detail::readCount(file.get(), "particle_count");
    CartesianGridLayout const& layout = grid.layout();
    if (detail::readCount(file.get(), "grid_global_side") != layout.globalSideLength() ||
        detail::readCount(file.get(), "grid_locality_side") != layout.localitiesPerAxis() ||
        detail::readCount(file.get(), "grid_locality_index") != layout.localityIndex()) {
        throw std::runtime_error("Silo particle checkpoint grid decomposition does not match");
    }
    if (detail::blockCount(count) > INT_MAX) {
        throw std::length_error("Silo particle checkpoint has too many point-mesh blocks");
    }
    int const hasAcceleration = formatVersion == 1 ? 0 : detail::readInt(file.get(), "particle_has_acceleration");
    int const hasPotential = formatVersion == 1 ? 0 : detail::readInt(file.get(), "particle_has_potential");
    if ((hasAcceleration != 0 && hasAcceleration != 1) || (hasPotential != 0 && hasPotential != 1)) {
        throw std::runtime_error("Silo gravity output field metadata is invalid");
    }
    ParticleStorage replacement;
    replacement.configureGravityFields(hasAcceleration != 0, hasPotential != 0);
    replacement.resize(count);
    detail::readBlocks(file.get(), replacement, formatVersion);
    detail::closeSilo(file);
    size_t const cellExponent = std::countr_zero(layout.globalSideLength());
    size_t const localityExponent = std::countr_zero(layout.localitiesPerAxis());
    CartesianGrid replacementGrid(cellExponent, localityExponent, layout.localityIndex(), layout.ghostWidth());
    detail::rebuildInteriorRanges(replacement, replacementGrid);
    grid = std::move(replacementGrid);
    particles.swap(replacement);
}

}    // namespace cosmictiger
