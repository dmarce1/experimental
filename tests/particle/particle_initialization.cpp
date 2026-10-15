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

#include <cosmictiger/camb.hpp>
#include <cosmictiger/friedmann.hpp>
#include <cosmictiger/particle/particle_initialization.hpp>
#include <cosmictiger/units.hpp>

#include <gtest/gtest.h>
#include <hpx/algorithm.hpp>
#include <hpx/bit.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/runtime_distributed/find_here.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <numbers>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using cosmictiger::CosmologyParameters;
using cosmictiger::EinsteinBoltzmannSolver;
using cosmictiger::GridIndices;
using cosmictiger::MatterPowerRequest;
using cosmictiger::MatterPowerSpectrum;
using cosmictiger::ParticleCount;
using cosmictiger::ParticleInitialConditions;
using cosmictiger::ParticleInitializationOptions;
using cosmictiger::ParticleStorage;

// The mock supplies scale-independent growth, P(k,a) = A*a^(2*g).
// Velocity and crossing expectations below follow directly from that law,
// independently of the implementation's random-number generator and FFT.
class AnalyticPowerSolver final : public EinsteinBoltzmannSolver {
public:
    explicit AnalyticPowerSolver(double basePower = 1.0e4, double growthIndex = 1.0)
      : basePower_(basePower)
      , growthIndex_(growthIndex) {
        PROFILE();
    }

    MatterPowerSpectrum compute(CosmologyParameters const& parameters, MatterPowerRequest const& request) const override {
        PROFILE();
        calls_++;
        double const scaleFactor = 1.0 / (1.0 + request.redshift);
        double const power = basePower_ * (parameters.scalarAmplitude / 2.1e-9) * std::pow(scaleFactor, 2.0 * growthIndex_);
        return {request.redshift, parameters.hubble / 100.0, request.component,
            {{request.maximumWaveNumber * 1.0e-9, power}, {request.maximumWaveNumber * 1.1, power}}};
    }

    size_t calls() const noexcept {
        PROFILE();
        return calls_;
    }

private:
    double basePower_;
    double growthIndex_;
    mutable size_t calls_ = 0;
};

class FailingPowerSolver final : public EinsteinBoltzmannSolver {
public:
    MatterPowerSpectrum compute(CosmologyParameters const&, MatterPowerRequest const&) const override {
        PROFILE();
        calls_++;
        throw std::runtime_error("synthetic root EB failure");
    }

    size_t calls() const noexcept {
        PROFILE();
        return calls_;
    }

private:
    mutable size_t calls_ = 0;
};

class TruncatedPowerSolver final : public EinsteinBoltzmannSolver {
public:
    explicit TruncatedPowerSolver(double minimumWaveNumber)
      : minimumWaveNumber_(minimumWaveNumber) {
        PROFILE();
    }

    MatterPowerSpectrum compute(CosmologyParameters const& parameters, MatterPowerRequest const& request) const override {
        PROFILE();
        calls_++;
        double const scaleFactor = 1.0 / (1.0 + request.redshift);
        double const power = 1.0e4 * scaleFactor * scaleFactor;
        return {request.redshift, parameters.hubble / 100.0, request.component, {{minimumWaveNumber_, power}, {request.maximumWaveNumber * 1.1, power}}};
    }

    size_t calls() const noexcept {
        PROFILE();
        return calls_;
    }

private:
    double minimumWaveNumber_;
    mutable size_t calls_ = 0;
};

// Test callers allocate IDs sequentially before entering collective tasks.
std::uint64_t nextOperationId() {
    PROFILE();
    static std::uint64_t sequence = 0x4943540000000000ULL;
    return sequence++;
}

ParticleInitializationOptions smallOptions() {
    PROFILE();
    ParticleInitializationOptions options;
    options.meshExponent = 3;
    options.boxLength = 100.0;
    options.seed = 31;
    options.maximumDensityDeviation = 0.025;
    options.earliestScaleFactor = 1.0e-4;
    options.latestScaleFactor = 1.0;
    options.maximumLogScaleFactorStep = 0.5;
    options.logScaleFactorTolerance = 1.0e-6;
    options.growthDerivativeStep = 1.0e-3;
    options.maximumSpectrumEvaluations = 256;
    return options;
}

ParticleInitialConditions initializeSingle(
    CosmologyParameters const& parameters, ParticleInitializationOptions const& options, EinsteinBoltzmannSolver const& solver) {
    PROFILE();
    std::vector<hpx::id_type> const sites{hpx::find_here()};
    auto const operationId = nextOperationId();
    return hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                return cosmictiger::initializeParticles(parameters, options, sites, 0, operationId, solver);
            },
            "test.particle.initializeSingle"))
        .get();
}

std::vector<ParticleInitialConditions> initializeLogicalSites(CosmologyParameters const& parameters, ParticleInitializationOptions const& options,
    EinsteinBoltzmannSolver const& solver, size_t siteCount, size_t* failedSites = nullptr) {
    PROFILE();
    std::vector<hpx::id_type> const sites(siteCount, hpx::find_here());
    auto const operationId = nextOperationId();
    std::vector<hpx::future<ParticleInitialConditions>> tasks;
    tasks.reserve(siteCount);
    for (size_t site = 0; site < siteCount; site++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, site] {
                    PROFILE();
                    return cosmictiger::initializeParticles(parameters, options, sites, site, operationId, solver);
                },
                "test.particle.initializeLogicalSite")));
    }
    std::vector<ParticleInitialConditions> results;
    results.reserve(siteCount);
    std::string firstError;
    size_t failureCount = 0;
    for (auto& task : tasks) {
        try {
            results.emplace_back(task.get());
        } catch (std::exception const& error) {
            failureCount++;
            if (firstError.empty()) {
                firstError = error.what();
            }
        } catch (...) {
            failureCount++;
            if (firstError.empty()) {
                firstError = "unknown initialization failure";
            }
        }
    }
    // Every task captures this helper's inputs. Drain them before propagating
    // an error so their references remain alive throughout collective failure.
    if (failedSites != nullptr) {
        *failedSites = failureCount;
    }
    if (!firstError.empty()) {
        throw std::runtime_error(firstError);
    }
    return results;
}

using ParticleRecord = std::tuple<std::array<std::uint32_t, NDIM>, std::array<std::uint32_t, NDIM>, std::uint8_t, bool>;

std::vector<ParticleRecord> records(ParticleStorage const& particles) {
    PROFILE();
    std::vector<ParticleRecord> result;
    result.reserve(static_cast<size_t>(particles.size()));
    for (ParticleCount particle = 0; particle < particles.size(); particle++) {
        std::array<std::uint32_t, NDIM> position{};
        std::array<std::uint32_t, NDIM> velocity{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            position[axis] = particles.position(axis, particle).raw();
            velocity[axis] = hpx::bit_cast<std::uint32_t>(particles.velocity(axis, particle));
        }
        result.emplace_back(position, velocity, particles.flags(particle).rung(), particles.flags(particle).isGhost());
    }
    hpx::sort(hpx::execution::seq, result.begin(), result.end());
    return result;
}

struct LatticeRecord {
    size_t latticeIndex;
    std::array<double, NDIM> displacement;
    std::array<float, NDIM> velocity;
};

std::vector<LatticeRecord> latticeRecords(ParticleStorage const& particles, size_t side) {
    PROFILE();
    std::vector<LatticeRecord> result;
    result.reserve(static_cast<size_t>(particles.size()));
    for (ParticleCount particle = 0; particle < particles.size(); particle++) {
        LatticeRecord entry{};
        size_t stride = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const position = particles.position(axis, particle).toReal<double>();
            size_t const latticeCoordinate = static_cast<size_t>(std::floor(position * static_cast<double>(side)));
            double const latticePosition = (static_cast<double>(latticeCoordinate) + 0.5) / static_cast<double>(side);
            entry.latticeIndex += latticeCoordinate * stride;
            stride *= side;
            entry.displacement[axis] = position - latticePosition;
            entry.velocity[axis] = particles.velocity(axis, particle);
        }
        result.push_back(entry);
    }
    hpx::sort(hpx::execution::seq, result.begin(), result.end(),
        hpx::annotated_function(
            [](LatticeRecord const& left, LatticeRecord const& right) {
                PROFILE();
                return left.latticeIndex < right.latticeIndex;
            },
            "test.particle.compareLatticeRecords"));
    return result;
}

struct ReconstructedDensity {
    double maximumDeviation = 0.0;
    double maximumImaginaryResidual = 0.0;
};

// Independent direct DFT: delta = -div(Psi). Recover the Fourier density
// from stored positions, without using any project FFT or seeded-mode helper.
// The fields and their density maximum are sampled at the lattice centers.
ReconstructedDensity reconstructDensity(std::vector<LatticeRecord> const& particles, size_t side) {
    PROFILE();
    size_t const count = particles.size();
    std::vector<std::array<double, NDIM>> positions(count);
    for (size_t sample = 0; sample < count; sample++) {
        size_t index = particles[sample].latticeIndex;
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions[sample][axis] = (static_cast<double>(index % side) + 0.5) / static_cast<double>(side);
            index /= side;
        }
    }
    std::vector<std::array<double, NDIM>> waves(count);
    std::vector<bool> active(count, true);
    std::vector<std::complex<double>> densitySpectrum(count);
    for (size_t frequency = 0; frequency < count; frequency++) {
        size_t index = frequency;
        for (size_t axis = 0; axis < NDIM; axis++) {
            size_t const coordinate = index % side;
            index /= side;
            if (coordinate == side / 2) {
                active[frequency] = false;
            }
            auto const signedCoordinate =
                coordinate > side / 2 ? static_cast<std::int64_t>(coordinate) - static_cast<std::int64_t>(side) : static_cast<std::int64_t>(coordinate);
            waves[frequency][axis] = 2.0 * std::numbers::pi_v<double> * static_cast<double>(signedCoordinate);
        }
        if (!active[frequency]) {
            continue;
        }
        for (size_t sample = 0; sample < count; sample++) {
            double phase = 0.0;
            double divergence = 0.0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                phase += waves[frequency][axis] * positions[sample][axis];
                divergence += waves[frequency][axis] * particles[sample].displacement[axis];
            }
            densitySpectrum[frequency] += std::complex<double>(0.0, -divergence) * std::complex<double>(std::cos(phase), -std::sin(phase));
        }
    }
    ReconstructedDensity result;
    for (size_t sample = 0; sample < count; sample++) {
        std::complex<double> density{};
        for (size_t frequency = 0; frequency < count; frequency++) {
            if (!active[frequency]) {
                continue;
            }
            double phase = 0.0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                phase += waves[frequency][axis] * positions[sample][axis];
            }
            density += densitySpectrum[frequency] * std::complex<double>(std::cos(phase), std::sin(phase));
        }
        density /= static_cast<double>(count);
        if (std::abs(density.real()) > result.maximumDeviation) {
            result.maximumDeviation = std::abs(density.real());
        }
        if (std::abs(density.imag()) > result.maximumImaginaryResidual) {
            result.maximumImaginaryResidual = std::abs(density.imag());
        }
    }
    return result;
}

void expectValidOwnedRanges(ParticleInitialConditions const& initial) {
    PROFILE();
    auto const& particles = initial.particles;
    auto const& layout = initial.grid.layout();
    EXPECT_EQ(layout.globalCellCount(), initial.report.globalParticleCount);
    ParticleCount cursor = 0;
    for (size_t z = 0; z < layout.interiorSideLength(); z++) {
        for (size_t y = 0; y < layout.interiorSideLength(); y++) {
            for (size_t x = 0; x < layout.interiorSideLength(); x++) {
                GridIndices const coordinate{static_cast<std::int64_t>(x), static_cast<std::int64_t>(y), static_cast<std::int64_t>(z)};
                auto const& cell = initial.grid.cell(coordinate);
                ASSERT_EQ(cell.localParticles.begin, cursor);
                ASSERT_EQ(cell.ownerParticles.begin, cell.localParticles.begin);
                ASSERT_EQ(cell.ownerParticles.end, cell.localParticles.end);
                ASSERT_LE(cell.localParticles.end, particles.size());
                ASSERT_TRUE(cell.rungRangesValid);
                EXPECT_EQ(cell.rungParticles[0].begin, cell.localParticles.begin);
                EXPECT_EQ(cell.rungParticles[0].end, cell.localParticles.end);
                for (size_t rung = 1; rung < cell.rungParticles.size(); rung++) {
                    EXPECT_EQ(cell.rungParticles[rung].begin, cell.localParticles.end);
                    EXPECT_EQ(cell.rungParticles[rung].end, cell.localParticles.end);
                }
                for (ParticleCount particle = cell.localParticles.begin; particle < cell.localParticles.end; particle++) {
                    std::array<cosmictiger::Coordinate, NDIM> position{};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        position[axis] = particles.position(axis, particle);
                    }
                    auto const owner = layout.particleOwner(position);
                    EXPECT_EQ(owner.localityIndex, layout.localityIndex());
                    EXPECT_EQ(owner.coordinates, coordinate);
                    EXPECT_EQ(particles.flags(particle).rung(), 0);
                    EXPECT_FALSE(particles.flags(particle).isGhost());
                }
                cursor = cell.localParticles.end;
            }
        }
    }
    EXPECT_EQ(cursor, particles.size());
    for (size_t cell = 0; cell < layout.storedCellCount(); cell++) {
        auto const coordinate = layout.cellCoordinates(cell);
        if (!layout.isInterior(coordinate)) {
            auto const& ghost = initial.grid.cell(coordinate);
            EXPECT_EQ(ghost.localParticles.begin, ghost.localParticles.end);
            EXPECT_EQ(ghost.ownerParticles.begin, ghost.ownerParticles.end);
        }
    }
}

void expectGrowingMode(ParticleInitialConditions const& initial, ParticleInitializationOptions const& options, double growthIndex) {
    PROFILE();
    size_t const side = size_t(1) << options.meshExponent;
    EXPECT_EQ(initial.grid.layout().globalSideLength(), side);
    auto const entries = latticeRecords(initial.particles, side);
    double const derivativeFactor = std::sinh(growthIndex * options.growthDerivativeStep) / options.growthDerivativeStep;
    double displacementSquared = 0.0;
    for (size_t particle = 0; particle < entries.size(); particle++) {
        ASSERT_EQ(entries[particle].latticeIndex, particle);
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const displacement = entries[particle].displacement[axis];
            ASSERT_LT(std::abs(displacement), 0.5 / static_cast<double>(side));
            EXPECT_NEAR(entries[particle].velocity[axis], derivativeFactor * displacement, 2.0e-8);
            displacementSquared += displacement * displacement;
        }
    }
    EXPECT_GT(displacementSquared, 1.0e-8);
}

TEST(ParticleInitialization, BracketsTheLinearThresholdAndCreatesGrowingModeParticles) {
    constexpr double kCriticalDensityNumerator = 3.0;
    constexpr double kCriticalDensityDenominator = 8.0;
    constexpr double kMassTolerance = 1.0e-14;
    PROFILE();
    CosmologyParameters const parameters;
    auto const options = smallOptions();
    AnalyticPowerSolver const solver;
    auto const initial = initializeSingle(parameters, options, solver);
    auto const& report = initial.report;
    EXPECT_GE(report.maximumDensityDeviation, options.maximumDensityDeviation);
    EXPECT_LT(report.previousDensityDeviation, options.maximumDensityDeviation);
    EXPECT_GE(report.previousScaleFactor, options.earliestScaleFactor);
    EXPECT_LT(report.previousScaleFactor, report.scaleFactor);
    EXPECT_LT(report.scaleFactor, options.latestScaleFactor);
    EXPECT_LE(std::log(report.scaleFactor / report.previousScaleFactor), options.logScaleFactorTolerance * 1.01);
    EXPECT_NEAR(report.previousDensityDeviation / report.maximumDensityDeviation, report.previousScaleFactor / report.scaleFactor, 1.0e-11);
    EXPECT_NEAR(report.redshift, 1.0 / report.scaleFactor - 1.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(report.boxLength, options.boxLength);
    EXPECT_EQ(report.globalParticleCount, 512U);
    EXPECT_EQ(initial.particles.size(), report.globalParticleCount);
    EXPECT_EQ(report.spectrumEvaluations, solver.calls());
    double const h = parameters.hubble / 100.0;
    double const omegaCold = (parameters.omegaBaryonH2 + parameters.omegaCdmH2) / (h * h);
    double const expectedNormalizedMass =
        kCriticalDensityNumerator * omegaCold / (kCriticalDensityDenominator * std::numbers::pi * static_cast<double>(report.globalParticleCount));
    cosmictiger::units::UnitScales const scales(parameters.hubble, options.boxLength);
    // The legacy report is Msun/h; removing h must agree with the mass used by
    // the normalized G=1 force kernels and the physical output conversion.
    double const reportedNormalizedMass = report.particleMass / (h * scales.massSolar());
    EXPECT_NEAR(reportedNormalizedMass / expectedNormalizedMass, 1.0, kMassTolerance);
    expectValidOwnedRanges(initial);
    expectGrowingMode(initial, options, 1.0);
    auto const reconstructed = reconstructDensity(latticeRecords(initial.particles, 8), 8);
    EXPECT_NEAR(reconstructed.maximumDeviation, report.maximumDensityDeviation, 1.0e-7);
    EXPECT_LT(reconstructed.maximumImaginaryResidual, 1.0e-11);
}

TEST(ParticleInitialization, SharesCartesianCellsWithTheInitialParticleLattice) {
    PROFILE();
    auto options = smallOptions();
    options.meshExponent = 2;
    options.maximumDensityDeviation = 1.0e-6;
    options.earliestScaleFactor = 1.0e-5;
    AnalyticPowerSolver const solver(1.0e-4);
    auto const initial = initializeSingle({}, options, solver);
    size_t const side = size_t(1) << options.meshExponent;
    auto const& layout = initial.grid.layout();
    ASSERT_EQ(layout.globalSideLength(), side);
    ASSERT_EQ(layout.globalCellCount(), initial.particles.size());
    auto const entries = latticeRecords(initial.particles, side);
    ASSERT_EQ(entries.size(), layout.globalCellCount());
    for (size_t particle = 0; particle < entries.size(); particle++) {
        EXPECT_EQ(entries[particle].latticeIndex, particle);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_LT(std::abs(entries[particle].displacement[axis]), 1.0e-4 * layout.spacing());
        }
    }
    // This gentle realization stays within its original cells. General
    // Zel'dovich displacements are still allowed to change cell occupancy.
    for (size_t cell = 0; cell < layout.storedCellCount(); cell++) {
        auto const coordinates = layout.cellCoordinates(cell);
        if (layout.isInterior(coordinates)) {
            auto const& range = initial.grid.cell(coordinates).localParticles;
            EXPECT_EQ(range.end - range.begin, 1);
        }
    }
    expectValidOwnedRanges(initial);
}

TEST(ParticleInitialization, UsesSpectrumGrowthRatherThanAssumingUnitGrowthRate) {
    constexpr double kGrowthIndex = 1.7;
    PROFILE();

    auto const options = smallOptions();
    AnalyticPowerSolver const solver(1.0e4, kGrowthIndex);
    auto const initial = initializeSingle({}, options, solver);
    expectGrowingMode(initial, options, kGrowthIndex);
    EXPECT_NEAR(initial.report.previousDensityDeviation / initial.report.maximumDensityDeviation,
        std::pow(initial.report.previousScaleFactor / initial.report.scaleFactor, kGrowthIndex), 1.0e-11);
}

TEST(ParticleInitialization, AcceptsAUserSelectedUnityDensityThreshold) {
    PROFILE();
    auto options = smallOptions();
    options.maximumDensityDeviation = 1.0;
    AnalyticPowerSolver const solver;
    auto const initial = initializeSingle({}, options, solver);
    EXPECT_GT(initial.report.maximumDensityDeviation, options.maximumDensityDeviation);
    EXPECT_LE(initial.report.previousDensityDeviation, options.maximumDensityDeviation);
    EXPECT_EQ(initial.particles.size(), 512U);
    expectValidOwnedRanges(initial);
}

TEST(ParticleInitialization, UsesBackwardGrowthDerivativeNearThePresentEpoch) {
    constexpr double kTargetScaleFactor = 0.999;
    PROFILE();

    auto options = smallOptions();
    options.maximumDensityDeviation = 1.0e-3;
    AnalyticPowerSolver const solver(0.1);
    auto const calibration = initializeSingle({}, options, solver);
    double const presentDeviation = calibration.report.maximumDensityDeviation / calibration.report.scaleFactor;
    ASSERT_GT(presentDeviation, 0.0);
    ASSERT_LT(presentDeviation, 1.0);
    options.maximumDensityDeviation = presentDeviation * kTargetScaleFactor;
    options.growthDerivativeStep = 0.02;
    auto const initial = initializeSingle({}, options, solver);
    EXPECT_NEAR(initial.report.scaleFactor, kTargetScaleFactor, 2.0e-6);
    ASSERT_GT(std::log(initial.report.scaleFactor) + options.growthDerivativeStep, 0.0);
    double const step = options.growthDerivativeStep;
    double const derivativeFactor = (3.0 - 4.0 * std::exp(-step) + std::exp(-2.0 * step)) / (2.0 * step);
    auto const entries = latticeRecords(initial.particles, 8);
    double displacementSquared = 0.0;
    for (size_t particle = 0; particle < entries.size(); particle++) {
        ASSERT_EQ(entries[particle].latticeIndex, particle);
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const displacement = entries[particle].displacement[axis];
            EXPECT_NEAR(entries[particle].velocity[axis], derivativeFactor * displacement, 1.0e-9);
            displacementSquared += displacement * displacement;
        }
    }
    EXPECT_GT(displacementSquared, 1.0e-8);
}

TEST(ParticleInitialization, CoarseTasksReproduceTheSeedAndChangeTheRealization) {
    PROFILE();
    auto options = smallOptions();
    options.meshExponent = 4;
    AnalyticPowerSolver const solver;
    auto const first = initializeSingle({}, options, solver);
    auto const repeated = initializeSingle({}, options, solver);
    ASSERT_EQ(first.particles.size(), 4096U);
    EXPECT_EQ(first.report.globalParticleCount, first.particles.size());
    EXPECT_EQ(records(first.particles), records(repeated.particles));
    EXPECT_DOUBLE_EQ(first.report.scaleFactor, repeated.report.scaleFactor);
    EXPECT_DOUBLE_EQ(first.report.redshift, repeated.report.redshift);
    EXPECT_DOUBLE_EQ(first.report.maximumDensityDeviation, repeated.report.maximumDensityDeviation);
    EXPECT_DOUBLE_EQ(first.report.previousScaleFactor, repeated.report.previousScaleFactor);
    EXPECT_DOUBLE_EQ(first.report.previousDensityDeviation, repeated.report.previousDensityDeviation);
    EXPECT_DOUBLE_EQ(first.report.boxLength, repeated.report.boxLength);
    EXPECT_DOUBLE_EQ(first.report.particleMass, repeated.report.particleMass);
    EXPECT_EQ(first.report.globalParticleCount, repeated.report.globalParticleCount);
    EXPECT_EQ(first.report.spectrumEvaluations, repeated.report.spectrumEvaluations);
    EXPECT_EQ(solver.calls(), first.report.spectrumEvaluations + repeated.report.spectrumEvaluations);
    expectValidOwnedRanges(first);
    expectGrowingMode(first, options, 1.0);
    options.seed++;
    auto const different = initializeSingle({}, options, solver);
    EXPECT_NE(records(first.particles), records(different.particles));
}

TEST(ParticleInitialization, PreservesPhasesWhenChangingThresholdAndPrimordialAmplitude) {
    PROFILE();
    CosmologyParameters parameters;
    auto options = smallOptions();
    AnalyticPowerSolver const solver;
    auto const first = initializeSingle(parameters, options, solver);
    auto const firstEntries = latticeRecords(first.particles, 8);
    options.maximumDensityDeviation *= 2.0;
    auto const later = initializeSingle(parameters, options, solver);
    auto const laterEntries = latticeRecords(later.particles, 8);
    double const amplitudeRatio = later.report.scaleFactor / first.report.scaleFactor;
    EXPECT_NEAR(amplitudeRatio, 2.0, 5.0e-6);
    ASSERT_EQ(firstEntries.size(), laterEntries.size());
    for (size_t particle = 0; particle < firstEntries.size(); particle++) {
        ASSERT_EQ(firstEntries[particle].latticeIndex, laterEntries[particle].latticeIndex);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(laterEntries[particle].displacement[axis], amplitudeRatio * firstEntries[particle].displacement[axis], 2.0e-9);
            EXPECT_NEAR(laterEntries[particle].velocity[axis], amplitudeRatio * firstEntries[particle].velocity[axis], 2.0e-8);
        }
    }
    options.maximumDensityDeviation *= 0.5;
    parameters.scalarAmplitude *= 4.0;
    auto const stronger = initializeSingle(parameters, options, solver);
    auto const strongerEntries = latticeRecords(stronger.particles, 8);
    EXPECT_NEAR(stronger.report.scaleFactor / first.report.scaleFactor, 0.5, 2.0e-6);
    double const equalEpochAmplitudeRatio = 2.0 * stronger.report.scaleFactor / first.report.scaleFactor;
    ASSERT_EQ(firstEntries.size(), strongerEntries.size());
    for (size_t particle = 0; particle < firstEntries.size(); particle++) {
        ASSERT_EQ(firstEntries[particle].latticeIndex, strongerEntries[particle].latticeIndex);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(strongerEntries[particle].displacement[axis], equalEpochAmplitudeRatio * firstEntries[particle].displacement[axis], 2.0e-9);
            EXPECT_NEAR(strongerEntries[particle].velocity[axis], equalEpochAmplitudeRatio * firstEntries[particle].velocity[axis], 2.0e-8);
        }
    }
}

TEST(ParticleInitialization, CoarseTasksPreserveTheFullParticleRealizationAcrossLogicalLocalities) {
    PROFILE();
    auto options = smallOptions();
    // Each of eight logical sites starts with 4096 samples, so both layouts
    // exercise coarse tasks when the runtime has multiple workers.
    options.meshExponent = 5;
    AnalyticPowerSolver const singleSolver;
    auto const single = initializeSingle({}, options, singleSolver);
    AnalyticPowerSolver const distributedSolver;
    auto const distributed = initializeLogicalSites({}, options, distributedSolver, 8);
    ASSERT_EQ(distributed.size(), 8U);
    ASSERT_EQ(single.particles.size(), 32768U);
    EXPECT_EQ(single.report.spectrumEvaluations, singleSolver.calls());
    EXPECT_EQ(distributedSolver.calls(), singleSolver.calls());
    std::vector<ParticleRecord> distributedRecords;
    for (auto const& initial : distributed) {
        expectValidOwnedRanges(initial);
        EXPECT_NEAR(initial.report.scaleFactor, single.report.scaleFactor, 1.0e-12);
        EXPECT_NEAR(initial.report.maximumDensityDeviation, single.report.maximumDensityDeviation, 1.0e-12);
        EXPECT_NEAR(initial.report.previousScaleFactor, single.report.previousScaleFactor, 1.0e-12);
        EXPECT_NEAR(initial.report.previousDensityDeviation, single.report.previousDensityDeviation, 1.0e-12);
        EXPECT_DOUBLE_EQ(initial.report.boxLength, single.report.boxLength);
        EXPECT_DOUBLE_EQ(initial.report.particleMass, single.report.particleMass);
        EXPECT_EQ(initial.report.globalParticleCount, single.report.globalParticleCount);
        EXPECT_DOUBLE_EQ(initial.report.scaleFactor, distributed.front().report.scaleFactor);
        EXPECT_DOUBLE_EQ(initial.report.redshift, distributed.front().report.redshift);
        EXPECT_DOUBLE_EQ(initial.report.maximumDensityDeviation, distributed.front().report.maximumDensityDeviation);
        EXPECT_DOUBLE_EQ(initial.report.previousScaleFactor, distributed.front().report.previousScaleFactor);
        EXPECT_DOUBLE_EQ(initial.report.previousDensityDeviation, distributed.front().report.previousDensityDeviation);
        EXPECT_EQ(initial.report.spectrumEvaluations, distributedSolver.calls());
        auto localRecords = records(initial.particles);
        distributedRecords.insert(distributedRecords.end(), localRecords.begin(), localRecords.end());
    }
    hpx::sort(hpx::execution::seq, distributedRecords.begin(), distributedRecords.end());
    auto const singleRecords = records(single.particles);
    ASSERT_EQ(distributedRecords.size(), singleRecords.size());
    for (size_t particle = 0; particle < singleRecords.size(); particle++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const firstPosition = static_cast<std::int64_t>(std::get<0>(singleRecords[particle])[axis]);
            auto const secondPosition = static_cast<std::int64_t>(std::get<0>(distributedRecords[particle])[axis]);
            EXPECT_LE(std::abs(firstPosition - secondPosition), 1);
            EXPECT_NEAR(hpx::bit_cast<float>(std::get<1>(singleRecords[particle])[axis]), hpx::bit_cast<float>(std::get<1>(distributedRecords[particle])[axis]),
                2.0e-8);
        }
        EXPECT_EQ(std::get<2>(singleRecords[particle]), std::get<2>(distributedRecords[particle]));
        EXPECT_EQ(std::get<3>(singleRecords[particle]), std::get<3>(distributedRecords[particle]));
    }
}

TEST(ParticleInitialization, RefusesAlreadyNonlinearAndUnreachedThresholdBrackets) {
    PROFILE();
    auto const options = smallOptions();
    AnalyticPowerSolver const alreadyNonlinear(1.0e18);
    EXPECT_ANY_THROW(initializeSingle({}, options, alreadyNonlinear));
    EXPECT_GT(alreadyNonlinear.calls(), 0U);
    AnalyticPowerSolver const alwaysLinear(1.0e-24);
    EXPECT_ANY_THROW(initializeSingle({}, options, alwaysLinear));
    EXPECT_GT(alwaysLinear.calls(), 0U);
}

TEST(ParticleInitialization, EnforcesSpectrumEvaluationBudget) {
    PROFILE();
    auto options = smallOptions();
    options.maximumSpectrumEvaluations = 5;
    AnalyticPowerSolver const solver;
    EXPECT_ANY_THROW(initializeSingle({}, options, solver));
    EXPECT_LE(solver.calls(), options.maximumSpectrumEvaluations);
}

TEST(ParticleInitialization, PropagatesTheRootSolverFailureToAllLogicalSites) {
    PROFILE();
    auto const options = smallOptions();
    FailingPowerSolver const solver;
    size_t failedSites = 0;
    try {
        (void) initializeLogicalSites({}, options, solver, 8, &failedSites);
        FAIL() << "initialization accepted the failing root EB solver";
    } catch (std::runtime_error const& error) {
        EXPECT_NE(std::string(error.what()).find("synthetic root EB failure"), std::string::npos);
    }
    EXPECT_EQ(solver.calls(), 1U);
    EXPECT_EQ(failedSites, 8U);
}

TEST(ParticleInitialization, PropagatesCoarseDensityTaskFailuresToAllLogicalSites) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    auto options = smallOptions();
    options.meshExponent = 5;
    // Only sites carrying the fundamental modes lack spectrum coverage.
    // Other sites must finish their chunks and join the same error agreement.
    TruncatedPowerSolver const solver(1.1 * 2.0 * std::numbers::pi_v<double> / options.boxLength);
    size_t failedSites = 0;
    try {
        (void) initializeLogicalSites({}, options, solver, 8, &failedSites);
        FAIL() << "initialization accepted a spectrum missing the fundamental modes";
    } catch (std::runtime_error const& error) {
        EXPECT_NE(std::string(error.what()).find("density_fields/"), std::string::npos);
    }
    EXPECT_EQ(solver.calls(), 1U);
    EXPECT_EQ(failedSites, 8U);
}

TEST(ParticleInitialization, RealCambSmoke) {
    PROFILE();
    auto const* enabled = std::getenv("COSMICTIGER_TEST_CAMB_INITIALIZATION");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "set COSMICTIGER_TEST_CAMB_INITIALIZATION=1 to run real CAMB initialization";
    }
    ParticleInitializationOptions options;
    options.meshExponent = 2;
    options.boxLength = 100.0;
    options.maximumDensityDeviation = 0.05;
    options.earliestScaleFactor = 0.002;
    options.latestScaleFactor = 1.0;
    options.maximumLogScaleFactorStep = 0.5;
    options.logScaleFactorTolerance = 1.0e-3;
    options.maximumSpectrumEvaluations = 64;
    cosmictiger::CambSolver const solver;
    size_t threadCount = 0;
    for (auto const& task : hpx::filesystem::directory_iterator("/proc/self/task")) {
        (void) task;
        threadCount++;
    }
    GTEST_LOG_(INFO) << "real CAMB initialization runtime OS threads: " << threadCount;
    ASSERT_LE(threadCount + 1, 6U) << "the runtime and one CAMB child must fit the shared thread limit";
    auto const initial = initializeSingle({}, options, solver);
    EXPECT_GT(initial.report.maximumDensityDeviation, options.maximumDensityDeviation);
    EXPECT_LE(initial.report.previousDensityDeviation, options.maximumDensityDeviation);
    EXPECT_LE(std::log(initial.report.scaleFactor / initial.report.previousScaleFactor), options.logScaleFactorTolerance * 1.01);
    EXPECT_GE(initial.report.scaleFactor, options.earliestScaleFactor);
    EXPECT_LE(initial.report.scaleFactor, options.latestScaleFactor);
    EXPECT_EQ(initial.report.globalParticleCount, 64U);
    EXPECT_EQ(initial.particles.size(), 64U);
    EXPECT_DOUBLE_EQ(initial.report.boxLength, options.boxLength);
    EXPECT_GT(initial.report.particleMass, 0.0);
    EXPECT_GT(initial.report.spectrumEvaluations, 0U);
    EXPECT_LE(initial.report.spectrumEvaluations, options.maximumSpectrumEvaluations);
    expectValidOwnedRanges(initial);
}

TEST(ParticleInitialization, ChecksInvalidOptionsAsTheyAreUsed) {
    PROFILE();
    if constexpr (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    auto check = [](auto change) {
        PROFILE();
        auto options = smallOptions();
        change(options);
        AnalyticPowerSolver const solver;
        EXPECT_ANY_THROW(initializeSingle({}, options, solver));
    };
    check([](auto& options) {
        PROFILE();
        options.meshExponent = 0;
    });
    check([](auto& options) {
        PROFILE();
        options.meshExponent = std::numeric_limits<size_t>::digits;
    });
    check([](auto& options) {
        PROFILE();
        options.boxLength = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.maximumDensityDeviation = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.maximumDensityDeviation = -1.0;
    });
    check([](auto& options) {
        PROFILE();
        options.earliestScaleFactor = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.latestScaleFactor = options.earliestScaleFactor;
    });
    check([](auto& options) {
        PROFILE();
        options.maximumLogScaleFactorStep = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.logScaleFactorTolerance = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.growthDerivativeStep = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.growthDerivativeStep = 1.0e-7;
    });
    check([](auto& options) {
        PROFILE();
        options.maximumSpectrumEvaluations = 0;
    });
    check([](auto& options) {
        PROFILE();
        options.samplesPerLogInterval = -1;
    });
    check([](auto& options) {
        PROFILE();
        options.accuracyBoost = 0.0;
    });
    check([](auto& options) {
        PROFILE();
        options.component = static_cast<cosmictiger::MatterComponent>(-1);
    });
}

}    // namespace
