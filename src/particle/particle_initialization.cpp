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
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/fft/fft_vector_radix.hpp>
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/particle/particle_initialization.hpp>
#include <cosmictiger/particle/particle_load_balance.hpp>
#include <cosmictiger/units.hpp>

#include <hpx/collectives/all_gather.hpp>
#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/datastructures/serialization/tuple.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cosmictiger::detail {

namespace {

    constexpr size_t kInitializationLanes = 2 * NDIM;
    constexpr size_t kInitializationMinimumChunkSize = 1024;
    using InitializationFft = FftVectorRadix<kInitializationLanes>;
    using InitializationBatch = InitializationFft::RealBatch;
    using InitializationSpectrum = InitializationFft::ComplexBatch;
    using DensityFft = FftVectorRadix<1>;
    using DensitySpectrum = DensityFft::ComplexBatch;

    std::string initializationError() {
        PROFILE();
        try {
            throw;
        } catch (std::exception const& error) {
            return error.what();
        } catch (...) {
            return "unknown particle initialization error";
        }
    }

    size_t reverseInitializationBits(size_t value, size_t bits) {
        PROFILE();
        size_t result = 0;
        for (size_t bit = 0; bit < bits; bit++) {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }

    std::array<size_t, NDIM> initializationCoordinates(size_t index, size_t side) {
        PROFILE();
        std::array<size_t, NDIM> result{};
        for (auto& coordinate : result) {
            coordinate = index % side;
            index /= side;
        }
        return result;
    }

    size_t initializationFftRank(size_t gridRank, size_t side) {
        PROFILE();
        auto const coordinates = initializationCoordinates(gridRank, side);
        size_t rank = 0;
        for (auto const coordinate : coordinates) {
            rank = rank * side + coordinate;
        }
        return rank;
    }

    // A counter-based Gaussian makes the realization independent of site count
    // and execution order. Each conjugate pair uses its canonical global index.
    std::uint64_t initializationHash(std::uint64_t value) {
        PROFILE();
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    double initializationUniform(std::uint64_t value) {
        PROFILE();
        return (static_cast<double>(value >> 12) + 0.5) * 0x1.0p-52;
    }

    double initializationExp(double value) {
        PROFILE();
        FpeGuard guard;
        return std::exp(value);
    }

    struct InitializationMode {
        double waveNumber = 0.0;
        std::array<double, NDIM> displacementMultiplier{};
        std::complex<double> amplitude{};
    };

    class ParticleInitializer {
    public:
        ParticleInitializer(CosmologyParameters const& parameters, ParticleInitializationOptions const& options, std::vector<hpx::id_type> const& localities,
            size_t rank, std::uint64_t operationId, EinsteinBoltzmannSolver const& solver, ParticleInitializationObserver const& progressObserver)
          : parameters_(parameters)
          , options_(options)
          , localities_(localities)
          , rank_(rank)
          , operationId_(operationId)
          , solver_(solver)
          , progressObserver_(progressObserver)
          , basename_("/cosmictiger/particle_initialization/" + std::to_string(operationId) + "/") {
            PROFILE();
        }

        ParticleInitialConditions run() {
            PROFILE();
            report("preparing FFT mesh and seeded modes");
            prepare();
            report("FFT mesh and seeded modes ready; scanning handoff epoch");
            double lowerLog;
            double lastLog;
            {
                FpeGuard guard;
                lowerLog = std::log(expectPositive(options_.earliestScaleFactor));
                expect(options_.earliestScaleFactor < options_.latestScaleFactor);
                lastLog = std::log(expectRange(0, expectPositive(options_.latestScaleFactor), 1));
            }
            expectPositive(options_.maximumDensityDeviation);
            expectPositive(options_.maximumLogScaleFactorStep);
            expectPositive(options_.logScaleFactorTolerance);
            expect(options_.logScaleFactorTolerance < options_.maximumLogScaleFactorStep);
            double lowerDeviation = evaluate(initializationExp(lowerLog), "scan");
            if (lowerDeviation > options_.maximumDensityDeviation) {
                throw std::runtime_error("initial linear density already exceeds the handoff threshold; choose an earlier scale factor");
            }
            double upperLog = lowerLog;
            double upperDeviation = lowerDeviation;
            while (upperDeviation <= options_.maximumDensityDeviation && upperLog < lastLog) {
                lowerLog = upperLog;
                lowerDeviation = upperDeviation;
                double const nextLog = std::min(lastLog, upperLog + options_.maximumLogScaleFactorStep);
                if (nextLog <= upperLog) {
                    throw std::runtime_error("initialization scale-factor scan step is below floating-point resolution");
                }
                upperLog = nextLog;
                upperDeviation = evaluate(initializationExp(upperLog), "scan");
            }
            if (upperDeviation <= options_.maximumDensityDeviation) {
                throw std::runtime_error("linear density does not exceed the handoff threshold in the requested scale-factor interval");
            }
            while (upperLog - lowerLog > options_.logScaleFactorTolerance) {
                double const middleLog = lowerLog + 0.5 * (upperLog - lowerLog);
                if (middleLog == lowerLog || middleLog == upperLog) {
                    throw std::runtime_error("initialization bracket tolerance is below floating-point resolution");
                }
                double const deviation = evaluate(initializationExp(middleLog), "refine");
                if (deviation > options_.maximumDensityDeviation) {
                    upperLog = middleLog;
                    upperDeviation = deviation;
                } else {
                    lowerLog = middleLog;
                    lowerDeviation = deviation;
                }
            }

            double const scaleFactor = initializationExp(upperLog);
            report("handoff bracket found; computing displacement and velocity spectra");
            auto const center = spectrum(scaleFactor, "handoff");
            expectRange(1.0e-6, options_.growthDerivativeStep, 0.1);
            double const before = scaleFactor * initializationExp(-options_.growthDerivativeStep);
            bool const centeredDerivative = upperLog + options_.growthDerivativeStep <= 0.0;
            double const after = scaleFactor * initializationExp(centeredDerivative ? options_.growthDerivativeStep : -2.0 * options_.growthDerivativeStep);
            auto const previous = spectrum(before, "growth before");
            auto const next = spectrum(after, "growth after");
            report("assembling Fourier displacement and velocity fields");
            double const derivativeDenominator = 2.0 * options_.growthDerivativeStep;
            std::vector<InitializationSpectrum> fields;
            std::string error;
            try {
                fields.resize(modes_.size());
                ParallelChunkPlan const chunks(modes_.size(), kInitializationMinimumChunkSize);
                parallelForChunks(chunks,
                    hpx::annotated_function(
                        [&](size_t, size_t begin, size_t end) {
                            PROFILE();
                            FpeGuard guard;
                            for (size_t index = begin; index < end; index++) {
                                auto const& mode = modes_[index];
                                if (mode.waveNumber == 0.0) {
                                    continue;
                                }
                                auto const density = mode.amplitude * std::sqrt(center.power(mode.waveNumber));
                                double const previousAmplitude = std::sqrt(previous.power(mode.waveNumber));
                                double const nextAmplitude = std::sqrt(next.power(mode.waveNumber));
                                double const amplitudeDerivative = centeredDerivative ?
                                    (nextAmplitude - previousAmplitude) / derivativeDenominator :
                                    (3.0 * std::sqrt(center.power(mode.waveNumber)) - 4.0 * previousAmplitude + nextAmplitude) / derivativeDenominator;
                                auto const derivative = mode.amplitude * amplitudeDerivative;
                                for (size_t axis = 0; axis < NDIM; axis++) {
                                    std::complex<double> const multiplier(0.0, mode.displacementMultiplier[axis]);
                                    fields[index][axis] = multiplier * density;
                                    fields[index][NDIM + axis] = multiplier * derivative;
                                }
                            }
                        },
                        "particleInitialization.parallelForChunks"));
            } catch (...) {
                error = initializationError();
            }
            agree("final_fields", std::move(error));
            report("inverse-transforming displacement and velocity fields");
            auto const realFields = inverse(fields);

            std::optional<ParticleInitialConditions> result;
            error.clear();
            report("placing particles on the displaced lattice");
            try {
                size_t const count = modes_.size();
                double const mass = [&] {
                    constexpr double kReferenceHubble = 100.0;
                    constexpr double kCriticalDensityNumerator = 3.0;
                    constexpr double kCriticalDensityDenominator = 8.0;
                    PROFILE();
                    FpeGuard guard;
                    double const hubbleParameter = expectPositive(parameters_.hubble) / kReferenceHubble;
                    double densityH2 = parameters_.omegaCdmH2;
                    if (options_.component != MatterComponent::cdm) {
                        densityH2 += parameters_.omegaBaryonH2;
                    }
                    units::UnitScales const scales(parameters_.hubble, options_.boxLength);
                    double const normalizedMass = kCriticalDensityNumerator * expectPositive(densityH2) /
                        (kCriticalDensityDenominator * std::numbers::pi * hubbleParameter * hubbleParameter * static_cast<double>(globalCount_));
                    // The report/checkpoint convention remains Msun/h. Derive
                    // it from exactly the G=1 mass used by the gravity kernels.
                    return expectPositive(normalizedMass * scales.massSolar() * hubbleParameter);
                }();
                result.emplace(ParticleInitialConditions{ParticleStorage(count), CartesianGrid(options_.meshExponent, localityExponent_, rank_),
                    {scaleFactor, 1.0 / scaleFactor - 1.0, upperDeviation, initializationExp(lowerLog), lowerDeviation, options_.boxLength, mass, globalCount_,
                        evaluations_}});
                auto const origin = initializationCoordinates(rank_, localitySide_);
                ParallelChunkPlan const chunks(count, kInitializationMinimumChunkSize);
                parallelForChunks(chunks,
                    hpx::annotated_function(
                        [&](size_t, size_t begin, size_t end) {
                            PROFILE();
                            FpeGuard guard;
                            for (size_t index = begin; index < end; index++) {
                                auto const coordinates = initializationCoordinates(index, blockSide_);
                                for (size_t axis = 0; axis < NDIM; axis++) {
                                    double const displacement = realFields[index][axis];
                                    double const velocity = realFields[index][NDIM + axis];
                                    double const lattice =
                                        (static_cast<double>(origin[axis] * blockSide_ + coordinates[axis]) + 0.5) / static_cast<double>(side_);
                                    result->particles.position(axis, index) = Coordinate(lattice + displacement);
                                    result->particles.velocity(axis, index) = static_cast<float>(velocity);
                                }
                                result->particles.flags(index) = Rung{0, false};
                            }
                        },
                        "particleInitialization.parallelForChunks"));
            } catch (...) {
                error = initializationError();
            }
            agree("particles", std::move(error));
            report("redistributing particles to their owning cells");
            loadBalanceParticles(result->particles, result->grid, localities_, operationId_);
            result->grid.sortParticlesByRung(result->particles);
            report("initial particles ready");
            return std::move(*result);
        }

    private:
        void report(std::string const& message) const {
            PROFILE();
            if (rank_ == 0 && progressObserver_) {
                progressObserver_(message);
            }
        }

        void agree(std::string const& phase, std::string error) const {
            PROFILE();
            auto const name = basename_ + phase;
            auto const errors = hpx::collectives::all_gather(
                name.c_str(), std::move(error), hpx::collectives::num_sites_arg(localities_.size()), hpx::collectives::this_site_arg(rank_))
                                    .get();
            for (auto const& siteError : errors) {
                if (!siteError.empty()) {
                    throw std::runtime_error("particle initialization " + phase + ": " + siteError);
                }
            }
        }

        void initializeGeometry() {
            PROFILE();
            expect(std::has_single_bit(localities_.size()));
            expectEqual(std::countr_zero(localities_.size()) % NDIM, 0);
            localityExponent_ = std::countr_zero(localities_.size()) / NDIM;
            side_ = size_t(1) << expectRange(2, options_.meshExponent, (std::numeric_limits<size_t>::digits - 1) / NDIM);
            expectRange(1, localities_.size(), side_);
            localitySide_ = size_t(1) << localityExponent_;
            blockSide_ = side_ / localitySide_;
            globalCount_ = size_t(1) << (NDIM * options_.meshExponent);
        }

        void prepare() {
            PROFILE();
            std::string error;
            try {
                initializeGeometry();
            } catch (...) {
                error = initializationError();
            }
            // Compare every input before entering a transform, so a mismatch
            // cannot make sites choose different iteration or exchange counts.
            std::vector<double> const realInputs{parameters_.hubble, parameters_.omegaBaryonH2, parameters_.omegaCdmH2, parameters_.omegaNeutrinoH2,
                parameters_.omegaCurvature, parameters_.effectiveNeutrinos, parameters_.scalarAmplitude, parameters_.spectralIndex, parameters_.spectralRunning,
                parameters_.pivotMpcInverse, parameters_.darkEnergyW, parameters_.darkEnergyWa, parameters_.temperatureCmb, parameters_.heliumFraction,
                parameters_.opticalDepth, options_.boxLength, options_.maximumDensityDeviation, options_.earliestScaleFactor, options_.latestScaleFactor,
                options_.maximumLogScaleFactorStep, options_.logScaleFactorTolerance, options_.growthDerivativeStep, options_.accuracyBoost};
            std::vector<std::uint64_t> const integerInputs{options_.meshExponent, options_.seed, options_.maximumSpectrumEvaluations,
                static_cast<std::uint64_t>(options_.samplesPerLogInterval), static_cast<std::uint64_t>(options_.component),
                static_cast<std::uint64_t>(parameters_.massiveNeutrinos), static_cast<std::uint64_t>(parameters_.masslessNeutrinos)};
            using Preparation = std::tuple<std::vector<double>, std::vector<std::uint64_t>, std::vector<hpx::id_type>, std::string>;
            auto const name = basename_ + "preparation";
            auto const preparation = hpx::collectives::all_gather(name.c_str(), Preparation{realInputs, integerInputs, localities_, std::move(error)},
                hpx::collectives::num_sites_arg(localities_.size()), hpx::collectives::this_site_arg(rank_))
                                         .get();
            for (auto const& [siteReal, siteInteger, siteLocalities, siteError] : preparation) {
                if (!siteError.empty()) {
                    throw std::runtime_error("particle initialization preparation: " + siteError);
                }
                expect(!(siteReal != realInputs || siteInteger != integerInputs || siteLocalities != localities_));
            }
            error.clear();
            try {
                size_t const fftRank = initializationFftRank(rank_, localitySide_);
                std::vector<hpx::id_type> fftLocalities(localities_.size());
                for (size_t site = 0; site < localities_.size(); site++) {
                    fftLocalities[initializationFftRank(site, localitySide_)] = localities_[site];
                }
                fft_ = std::make_unique<InitializationFft>(side_, localities_.size(), fftRank);
                densityFft_ = std::make_unique<DensityFft>(side_, localities_.size(), fftRank);
                exchange_ = std::make_unique<FftHpxExchange>(std::move(fftLocalities), operationId_, fftRank);
                modes_.resize(fft_->localValueCount());
                auto const origin = initializationCoordinates(rank_, localitySide_);
                double const normalization = [&] {
                    PROFILE();
                    FpeGuard guard;
                    return static_cast<double>(globalCount_) / std::sqrt(std::pow(expectPositive(options_.boxLength), NDIM));
                }();
                ParallelChunkPlan const chunks(modes_.size(), kInitializationMinimumChunkSize);
                parallelForChunks(chunks,
                    hpx::annotated_function(
                        [&](size_t, size_t begin, size_t end) {
                            PROFILE();
                            FpeGuard guard;
                            for (size_t index = begin; index < end; index++) {
                                auto const local = initializationCoordinates(index, blockSide_);
                                std::array<size_t, NDIM> frequency{};
                                std::array<double, NDIM> wave{};
                                size_t globalIndex = 0;
                                size_t conjugateIndex = 0;
                                size_t stride = 1;
                                double waveSquared = 0.0;
                                double centerPhase = 0.0;
                                bool nyquist = false;
                                for (size_t axis = 0; axis < NDIM; axis++) {
                                    frequency[axis] = reverseInitializationBits(origin[axis], localityExponent_) + localitySide_ * local[axis];
                                    size_t const conjugate = (side_ - frequency[axis]) % side_;
                                    globalIndex += stride * frequency[axis];
                                    conjugateIndex += stride * conjugate;
                                    stride *= side_;
                                    auto const signedFrequency = frequency[axis] < side_ / 2 ?
                                        static_cast<std::int64_t>(frequency[axis]) :
                                        static_cast<std::int64_t>(frequency[axis]) - static_cast<std::int64_t>(side_);
                                    wave[axis] = 2.0 * std::numbers::pi * static_cast<double>(signedFrequency);
                                    waveSquared += wave[axis] * wave[axis];
                                    centerPhase += wave[axis] * 0.5 / static_cast<double>(side_);
                                    nyquist = nyquist || frequency[axis] == side_ / 2;
                                }
                                if (waveSquared == 0.0 || nyquist) {
                                    continue;
                                }
                                auto& mode = modes_[index];
                                mode.waveNumber = std::sqrt(waveSquared) / options_.boxLength;
                                for (size_t axis = 0; axis < NDIM; axis++) {
                                    mode.displacementMultiplier[axis] = wave[axis] / waveSquared;
                                }
                                auto const canonical = static_cast<std::uint64_t>(std::min(globalIndex, conjugateIndex));
                                auto const firstHash = initializationHash(canonical ^ initializationHash(options_.seed));
                                auto const secondHash = initializationHash(firstHash);
                                double const radius = std::sqrt(-std::log(initializationUniform(firstHash)));
                                double const angle = 2.0 * std::numbers::pi * initializationUniform(secondHash);
                                std::complex<double> gaussian(radius * std::cos(angle), radius * std::sin(angle));
                                if (globalIndex > conjugateIndex) {
                                    gaussian = std::conj(gaussian);
                                }
                                mode.amplitude = normalization * gaussian * std::complex<double>(std::cos(centerPhase), std::sin(centerPhase));
                            }
                        },
                        "particleInitialization.parallelForChunks"));
            } catch (...) {
                error = initializationError();
            }
            agree("fft_setup", std::move(error));
        }

        MatterPowerSpectrum spectrum(double scaleFactor, char const* phase) {
            PROFILE();
            expectRange(5, options_.maximumSpectrumEvaluations, std::numeric_limits<size_t>::max());
            if (evaluations_ >= options_.maximumSpectrumEvaluations) {
                throw std::runtime_error("particle initialization exceeded its spectrum evaluation budget");
            }
            if (rank_ == 0 && progressObserver_) {
                std::ostringstream message;
                message << "spectrum " << evaluations_ + 1 << '/' << options_.maximumSpectrumEvaluations << " (" << phase << ") a=" << std::setprecision(12)
                        << scaleFactor << ": running CAMB";
                report(message.str());
            }
            using Packet = std::tuple<std::vector<double>, std::string>;
            Packet packet;
            auto const name = basename_ + "spectrum/" + std::to_string(evaluations_++);
            if (rank_ == 0) {
                try {
                    MatterPowerRequest request;
                    {
                        FpeGuard guard;
                        request.redshift = 1.0 / expectPositive(scaleFactor) - 1.0;
                        request.maximumWaveNumber =
                            std::sqrt(static_cast<double>(NDIM)) * std::numbers::pi * static_cast<double>(side_) / expectPositive(options_.boxLength);
                    }
                    request.samplesPerLogInterval = expectNonNegative(options_.samplesPerLogInterval);
                    request.accuracyBoost = expectPositive(options_.accuracyBoost);
                    expect(options_.component == MatterComponent::cdm || options_.component == MatterComponent::cdmAndBaryons ||
                        options_.component == MatterComponent::totalMatter);
                    expect(options_.component != MatterComponent::totalMatter || parameters_.omegaNeutrinoH2 == 0.0);
                    request.component = options_.component;
                    auto const power = solver_.compute(parameters_, request);
                    expect(!(power.redshift() != request.redshift || power.component() != request.component ||
                        power.hubbleParameter() != parameters_.hubble / 100.0));
                    auto& values = std::get<0>(packet);
                    values.reserve(2 * power.samples().size());
                    for (auto const& sample : power.samples()) {
                        values.push_back(sample.waveNumber);
                        values.push_back(sample.power);
                    }
                } catch (...) {
                    std::get<1>(packet) = initializationError();
                }
                packet = hpx::collectives::broadcast_to(
                    name.c_str(), std::move(packet), hpx::collectives::num_sites_arg(localities_.size()), hpx::collectives::this_site_arg(rank_))
                             .get();
            } else {
                packet = hpx::collectives::broadcast_from<Packet>(name.c_str(), hpx::collectives::this_site_arg(rank_)).get();
            }
            if (!std::get<1>(packet).empty()) {
                throw std::runtime_error("particle initialization EB spectrum: " + std::get<1>(packet));
            }
            std::optional<MatterPowerSpectrum> result;
            std::string error;
            try {
                auto const& values = std::get<0>(packet);
                std::vector<MatterPowerSample> samples;
                samples.reserve(values.size() / 2);
                for (size_t index = 0; index < values.size(); index += 2) {
                    samples.push_back({values[index], values[index + 1]});
                }
                result.emplace(1.0 / scaleFactor - 1.0, parameters_.hubble / 100.0, options_.component, std::move(samples));
            } catch (...) {
                error = initializationError();
            }
            agree("spectrum_check/" + std::to_string(evaluations_), std::move(error));
            report("spectrum " + std::to_string(evaluations_) + " ready");
            return std::move(*result);
        }

        std::vector<InitializationBatch> inverse(std::vector<InitializationSpectrum> const& fields) {
            PROFILE();
            return fft_->inverse(fields, *exchange_);
        }

        double evaluate(double scaleFactor, char const* phase) {
            PROFILE();
            auto const power = spectrum(scaleFactor, phase);
            std::vector<DensitySpectrum> fields;
            std::string error;
            try {
                fields.resize(modes_.size());
                ParallelChunkPlan const chunks(modes_.size(), kInitializationMinimumChunkSize);
                parallelForChunks(chunks,
                    hpx::annotated_function(
                        [&](size_t, size_t begin, size_t end) {
                            PROFILE();
                            FpeGuard guard;
                            for (size_t index = begin; index < end; index++) {
                                auto const& mode = modes_[index];
                                if (mode.waveNumber != 0.0) {
                                    fields[index][0] = mode.amplitude * std::sqrt(power.power(mode.waveNumber));
                                }
                            }
                        },
                        "particleInitialization.parallelForChunks"));
            } catch (...) {
                error = initializationError();
            }
            agree("density_fields/" + std::to_string(evaluations_), std::move(error));
            auto const density = densityFft_->inverse(fields, *exchange_);
            double localMaximum = 0.0;
            error.clear();
            try {
                ParallelChunkPlan const chunks(density.size(), kInitializationMinimumChunkSize);
                std::vector<double> maxima(chunks.size(), 0.0);
                parallelForChunks(chunks,
                    hpx::annotated_function(
                        [&](size_t chunk, size_t begin, size_t end) {
                            PROFILE();
                            FpeGuard guard;
                            double maximum = 0.0;
                            for (size_t index = begin; index < end; index++) {
                                maximum = std::max(maximum, std::abs(density[index][0]));
                            }
                            maxima[chunk] = maximum;
                        },
                        "particleInitialization.parallelForChunks"));
                FpeGuard guard;
                for (double const maximum : maxima) {
                    localMaximum = std::max(localMaximum, maximum);
                }
            } catch (...) {
                error = initializationError();
            }
            agree("density_check/" + std::to_string(evaluations_), std::move(error));
            auto const name = basename_ + "maximum/" + std::to_string(evaluations_);
            double const maximum = hpx::collectives::all_reduce(name.c_str(), localMaximum,
                hpx::annotated_function(
                    [](double first, double second) {
                        PROFILE();
                        return std::max(first, second);
                    },
                    "particleInitialization.reduceDensityMaximum"),
                hpx::collectives::num_sites_arg(localities_.size()), hpx::collectives::this_site_arg(rank_))
                                       .get();
            if (rank_ == 0 && progressObserver_) {
                std::ostringstream message;
                message << "density " << evaluations_ << " (" << phase << ") a=" << std::setprecision(12) << scaleFactor << " max|delta|=" << maximum;
                report(message.str());
            }
            return maximum;
        }

        CosmologyParameters const& parameters_;
        ParticleInitializationOptions const& options_;
        std::vector<hpx::id_type> const& localities_;
        size_t rank_;
        std::uint64_t operationId_;
        EinsteinBoltzmannSolver const& solver_;
        ParticleInitializationObserver progressObserver_;
        std::string basename_;
        size_t side_ = 0;
        size_t localityExponent_ = 0;
        size_t localitySide_ = 0;
        size_t blockSide_ = 0;
        ParticleCount globalCount_ = 0;
        size_t evaluations_ = 0;
        std::unique_ptr<InitializationFft> fft_;
        std::unique_ptr<DensityFft> densityFft_;
        std::unique_ptr<FftHpxExchange> exchange_;
        std::vector<InitializationMode> modes_;
    };

}    // namespace
}    // namespace cosmictiger::detail

namespace cosmictiger {

ParticleInitialConditions initializeParticles(CosmologyParameters const& parameters, ParticleInitializationOptions const& options,
    std::vector<hpx::id_type> const& localities, size_t localityIndex, std::uint64_t operationId, EinsteinBoltzmannSolver const& solver,
    ParticleInitializationObserver const& progressObserver) {
    PROFILE();
    expect(!(localities.empty() || localityIndex >= localities.size() || localities[localityIndex] != hpx::find_here()));
    detail::ParticleInitializer initializer(parameters, options, localities, localityIndex, operationId, solver, progressObserver);
    return initializer.run();
}

ParticleInitialConditions initializeParticles(CosmologyParameters const& parameters, ParticleInitializationOptions const& options, std::uint64_t operationId,
    ParticleInitializationObserver const& progressObserver) {
    PROFILE();
    auto const localities = hpx::find_all_localities();
    size_t rank = 0;
    while (rank < localities.size() && localities[rank] != hpx::find_here()) {
        rank++;
    }
    CambSolver const solver;
    return initializeParticles(parameters, options, localities, rank, operationId, solver, progressObserver);
}

}    // namespace cosmictiger
