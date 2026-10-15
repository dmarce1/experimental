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
#include <cosmictiger/healpix/healpix.hpp>
#include <cosmictiger/lightcone/lightcone.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/modules/filesystem.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cosmictiger {
namespace detail {

    double lightconeNorm(std::array<double, NDIM> const& value) {
        PROFILE();
        FpeGuard guard;
        double squared = 0.0;
        for (double const coordinate : value) {
            squared += coordinate * coordinate;
        }
        return std::sqrt(squared);
    }

    // Shared background evaluations substantially reduce work for identical
    // radii. This bounded cache never grows with the number of particles.
    class LightconeBackground {
    public:
        LightconeBackground(SimulationBackground const& background, LightconeStep const& step)
          : background_(background)
          , step_(step) {
            PROFILE();
        }

        SimulationBackgroundStep at(double offset) {
            constexpr size_t kMaximumCachedStates = 4096;
            PROFILE();
            auto const found = states_.find(offset);
            if (found != states_.end()) {
                return found->second;
            }
            auto result = step_.backgroundTable ? step_.backgroundTable->interval(step_.tableTickOffset, step_.tableTickOffset + offset) :
                                                  background_.advance(step_.tickBackground, offset);
            if (offset == step_.beginOffset) {
                result.driftFactor = step_.oldDrift;
            } else if (offset == step_.endOffset) {
                result.driftFactor = step_.endDrift;
            }
            if (states_.size() == kMaximumCachedStates) {
                states_.clear();
            }
            states_.emplace(offset, result);
            return result;
        }

    private:
        SimulationBackground const& background_;
        LightconeStep const& step_;
        std::map<double, SimulationBackgroundStep> states_;
    };

    std::array<double, NDIM> lightconePosition(std::array<double, NDIM> const& begin, std::array<double, NDIM> const& momentum, double drift) {
        PROFILE();
        FpeGuard guard;
        std::array<double, NDIM> result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = std::fma(momentum[axis], drift, begin[axis]);
        }
        return result;
    }

    // A subluminal trajectory makes |x|+c*t strictly increasing. Bracketing
    // therefore finds its unique crossing without a straight-chord assumption.
    LightconeCrossing solveLightconeCrossing(std::array<double, NDIM> const& begin, std::array<double, NDIM> const& momentum, double momentumNorm,
        LightconeStep const& step, double remainingAtTick, double lightSpeed, LightconeBackground& background) {
        constexpr size_t kMaximumIterations = 96;
        constexpr double kBracketFraction = 0.1;
        PROFILE();
        FpeGuard guard;
        double lo = step.beginOffset;
        double hi = step.endOffset;
        double const firstRadius = lightconeNorm(begin);
        double offset = momentumNorm == 0.0 ? remainingAtTick - firstRadius / lightSpeed : lo + (hi - lo) / 2.0;
        if (offset < lo) {
            offset = momentumNorm == 0.0 ? lo : lo + (hi - lo) / 2.0;
        } else if (offset > hi) {
            offset = momentumNorm == 0.0 ? hi : lo + (hi - lo) / 2.0;
        }
        for (size_t iteration = 0; iteration < kMaximumIterations; iteration++) {
            auto const state = background.at(offset);
            auto const position = lightconePosition(begin, momentum, state.driftFactor - step.oldDrift);
            double const radius = lightconeNorm(position);
            double const residual = radius - lightSpeed * (remainingAtTick - offset);
            if (momentumNorm == 0.0 || residual == 0.0 || std::nextafter(lo, hi) >= hi) {
                LightconeCrossing crossing;
                crossing.position = position;
                crossing.radius = radius;
                crossing.tickOffset = offset;
                crossing.conformalTime = step.tickConformalTime + offset;
                crossing.scaleFactor = state.state.scaleFactor;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    crossing.velocity[axis] = momentum[axis] / crossing.scaleFactor;
                }
                if (radius > 0.0) {
                    crossing.inverseRadiusSquared = 1.0 / (radius * radius);
                }
                return crossing;
            }
            if (residual < 0.0) {
                lo = offset;
            } else {
                hi = offset;
            }
            double derivative = lightSpeed;
            if (radius > 0.0) {
                for (size_t axis = 0; axis < NDIM; axis++) {
                    derivative += position[axis] / radius * momentum[axis] / state.state.scaleFactor;
                }
            }
            double const trial = offset - residual / derivative;
            double const margin = kBracketFraction * (hi - lo);
            offset = trial > lo + margin && trial < hi - margin ? trial : lo + (hi - lo) / 2.0;
        }
        throw std::runtime_error("lightcone crossing root did not converge");
    }

    void lightconeMetadata(std::ofstream& stream, LightconeOptions const& options, LightconeStep const& step, double observationTime, double lightSpeed,
        size_t locality, std::uint64_t operation, units::UnitScales const& scales) {
        PROFILE();
        stream << std::setprecision(std::numeric_limits<double>::max_digits10);
        stream << "# CosmicTiger lightcone CSV version=2\n# ordering=NESTED nside=" << options.nside << "\n# locality=" << locality
               << " operation=" << operation << "\n# interval=(begin,end] tickConformalTime=" << step.tickConformalTime << " beginOffset=" << step.beginOffset
               << " endOffset=" << step.endOffset << "\n# observationConformalTime=" << observationTime
               << " observationTimeYears=" << options.observationTimeYears << "\n# maximumRadius=" << options.maximumRadius << " observer=";
        for (size_t axis = 0; axis < NDIM; axis++) {
            stream << (axis == 0 ? "" : ",") << options.observer[axis];
        }
        stream << "\n# position=observer_relative_comoving_box_lengths radius=box_lengths time=H0^-1"
               << " velocity=peculiar_p_over_a_in_H0_L inverseRadiusSquared=box_lengths^-2"
               << "\n# lightSpeed=" << lightSpeed << " velocityToKmPerSecond=" << scales.velocityKmPerSecond()
               << "\n# boxLengthComovingMpc=" << scales.lengthMpc() << " boxLengthComovingCm=" << scales.lengthCgs()
               << " timeToSeconds=" << scales.timeSeconds() << "\n# converted_positions=observer_relative proper_scale_factor=each_crossing"
               << " converted_velocities=physical_peculiar hubble_flow=excluded"
               << "\n# conformalTimeSeconds=conformal_time_since_original_handoff tickOffsetSeconds=conformal_interval_from_tick"
               << "\n# localParticleIndex=step_local_storage_index_not_persistent_id origin_crossings=omitted\n";
    }

}    // namespace detail

Lightcone::Lightcone(LightconeOptions options, double observationConformalTime, double lightSpeed)
  : options_(std::move(options))
  , observationConformalTime_(observationConformalTime)
  , lightSpeed_(lightSpeed) {
    constexpr double kLargestExactImage = 0x1p52;
    constexpr double kMaximumDouble = std::numeric_limits<double>::max();
    PROFILE();
    if (!(options_.maximumRadius > 0.0 && options_.maximumRadius < kLargestExactImage)) {
        throw std::invalid_argument("lightcone maximum radius must be positive and representable in box lengths");
    }
    if (!(lightSpeed_ > 0.0 && lightSpeed_ <= kMaximumDouble) || options_.outputPrefix.empty()) {
        throw std::invalid_argument("lightcone requires positive light speed and a nonempty output prefix");
    }
    if (!(observationConformalTime_ >= -kMaximumDouble && observationConformalTime_ <= kMaximumDouble)) {
        throw std::invalid_argument("lightcone observation conformal time is outside the representable range");
    }
    for (double const coordinate : options_.observer) {
        if (!(coordinate >= 0.0 && coordinate < 1.0)) {
            throw std::invalid_argument("lightcone observer coordinates must lie in the periodic unit box [0,1)");
        }
    }
    healpix::Healpix const pixels(options_.nside, healpix::Ordering::nested);
}

LightconeSample Lightcone::sample(
    ParticleStorage const& particles, CartesianGrid const& grid, SimulationBackground const& background, LightconeStep const& step) const {
    constexpr double kLargestExactImage = 0x1p52;
    constexpr size_t kPixelBatchSize = 256;
    PROFILE();
    LightconeSample result;
    if (!options_.enabled || step.beginOffset == step.endOffset) {
        return result;
    }
    if (!(step.beginOffset >= 0.0 && step.endOffset > step.beginOffset && step.endDrift >= step.oldDrift)) {
        throw std::invalid_argument("lightcone step must advance time and its drift integral");
    }
    double const remainingAtTick = observationConformalTime_ - step.tickConformalTime;
    double const coneBegin = lightSpeed_ * (remainingAtTick - step.beginOffset);
    if (coneBegin <= 0.0) {
        return result;
    }
    double const coneEnd = lightSpeed_ * (remainingAtTick - step.endOffset);
    if (coneEnd > options_.maximumRadius) {
        return result;
    }
    detail::LightconeBackground evaluations(background, step);
    double const scaleBegin = evaluations.at(step.beginOffset).state.scaleFactor;
    double const scaleEnd = evaluations.at(step.endOffset).state.scaleFactor;
    double const minimumScale = scaleBegin < scaleEnd ? scaleBegin : scaleEnd;
    double const drift = step.endDrift - step.oldDrift;
    auto const& layout = grid.layout();
    for (size_t cellIndex = 0; cellIndex < layout.storedCellCount(); cellIndex++) {
        auto const coordinates = layout.cellCoordinates(cellIndex);
        if (!layout.isInterior(coordinates)) {
            continue;
        }
        auto const range = grid.cell(coordinates).localParticles;
        expect(range.begin <= range.end && range.end <= particles.size());
        for (ParticleCount particle = range.begin; particle < range.end; particle++) {
            if (particles.flags(particle).isGhost()) {
                continue;
            }
            FpeGuard guard;
            std::array<double, NDIM> begin{};
            std::array<double, NDIM> momentum{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                begin[axis] = particles.position(axis, particle).toDouble() - options_.observer[axis];
                momentum[axis] = particles.velocity(axis, particle);
            }
            double const momentumNorm = detail::lightconeNorm(momentum);
            if (!(momentumNorm < lightSpeed_ * minimumScale)) {
                throw std::runtime_error("lightcone sampling requires subluminal particle velocity throughout the drift");
            }
            auto const end = detail::lightconePosition(begin, momentum, drift);
            std::array<std::int64_t, NDIM> lower{};
            std::array<std::int64_t, NDIM> upper{};
            bool empty = false;
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const minimum = begin[axis] < end[axis] ? begin[axis] : end[axis];
                double const maximum = begin[axis] > end[axis] ? begin[axis] : end[axis];
                double const lo = std::ceil(-options_.maximumRadius - maximum);
                double const hi = std::floor(options_.maximumRadius - minimum);
                if (!(lo > -kLargestExactImage && hi < kLargestExactImage)) {
                    throw std::length_error("lightcone periodic-image bounds exceed exact integer arithmetic");
                }
                lower[axis] = static_cast<std::int64_t>(lo);
                upper[axis] = static_cast<std::int64_t>(hi);
                empty = empty || lower[axis] > upper[axis];
            }
            if (empty) {
                continue;
            }
            auto image = lower;
            bool moreImages = true;
            while (moreImages) {
                std::array<double, NDIM> imageBegin{};
                for (size_t axis = 0; axis < NDIM; axis++) {
                    imageBegin[axis] = begin[axis] + static_cast<double>(image[axis]);
                }
                auto const imageEnd = detail::lightconePosition(imageBegin, momentum, drift);
                double const firstRadius = detail::lightconeNorm(imageBegin);
                double const lastRadius = detail::lightconeNorm(imageEnd);
                if (firstRadius < coneBegin && lastRadius >= coneEnd) {
                    auto crossing = detail::solveLightconeCrossing(imageBegin, momentum, momentumNorm, step, remainingAtTick, lightSpeed_, evaluations);
                    if (crossing.radius > 0.0 && crossing.radius <= options_.maximumRadius) {
                        crossing.localParticleIndex = particle;
                        crossing.image = image;
                        result.crossings.push_back(crossing);
                    }
                }
                for (size_t axis = 0; axis < NDIM; axis++) {
                    if (image[axis] < upper[axis]) {
                        image[axis]++;
                        break;
                    }
                    image[axis] = lower[axis];
                    if (axis == NDIM - 1) {
                        moreImages = false;
                    }
                }
            }
        }
    }
    healpix::Healpix const pixels(options_.nside, healpix::Ordering::nested);
    std::array<std::array<double, NDIM>, kPixelBatchSize> directions{};
    std::array<std::uint64_t, kPixelBatchSize> pixelIndices{};
    std::map<std::uint64_t, LightconeMapBin> occupied;
    for (size_t begin = 0; begin < result.crossings.size(); begin += kPixelBatchSize) {
        size_t const remaining = result.crossings.size() - begin;
        size_t const count = remaining < kPixelBatchSize ? remaining : kPixelBatchSize;
        for (size_t index = 0; index < count; index++) {
            directions[index] = result.crossings[begin + index].position;
        }
        pixels.vectorsToPixels(std::span(directions.data(), count), std::span(pixelIndices.data(), count));
        for (size_t index = 0; index < count; index++) {
            auto& crossing = result.crossings[begin + index];
            crossing.pixel = pixelIndices[index];
            auto& bin = occupied[crossing.pixel];
            bin.pixel = crossing.pixel;
            bin.count++;
            FpeGuard guard;
            bin.inverseRadiusSquared += crossing.inverseRadiusSquared;
            double const scaleSquared = crossing.scaleFactor * crossing.scaleFactor;
            bin.inverseProperRadiusSquared += crossing.inverseRadiusSquared / scaleSquared;
        }
    }
    result.bins.reserve(occupied.size());
    for (auto const& entry : occupied) {
        result.bins.push_back(entry.second);
    }
    return result;
}

void Lightcone::write(
    LightconeSample const& sample, LightconeStep const& step, size_t localityIndex, std::uint64_t operationId, units::UnitScales const& scales) const {
    PROFILE();
    if (!options_.enabled) {
        return;
    }
    std::string const base = options_.outputPrefix + ".step" + std::to_string(operationId) + ".rank" + std::to_string(localityIndex);
    hpx::filesystem::path const catalog = base + ".crossings.csv";
    hpx::filesystem::path const map = base + ".map.csv";
    hpx::filesystem::path const pending = base + ".pending";
    hpx::filesystem::path const complete = base + ".complete";
    if (!catalog.parent_path().empty()) {
        hpx::filesystem::create_directories(catalog.parent_path());
    }
    if (!hpx::filesystem::create_directory(pending)) {
        throw std::runtime_error("lightcone output segment is already being written: " + base);
    }
    bool publishedCatalog = false;
    bool publishedMap = false;
    try {
        if (hpx::filesystem::exists(catalog) || hpx::filesystem::exists(map) || hpx::filesystem::exists(complete)) {
            throw std::runtime_error("refusing to overwrite existing lightcone segment: " + base);
        }
        hpx::filesystem::path const temporaryCatalog = pending / "crossings.csv";
        hpx::filesystem::path const temporaryMap = pending / "map.csv";
        hpx::filesystem::path const temporaryComplete = pending / "complete";
        std::ofstream crossings(temporaryCatalog.string());
        std::ofstream bins(temporaryMap.string());
        crossings.exceptions(std::ios::failbit | std::ios::badbit);
        bins.exceptions(std::ios::failbit | std::ios::badbit);
        detail::lightconeMetadata(crossings, options_, step, observationConformalTime_, lightSpeed_, localityIndex, operationId, scales);
        detail::lightconeMetadata(bins, options_, step, observationConformalTime_, lightSpeed_, localityIndex, operationId, scales);
        crossings << "pixel,localParticleIndex,imageX,imageY,imageZ,x,y,z,vx,vy,vz,conformalTime,tickOffset,scaleFactor,radius,inverseRadiusSquared"
                  << ",xComovingMpc,yComovingMpc,zComovingMpc,peculiarVxKmPerSecond,peculiarVyKmPerSecond,peculiarVzKmPerSecond"
                  << ",conformalTimeSeconds,tickOffsetSeconds,radiusComovingMpc,inverseComovingRadiusSquaredPerMpc2"
                  << ",xProperCm,yProperCm,zProperCm,peculiarVxCmPerSecond,peculiarVyCmPerSecond,peculiarVzCmPerSecond"
                  << ",radiusProperCm,inverseProperRadiusSquaredPerCm2\n";
        double const inverseLengthMpcSquared = 1.0 / (scales.lengthMpc() * scales.lengthMpc());
        double const inverseLengthCgsSquared = 1.0 / (scales.lengthCgs() * scales.lengthCgs());
        for (auto const& crossing : sample.crossings) {
            FpeGuard guard;
            double const properLength = crossing.scaleFactor * scales.lengthCgs();
            double const inverseScaleSquared = 1.0 / (crossing.scaleFactor * crossing.scaleFactor);
            crossings << crossing.pixel << ',' << crossing.localParticleIndex;
            for (auto const value : crossing.image) {
                crossings << ',' << value;
            }
            for (double const value : crossing.position) {
                crossings << ',' << value;
            }
            for (double const value : crossing.velocity) {
                crossings << ',' << value;
            }
            crossings << ',' << crossing.conformalTime << ',' << crossing.tickOffset << ',' << crossing.scaleFactor << ',' << crossing.radius << ','
                      << crossing.inverseRadiusSquared;
            for (double const value : crossing.position) {
                crossings << ',' << value * scales.lengthMpc();
            }
            for (double const value : crossing.velocity) {
                crossings << ',' << value * scales.velocityKmPerSecond();
            }
            crossings << ',' << crossing.conformalTime * scales.timeSeconds() << ',' << crossing.tickOffset * scales.timeSeconds() << ','
                      << crossing.radius * scales.lengthMpc() << ',' << crossing.inverseRadiusSquared * inverseLengthMpcSquared;
            for (double const value : crossing.position) {
                crossings << ',' << value * properLength;
            }
            for (double const value : crossing.velocity) {
                crossings << ',' << value * scales.velocityCgs();
            }
            crossings << ',' << crossing.radius * properLength << ',' << crossing.inverseRadiusSquared * inverseLengthCgsSquared * inverseScaleSquared << '\n';
        }
        bins << "pixel,count,inverseRadiusSquared,inverseComovingRadiusSquaredPerMpc2,inverseProperRadiusSquaredPerCm2\n";
        for (auto const& bin : sample.bins) {
            bins << bin.pixel << ',' << bin.count << ',' << bin.inverseRadiusSquared << ',' << bin.inverseRadiusSquared * inverseLengthMpcSquared << ','
                 << bin.inverseProperRadiusSquared * inverseLengthCgsSquared << '\n';
        }
        crossings.close();
        bins.close();
        std::ofstream marker(temporaryComplete.string());
        marker.exceptions(std::ios::failbit | std::ios::badbit);
        marker << "CosmicTiger lightcone CSV version=2\n" << sample.crossings.size() << ' ' << sample.bins.size() << '\n';
        marker.close();
        hpx::filesystem::rename(temporaryCatalog, catalog);
        publishedCatalog = true;
        hpx::filesystem::rename(temporaryMap, map);
        publishedMap = true;
        hpx::filesystem::rename(temporaryComplete, complete);
        std::error_code ignored;
        hpx::filesystem::remove(pending, ignored);
    } catch (...) {
        std::error_code ignored;
        if (publishedCatalog) {
            hpx::filesystem::remove(catalog, ignored);
        }
        if (publishedMap) {
            hpx::filesystem::remove(map, ignored);
        }
        hpx::filesystem::remove_all(pending, ignored);
        throw;
    }
}

}    // namespace cosmictiger
