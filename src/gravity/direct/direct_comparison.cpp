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

#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/collectives/all_gather.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/vector.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger::gravity {
namespace detail {

    struct DirectNormSums {
        double errorSum = 0.0;
        double errorSquared = 0.0;
        double errorMaximum = 0.0;
        double referenceSum = 0.0;
        double referenceSquared = 0.0;
        double referenceMaximum = 0.0;
    };

    template <typename Archive>
    void serialize(Archive& archive, DirectNormSums& sums, unsigned) {
        PROFILE();
        archive & sums.errorSum & sums.errorSquared & sums.errorMaximum & sums.referenceSum & sums.referenceSquared & sums.referenceMaximum;
    }

    struct DirectComparisonSums {
        size_t count = 0;
        DirectNormSums acceleration;
        DirectNormSums potential;
        std::string error;
    };

    template <typename Archive>
    void serialize(Archive& archive, DirectComparisonSums& sums, unsigned) {
        PROFILE();
        archive & sums.count & sums.acceleration & sums.potential & sums.error;
    }

    void addNormSample(DirectNormSums& sums, double error, double reference) {
        PROFILE();
        sums.errorSum += error;
        sums.errorSquared += error * error;
        sums.errorMaximum = error > sums.errorMaximum ? error : sums.errorMaximum;
        sums.referenceSum += reference;
        sums.referenceSquared += reference * reference;
        sums.referenceMaximum = reference > sums.referenceMaximum ? reference : sums.referenceMaximum;
    }

    void mergeNormSums(DirectNormSums& sums, DirectNormSums const& other) {
        PROFILE();
        FpeGuard guard;
        sums.errorSum += other.errorSum;
        sums.errorSquared += other.errorSquared;
        sums.errorMaximum = other.errorMaximum > sums.errorMaximum ? other.errorMaximum : sums.errorMaximum;
        sums.referenceSum += other.referenceSum;
        sums.referenceSquared += other.referenceSquared;
        sums.referenceMaximum = other.referenceMaximum > sums.referenceMaximum ? other.referenceMaximum : sums.referenceMaximum;
    }

    double directNormRatio(double error, double reference) {
        PROFILE();
        return reference != 0.0 ? error / reference : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    }

    DirectErrorNorms finishNorms(DirectNormSums const& sums, size_t count) {
        PROFILE();
        if (count == 0) {
            return {};
        }
        FpeGuard guard;
        double const inverseCount = 1.0 / static_cast<double>(count);
        DirectErrorNorms result;
        result.l1 = sums.errorSum * inverseCount;
        result.l2 = std::sqrt(sums.errorSquared * inverseCount);
        result.linf = sums.errorMaximum;
        result.referenceL1 = sums.referenceSum * inverseCount;
        result.referenceL2 = std::sqrt(sums.referenceSquared * inverseCount);
        result.referenceLinf = sums.referenceMaximum;
        result.relativeL1 = directNormRatio(result.l1, result.referenceL1);
        result.relativeL2 = directNormRatio(result.l2, result.referenceL2);
        result.relativeLinf = directNormRatio(result.linf, result.referenceLinf);
        return result;
    }

    DirectComparisonSums fieldSums(std::span<DirectSolveField const> reference, std::span<DirectSolveField const> approximation) {
        PROFILE();
        if (reference.size() != approximation.size()) {
            throw std::invalid_argument("direct comparison field counts differ");
        }
        DirectComparisonSums result;
        result.count = reference.size();
        FpeGuard guard;
        for (size_t index = 0; index < reference.size(); index++) {
            double errorSquared = 0.0;
            double referenceSquared = 0.0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const error = approximation[index].acceleration[axis] - reference[index].acceleration[axis];
                errorSquared += error * error;
                referenceSquared += reference[index].acceleration[axis] * reference[index].acceleration[axis];
            }
            addNormSample(result.acceleration, std::sqrt(errorSquared), std::sqrt(referenceSquared));
            addNormSample(result.potential, std::abs(approximation[index].potential - reference[index].potential), std::abs(reference[index].potential));
        }
        return result;
    }

    std::string directComparisonError() {
        PROFILE();
        try {
            throw;
        } catch (std::exception const& error) {
            return error.what();
        } catch (...) {
            return "unknown direct comparison failure";
        }
    }

}    // namespace detail

DirectComparison compareDirectFields(std::span<DirectSolveField const> reference, std::span<DirectSolveField const> approximation) {
    PROFILE();
    auto const sums = detail::fieldSums(reference, approximation);
    return {0, sums.count, detail::finishNorms(sums.acceleration, sums.count), detail::finishNorms(sums.potential, sums.count)};
}

DirectComparison compareDirectGravity(
    ParticleStorage const& particles, CartesianGrid const& grid, GlobalConstantsData const& constants, std::uint64_t operationId) {
    using PackedPosition = std::array<std::uint32_t, NDIM>;
    PROFILE();
    auto const localities = hpx::find_all_localities();
    auto const here = hpx::find_here();
    size_t rank = 0;
    while (rank < localities.size() && localities[rank] != here) {
        rank++;
    }
    if (rank == localities.size()) {
        throw std::runtime_error("direct comparison locality is not registered");
    }
    auto const sites = hpx::collectives::num_sites_arg(localities.size());
    auto const site = hpx::collectives::this_site_arg(rank);
    std::string const basename = "/cosmictiger/direct/comparison/" + std::to_string(operationId);
    std::vector<PackedPosition> localSources;
    std::vector<ParticleCount> owned;
    std::string error;
    try {
        if (!particles.storesAcceleration() || !particles.storesPotential()) {
            throw std::invalid_argument("direct comparison requires current PMMM acceleration and potential fields");
        }
        for (size_t cellIndex = 0; cellIndex < grid.layout().storedCellCount(); cellIndex++) {
            auto const coordinates = grid.layout().cellCoordinates(cellIndex);
            if (!grid.layout().isInterior(coordinates)) {
                continue;
            }
            auto const range = grid.cell(coordinates).localParticles;
            expectRange(range.begin, range.end, particles.size());
            for (ParticleCount index = range.begin; index < range.end; index++) {
                localSources.push_back({particles.position(0, index).raw(), particles.position(1, index).raw(), particles.position(2, index).raw()});
                owned.push_back(index);
            }
        }
    } catch (...) {
        error = detail::directComparisonError();
    }
    auto const errors = hpx::collectives::all_gather((basename + "/prepare").c_str(), std::move(error), sites, site).get();
    for (auto const& failure : errors) {
        if (!failure.empty()) {
            throw std::runtime_error("direct comparison preparation failed: " + failure);
        }
    }
    auto const sourceBlocks = hpx::collectives::all_gather((basename + "/sources").c_str(), std::move(localSources), sites, site).get();
    detail::DirectComparisonSums localSums;
    size_t sourceCount = 0;
    try {
        size_t offset = 0;
        for (size_t index = 0; index < sourceBlocks.size(); index++) {
            if (sourceBlocks[index].size() > std::numeric_limits<size_t>::max() - sourceCount) {
                throw std::length_error("direct source count overflow");
            }
            if (index < rank) {
                offset += sourceBlocks[index].size();
            }
            sourceCount += sourceBlocks[index].size();
        }
        if (sourceCount != constants.globalParticleCount) {
            throw std::runtime_error("direct comparison did not gather every owned particle exactly once");
        }
        std::vector<Position> sources;
        sources.reserve(sourceCount);
        for (auto const& block : sourceBlocks) {
            for (auto const& position : block) {
                sources.push_back({Coordinate::fromRaw(position[0]), Coordinate::fromRaw(position[1]), Coordinate::fromRaw(position[2])});
            }
        }
        auto const sample = sampleDirectSinks(sourceCount, constants.directSinkFraction, constants.directSeed);
        std::vector<size_t> sinks;
        std::vector<DirectSolveField> approximation;
        for (size_t index : sample) {
            if (index >= offset && index - offset < owned.size()) {
                sinks.push_back(index);
                ParticleCount const particle = owned[index - offset];
                DirectSolveField field;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    field.acceleration[axis] = particles.acceleration(axis, particle);
                }
                field.potential = particles.potential(particle);
                approximation.push_back(field);
            }
        }
        std::vector<DirectSolveField> fields(sinks.size());
        DirectSolveRequest request{sources.data(), sources.size(), sinks.data(), sinks.size(), fields.data(), constants.gravityParticleMass,
            constants.softeningLength, constants.softeningKernelOrder, constants.softeningModel};
#ifdef COSMICTIGER_ENABLE_CUDA
        if (constants.cudaDirect && !sinks.empty()) {
            ::cosmictiger::detail::ParticleField<Position> deviceSources;
            ::cosmictiger::detail::ParticleField<size_t> deviceSinks;
            ::cosmictiger::detail::ParticleField<DirectSolveField> deviceFields;
            deviceSources.reserve(sources.size(), 0);
            deviceSinks.reserve(sinks.size(), 0);
            deviceFields.reserve(fields.size(), 0);
            deviceSources.initialize(0, sources.size());
            deviceSinks.initialize(0, sinks.size());
            deviceFields.initialize(0, fields.size());
            for (size_t index = 0; index < sources.size(); index++) {
                deviceSources.data()[index] = sources[index];
            }
            for (size_t index = 0; index < sinks.size(); index++) {
                deviceSinks.data()[index] = sinks[index];
            }
            request.sources = deviceSources.data();
            request.sinks = deviceSinks.data();
            request.fields = deviceFields.data();
            launchDirectSolveCuda(request).get();
            for (size_t index = 0; index < fields.size(); index++) {
                fields[index] = deviceFields.data()[index];
            }
        } else
#endif
        {
#ifndef COSMICTIGER_ENABLE_CUDA
            if (constants.cudaDirect) {
                throw std::invalid_argument("CUDA direct solve requested in a CPU-only build");
            }
#endif
            directSolveCpu(request);
        }
        localSums = detail::fieldSums(fields, approximation);
    } catch (...) {
        localSums.error = detail::directComparisonError();
    }
    auto const results = hpx::collectives::all_gather((basename + "/results").c_str(), std::move(localSums), sites, site).get();
    detail::DirectComparisonSums sums;
    for (auto const& result : results) {
        if (!result.error.empty()) {
            throw std::runtime_error("direct comparison failed: " + result.error);
        }
        sums.count += result.count;
        detail::mergeNormSums(sums.acceleration, result.acceleration);
        detail::mergeNormSums(sums.potential, result.potential);
    }
    return {sourceCount, sums.count, detail::finishNorms(sums.acceleration, sums.count), detail::finishNorms(sums.potential, sums.count)};
}

std::string formatDirectComparison(DirectComparison const& comparison) {
    PROFILE();
    std::ostringstream output;
    output << "Direct vs PMMM: " << comparison.sinkCount << " sampled sinks, " << comparison.sourceCount << " owned sources (periodic Ewald)\n";
    output << std::scientific << std::setprecision(6);
    auto write = [&](char const* label, DirectErrorNorms const& norms) {
        PROFILE();
        output << "  " << label << " relative L1=" << norms.relativeL1 << " L2=" << norms.relativeL2 << " Linf=" << norms.relativeLinf << '\n';
        output << "  " << label << " absolute L1=" << norms.l1 << " L2=" << norms.l2 << " Linf=" << norms.linf << '\n';
        output << "  " << label << " reference L1=" << norms.referenceL1 << " L2=" << norms.referenceL2 << " Linf=" << norms.referenceLinf << '\n';
    };
    write("acceleration", comparison.acceleration);
    write("potential", comparison.potential);
    return output.str();
}

}    // namespace cosmictiger::gravity
