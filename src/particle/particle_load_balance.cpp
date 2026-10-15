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
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/particle/particle_load_balance.hpp>

#include <hpx/parallel/algorithms/sort.hpp>

#include <hpx/collectives/all_gather.hpp>
#include <hpx/collectives/all_to_all.hpp>
#include <hpx/collectives/channel_communicator.hpp>
#include <hpx/datastructures/serialization/tuple.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/serialize_buffer.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <optional>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cosmictiger::detail {

// Raw fixed-point words avoid requiring serialization on Fixed itself.
struct ParticleTransfer {
    std::array<std::uint32_t, NDIM> positions;
    std::array<float, NDIM> velocities;
    std::array<float, NDIM> accelerations;
    float potential;
    std::uint8_t rung;
};
static_assert(std::is_trivially_copyable_v<ParticleTransfer> && sizeof(float) == sizeof(std::uint32_t));

}    // namespace cosmictiger::detail

HPX_IS_BITWISE_SERIALIZABLE(cosmictiger::detail::ParticleTransfer)

namespace cosmictiger::detail {

using ParticlePayload = hpx::serialization::serialize_buffer<ParticleTransfer>;

std::string currentError() {
    PROFILE();
    try {
        throw;
    } catch (std::exception const& error) {
        return error.what();
    } catch (...) {
        return "unknown particle load balancing error";
    }
}

void agree(std::string const& basename, char const* phase, size_t sites, size_t rank, std::string error) {
    PROFILE();
    auto const name = basename + phase;
    auto const errors =
        hpx::collectives::all_gather(name.c_str(), std::move(error), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank)).get();
    for (size_t site = 0; site < errors.size(); site++) {
        if (!errors[site].empty()) {
            throw std::runtime_error("particle load balance " + std::string(phase) + " failed on locality " + std::to_string(site) + ": " + errors[site]);
        }
    }
}

ParticleCount checkedParticleSum(ParticleCount first, ParticleCount second) {
    PROFILE();
    if (second > std::numeric_limits<ParticleCount>::max() - first) {
        throw std::length_error("redistributed particle count overflows");
    }
    return first + second;
}

GridIndices interiorCoordinates(size_t index, size_t side) noexcept {
    PROFILE();
    GridIndices coordinates{};
    for (auto& coordinate : coordinates) {
        coordinate = static_cast<std::int64_t>(index % side);
        index /= side;
    }
    return coordinates;
}

size_t interiorIndex(GridIndices const& coordinates, size_t side) noexcept {
    PROFILE();
    return static_cast<size_t>(coordinates[0]) + side * (static_cast<size_t>(coordinates[1]) + side * static_cast<size_t>(coordinates[2]));
}

std::array<Coordinate, NDIM> transferPosition(ParticleTransfer const& particle) noexcept {
    PROFILE();
    return {Coordinate::fromRaw(particle.positions[0]), Coordinate::fromRaw(particle.positions[1]), Coordinate::fromRaw(particle.positions[2])};
}

ParticleTransfer readParticle(ParticleStorage const& particles, ParticleCount index) {
    PROFILE();
    ParticleTransfer result{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        result.positions[axis] = particles.position(axis, index).raw();
        result.velocities[axis] = particles.velocity(axis, index);
        if (particles.storesAcceleration()) {
            result.accelerations[axis] = particles.acceleration(axis, index);
        }
    }
    if (particles.storesPotential()) {
        result.potential = particles.potential(index);
    }
    result.rung = particles.flags(index).rung();
    return result;
}

std::array<Coordinate, NDIM> readPosition(ParticleStorage const& particles, ParticleCount index) {
    PROFILE();
    return {particles.position(0, index), particles.position(1, index), particles.position(2, index)};
}

void writeParticle(ParticleStorage& particles, ParticleCount index, ParticleTransfer const& particle) {
    PROFILE();
    for (size_t axis = 0; axis < NDIM; axis++) {
        particles.position(axis, index) = Coordinate::fromRaw(particle.positions[axis]);
        particles.velocity(axis, index) = particle.velocities[axis];
        if (particles.storesAcceleration()) {
            particles.acceleration(axis, index) = particle.accelerations[axis];
        }
    }
    if (particles.storesPotential()) {
        particles.potential(index) = particle.potential;
    }
    particles.flags(index) = {particle.rung, false};
}

std::vector<ParticleRange> ownedRanges(ParticleStorage const& particles, CartesianGrid const& grid) {
    PROFILE();
    auto const& layout = grid.layout();
    std::vector<ParticleRange> ghosts;
    for (size_t index = 0; index < layout.storedCellCount(); index++) {
        auto const coordinates = layout.cellCoordinates(index);
        if (!layout.isInterior(coordinates)) {
            auto const range = grid.cell(coordinates).localParticles;
            expect(!(range.end > particles.size()));
            if (range.begin != range.end) {
                ghosts.push_back(range);
            }
        }
    }
    hpx::sort(hpx::execution::seq, ghosts.begin(), ghosts.end(),
        hpx::annotated_function(
            [](ParticleRange first, ParticleRange second) {
                PROFILE();
                return first.begin < second.begin;
            },
            "particleLoadBalance.sort"));
    std::vector<ParticleRange> owned;
    ParticleCount end = 0;
    for (auto const ghost : ghosts) {
        if (end < ghost.begin) {
            owned.push_back({end, ghost.begin});
        }
        end = std::max(end, ghost.end);
    }
    if (end < particles.size()) {
        owned.push_back({end, particles.size()});
    }
    return owned;
}

// Chunks intersect the sorted owned ranges, skipping ghost intervals without
// copying particle indices into another O(N) array.
template <typename Function>
void forEachOwnedParticle(std::vector<ParticleRange> const& owned, size_t begin, size_t end, Function&& function) {
    PROFILE();
    auto range = std::upper_bound(owned.begin(), owned.end(), static_cast<ParticleCount>(begin),
        hpx::annotated_function(
            [](ParticleCount index, ParticleRange const& candidate) {
                PROFILE();
                return index < candidate.end;
            },
            "particleLoadBalance.compareOwnedRange"));
    for (; range != owned.end() && range->begin < end; range++) {
        ParticleCount const first = std::max(range->begin, static_cast<ParticleCount>(begin));
        ParticleCount const last = std::min(range->end, static_cast<ParticleCount>(end));
        for (ParticleCount index = first; index < last; index++) {
            function(index);
        }
    }
}

// Incoming payloads form a logical contiguous stream in source-site order.
// A chunk can cross empty payloads and site boundaries without concatenation.
template <typename Function>
void forEachReceivedParticle(
    std::vector<ParticlePayload> const& incoming, std::vector<size_t> const& payloadEnds, size_t begin, size_t end, Function&& function) {
    PROFILE();
    size_t site = static_cast<size_t>(std::upper_bound(payloadEnds.begin(), payloadEnds.end(), begin) - payloadEnds.begin());
    while (begin < end) {
        auto const& payload = incoming[expectIndex(site, incoming.size())];
        size_t const payloadBegin = site == 0 ? 0 : payloadEnds[site - 1];
        size_t const stop = std::min(end, payloadEnds[site]);
        for (size_t index = begin - payloadBegin; index < stop - payloadBegin; index++) {
            function(payload[expectIndex(index, payload.size())]);
        }
        begin = stop;
        site++;
    }
}

}    // namespace cosmictiger::detail

namespace cosmictiger {

void loadBalanceParticles(ParticleStorage& particles, CartesianGrid& grid, std::vector<hpx::id_type> const& localities, std::uint64_t operationId) {
    PROFILE();
    auto const& layout = grid.layout();
    size_t const sites = localities.size();
    size_t const rank = layout.localityIndex();
    // These are collective participation preconditions: without a shared site
    // list and valid site indices there is no well-defined collective to join.
    expect(!(sites == 0 || rank >= sites || localities[rank] != hpx::find_here()));
    std::string const basename = "/cosmictiger/particle_load_balance/" + std::to_string(operationId) + "/";
    std::vector<detail::ParticleTransfer> outgoing;
    std::vector<ParticleCount> sendCounts;
    std::vector<ParticleCount> countValues;
    std::vector<ParticleCount> sendOffsets;
    std::string error;
    try {
        expect(!(sites != layout.localityCount()));
        auto const owned = detail::ownedRanges(particles, grid);
        sendCounts.resize(sites);
        sendOffsets.resize(sites);
        detail::ParallelChunkPlan const sourcePlan(static_cast<size_t>(particles.size()));
        // Destination ranks are few relative to local particles. Each chunk
        // owns one dense rank histogram, later reused as disjoint cursors.
        std::vector<std::vector<ParticleCount>> chunkCounts(sourcePlan.size(), std::vector<ParticleCount>(sites));
        detail::parallelForChunks(sourcePlan,
            hpx::annotated_function(
                [&](size_t chunk, size_t begin, size_t end) {
                    PROFILE();
                    auto& counts = chunkCounts[chunk];
                    detail::forEachOwnedParticle(owned, begin, end,
                        hpx::annotated_function(
                            [&](ParticleCount index) {
                                PROFILE();
                                if (!particles.flags(index).isGhost()) {
                                    size_t const site = layout.particleOwner(detail::readPosition(particles, index)).localityIndex;
                                    counts[expectIndex(site, counts.size())]++;
                                }
                            },
                            "particleLoadBalance.countOwnedDestinations"));
                },
                "particleLoadBalance.parallelForChunks"));
        ParticleCount total = 0;
        for (size_t site = 0; site < sites; site++) {
            sendOffsets[site] = total;
            for (size_t chunk = 0; chunk < sourcePlan.size(); chunk++) {
                ParticleCount const count = chunkCounts[chunk][site];
                chunkCounts[chunk][site] = total;
                total = detail::checkedParticleSum(total, count);
                sendCounts[site] = detail::checkedParticleSum(sendCounts[site], count);
            }
        }
        if (total > outgoing.max_size()) {
            throw std::length_error("particle transfer buffer exceeds addressable memory");
        }
        outgoing.resize(static_cast<size_t>(total));
        detail::parallelForChunks(sourcePlan,
            hpx::annotated_function(
                [&](size_t chunk, size_t begin, size_t end) {
                    PROFILE();
                    auto& cursors = chunkCounts[chunk];
                    detail::forEachOwnedParticle(owned, begin, end,
                        hpx::annotated_function(
                            [&](ParticleCount index) {
                                PROFILE();
                                if (!particles.flags(index).isGhost()) {
                                    auto const particle = detail::readParticle(particles, index);
                                    size_t const site = layout.particleOwner(detail::transferPosition(particle)).localityIndex;
                                    ParticleCount const destination = cursors[expectIndex(site, cursors.size())]++;
                                    outgoing[expectIndex(static_cast<size_t>(destination), outgoing.size())] = particle;
                                }
                            },
                            "particleLoadBalance.packOwnedParticles"));
                },
                "particleLoadBalance.parallelForChunks"));
        countValues = sendCounts;
    } catch (...) {
        error = detail::currentError();
    }
    // Small topology/error metadata may use a centralized collective; particle
    // payloads below always travel directly between source and destination.
    using Preparation = hpx::tuple<size_t, size_t, bool, bool, std::string>;
    auto const preparationName = basename + "preparation";
    auto const preparation = hpx::collectives::all_gather(preparationName.c_str(),
        Preparation{layout.globalSideLength(), layout.localitiesPerAxis(), particles.storesAcceleration(), particles.storesPotential(), std::move(error)},
        hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
                                 .get();
    for (size_t site = 0; site < sites; site++) {
        auto const& [globalSide, localitySide, acceleration, potential, siteError] = preparation[site];
        if (!siteError.empty()) {
            throw std::runtime_error("particle load balance preparation failed on locality " + std::to_string(site) + ": " + siteError);
        }
        expect(!(globalSide != layout.globalSideLength() || localitySide != layout.localitiesPerAxis()));
        if (acceleration != particles.storesAcceleration() || potential != particles.storesPotential()) {
            throw std::runtime_error("particle load balance requires matching gravity output fields on every locality");
        }
    }

    auto const countName = basename + "counts";
    auto const receiveCounts =
        hpx::collectives::all_to_all(countName.c_str(), std::move(countValues), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
            .get();
    ParticleStorage replacement;
    std::vector<ParticleCount> cellCounts;
    std::vector<ParticleRange> ranges;
    std::vector<detail::ParticlePayload> incoming;
    std::vector<hpx::future<void>> sends;
    std::optional<hpx::collectives::channel_communicator> channel;
    ParticleCount total = 0;
    error.clear();
    auto const channelName = basename + "payload";
    try {
        for (auto const count : receiveCounts) {
            total = detail::checkedParticleSum(total, count);
        }
        replacement.configureGravityFields(particles.storesAcceleration(), particles.storesPotential());
        replacement.resize(total);
        cellCounts.resize(layout.interiorCellCount());
        ranges.resize(layout.interiorCellCount());
        incoming.resize(sites);
        sends.reserve(sites);
        channel.emplace(
            hpx::collectives::create_channel_communicator(channelName.c_str(), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
                .get());
    } catch (...) {
        error = detail::currentError();
    }
    detail::agree(basename, "allocation", sites, rank, std::move(error));

    error.clear();
    for (size_t site = 0; site < sites; site++) {
        if (site != rank && sendCounts[site] != 0) {
            try {
                // References remain live through the final barrier. Local
                // logical sites share an address space, so no serialization
                // occurs; copy their payload to keep ownership unambiguous.
                auto const mode = localities[site] == hpx::find_here() ? detail::ParticlePayload::copy : detail::ParticlePayload::reference;
                detail::ParticlePayload payload(outgoing.data() + static_cast<size_t>(sendOffsets[site]), static_cast<size_t>(sendCounts[site]), mode);
                sends.push_back(hpx::collectives::set(*channel, hpx::collectives::that_site_arg(site), std::move(payload)));
            } catch (...) {
                if (error.empty()) {
                    error = detail::currentError();
                }
            }
        }
    }
    for (auto& send : sends) {
        try {
            send.get();
        } catch (...) {
            if (error.empty()) {
                error = detail::currentError();
            }
        }
    }
    // Fresh channel slots accept a send without a waiting receive. Agree that
    // every send completed before waiting for any expected incoming payload.
    detail::agree(basename, "transfer", sites, rank, std::move(error));

    error.clear();
    try {
        for (size_t site = 0; site < sites; site++) {
            if (receiveCounts[site] == 0) {
                continue;
            }
            if (site == rank) {
                incoming[site] = detail::ParticlePayload(
                    outgoing.data() + static_cast<size_t>(sendOffsets[site]), static_cast<size_t>(sendCounts[site]), detail::ParticlePayload::reference);
            } else {
                incoming[site] = hpx::collectives::get<detail::ParticlePayload>(*channel, hpx::collectives::that_site_arg(site)).get();
            }
            expect(!(incoming[site].size() != receiveCounts[site]));
        }
        std::vector<size_t> payloadEnds(sites);
        ParticleCount received = 0;
        for (size_t site = 0; site < sites; site++) {
            received = detail::checkedParticleSum(received, incoming[site].size());
            payloadEnds[site] = static_cast<size_t>(received);
        }
        expectEqual(received, total);
        detail::ParallelChunkPlan const partitionPlan(static_cast<size_t>(total));
        // Sparse cell histograms avoid a workers-times-cell-count allocation
        // when the grid has one cell for each initial particle.
        std::vector<std::unordered_map<size_t, ParticleCount>> chunkCells(partitionPlan.size());
        detail::parallelForChunks(partitionPlan,
            hpx::annotated_function(
                [&](size_t chunk, size_t begin, size_t end) {
                    PROFILE();
                    auto& counts = chunkCells[chunk];
                    counts.reserve(std::min(end - begin, cellCounts.size()));
                    detail::forEachReceivedParticle(incoming, payloadEnds, begin, end,
                        hpx::annotated_function(
                            [&](detail::ParticleTransfer const& particle) {
                                PROFILE();
                                expectRange(0, particle.rung, Rung::maximumRung());
                                auto const owner = layout.particleOwner(detail::transferPosition(particle));
                                expectEqual(owner.localityIndex, rank);
                                size_t const cell = detail::interiorIndex(owner.coordinates, layout.interiorSideLength());
                                counts[expectIndex(cell, cellCounts.size())]++;
                            },
                            "particleLoadBalance.countReceivedCells"));
                },
                "particleLoadBalance.parallelForChunks"));
        for (auto const& counts : chunkCells) {
            for (auto const& [cell, count] : counts) {
                cellCounts[cell] = detail::checkedParticleSum(cellCounts[cell], count);
            }
        }
        ParticleCount offset = 0;
        for (size_t cell = 0; cell < cellCounts.size(); cell++) {
            ranges[cell] = {offset, detail::checkedParticleSum(offset, cellCounts[cell])};
            cellCounts[cell] = offset;
            offset = ranges[cell].end;
        }
        expectEqual(offset, total);
        // Assign each chunk a private interval within every cell it contains.
        // Chunk traversal follows source order, independent of task completion.
        for (auto& counts : chunkCells) {
            for (auto& [cell, count] : counts) {
                ParticleCount const size = count;
                count = cellCounts[cell];
                cellCounts[cell] += size;
            }
        }
        detail::parallelForChunks(partitionPlan,
            hpx::annotated_function(
                [&](size_t chunk, size_t begin, size_t end) {
                    PROFILE();
                    auto& cursors = chunkCells[chunk];
                    detail::forEachReceivedParticle(incoming, payloadEnds, begin, end,
                        hpx::annotated_function(
                            [&](detail::ParticleTransfer const& particle) {
                                PROFILE();
                                auto const owner = layout.particleOwner(detail::transferPosition(particle));
                                size_t const cell = detail::interiorIndex(owner.coordinates, layout.interiorSideLength());
                                auto const cursor = cursors.find(cell);
                                expect(cursor != cursors.end());
                                detail::writeParticle(replacement, cursor->second++, particle);
                            },
                            "particleLoadBalance.unpackReceivedParticles"));
                },
                "particleLoadBalance.parallelForChunks"));
    } catch (...) {
        error = detail::currentError();
    }
    detail::agree(basename, "partition", sites, rank, std::move(error));

    // Replacement preparation is complete on every participant. The swap and
    // these prevalidated cell assignments require no allocation.
    particles.swap(replacement);
    for (size_t cell = 0; cell < ranges.size(); cell++) {
        grid.setInteriorRange(detail::interiorCoordinates(cell, layout.interiorSideLength()), ranges[cell]);
    }
    grid.clearGhostRanges();
    error.clear();
    try {
        grid.sortParticlesByRung(particles);
    } catch (...) {
        error = detail::currentError();
    }
    // Sorting may allocate or fail to launch tasks after replacement is
    // committed. Every participant must still enter the completion agreement.
    detail::agree(basename, "complete", sites, rank, std::move(error));
}

void loadBalanceParticles(CartesianGrid& grid, std::uint64_t operationId) {
    PROFILE();
    loadBalanceParticles(localParticleStorage(), grid, hpx::find_all_localities(), operationId);
}

}    // namespace cosmictiger
