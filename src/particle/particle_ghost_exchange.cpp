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
#include <cosmictiger/particle/particle_ghost_exchange.hpp>

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
#include <utility>
#include <vector>

namespace cosmictiger::detail {

struct GhostParticleTransfer {
    std::array<std::uint32_t, NDIM> positions;
    std::array<float, NDIM> velocities;
    std::array<float, NDIM> accelerations;
    float potential;
    std::uint8_t rung;
};
static_assert(std::is_trivially_copyable_v<GhostParticleTransfer>);

}    // namespace cosmictiger::detail

HPX_IS_BITWISE_SERIALIZABLE(cosmictiger::detail::GhostParticleTransfer)

namespace cosmictiger::detail {
namespace {

    using GhostPayload = hpx::serialization::serialize_buffer<GhostParticleTransfer>;
    using OwnerRange = std::array<ParticleCount, 2>;

    struct GhostTarget {
        GridIndices coordinates;
        GridCellOwner owner;
        size_t requestIndex;
        ParticleRange localParticles;
        ParticleRange ownerParticles;
        ParticleCount payloadOffset;
    };

    struct GhostPackingRange {
        GridIndices coordinates;
        size_t site;
        ParticleRange particles;
        ParticleCount payloadOffset;
        ParticleCount workEnd;
    };

    std::string ghostError() {
        PROFILE();
        try {
            throw;
        } catch (std::exception const& error) {
            return error.what();
        } catch (...) {
            return "unknown ghost particle exchange error";
        }
    }

    void agreeGhosts(std::string const& basename, char const* phase, size_t sites, size_t rank, std::string error) {
        PROFILE();
        auto const name = basename + phase;
        auto const errors =
            hpx::collectives::all_gather(name.c_str(), std::move(error), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank)).get();
        for (size_t site = 0; site < errors.size(); site++) {
            if (!errors[site].empty()) {
                throw std::runtime_error("ghost particle exchange " + std::string(phase) + " failed on locality " + std::to_string(site) + ": " + errors[site]);
            }
        }
    }

    ParticleCount ghostSum(ParticleCount first, ParticleCount second) {
        PROFILE();
        if (second > std::numeric_limits<ParticleCount>::max() - first) {
            throw std::length_error("ghost particle count overflows");
        }
        return first + second;
    }

    // Compute the owned prefix while consuming interior range metadata.
    ParticleCount ownedPrefix([[maybe_unused]] ParticleStorage const& particles, CartesianGrid const& grid) {
        PROFILE();
        auto const& layout = grid.layout();
        ParticleCount owned = 0;
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            auto const coordinates = layout.cellCoordinates(index);
            if (layout.isInterior(coordinates)) {
                auto const& cell = grid.cell(coordinates);
                auto const range = cell.localParticles;
                expectRange(range.begin, range.end, particles.size());
                expectEqual(range.begin, owned);
                expectEqual(cell.ownerParticles.begin, range.begin);
                expectEqual(cell.ownerParticles.end, range.end);
                owned = range.end;
            }
        }
        return owned;
    }

    [[maybe_unused]] std::array<Coordinate, NDIM> ghostPosition(GhostParticleTransfer const& particle) {
        PROFILE();
        return {Coordinate::fromRaw(particle.positions[0]), Coordinate::fromRaw(particle.positions[1]), Coordinate::fromRaw(particle.positions[2])};
    }

    GhostParticleTransfer packGhost(ParticleStorage const& particles, ParticleCount index) {
        PROFILE();
        GhostParticleTransfer result{};
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

    void unpackGhost(ParticleStorage& particles, ParticleCount index, GhostParticleTransfer const& particle, bool isGhost) {
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
        particles.flags(index) = {particle.rung, isGhost};
    }

}    // namespace
}    // namespace cosmictiger::detail

namespace cosmictiger {

void exchangeGhostParticles(ParticleStorage& particles, CartesianGrid& grid, std::vector<hpx::id_type> const& localities, std::uint64_t operationId) {
    PROFILE();
    auto const& layout = grid.layout();
    size_t const sites = localities.size();
    size_t const rank = layout.localityIndex();
    expect(!(sites == 0 || rank >= sites || localities[rank] != hpx::find_here()));
    std::string const basename = "/cosmictiger/particle_ghost_exchange/" + std::to_string(operationId) + "/";
    ParticleCount owned = 0;
    std::vector<detail::GhostTarget> targets;
    std::vector<std::vector<size_t>> requests;
    std::string error;
    try {
        expect(!(sites != layout.localityCount()));
        owned = detail::ownedPrefix(particles, grid);
        requests.resize(sites);
        targets.reserve(layout.storedCellCount() - layout.interiorCellCount());
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            auto const coordinates = layout.cellCoordinates(index);
            if (!layout.isInterior(coordinates)) {
                auto const owner = layout.cellOwner(coordinates);
                auto& request = requests[owner.localityIndex];
                targets.push_back({coordinates, owner, request.size(), {}, {}, 0});
                request.push_back(layout.cellIndex(owner.coordinates));
            }
        }
    } catch (...) {
        error = detail::ghostError();
    }
    using Preparation = hpx::tuple<size_t, size_t, size_t, bool, bool, std::string>;
    auto const preparationName = basename + "preparation";
    auto const preparation = hpx::collectives::all_gather(preparationName.c_str(),
        Preparation{layout.globalSideLength(), layout.localitiesPerAxis(), layout.ghostWidth(), particles.storesAcceleration(), particles.storesPotential(),
            std::move(error)},
        hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
                                 .get();
    for (size_t site = 0; site < sites; site++) {
        auto const& [globalSide, localitySide, ghostWidth, acceleration, potential, siteError] = preparation[site];
        if (!siteError.empty()) {
            throw std::runtime_error("ghost particle exchange preparation failed on locality " + std::to_string(site) + ": " + siteError);
        }
        expect(!(globalSide != layout.globalSideLength() || localitySide != layout.localitiesPerAxis()));
        if (ghostWidth != layout.ghostWidth()) {
            throw std::runtime_error("ghost exchange requires matching halo widths on every locality");
        }
        if (acceleration != particles.storesAcceleration() || potential != particles.storesPotential()) {
            throw std::runtime_error("ghost exchange requires matching gravity output fields on every locality");
        }
    }

    auto const requestName = basename + "requests";
    auto const incomingRequests =
        hpx::collectives::all_to_all(requestName.c_str(), std::move(requests), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
            .get();
    std::vector<std::vector<detail::OwnerRange>> responses;
    std::vector<std::vector<detail::GhostParticleTransfer>> outgoing;
    error.clear();
    try {
        responses.resize(sites);
        outgoing.resize(sites);
        std::vector<detail::GhostPackingRange> packingRanges;
        ParticleCount workCount = 0;
        for (size_t site = 0; site < sites; site++) {
            responses[site].reserve(incomingRequests[site].size());
            ParticleCount total = 0;
            for (auto const cellIndex : incomingRequests[site]) {
                auto const coordinates = layout.cellCoordinates(cellIndex);
                expect(!(!layout.isInterior(coordinates)));
                auto const range = grid.cell(coordinates).localParticles;
                expectRange(range.begin, range.end, owned);
                responses[site].push_back({range.begin, range.end});
                ParticleCount const count = range.end - range.begin;
                if (count != 0) {
                    workCount = detail::ghostSum(workCount, count);
                    packingRanges.push_back({coordinates, site, range, total, workCount});
                }
                total = detail::ghostSum(total, count);
            }
            if (total > outgoing[site].max_size()) {
                throw std::length_error("ghost transfer buffer exceeds addressable memory");
            }
            outgoing[site].resize(static_cast<size_t>(total));
        }
        if (workCount > std::numeric_limits<size_t>::max()) {
            throw std::length_error("ghost packing work exceeds addressable range");
        }
        // Flatten requested particles across peers, retaining each peer's cell
        // order. Large cells can span chunks; empty requests keep their response
        // entries without adding copy work or a task per cell/peer.
        detail::ParallelChunkPlan const chunks(static_cast<size_t>(workCount));
        detail::parallelForChunks(chunks,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    auto range = std::upper_bound(packingRanges.begin(), packingRanges.end(), static_cast<ParticleCount>(begin),
                        hpx::annotated_function(
                            [](ParticleCount index, detail::GhostPackingRange const& candidate) {
                                PROFILE();
                                return index < candidate.workEnd;
                            },
                            "particleGhostExchange.comparePackingRange"));
                    ParticleCount cursor = begin;
                    while (cursor < end) {
                        expect(range != packingRanges.end());
                        ParticleCount const workBegin = range->workEnd - (range->particles.end - range->particles.begin);
                        ParticleCount const stop = std::min(static_cast<ParticleCount>(end), range->workEnd);
                        auto& payload = outgoing[range->site];
                        for (; cursor < stop; cursor++) {
                            ParticleCount const offset = cursor - workBegin;
                            ParticleCount const index = range->particles.begin + offset;
                            if constexpr (validationChecksEnabled()) {
                                auto const owner =
                                    layout.particleOwner({particles.position(0, index), particles.position(1, index), particles.position(2, index)});
                                expect(!particles.flags(index).isGhost());
                                expectEqual(owner.localityIndex, rank);
                                expectEqual(owner.coordinates, range->coordinates);
                            }
                            payload[expectIndex(static_cast<size_t>(range->payloadOffset + offset), payload.size())] = detail::packGhost(particles, index);
                        }
                        range++;
                    }
                },
                "particleGhostExchange.parallelForChunks"));
    } catch (...) {
        error = detail::ghostError();
    }
    detail::agreeGhosts(basename, "packing", sites, rank, std::move(error));

    auto const responseName = basename + "ranges";
    auto const ownerRanges =
        hpx::collectives::all_to_all(responseName.c_str(), std::move(responses), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
            .get();
    std::vector<ParticleCount> receiveCounts;
    std::vector<detail::GhostPayload> incoming;
    std::vector<hpx::future<void>> sends;
    std::optional<hpx::collectives::channel_communicator> channel;
    ParticleCount total = owned;
    error.clear();
    auto const channelName = basename + "payload";
    try {
        receiveCounts.resize(sites);
        incoming.resize(sites);
        sends.reserve(sites);
        for (auto& target : targets) {
            size_t const site = target.owner.localityIndex;
            auto const range = ownerRanges[site][expectIndex(target.requestIndex, ownerRanges[site].size())];
            expect(!(range[0] > range[1]));
            ParticleCount const count = range[1] - range[0];
            target.ownerParticles = {range[0], range[1]};
            target.payloadOffset = receiveCounts[site];
            receiveCounts[site] = detail::ghostSum(receiveCounts[site], count);
            target.localParticles = {total, detail::ghostSum(total, count)};
            total = target.localParticles.end;
        }
        // Validate addressability before any particle messages are sent.
        (void) detail::particleAllocationBytes<Coordinate>(ParticleStorage::capacityForSize(total));
        channel.emplace(
            hpx::collectives::create_channel_communicator(channelName.c_str(), hpx::collectives::num_sites_arg(sites), hpx::collectives::this_site_arg(rank))
                .get());
    } catch (...) {
        error = detail::ghostError();
    }
    detail::agreeGhosts(basename, "allocation", sites, rank, std::move(error));

    error.clear();
    for (size_t site = 0; site < sites; site++) {
        if (site != rank && !outgoing[site].empty()) {
            try {
                // Same-process logical peers need an owning copy; remote
                // reference buffers stay live through the completion barrier.
                auto const mode = localities[site] == hpx::find_here() ? detail::GhostPayload::copy : detail::GhostPayload::reference;
                detail::GhostPayload payload(outgoing[site].data(), outgoing[site].size(), mode);
                sends.push_back(hpx::collectives::set(*channel, hpx::collectives::that_site_arg(site), std::move(payload)));
            } catch (...) {
                if (error.empty()) {
                    error = detail::ghostError();
                }
            }
        }
    }
    for (auto& send : sends) {
        try {
            send.get();
        } catch (...) {
            if (error.empty()) {
                error = detail::ghostError();
            }
        }
    }
    detail::agreeGhosts(basename, "transfer", sites, rank, std::move(error));

    error.clear();
    try {
        for (size_t site = 0; site < sites; site++) {
            if (receiveCounts[site] == 0) {
                continue;
            }
            if (site == rank) {
                incoming[site] = detail::GhostPayload(outgoing[site].data(), outgoing[site].size(), detail::GhostPayload::reference);
            } else {
                incoming[site] = hpx::collectives::get<detail::GhostPayload>(*channel, hpx::collectives::that_site_arg(site)).get();
            }
            expect(!(incoming[site].size() != receiveCounts[site]));
        }
        // Payload ownership is checked when each ghost is consumed below.
        particles.reserve(total);
    } catch (...) {
        error = detail::ghostError();
    }
    detail::agreeGhosts(basename, "ready", sites, rank, std::move(error));

    error.clear();
    try {
        particles.resize(total);
        // Allocation established addressability above. Assigned suffix ranges
        // are disjoint, including independent copies of one periodic cell.
        detail::ParallelChunkPlan const chunks(static_cast<size_t>(total - owned));
        detail::parallelForChunks(chunks,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    ParticleCount cursor = owned + begin;
                    ParticleCount const last = owned + end;
                    auto target = std::upper_bound(targets.begin(), targets.end(), cursor,
                        hpx::annotated_function(
                            [](ParticleCount index, detail::GhostTarget const& candidate) {
                                PROFILE();
                                return index < candidate.localParticles.end;
                            },
                            "particleGhostExchange.compareTargetRange"));
                    while (cursor < last) {
                        expect(target != targets.end());
                        auto const& payload = incoming[target->owner.localityIndex];
                        ParticleCount const stop = std::min(last, target->localParticles.end);
                        for (; cursor < stop; cursor++) {
                            auto const offset = static_cast<size_t>(target->payloadOffset + (cursor - target->localParticles.begin));
                            auto const& particle = payload[expectIndex(offset, payload.size())];
                            if constexpr (validationChecksEnabled()) {
                                auto const owner = layout.particleOwner(detail::ghostPosition(particle));
                                expectRange(0, particle.rung, Rung::maximumRung());
                                expectEqual(owner.localityIndex, target->owner.localityIndex);
                                expectEqual(owner.coordinates, target->owner.coordinates);
                            }
                            detail::unpackGhost(particles, cursor, particle, target->owner.localityIndex != rank);
                        }
                        target++;
                    }
                },
                "particleGhostExchange.parallelForChunks"));
        // A cell may have been copied by several chunks. Publish its ranges
        // once, only after all copies have finished, then sort cells by rung.
        for (auto const& target : targets) {
            grid.setGhostRanges(target.coordinates, target.localParticles, target.ownerParticles);
        }
        grid.sortParticlesByRung(particles);
    } catch (...) {
        error = detail::ghostError();
    }
    detail::agreeGhosts(basename, "complete", sites, rank, std::move(error));
}

void exchangeGhostParticles(CartesianGrid& grid, std::uint64_t operationId) {
    PROFILE();
    exchangeGhostParticles(localParticleStorage(), grid, hpx::find_all_localities(), operationId);
}

void clearGhostParticles(ParticleStorage& particles, CartesianGrid& grid) {
    PROFILE();
    ParticleCount const owned = grid.ownedParticleEnd();
    expectRange(0, owned, particles.size());
    particles.resize(owned);
    grid.clearGhostRanges();
}

void clearGhostParticles(CartesianGrid& grid) {
    PROFILE();
    clearGhostParticles(localParticleStorage(), grid);
}

}    // namespace cosmictiger
