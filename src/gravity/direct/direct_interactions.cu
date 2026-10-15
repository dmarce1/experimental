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

#include <cosmictiger/gravity/direct/direct_interactions_cuda.cuh>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/debug.hpp>

#include <exception>

namespace cosmictiger::gravity {

namespace detail {

    template <int order, bool computePotential>
    hpx::future<void> launchDirectInteractionOrder(DirectInteractionRequest request, unsigned blocks, cudaStream_t stream, int deviceId) {
        PROFILE();
        auto const backend = makeDirectKickBackend<order, computePotential>(request);
        return cuda::launch(hpx::annotated_function(
                                [request, blocks, backend](cudaStream_t launchStream) {
                                    PROFILE();
                                    directInteractionsKernel<DirectKickBackend, order, computePotential>
                                        <<<blocks, kDirectInteractionTileSize, 0, launchStream>>>(request, backend);
                                },
                                "gravity::launchDirectInteractionOrder"),
            stream, deviceId);
    }

    template <bool computePotential>
    hpx::future<void> launchDirectInteractionPotential(DirectInteractionRequest request, int ferrersOrder, unsigned blocks, cudaStream_t stream, int deviceId) {
        PROFILE();
        switch (ferrersOrder) {
        case 0:
            return launchDirectInteractionOrder<0, computePotential>(request, blocks, stream, deviceId);
        case 1:
            return launchDirectInteractionOrder<1, computePotential>(request, blocks, stream, deviceId);
        case 2:
            return launchDirectInteractionOrder<2, computePotential>(request, blocks, stream, deviceId);
        default:
            expectEqual(ferrersOrder, 3);
            return launchDirectInteractionOrder<3, computePotential>(request, blocks, stream, deviceId);
        }
    }

}    // namespace detail

hpx::future<void> launchDirectInteractionsCuda(DirectInteractionRequest request, int ferrersOrder, bool computePotential, cudaStream_t stream, int deviceId) {
    ParticleCount constexpr kMaximumBlocks = 65535;
    PROFILE();
    try {
        expect(request.source.begin <= request.source.end);
        expect(request.sink.begin <= request.sink.end);
        expectRange(0, request.rung, Rung::maximumRung());
        expectRange(0, ferrersOrder, 3);
        ParticleCount const sinkCount = request.sink.end - request.sink.begin;
        if (sinkCount == 0) {
            return hpx::make_ready_future();
        }
        if constexpr (validationChecksEnabled()) {
            request.particles.flags = expectNonZero(request.particles.flags);
            for (unsigned axis = 0; axis < NDIM; axis++) {
                request.particles.positions[axis] = expectNonZero(request.particles.positions[axis]);
                request.particles.velocities[axis] = expectNonZero(request.particles.velocities[axis]);
            }
        }

        ParticleCount const neededBlocks = (sinkCount - 1) / detail::kDirectInteractionTileSize + 1;
        unsigned const blocks = static_cast<unsigned>(neededBlocks < kMaximumBlocks ? neededBlocks : kMaximumBlocks);
        if (computePotential) {
            return detail::launchDirectInteractionPotential<true>(request, ferrersOrder, blocks, stream, deviceId);
        }
        return detail::launchDirectInteractionPotential<false>(request, ferrersOrder, blocks, stream, deviceId);
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger::gravity
