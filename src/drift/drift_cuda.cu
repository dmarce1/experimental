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

#include <cosmictiger/drift/drift.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cuda/cuda_runtime.hpp>

#include <cmath>
#include <cstdint>
#include <exception>

namespace cosmictiger {

namespace detail {

    inline constexpr unsigned kDriftWarpSize = 32;
    inline constexpr unsigned kDriftBlockSize = 256;
    static_assert(kDriftBlockSize % kDriftWarpSize == 0);

    __global__ void driftParticlesKernel(DriftRequest request) {
        PROFILE();
        ParticleCount const count = request.range.end - request.range.begin;
        ParticleCount const stride = static_cast<ParticleCount>(gridDim.x) * blockDim.x;
        ParticleCount offset = static_cast<ParticleCount>(blockIdx.x) * blockDim.x + threadIdx.x;
        while (offset < count) {
            ParticleCount const index = request.range.begin + offset;
            if (request.ownedOnly || !request.particles.flags[index].isGhost()) {
                FpeGuard guard;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    double const displacement = double(request.particles.momenta[axis][index]) * request.factor;
                    // Every finite double with magnitude >= 2^52 is integral.
                    // Otherwise subtraction of trunc(displacement) gives the
                    // same signed fraction as fmod(displacement, 1), without division.
                    double const fractional = displacement - trunc(displacement);
                    double const scaled = ldexp(fractional, Coordinate::fractionalBits());
                    [[maybe_unused]] double const period = ldexp(1.0, Coordinate::fractionalBits());
                    expectRange(-period, scaled, period);
                    auto const shift = static_cast<std::uint32_t>(static_cast<std::int64_t>(scaled));
                    auto& position = request.particles.positions[axis][index];
                    position = Coordinate::fromRaw(static_cast<std::uint32_t>(std::uint64_t(position.raw()) + shift));
                }
            }
            if (count - offset <= stride) {
                break;
            }
            offset += stride;
        }
    }

}    // namespace detail

hpx::future<void> launchDriftParticlesCuda(DriftRequest request, cudaStream_t stream, int deviceId) {
    ParticleCount constexpr kMaximumBlocks = 65535;
    PROFILE();
    try {
        detail::validateDriftRequest(request);
        ParticleCount const count = request.range.end - request.range.begin;
        if (count == 0 || request.factor == 0.0) {
            return hpx::make_ready_future();
        }
        // Whole 256-thread blocks keep eight complete warps on contiguous
        // particles. Only the final range tail has inactive lanes.
        ParticleCount const neededBlocks = (count - 1) / detail::kDriftBlockSize + 1;
        unsigned const blocks = static_cast<unsigned>(neededBlocks < kMaximumBlocks ? neededBlocks : kMaximumBlocks);
        return cuda::launch(hpx::annotated_function(
                                [request, blocks](cudaStream_t launchStream) {
                                    PROFILE();
                                    detail::driftParticlesKernel<<<blocks, detail::kDriftBlockSize, 0, launchStream>>>(request);
                                },
                                "drift::launchDriftCuda"),
            stream, deviceId);
    } catch (...) {
        return hpx::make_exceptional_future<void>(std::current_exception());
    }
}

}    // namespace cosmictiger
