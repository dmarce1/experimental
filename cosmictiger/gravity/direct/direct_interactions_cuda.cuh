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

#pragma once

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/gravity/plummer.hpp>
#include <cosmictiger/profiler.hpp>

#include <cstdint>

namespace cosmictiger::gravity::detail {

inline constexpr unsigned kDirectInteractionTileSize = 128;

// A block owns a tile of sinks. Its threads cooperatively load each source
// tile once, then each thread accumulates the field for its own sink.
// Inactive and out-of-range threads still participate in both barriers.
template <typename Backend, int order, bool computePotential>
__global__ void directInteractionsKernel(DirectInteractionRequest request, Backend backend) {
    PROFILE();
    __shared__ std::uint32_t sourceRaw[NDIM][kDirectInteractionTileSize];

    ParticleCount const sinkCount = request.sink.end - request.sink.begin;
    ParticleCount const sourceCount = request.source.end - request.source.begin;
    ParticleCount const sinkStride = static_cast<ParticleCount>(gridDim.x) * blockDim.x;
    ParticleCount sinkBlockOffset = static_cast<ParticleCount>(blockIdx.x) * blockDim.x;

    while (sinkBlockOffset < sinkCount) {
        bool const inRange = threadIdx.x < sinkCount - sinkBlockOffset;
        ParticleCount const sinkIndex = inRange ? request.sink.begin + sinkBlockOffset + threadIdx.x : 0;
        Rung sinkFlags{};
        if (inRange) {
            sinkFlags = request.particles.flags[sinkIndex];
        }
        bool const active = inRange && !sinkFlags.isGhost() && sinkFlags.rung() >= request.rung;
        Position sinkPosition{};
        if (active) {
            sinkPosition = {request.particles.positions[0][sinkIndex], request.particles.positions[1][sinkIndex], request.particles.positions[2][sinkIndex]};
        }

        float accelerationX = 0.0f;
        float accelerationY = 0.0f;
        float accelerationZ = 0.0f;
        float potential = 0.0f;
        ParticleCount sourceOffset = 0;
        while (sourceOffset < sourceCount) {
            ParticleCount const remainingSources = sourceCount - sourceOffset;
            unsigned const tileCount = remainingSources < kDirectInteractionTileSize ? static_cast<unsigned>(remainingSources) : kDirectInteractionTileSize;
            if (threadIdx.x < tileCount) {
                ParticleCount const sourceIndex = request.source.begin + sourceOffset + threadIdx.x;
                for (unsigned axis = 0; axis < NDIM; axis++) {
                    sourceRaw[axis][threadIdx.x] = request.particles.positions[axis][sourceIndex].raw();
                }
            }
            __syncthreads();

            if (active) {
                for (unsigned lane = 0; lane < tileCount; lane++) {
                    ParticleCount const sourceIndex = request.source.begin + sourceOffset + lane;
                    if (sourceIndex == sinkIndex) {
                        continue;
                    }
                    Position const sourcePosition{
                        Coordinate::fromRaw(sourceRaw[0][lane]), Coordinate::fromRaw(sourceRaw[1][lane]), Coordinate::fromRaw(sourceRaw[2][lane])};
                    if constexpr (computePotential) {
                        auto const field = softenedAccelerationAndPotential<order>(sinkPosition, sourcePosition, request.softening);
                        accelerationX += field.acceleration.x;
                        accelerationY += field.acceleration.y;
                        accelerationZ += field.acceleration.z;
                        potential += field.potential;
                    } else {
                        auto const acceleration = softenedAcceleration<order>(sinkPosition, sourcePosition, request.softening);
                        accelerationX += acceleration.x;
                        accelerationY += acceleration.y;
                        accelerationZ += acceleration.z;
                    }
                }
            }
            __syncthreads();
            sourceOffset += tileCount;
        }

        if (active) {
            Acceleration const acceleration{request.particleMass * accelerationX, request.particleMass * accelerationY, request.particleMass * accelerationZ};
            backend(sinkIndex, acceleration, request.particleMass * potential, sinkFlags, request.rungZeroTimestep, request.particles);
        }

        if (sinkCount - sinkBlockOffset <= sinkStride) {
            break;
        }
        sinkBlockOffset += sinkStride;
    }
}

}    // namespace cosmictiger::gravity::detail
