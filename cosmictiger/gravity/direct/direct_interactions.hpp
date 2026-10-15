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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/gravity/plummer.hpp>
#include <cosmictiger/parallel.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <cstdint>
#include <stddef.h>
#include <stdexcept>

#if defined(__CUDACC__)
#define COSMICTIGER_DIRECT_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_DIRECT_HOST_DEVICE
#endif

namespace cosmictiger::gravity {

struct DirectInteractionParticles {
    Coordinate const* positions[NDIM];
    Rung* flags;
    float* velocities[NDIM];
    float* accelerations[NDIM]{};
    float* potentials = nullptr;
};

// Both ranges index the same local particle arrays. The sink range must refer
// to owned particles: same-locality periodic copies need not set isGhost.
// After sorting a cell by rung, and before changing its rungs, callers may
// select the active suffix with {cell.rungParticles[rung].begin,
// cell.localParticles.end}. Flags are still checked within the supplied range.
struct DirectInteractionRequest {
    ParticleRange source;
    ParticleRange sink;
    DirectInteractionParticles particles;
    SofteningLength softening;
    float particleMass;
    double rungZeroTimestep;
    std::uint8_t rung;
    KickParameters kick{};
    KickSums* sums = nullptr;
};

// The source range must include the complete field when using this backend.
// For PMMM plus multiple near cells use kickPmmmParticlesCpu instead.
template <int order, bool computePotential>
DirectKickBackend makeDirectKickBackend(DirectInteractionRequest const& request) {
    PROFILE();
    if (!computePotential && (request.kick.sumPotential || request.particles.potentials != nullptr)) {
        throw std::invalid_argument("kick potential output/sums require potential evaluation");
    }
    auto parameters = request.kick;
    parameters.epsilon = softeningEpsilon<order>(request.softening);
    return DirectKickBackend(parameters, request.sums);
}

// Uses the published constants and configures optional particle output fields.
DirectInteractionRequest makeDirectKickRequest(ParticleStorage& particles, ParticleRange source, ParticleRange sink, SofteningLength softening,
    std::uint8_t activeRung, double scaleFactor, std::uint64_t timeTick, KickPhase phase, KickSums* sums = nullptr, bool sumPotential = false);

// Collective convenience entry point: computes the requested potential when
// needed, kicks locally, then returns global sums of the evaluated owned sinks.
// The source range must supply the full field. All localities participate,
// including those with empty sink ranges, using a fresh operationId.
KickSums kickDirectInteractionsCpu(DirectInteractionRequest request, int ferrersOrder, std::uint64_t operationId);

// Store fields without applying a timestep or velocity kick. Each output array
// must contain at least sinks.end-sinks.begin entries and remain live until the
// interaction call completes. Output entry zero corresponds to sinks.begin;
// inactive and ghost sinks remain untouched. Potentials may be omitted.
// Output arrays must be disjoint and must not alias particle input arrays.
class DirectFieldBackend {
public:
    COSMICTIGER_DIRECT_HOST_DEVICE DirectFieldBackend(ParticleRange sinks, Acceleration* accelerations, float* potentials = nullptr)
      : sinks_(sinks)
      , accelerations_(accelerations)
      , potentials_(potentials) {
        PROFILE();
        expect(sinks_.begin <= sinks_.end);
    }

    static constexpr bool supportsConcurrentSinks() noexcept {
        PROFILE();
        return true;
    }

    COSMICTIGER_DIRECT_HOST_DEVICE void operator()(
        ParticleCount sinkIndex, Acceleration acceleration, float potential, Rung, float, DirectInteractionParticles) const {
        PROFILE();
        ParticleCount const offset = expectIndex(sinkIndex - sinks_.begin, sinks_.end - sinks_.begin);
        expectNonZero(accelerations_)[offset] = acceleration;
        if (potentials_ != nullptr) {
            potentials_[offset] = potential;
        }
    }

private:
    ParticleRange sinks_;
    Acceleration* accelerations_;
    float* potentials_;
};

namespace detail {

    template <typename Backend>
    constexpr bool directBackendSupportsConcurrentSinks() noexcept {
        PROFILE();
        if constexpr (requires { Backend::supportsConcurrentSinks(); }) {
            return Backend::supportsConcurrentSinks();
        } else {
            return false;
        }
    }

#if defined(__AVX512F__)
    inline constexpr size_t kDirectFloatLanes = 16;
#elif defined(__AVX2__)
    inline constexpr size_t kDirectFloatLanes = 8;
#elif defined(__SSE__) || defined(__ARM_NEON)
    inline constexpr size_t kDirectFloatLanes = 4;
#else
    inline constexpr size_t kDirectFloatLanes = 1;
#endif

    struct DirectInteractionLanes {
        float x[kDirectFloatLanes]{};
        float y[kDirectFloatLanes]{};
        float z[kDirectFloatLanes]{};
        float potential[kDirectFloatLanes]{};
    };

    template <int order, bool computePotential>
    inline void accumulateDirectParticle(
        Position sinkPosition, ParticleCount sourceIndex, size_t lane, DirectInteractionRequest const& request, DirectInteractionLanes& lanes) {
        PROFILE();
        auto const& particles = request.particles;
        FpeGuard fpeGuard;
        Position const sourcePosition{expectNonZero(particles.positions[0])[sourceIndex], expectNonZero(particles.positions[1])[sourceIndex],
            expectNonZero(particles.positions[2])[sourceIndex]};
        if constexpr (computePotential) {
            auto const interaction = softenedAccelerationAndPotential<order>(sinkPosition, sourcePosition, request.softening);
            lanes.x[lane] += interaction.acceleration.x;
            lanes.y[lane] += interaction.acceleration.y;
            lanes.z[lane] += interaction.acceleration.z;
            lanes.potential[lane] += interaction.potential;
        } else {
            auto const acceleration = softenedAcceleration<order>(sinkPosition, sourcePosition, request.softening);
            lanes.x[lane] += acceleration.x;
            lanes.y[lane] += acceleration.y;
            lanes.z[lane] += acceleration.z;
        }
    }

    template <int order, bool computePotential>
    inline void accumulateDirectRange(
        Position sinkPosition, ParticleCount begin, ParticleCount end, DirectInteractionRequest const& request, DirectInteractionLanes& lanes) {
        PROFILE();
        ParticleCount sourceIndex = begin;
        while (end - sourceIndex >= kDirectFloatLanes) {
            for (size_t lane = 0; lane < kDirectFloatLanes; lane++) {
                accumulateDirectParticle<order, computePotential>(sinkPosition, sourceIndex + lane, lane, request, lanes);
            }
            sourceIndex += kDirectFloatLanes;
        }
        for (size_t lane = 0; sourceIndex < end; lane++, sourceIndex++) {
            accumulateDirectParticle<order, computePotential>(sinkPosition, sourceIndex, lane, request, lanes);
        }
    }

}    // namespace detail

// Every qualifying owned sink receives all sources, including sources on slower
// rungs. Splitting the source range excludes self without a branch in its loop.
// A backend may opt in with static constexpr bool supportsConcurrentSinks()
// returning true.
// That promises concurrent calls on the same backend instance are safe: writes
// for distinct sinks must be disjoint, and callbacks must not read another
// sink's changing output or modify source positions or the request. A sink may
// update its own flags after evaluating its force.
// Aliased particle velocity/output arrays must not defeat that disjointness.
// Other backends retain increasing sink order. Opted-in backends use bounded
// chunks when multiple HPX workers are available. The unbenchmarked grain
// heuristic targets 4096 source/sink pairs, with a minimum grain between 8 and
// 64 sinks. Source and reduction order within a sink is unchanged. Callbacks
// may yield because arithmetic guards have ended. All launched chunks finish
// before returning or rethrowing an error; completed writes are not rolled back.
template <typename Backend, int order, bool computePotential>
void directInteractionsCpu(DirectInteractionRequest const& request, Backend backend) {
    [[maybe_unused]] constexpr ParticleCount kTargetPairs = 4096;
    [[maybe_unused]] constexpr ParticleCount kMinimumGrain = 8;
    [[maybe_unused]] constexpr ParticleCount kMaximumGrain = 64;

    static_assert(order >= 0 && order <= 3, "Ferrers order must be F0, F1, F2, or F3");
    PROFILE();
    expect(!(request.source.begin > request.source.end || request.sink.begin > request.sink.end || request.rung > Rung::maximumRung()));
    auto const& particles = request.particles;
    auto processSinks = [&](ParticleCount begin, ParticleCount end) {
        PROFILE();
        for (ParticleCount sinkIndex = begin; sinkIndex < end; sinkIndex++) {
            Rung const sinkFlags = expectNonZero(particles.flags)[sinkIndex];
            if (sinkFlags.isGhost() || sinkFlags.rung() < request.rung) {
                continue;
            }
            Position const sinkPosition{expectNonZero(particles.positions[0])[sinkIndex], expectNonZero(particles.positions[1])[sinkIndex],
                expectNonZero(particles.positions[2])[sinkIndex]};
            detail::DirectInteractionLanes lanes;
            if (sinkIndex >= request.source.begin && sinkIndex < request.source.end) {
                detail::accumulateDirectRange<order, computePotential>(sinkPosition, request.source.begin, sinkIndex, request, lanes);
                detail::accumulateDirectRange<order, computePotential>(sinkPosition, sinkIndex + 1, request.source.end, request, lanes);
            } else {
                detail::accumulateDirectRange<order, computePotential>(sinkPosition, request.source.begin, request.source.end, request, lanes);
            }
            Acceleration acceleration{};
            float potential = 0.0f;
            {
                FpeGuard fpeGuard;
                for (size_t lane = 0; lane < detail::kDirectFloatLanes; lane++) {
                    acceleration.x += lanes.x[lane];
                    acceleration.y += lanes.y[lane];
                    acceleration.z += lanes.z[lane];
                    if constexpr (computePotential) {
                        potential += lanes.potential[lane];
                    }
                }
                acceleration.x *= request.particleMass;
                acceleration.y *= request.particleMass;
                acceleration.z *= request.particleMass;
                if constexpr (computePotential) {
                    potential *= request.particleMass;
                }
            }
            DirectInteractionParticles backendParticles = particles;
            for (size_t axis = 0; axis < NDIM; axis++) {
                backendParticles.velocities[axis] = expectNonZero(particles.velocities[axis]);
            }
            backend(sinkIndex, acceleration, potential, sinkFlags, request.rungZeroTimestep, backendParticles);
        }
    };
    if constexpr (detail::directBackendSupportsConcurrentSinks<Backend>()) {
        ParticleCount const sourceCount = request.source.end - request.source.begin;
        ParticleCount const estimatedGrain = kTargetPairs / (sourceCount == 0 ? 1 : sourceCount);
        size_t const minimumSinks =
            static_cast<size_t>(estimatedGrain < kMinimumGrain ? kMinimumGrain : (estimatedGrain > kMaximumGrain ? kMaximumGrain : estimatedGrain));
        ::cosmictiger::detail::ParallelChunkPlan const plan(static_cast<size_t>(request.sink.end - request.sink.begin), minimumSinks);
        ::cosmictiger::detail::parallelForChunks(plan,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    processSinks(request.sink.begin + begin, request.sink.begin + end);
                },
                "directInteractions.parallelForChunks"));
    } else {
        processSinks(request.sink.begin, request.sink.end);
    }
}

template <typename Backend = DirectKickBackend, int order, bool computePotential>
void directInteractionsCpu(DirectInteractionRequest const& request) {
    static_assert(std::is_same_v<Backend, DirectKickBackend>);
    PROFILE();
    directInteractionsCpu<Backend, order, computePotential>(request, makeDirectKickBackend<order, computePotential>(request));
}

#ifdef COSMICTIGER_ENABLE_CUDA
// The future becomes ready after the GPU has completed the requested work.
// Invalid requests and CUDA failures are reported through the future.
// All pointers, including request.sums, must be CUDA-accessible and live until
// completion. Concurrent CPU/GPU launches must use separate KickSums objects;
// combine their values on the host after every completion future is ready.
hpx::future<void> launchDirectInteractionsCuda(
    DirectInteractionRequest request, int ferrersOrder, bool computePotential, cudaStream_t stream = nullptr, int deviceId = 0);
#endif

}    // namespace cosmictiger::gravity

#undef COSMICTIGER_DIRECT_HOST_DEVICE
