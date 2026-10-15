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

#include <cosmictiger/parallel.hpp>
#include <cosmictiger/particle/particle_ghost_exchange.hpp>

#include <cmath>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cosmictiger {
namespace detail {

    void driftParticleCpu(DriftRequest const& request, ParticleCount index) {
        PROFILE();
        if (!request.ownedOnly && request.particles.flags[index].isGhost()) {
            return;
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const displacement = double(request.particles.momenta[axis][index]) * request.factor;
            // Reduce before conversion so multi-box and negative displacements
            // never overflow the integer conversion. Quantize only the shift;
            // adding floating coordinates would discard their existing low bits.
            double const fractional = displacement - trunc(displacement);
            double const scaled = ldexp(fractional, Coordinate::fractionalBits());
            [[maybe_unused]] double const period = ldexp(1.0, Coordinate::fractionalBits());
            expectRange(-period, scaled, period);
            auto const shift = static_cast<std::uint32_t>(static_cast<std::int64_t>(scaled));
            auto& position = request.particles.positions[axis][index];
            position = Coordinate::fromRaw(static_cast<std::uint32_t>(std::uint64_t(position.raw()) + shift));
        }
    }

#if defined(__AVX2__)
    inline constexpr size_t kDriftSimdWidth = 4;

    void driftPackCpu(DriftRequest const& request, ParticleCount first) {
        constexpr double kPeriod = 0x1p32;
        constexpr double kHalfPeriod = 0x1p31;
        constexpr int kTruncate = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
        PROFILE();
        static_assert(Coordinate::fractionalBits() == 32);
        static_assert(sizeof(Coordinate) == sizeof(std::uint32_t));
        if (!request.ownedOnly) {
            for (size_t lane = 0; lane < kDriftSimdWidth; lane++) {
                if (request.particles.flags[first + lane].isGhost()) {
                    for (size_t offset = 0; offset < kDriftSimdWidth; offset++) {
                        driftParticleCpu(request, first + offset);
                    }
                    return;
                }
            }
        }
        auto const factor = _mm256_set1_pd(request.factor);
        auto const period = _mm256_set1_pd(kPeriod);
        auto const half = _mm256_set1_pd(kHalfPeriod);
        auto const negativeHalf = _mm256_set1_pd(-kHalfPeriod);
        for (size_t axis = 0; axis < NDIM; axis++) {
            auto const momentum = _mm256_cvtps_pd(_mm_loadu_ps(request.particles.momenta[axis] + first));
            auto const displacement = _mm256_mul_pd(momentum, factor);
            auto const fractional = _mm256_sub_pd(displacement, _mm256_round_pd(displacement, kTruncate));
            auto scaled = _mm256_round_pd(_mm256_mul_pd(fractional, period), kTruncate);
            // Map the already truncated shift into signed 32-bit range before
            // packed conversion. Integer addition then wraps modulo 2^32.
            auto const positiveWrap = _mm256_and_pd(_mm256_cmp_pd(scaled, half, _CMP_GE_OQ), period);
            auto const negativeWrap = _mm256_and_pd(_mm256_cmp_pd(scaled, negativeHalf, _CMP_LT_OQ), period);
            scaled = _mm256_add_pd(_mm256_sub_pd(scaled, positiveWrap), negativeWrap);
            auto const shift = _mm256_cvttpd_epi32(scaled);
            auto* position = reinterpret_cast<__m128i*>(request.particles.positions[axis] + first);
            _mm_storeu_si128(position, _mm_add_epi32(_mm_loadu_si128(position), shift));
        }
    }
#endif

}    // namespace detail

DriftRequest makeDriftRequest(ParticleStorage& particles, ParticleRange range, double factor) {
    PROFILE();
    expect(range.begin <= range.end);
    expectRange(0, range.end, particles.size());
    DriftRequest request{range, {}, factor};
    request.particles.flags = particles.rungData();
    for (size_t axis = 0; axis < NDIM; axis++) {
        request.particles.positions[axis] = particles.positionData(axis);
        request.particles.momenta[axis] = particles.velocityData(axis);
    }
    return request;
}

void driftParticlesCpu(DriftRequest const& request) {
    PROFILE();
    detail::validateDriftRequest(request);
    if (request.range.begin == request.range.end || request.factor == 0.0) {
        return;
    }
    auto const count = request.range.end - request.range.begin;
    if (count > size_t(-1)) {
        throw std::length_error("drift range exceeds addressable memory");
    }
    detail::ParallelChunkPlan const plan(static_cast<size_t>(count));
    detail::parallelForChunks(plan,
        hpx::annotated_function(
            [&request](size_t, size_t first, size_t last) {
                PROFILE();
                FpeGuard guard;
                size_t offset = first;
#if defined(__AVX2__)
                for (; last - offset >= detail::kDriftSimdWidth; offset += detail::kDriftSimdWidth) {
                    detail::driftPackCpu(request, request.range.begin + offset);
                }
#endif
                for (; offset < last; offset++) {
                    detail::driftParticleCpu(request, request.range.begin + offset);
                }
            },
            "drift.parallelForChunks"));
}

void driftParticles(ParticleStorage& particles, CartesianGrid& grid, double factor, DriftExecution execution, int deviceId) {
    PROFILE();
    switch (execution) {
    case DriftExecution::cpu:
        break;
    case DriftExecution::cuda:
#ifndef COSMICTIGER_ENABLE_CUDA
        throw std::runtime_error("CUDA drift requires a CUDA-enabled build");
#endif
        break;
    default:
        throw std::invalid_argument("unknown drift execution processor");
    }
    if (factor == 0.0) {
        return;
    }
    clearGhostParticles(particles, grid);
    auto request = makeDriftRequest(particles, {0, particles.size()}, factor);
    request.ownedOnly = true;
    if (execution == DriftExecution::cpu) {
        driftParticlesCpu(request);
    } else {
#ifdef COSMICTIGER_ENABLE_CUDA
        launchDriftParticlesCuda(request, nullptr, deviceId).get();
#else
        (void) deviceId;
#endif
    }
}

}    // namespace cosmictiger
