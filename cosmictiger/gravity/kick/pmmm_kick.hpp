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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/gravity/ferrers.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_progress.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/futures/future.hpp>

#include <cstdint>
#include <exception>
#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

namespace cosmictiger::gravity {

// Exclusive wall times: preparation joins both overlapping branches before
// the local kick begins. Global diagnostic reductions are excluded.
struct PmmmKickTimings {
    double gravity = 0.0;
    double kick = 0.0;
};

#ifdef COSMICTIGER_ENABLE_CUDA
namespace detail {

    inline constexpr size_t kPmmmCudaNearSide = 2 * kPmmmMaximumNeighborWidth + 1;
    inline constexpr size_t kPmmmCudaNearCellCount = [] {
        PROFILE();
        size_t count = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            count *= kPmmmCudaNearSide;
        }
        return count;
    }();

    struct PmmmCudaCandidate {
        ParticleCount sinkIndex;
        size_t cellIndex;
    };

    // The cell expansion has completed before the batched CUDA launch. Its
    // near ranges use exact owner indices for same-locality periodic copies.
    struct PmmmCudaCell {
        ParticleRange nearSources[kPmmmCudaNearCellCount];
        size_t nearSourceCount;
        double center[NDIM];
        double spacing;
        double expansion[symmetric::coefficientCount(kExpansionOrder)];
        bool fullExpansion;
    };

    void applyPmmmKickCuda(PmmmCudaCandidate const* candidates, size_t candidateCount, PmmmCudaCell const* cells, size_t cellCount,
        DirectInteractionParticles particles, SofteningLength softening, int ferrersOrder, bool computePotential, float particleMass, double rungZeroTimestep,
        std::uint8_t activeRung, KickParameters parameters, KickSums* sums, int deviceId);

}    // namespace detail
#endif

// Evaluate each owned active particle once, combining its complete cell
// expansion with all direct near cells before updating momentum/rung or sums.
// Multipoles, periodic expansions, and remote ghosts must already be current.
// Distributed grids require ghostWidth() >= parameters.ghostWidth. Local sources
// use their owned ranges, including periodic neighbors, to identify self exactly.
// The cell expansion already includes source masses and the periodic background;
// particleMass is applied only to the direct near interactions.
// Potential is evaluated when its persistent field or sum is requested.
// Diagnostics evaluates every requested sink without changing momentum/rungs.
// Other phases rebuild rung ranges after every field evaluation has completed.
void kickPmmmParticlesCpu(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, KickSums* sums = nullptr);

// Collective variant over all physical HPX localities. Returns global sums;
// every locality participates even after a local kick failure and then throws
// a shared error. Completed particle updates are not rolled back on failure.
KickSums kickPmmmParticles(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, std::uint64_t operationId);

// Temporary read-only near preparation, independent of cell moments/expansions.
// The CPU constructor computes raw SIMD lanes so kick() adds them to the
// completed far polynomial in the same order as the fused kick above. The CUDA
// constructor stages every masked near range per occupied cell; kickCuda() reads the
// completed expansions and evaluates both near P2P and L2P on the device.
// Scratch covers owned candidates using sorted active suffixes when available,
// never the appended ghost suffix.
// Keep grid/particles alive and their layout, ranges, positions, flags, storage,
// and output-field configuration fixed until kick() or destruction. Concurrent
// far work may read moments/trace and write expansions only. No kick or sums
// occur during construction; failed construction drains its local tasks.
class PmmmNearField {
public:
    PmmmNearField(CartesianGrid const& grid, ParticleStorage const& particles, SofteningLength softening, int ferrersOrder, float particleMass,
        double rungZeroTimestep, std::uint8_t activeRung, bool sumPotential = false, bool useCuda = false, PmmmProgressObserver const& observer = {},
        int ghostWidth = kPmmmNeighborWidth);
    PmmmNearField(PmmmNearField&&) noexcept;
    PmmmNearField& operator=(PmmmNearField&&) noexcept;
    PmmmNearField(PmmmNearField const&) = delete;
    PmmmNearField& operator=(PmmmNearField const&) = delete;

    // Call once after both CPU branches finish, on the original grid/storage.
    // Combine the complete field before the backend updates momentum/rungs,
    // optional output fields, or sums. Non-diagnostic phases sort cells once.
    void kick(
        CartesianGrid& grid, ParticleStorage& particles, KickParameters const& parameters, KickSums* sums = nullptr, PmmmProgressObserver const& observer = {});

#ifdef COSMICTIGER_ENABLE_CUDA
    // Stage completed cell expansions and near-cell ranges, then evaluate
    // both L2P and near P2P and update each sink in one batched CUDA launch.
    // Wait for GPU completion before sorting particles or returning sums.
    void kickCuda(CartesianGrid& grid, ParticleStorage& particles, KickParameters const& parameters, KickSums* sums = nullptr, int deviceId = 0,
        PmmmProgressObserver const& observer = {});
#endif

private:
    CartesianGrid const* grid_;
    ParticleStorage const* particles_;
    SofteningLength softening_;
    int ferrersOrder_;
    float particleMass_;
    double rungZeroTimestep_;
    std::uint8_t activeRung_;
    bool computePotential_;
    bool cudaEnabled_;
    int ghostWidth_;
    std::vector<ParticleRange> ranges_;
    std::vector<size_t> offsets_;
    std::vector<detail::DirectInteractionLanes> lanes_;
#ifdef COSMICTIGER_ENABLE_CUDA
    std::vector<detail::PmmmCudaCandidate> cudaCandidates_;
    std::vector<detail::PmmmCudaCell> cudaCells_;
#endif
};

// Call from an HPX task with exclusive particle/range access and completed GPU
// work. Every rank must enter computeFarField's matching FFT sequence even when
// near preparation or its task launch fails. The callback must drain its own
// work before returning/throwing. Always join the near task before committing
// the combined kick or propagating either branch's error. No FpeGuard may span
// this call. Distributed FFT failures still require runtime-level recovery.
template <typename FarField>
void kickPmmmParticlesOverlappedCpu(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, FarField&& computeFarField, KickSums* sums = nullptr,
    PmmmProgressObserver const& observer = {}, PmmmKickTimings* timings = nullptr) {
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();

    auto const gravityStart = timings ? hpx::chrono::high_resolution_clock::now() : 0;
    hpx::future<PmmmNearField> near;
    std::exception_ptr failure;
    try {
        near = hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&grid, &particles, &observer, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, sumPotential = parameters.sumPotential,
                    ghostWidth = parameters.ghostWidth] {
                    PROFILE();
                    return PmmmNearField(
                        grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, sumPotential, false, observer, ghostWidth);
                },
                "pmmmKick.async"));
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        computeFarField();
    } catch (...) {
        if (!failure) {
            failure = std::current_exception();
        }
    }
    if (near.valid()) {
        try {
            auto field = near.get();
            if (!failure) {
                auto const kickStart = timings ? hpx::chrono::high_resolution_clock::now() : 0;
                field.kick(grid, particles, parameters, sums, observer);
                if (timings) {
                    timings->gravity = double(kickStart - gravityStart) * kSecondsPerNanosecond;
                    timings->kick = double(hpx::chrono::high_resolution_clock::now() - kickStart) * kSecondsPerNanosecond;
                }
            }
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

// Collective variant over all physical HPX localities. Both branches are
// drained before every rank joins the existing force/potential/error reduction.
// Local branch failures skip the local kick; another rank's completed kick is
// not rolled back if the collective reports that failure.
template <typename FarField>
KickSums kickPmmmParticlesOverlapped(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, FarField&& computeFarField, std::uint64_t operationId,
    PmmmProgressObserver const& observer = {}, PmmmKickTimings* timings = nullptr) {
    PROFILE();
    KickSums local;
    std::string error;
    try {
        kickPmmmParticlesOverlappedCpu(
            grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, parameters, computeFarField, &local, observer, timings);
    } catch (std::exception const& failure) {
        error = failure.what();
    } catch (...) {
        error = "unknown overlapped PMMM gravity failure";
    }
    try {
        detail::reportPmmmProgress(observer, PmmmProgressStage::diagnosticReduction, 0, 1);
    } catch (std::exception const& failure) {
        if (error.empty()) {
            error = failure.what();
        }
    } catch (...) {
        if (error.empty()) {
            error = "unknown PMMM reduction progress failure";
        }
    }
    auto const result = allReduceKickSums(local, parameters.sumPotential, operationId, std::move(error));
    detail::reportPmmmProgress(observer, PmmmProgressStage::diagnosticReduction, 1, 1);
    return result;
}

#ifdef COSMICTIGER_ENABLE_CUDA
// P2M and CPU periodic M2L complete before one batched CUDA launch evaluates
// L2P and all near P2P, then updates momentum, rungs, optional output fields,
// and local diagnostic sums. Every locality still enters the matching
// far-field and diagnostic collective sequence if local CUDA work fails.
template <typename FarField>
KickSums kickPmmmParticlesOverlappedCuda(CartesianGrid& grid, ParticleStorage& particles, SofteningLength softening, int ferrersOrder, float particleMass,
    double rungZeroTimestep, std::uint8_t activeRung, KickParameters const& parameters, FarField&& computeFarField, std::uint64_t operationId, int deviceId = 0,
    PmmmProgressObserver const& observer = {}, PmmmKickTimings* timings = nullptr) {
    constexpr double kSecondsPerNanosecond = 1.0e-9;
    PROFILE();

    auto const gravityStart = timings ? hpx::chrono::high_resolution_clock::now() : 0;
    KickSums local;
    std::string error;
    hpx::future<PmmmNearField> near;
    std::exception_ptr failure;
    try {
        near = hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&grid, &particles, &observer, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, sumPotential = parameters.sumPotential,
                    ghostWidth = parameters.ghostWidth] {
                    PROFILE();
                    return PmmmNearField(
                        grid, particles, softening, ferrersOrder, particleMass, rungZeroTimestep, activeRung, sumPotential, true, observer, ghostWidth);
                },
                "pmmmKick.async"));
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        computeFarField();
    } catch (...) {
        if (!failure) {
            failure = std::current_exception();
        }
    }
    if (near.valid()) {
        try {
            auto field = near.get();
            if (!failure) {
                auto const kickStart = timings ? hpx::chrono::high_resolution_clock::now() : 0;
                field.kickCuda(grid, particles, parameters, &local, deviceId, observer);
                if (timings) {
                    timings->gravity = double(kickStart - gravityStart) * kSecondsPerNanosecond;
                    timings->kick = double(hpx::chrono::high_resolution_clock::now() - kickStart) * kSecondsPerNanosecond;
                }
            }
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        try {
            std::rethrow_exception(failure);
        } catch (std::exception const& caught) {
            error = caught.what();
        } catch (...) {
            error = "unknown overlapped PMMM CUDA gravity failure";
        }
    }
    try {
        detail::reportPmmmProgress(observer, PmmmProgressStage::diagnosticReduction, 0, 1);
    } catch (std::exception const& caught) {
        if (error.empty()) {
            error = caught.what();
        }
    } catch (...) {
        if (error.empty()) {
            error = "unknown PMMM CUDA reduction progress failure";
        }
    }
    auto const result = allReduceKickSums(local, parameters.sumPotential, operationId, std::move(error));
    detail::reportPmmmProgress(observer, PmmmProgressStage::diagnosticReduction, 1, 1);
    return result;
}
#endif

}    // namespace cosmictiger::gravity
