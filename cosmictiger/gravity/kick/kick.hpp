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
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/kick/kick_sums.hpp>
#include <cosmictiger/particle/particle_storage.hpp>
#include <cosmictiger/profiler.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define COSMICTIGER_KICK_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_KICK_HOST_DEVICE
#endif

namespace cosmictiger::gravity {

enum class KickPhase { initial, regular, final, diagnostics };

// tau = H0 integral(dt/a), q = x/L, stored momentum p = a dq/dtau.
// The completed comoving field g0 is dp/dtau; epsilon is Plummer-equivalent
// in unit-box comoving coordinates. A tick is dtau0 / 2^maximumRung.
struct KickParameters {
    double eta = 0.025;
    double epsilon = 0.0;
    double scaleFactor = 1.0;
    std::uint64_t timeTick = 0;
    KickPhase phase = KickPhase::initial;
    bool sumPotential = false;
    // Regular evaluations may include inactive sinks for global diagnostics.
    // Only rungs at or above this threshold receive the regular kick.
    std::uint8_t minimumKickRung = 0;
    bool sumKinetic = false;
    bool momentaStaggered = false;
    // Elapsed conformal time after timeTick, for a stop between block events.
    double conformalOffset = 0.0;
    int ghostWidth = kPmmmNeighborWidth;
    // Near-region solver. The tree walk opens nodes with openingAngle and
    // evaluates its multipole interactions at fmmOrder.
    NearFieldMethod nearFieldMethod = NearFieldMethod::direct;
    double openingAngle = kDefaultOpeningAngle;
    int fmmOrder = kDefaultFmmOrder;
    // Optional snapshot-only output, indexed like the particle arrays. Capture
    // synchronized canonical momentum before either integration half-kick.
    double* synchronizedMomenta[NDIM]{};
    ParticleCount synchronizedMomentumCount = 0;
};

COSMICTIGER_KICK_HOST_DEVICE inline std::uint64_t rungTicks(std::uint8_t rung) {
    PROFILE();
    return std::uint64_t(1) << (Rung::maximumRung() - expectRange(0, rung, Rung::maximumRung()));
}

COSMICTIGER_KICK_HOST_DEVICE inline bool rungSynchronized(std::uint64_t timeTick, std::uint8_t rung) {
    PROFILE();
    return (timeTick & (rungTicks(rung) - 1)) == 0;
}

COSMICTIGER_KICK_HOST_DEVICE inline double rungTimestep(double rungZeroTimestep, std::uint8_t rung) {
    PROFILE();
    return ldexp(expectPositive(rungZeroTimestep), -int(expectRange(0, rung, Rung::maximumRung())));
}

// Exact comparisons with dyadic step sizes avoid logarithmic rounding at rung
// boundaries. A request below the representable rung range is an error.
COSMICTIGER_KICK_HOST_DEVICE inline std::uint8_t allowedKickRung(Acceleration acceleration, double rungZeroTimestep, KickParameters const& parameters) {
    constexpr double kAccelerationFactor = 2.0;
    constexpr double kRungReduction = 0.5;
    PROFILE();

    FpeGuard guard;
    double const magnitude = hypot(hypot(double(acceleration.x), double(acceleration.y)), double(acceleration.z));
    double const numerator = kAccelerationFactor * expectPositive(parameters.eta) * expectPositive(parameters.scaleFactor) * expectPositive(parameters.epsilon);
    expectPositive(rungZeroTimestep);
    if (magnitude == 0.0) {
        return 0;
    }
    double const allowed = sqrt(numerator / magnitude);
    std::uint8_t rung = 0;
    double step = rungZeroTimestep;
    while (step > allowed && rung < Rung::maximumRung()) {
        step *= kRungReduction;
        rung++;
    }
    // This is a capacity failure, not a validation-only numerical check.
    if (step > allowed) {
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::range_error("kick timestep exceeds the available rung range");
#endif
    }
    return rung;
}

COSMICTIGER_KICK_HOST_DEVICE inline std::uint8_t selectKickRung(std::uint8_t current, std::uint8_t allowed, std::uint64_t timeTick) {
    PROFILE();
    expectRange(0, current, Rung::maximumRung());
    expectRange(0, allowed, Rung::maximumRung());
    if (allowed >= current) {
        return allowed;
    }
    while (current > allowed && rungSynchronized(timeTick, current - 1)) {
        current--;
    }
    return current;
}

namespace detail {

    // HPX has no atomic_ref replacement. Within one CPU or CUDA execution,
    // sinks share only these scalar totals; particle writes remain disjoint.
    // Concurrent host/device executions require separate accumulators. Combine
    // them only after all workers and CUDA completion futures have finished.
    COSMICTIGER_KICK_HOST_DEVICE inline void addKickScalar(double& destination, double value) {
        PROFILE();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        atomicAdd(&destination, value);
#elif defined(__CUDA_ARCH__)
        auto* bits = reinterpret_cast<unsigned long long*>(&destination);
        unsigned long long previous = *bits;
        unsigned long long expected;
        do {
            expected = previous;
            previous = atomicCAS(bits, expected, __double_as_longlong(__longlong_as_double(expected) + value));
        } while (previous != expected);
#else
        std::atomic_ref<double>(destination).fetch_add(value, std::memory_order_relaxed);
#endif
    }

    COSMICTIGER_KICK_HOST_DEVICE inline void countKickParticle(ParticleCount& destination) {
        PROFILE();
#if defined(__CUDA_ARCH__)
        atomicAdd(reinterpret_cast<unsigned long long*>(&destination), 1ULL);
#else
        std::atomic_ref<ParticleCount>(destination).fetch_add(1, std::memory_order_relaxed);
#endif
    }

    COSMICTIGER_KICK_HOST_DEVICE inline void maximizeKickRung(std::uint32_t& destination, std::uint8_t rung) {
        PROFILE();
#if defined(__CUDA_ARCH__)
        atomicMax(&destination, static_cast<std::uint32_t>(rung));
#else
        std::atomic_ref<std::uint32_t> maximum(destination);
        auto previous = maximum.load(std::memory_order_relaxed);
        while (previous < rung && !maximum.compare_exchange_weak(previous, rung, std::memory_order_relaxed)) {
        }
#endif
    }

}    // namespace detail

// Invoke once per owned sink with its COMPLETE field (all near sources and
// the far-field expansion combined). The same backend works on CPU and CUDA.
// It retains no per-particle fields unless the corresponding output pointers
// are supplied. The optional totals cover exactly the sinks evaluated here.
// Regular evaluation may include inactive sinks for complete diagnostics;
// those retain their momenta/rungs. Kinetic sums synchronize temporary values.
class DirectKickBackend {
public:
    COSMICTIGER_KICK_HOST_DEVICE explicit DirectKickBackend(KickParameters parameters, KickSums* sums = nullptr)
      : parameters_(parameters)
      , sums_(sums) {
        PROFILE();
    }

    static constexpr bool supportsConcurrentSinks() noexcept {
        PROFILE();
        return true;
    }

    template <typename Particles>
    COSMICTIGER_KICK_HOST_DEVICE void operator()(
        ParticleCount index, Acceleration acceleration, float potential, Rung oldFlags, double rungZeroTimestep, Particles particles) const {
        constexpr double kHalfStepFraction = 0.5;
        PROFILE();

        if (oldFlags.isGhost()) {
            return;
        }
        FpeGuard guard;
        bool const kickParticle = parameters_.phase != KickPhase::diagnostics &&
            (parameters_.phase != KickPhase::regular || oldFlags.rung() >= expectRange(0, parameters_.minimumKickRung, Rung::maximumRung()));
        expect(parameters_.timeTick <= rungTicks(0));
        expect(!kickParticle || rungSynchronized(parameters_.timeTick, oldFlags.rung()));
        expect(parameters_.phase != KickPhase::initial || parameters_.timeTick == 0);
        double const components[NDIM]{acceleration.x, acceleration.y, acceleration.z};
        double synchronizationInterval = 0.0;
        bool const captureMomentum = parameters_.synchronizedMomentumCount != 0;
        bool const synchronizeMomentum = (sums_ != nullptr && parameters_.sumKinetic) || captureMomentum;
        if (synchronizeMomentum && parameters_.momentaStaggered) {
            double const halfStep = kHalfStepFraction * rungTimestep(rungZeroTimestep, oldFlags.rung());
            if (kickParticle && (parameters_.phase == KickPhase::regular || parameters_.phase == KickPhase::final)) {
                synchronizationInterval = halfStep;
            } else {
                auto const elapsedTicks = parameters_.timeTick & (rungTicks(oldFlags.rung()) - 1);
                synchronizationInterval =
                    static_cast<double>(elapsedTicks) * ldexp(rungZeroTimestep, -int(Rung::maximumRung())) + parameters_.conformalOffset - halfStep;
            }
        }
        double momentumSquared = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (synchronizeMomentum) {
                double const synchronizedMomentum = double(expectNonZero(particles.velocities[axis])[index]) + synchronizationInterval * components[axis];
                momentumSquared += synchronizedMomentum * synchronizedMomentum;
                if (captureMomentum) {
                    expectNonZero(parameters_.synchronizedMomenta[axis])[expectIndex(index, parameters_.synchronizedMomentumCount)] = synchronizedMomentum;
                }
            }
        }
        // Complete the incoming half-step before selecting the outgoing rung.
        // Initial momenta are synchronized already, so there is no old half-step.
        if (kickParticle && (parameters_.phase == KickPhase::regular || parameters_.phase == KickPhase::final)) {
            double const oldHalfStep = kHalfStepFraction * rungTimestep(rungZeroTimestep, oldFlags.rung());
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto& momentum = expectNonZero(particles.velocities[axis])[index];
                momentum = static_cast<float>(double(momentum) + oldHalfStep * components[axis]);
            }
        }
        auto nextRung = oldFlags.rung();
        if (kickParticle && (parameters_.phase == KickPhase::regular || parameters_.phase == KickPhase::initial)) {
            nextRung = selectKickRung(oldFlags.rung(), allowedKickRung(acceleration, rungZeroTimestep, parameters_), parameters_.timeTick);
            expectNonZero(particles.flags)[index].setRung(nextRung);
            double const newHalfStep = kHalfStepFraction * rungTimestep(rungZeroTimestep, nextRung);
            for (size_t axis = 0; axis < NDIM; axis++) {
                auto& momentum = expectNonZero(particles.velocities[axis])[index];
                momentum = static_cast<float>(double(momentum) + newHalfStep * components[axis]);
            }
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (particles.accelerations[axis] != nullptr) {
                particles.accelerations[axis][index] = static_cast<float>(components[axis]);
            }
            if (sums_ != nullptr) {
                detail::addKickScalar(sums_->acceleration[axis], components[axis]);
            }
        }
        if (particles.potentials != nullptr) {
            particles.potentials[index] = potential;
        }
        if (sums_ != nullptr) {
            if (parameters_.sumKinetic) {
                detail::addKickScalar(sums_->momentumSquared, momentumSquared);
            }
            if (parameters_.sumPotential) {
                detail::addKickScalar(sums_->potential, potential);
            }
            detail::countKickParticle(sums_->particleCount);
            detail::maximizeKickRung(sums_->maximumRung, nextRung);
        }
    }

private:
    KickParameters parameters_;
    KickSums* sums_;
};

}    // namespace cosmictiger::gravity

#undef COSMICTIGER_KICK_HOST_DEVICE
