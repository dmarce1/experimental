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

#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/gravity/kick/kick.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/type_support/bit_cast.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>

namespace {

using cosmictiger::Acceleration;
using cosmictiger::Coordinate;
using cosmictiger::ParticleCount;
using cosmictiger::Rung;
using cosmictiger::gravity::allowedKickRung;
using cosmictiger::gravity::DirectInteractionParticles;
using cosmictiger::gravity::DirectInteractionRequest;
using cosmictiger::gravity::DirectKickBackend;
using cosmictiger::gravity::KickParameters;
using cosmictiger::gravity::KickPhase;
using cosmictiger::gravity::KickSums;
using cosmictiger::gravity::rungSynchronized;
using cosmictiger::gravity::rungTicks;
using cosmictiger::gravity::rungTimestep;
using cosmictiger::gravity::selectKickRung;

template <size_t count>
class KickParticles {
public:
    DirectInteractionParticles view(bool accelerationOutput = false, bool potentialOutput = false) {
        PROFILE();
        DirectInteractionParticles result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result.positions[axis] = positions_[axis].data();
            result.velocities[axis] = momenta_[axis].data();
            result.accelerations[axis] = accelerationOutput ? accelerations_[axis].data() : nullptr;
        }
        result.flags = flags_.data();
        result.potentials = potentialOutput ? potentials_.data() : nullptr;
        return result;
    }

    Rung& flags(size_t index = 0) {
        PROFILE();
        return flags_[index];
    }

    float& momentum(size_t axis = 0, size_t index = 0) {
        PROFILE();
        return momenta_[axis][index];
    }

    float& acceleration(size_t axis = 0, size_t index = 0) {
        PROFILE();
        return accelerations_[axis][index];
    }

    float& potential(size_t index = 0) {
        PROFILE();
        return potentials_[index];
    }

    void setPosition(size_t index, std::array<double, NDIM> const& coordinates) {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions_[axis][index] = Coordinate(coordinates[axis]);
        }
    }

private:
    std::array<Coordinate, count> positions_[NDIM]{};
    std::array<Rung, count> flags_{};
    std::array<float, count> momenta_[NDIM]{};
    std::array<float, count> accelerations_[NDIM]{};
    std::array<float, count> potentials_{};
};

KickParameters unitParameters() {
    PROFILE();
    KickParameters parameters;
    parameters.eta = 0.5;
    parameters.epsilon = 1.0;
    return parameters;
}

struct OscillatorResult {
    double error;
    double maximumEnergyError;
};

OscillatorResult integrateOscillator(std::uint8_t rung) {
    PROFILE();
    KickParticles<1> particles;
    auto parameters = unitParameters();
    double const step = rungTimestep(1.0, rung);
    parameters.epsilon = step * step;
    double position = 1.0;
    DirectKickBackend{parameters}(0, {-1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_EQ(particles.flags().rung(), rung);
    size_t const steps = size_t(1) << rung;
    double maximumEnergyError = 0.0;
    for (size_t index = 1; index <= steps; index++) {
        position += step * particles.momentum();
        float const acceleration = static_cast<float>(-position);
        parameters.timeTick = static_cast<std::uint64_t>(index) * rungTicks(rung);
        parameters.phase = index == steps ? KickPhase::final : KickPhase::regular;
        DirectKickBackend{parameters}(0, {acceleration, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
        EXPECT_EQ(particles.flags().rung(), rung);
        double const synchronizedMomentum = particles.momentum() - (index == steps ? 0.0 : 0.5 * step * acceleration);
        double const energyError = std::abs(0.5 * (position * position + synchronizedMomentum * synchronizedMomentum) - 0.5);
        if (energyError > maximumEnergyError) {
            maximumEnergyError = energyError;
        }
    }
    return {std::hypot(position - std::cos(1.0), particles.momentum() + std::sin(1.0)), maximumEnergyError};
}

}    // namespace

TEST(Kick, ConformalAccelerationCriterion) {
    PROFILE();
    auto parameters = unitParameters();
    EXPECT_EQ(allowedKickRung({1.0f, 0.0f, 0.0f}, 8.0, parameters), 3);
    parameters.scaleFactor = 0.25;
    EXPECT_EQ(allowedKickRung({1.0f, 0.0f, 0.0f}, 8.0, parameters), 4);
    parameters.scaleFactor = 4.0;
    EXPECT_EQ(allowedKickRung({1.0f, 0.0f, 0.0f}, 8.0, parameters), 2);
    parameters.scaleFactor = 1.0;
    parameters.epsilon = 5.0;
    EXPECT_EQ(allowedKickRung({3.0f, 4.0f, 0.0f}, 8.0, parameters), 3);
}

TEST(Kick, ExactDyadicBoundaryAndZeroForce) {
    PROFILE();
    auto const parameters = unitParameters();
    EXPECT_EQ(allowedKickRung({1.0f, 0.0f, 0.0f}, 8.0, parameters), 3);
    EXPECT_EQ(allowedKickRung({std::nextafter(1.0f, 2.0f), 0.0f, 0.0f}, 8.0, parameters), 4);
    EXPECT_EQ(allowedKickRung({std::nextafter(1.0f, 0.0f), 0.0f, 0.0f}, 8.0, parameters), 3);
    EXPECT_EQ(allowedKickRung({}, 8.0, parameters), 0);
}

TEST(Kick, FinestRungAndCapacityFailure) {
    PROFILE();
    auto parameters = unitParameters();
    parameters.epsilon = std::ldexp(1.0, -2 * int(Rung::maximumRung()));
    EXPECT_EQ(allowedKickRung({1.0f, 0.0f, 0.0f}, 1.0, parameters), Rung::maximumRung());
    parameters.epsilon *= 0.25;
    EXPECT_THROW(allowedKickRung({1.0f, 0.0f, 0.0f}, 1.0, parameters), std::range_error);
}

TEST(Kick, RefinementIsImmediateAndCoarseningRequiresExactSynchronization) {
    PROFILE();
    EXPECT_EQ(rungTicks(0), std::uint64_t(1) << Rung::maximumRung());
    EXPECT_EQ(rungTicks(Rung::maximumRung()), 1);
    EXPECT_EQ(selectKickRung(2, 6, rungTicks(2)), 6);
    EXPECT_EQ(selectKickRung(4, 0, rungTicks(4)), 4);
    EXPECT_EQ(selectKickRung(4, 0, 2 * rungTicks(4)), 3);
    EXPECT_EQ(selectKickRung(4, 0, 4 * rungTicks(4)), 2);
    EXPECT_EQ(selectKickRung(4, 0, rungTicks(1)), 1);
    EXPECT_EQ(selectKickRung(4, 0, 0), 0);
    EXPECT_EQ(selectKickRung(4, 0, rungTicks(0)), 0);
    EXPECT_EQ(selectKickRung(Rung::maximumRung(), 0, 1), Rung::maximumRung());
    EXPECT_EQ(selectKickRung(Rung::maximumRung(), 0, 2), Rung::maximumRung() - 1);
    EXPECT_TRUE(rungSynchronized(rungTicks(0), 0));
    EXPECT_FALSE(rungSynchronized(rungTicks(0) - 1, 0));
    EXPECT_TRUE(rungSynchronized(rungTicks(0) - 1, Rung::maximumRung()));
}

TEST(Kick, InitialKickUsesOnlyNewHalfStep) {
    PROFILE();
    KickParticles<1> particles;
    auto parameters = unitParameters();
    parameters.epsilon = 3.0 / 16.0;
    DirectKickBackend{parameters}(0, {1.0f, 2.0f, -2.0f}, -5.0f, particles.flags(), 1.0, particles.view());
    EXPECT_EQ(particles.flags().rung(), 2);
    EXPECT_FLOAT_EQ(particles.momentum(0), 0.125f);
    EXPECT_FLOAT_EQ(particles.momentum(1), 0.25f);
    EXPECT_FLOAT_EQ(particles.momentum(2), -0.25f);
}

TEST(Kick, RegularTransitionUsesOldAndNewHalfSteps) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(3);
    auto parameters = unitParameters();
    parameters.phase = KickPhase::regular;
    parameters.timeTick = rungTicks(2);
    parameters.epsilon = 1.0 / 16.0;
    DirectKickBackend{parameters}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_EQ(particles.flags().rung(), 2);
    EXPECT_FLOAT_EQ(particles.momentum(), 0.1875f);

    particles.flags().setRung(1);
    particles.momentum() = 0.0f;
    parameters.timeTick = rungTicks(1);
    parameters.epsilon = 1.0 / 64.0;
    DirectKickBackend{parameters}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_EQ(particles.flags().rung(), 3);
    EXPECT_FLOAT_EQ(particles.momentum(), 0.3125f);
}

TEST(Kick, RegularHalfStepsRoundSeparatelyForUnchangedRefinedAndCoarsenedRungs) {
    PROFILE();
    struct KickCase {
        char const* description;
        std::uint8_t oldRung;
        std::uint8_t nextRung;
        double epsilon;
        double synchronizedMomentum;
    };
    constexpr float kIncomingMomentum = 0x1p24f;
    constexpr std::array<KickCase, 3> kCases{{
        {"unchanged rung", 0, 0, 4.0, double(kIncomingMomentum) + 1.0},
        {"refinement", 0, 1, 1.0, double(kIncomingMomentum) + 1.0},
        {"coarsening", 1, 0, 4.0, double(kIncomingMomentum) + 0.5},
    }};

    for (auto const& kickCase : kCases) {
        SCOPED_TRACE(kickCase.description);
        KickParticles<1> particles;
        particles.flags().setRung(kickCase.oldRung);
        particles.momentum() = kIncomingMomentum;
        auto parameters = unitParameters();
        parameters.phase = KickPhase::regular;
        parameters.timeTick = rungTicks(0);
        parameters.epsilon = kickCase.epsilon;
        parameters.sumKinetic = true;
        parameters.momentaStaggered = true;
        KickSums sums;
        DirectKickBackend{parameters, &sums}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 2.0, particles.view());
        // At 2^24 the spacing is two: each half kick rounds away, whereas
        // fusing their intervals would increase the stored momentum by two.
        EXPECT_EQ(particles.momentum(), kIncomingMomentum);
        EXPECT_EQ(particles.flags().rung(), kickCase.nextRung);
        EXPECT_EQ(sums.maximumRung, kickCase.nextRung);
        // Diagnostics retain the double synchronized value before either
        // stored half kick rounds it, including the closing half step.
        EXPECT_DOUBLE_EQ(sums.momentumSquared, kickCase.synchronizedMomentum * kickCase.synchronizedMomentum);
    }
}

TEST(Kick, RegularTransitionCompletesOldHalfStepBeforeOpeningNewHalfStep) {
    constexpr float kIncomingMomentum = 0x1p24f - 2.0f;
    PROFILE();

    KickParticles<1> particles;
    particles.flags().setRung(0);
    particles.momentum() = kIncomingMomentum;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::regular;
    parameters.timeTick = rungTicks(0);
    parameters.epsilon = 1.0;
    DirectKickBackend{parameters}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 2.0, particles.view());
    // Refinement adds one first, then the half increment rounds upward.
    // Reversing the half steps would produce 2^24 - 1 instead.
    EXPECT_EQ(particles.momentum(), 0x1p24f);
    EXPECT_EQ(particles.flags().rung(), 1);

    particles.flags().setRung(1);
    particles.momentum() = kIncomingMomentum;
    parameters.epsilon = 4.0;
    DirectKickBackend{parameters}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 2.0, particles.view());
    // Coarsening rounds away the half increment first, then adds one.
    // A reversed or fused update would produce 2^24 instead.
    EXPECT_EQ(particles.momentum(), 0x1p24f - 1.0f);
    EXPECT_EQ(particles.flags().rung(), 0);
}

TEST(Kick, FinalKickUsesOnlyOldHalfStepAndPreservesRung) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(3);
    auto parameters = unitParameters();
    parameters.phase = KickPhase::final;
    parameters.timeTick = rungTicks(0);
    DirectKickBackend{parameters}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_EQ(particles.flags().rung(), 3);
    EXPECT_FLOAT_EQ(particles.momentum(), 0.0625f);
}

TEST(Kick, InitialKineticSumUsesSynchronizedMomentumBeforeKick) {
    PROFILE();
    KickParticles<1> particles;
    particles.momentum(0) = 1.0f;
    particles.momentum(1) = -2.0f;
    particles.momentum(2) = 3.0f;
    auto parameters = unitParameters();
    parameters.sumKinetic = true;
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {4.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 14.0);
    EXPECT_FLOAT_EQ(particles.momentum(), 2.0f);
    EXPECT_EQ(particles.flags().rung(), 1);
    EXPECT_EQ(sums.maximumRung, 1);
}

TEST(Kick, RegularGlobalDiagnosticsIncludeInactiveRungsWithoutKickingThem) {
    PROFILE();
    KickParticles<2> particles;
    particles.flags(0).setRung(1);
    particles.flags(1).setRung(3);
    particles.momentum(0, 0) = 1.0f;
    particles.momentum(0, 1) = 2.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::regular;
    parameters.timeTick = rungTicks(3);
    parameters.minimumKickRung = 3;
    parameters.sumPotential = true;
    parameters.sumKinetic = true;
    parameters.momentaStaggered = true;
    KickSums sums;
    auto const view = particles.view(true, true);
    DirectKickBackend{parameters, &sums}(0, {2.0f, 0.0f, 0.0f}, -5.0f, particles.flags(0), 1.0, view);
    DirectKickBackend{parameters, &sums}(1, {4.0f, 0.0f, 0.0f}, -7.0f, particles.flags(1), 1.0, view);
    // The inactive momentum midpoint is tau=1/4, and the active midpoint is
    // tau=1/16. Synchronize both to this event at tau=1/8 before squaring.
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 0.75 * 0.75 + 2.25 * 2.25);
    EXPECT_EQ(sums.acceleration, (std::array<double, NDIM>{6.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(sums.potential, -12.0);
    EXPECT_EQ(sums.particleCount, 2);
    EXPECT_EQ(sums.maximumRung, 3);
    EXPECT_FLOAT_EQ(particles.momentum(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(particles.momentum(0, 1), 2.5f);
    EXPECT_EQ(particles.flags(0).rung(), 1);
    EXPECT_EQ(particles.flags(1).rung(), 3);
    EXPECT_FLOAT_EQ(particles.acceleration(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(particles.potential(0), -5.0f);
}

TEST(Kick, FinalKineticSumSynchronizesTheIncomingHalfStep) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(2);
    particles.momentum() = 2.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::final;
    parameters.timeTick = rungTicks(0);
    parameters.sumKinetic = true;
    parameters.momentaStaggered = true;
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {4.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 6.25);
    EXPECT_FLOAT_EQ(particles.momentum(), 2.5f);
    EXPECT_EQ(sums.maximumRung, 2);
}

TEST(Kick, KineticSumUsesOldRungWhileMaximumReportsNewRung) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(3);
    particles.momentum() = 1.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::regular;
    parameters.timeTick = rungTicks(2);
    parameters.epsilon = 1.0 / 16.0;
    parameters.sumKinetic = true;
    parameters.momentaStaggered = true;
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {1.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 1.0625 * 1.0625);
    EXPECT_FLOAT_EQ(particles.momentum(), 1.1875f);
    EXPECT_EQ(particles.flags().rung(), 2);
    EXPECT_EQ(sums.maximumRung, 2);
}

TEST(Kick, DiagnosticKineticSumSynchronizesProcessedTicksAndPartialDrifts) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(2);
    particles.momentum() = 2.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::diagnostics;
    parameters.timeTick = rungTicks(2);
    parameters.sumKinetic = true;
    parameters.momentaStaggered = true;
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {4.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    // Diagnostics at an already processed event undo the outgoing half kick.
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 2.25);
    EXPECT_FLOAT_EQ(particles.momentum(), 2.0f);
    EXPECT_EQ(particles.flags().rung(), 2);

    particles.flags().setRung(1);
    parameters.conformalOffset = 0.0625;
    sums = {};
    DirectKickBackend{parameters, &sums}(0, {4.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 5.0625);
    EXPECT_FLOAT_EQ(particles.momentum(), 2.0f);
    EXPECT_EQ(particles.flags().rung(), 1);
    EXPECT_EQ(sums.maximumRung, 1);
}

TEST(Kick, DiagnosticsPreserveMomentumAndFlagsExactly) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(5);
    for (size_t axis = 0; axis < NDIM; axis++) {
        particles.momentum(axis) = -0.0f;
    }
    auto parameters = unitParameters();
    parameters.phase = KickPhase::diagnostics;
    parameters.timeTick = 1;
    parameters.sumPotential = true;
    KickSums sums;
    std::array<std::uint32_t, NDIM> previousMomentum;
    for (size_t axis = 0; axis < NDIM; axis++) {
        previousMomentum[axis] = hpx::bit_cast<std::uint32_t>(particles.momentum(axis));
    }
#ifdef COSMICTIGER_VALIDATION_CHECKS
    EXPECT_EQ(previousMomentum[0], 0x80000000u);
#endif
    auto const previousFlags = hpx::bit_cast<std::uint8_t>(particles.flags());
    DirectKickBackend{parameters, &sums}(0, {1.0f, 2.0f, 3.0f}, -4.0f, particles.flags(), 1.0, particles.view(true, true));
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_EQ(hpx::bit_cast<std::uint32_t>(particles.momentum(axis)), previousMomentum[axis]);
        EXPECT_FLOAT_EQ(particles.acceleration(axis), static_cast<float>(axis + 1));
        EXPECT_DOUBLE_EQ(sums.acceleration[axis], static_cast<double>(axis + 1));
    }
    EXPECT_EQ(hpx::bit_cast<std::uint8_t>(particles.flags()), previousFlags);
    EXPECT_FLOAT_EQ(particles.potential(), -4.0f);
    EXPECT_DOUBLE_EQ(sums.potential, -4.0);
    EXPECT_EQ(sums.particleCount, 1);
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 0.0);
    EXPECT_EQ(sums.maximumRung, 5);
}

TEST(Kick, GhostsAreExcludedFromUpdatesOutputAndSums) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags() = Rung{4, true};
    particles.momentum() = 7.0f;
    particles.acceleration() = 11.0f;
    particles.potential() = 13.0f;
    auto parameters = unitParameters();
    parameters.sumPotential = true;
    parameters.sumKinetic = true;
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {1.0f, 2.0f, 3.0f}, -4.0f, particles.flags(), 1.0, particles.view(true, true));
    EXPECT_EQ(particles.flags().rung(), 4);
    EXPECT_TRUE(particles.flags().isGhost());
    EXPECT_FLOAT_EQ(particles.momentum(), 7.0f);
    EXPECT_FLOAT_EQ(particles.acceleration(), 11.0f);
    EXPECT_FLOAT_EQ(particles.potential(), 13.0f);
    EXPECT_EQ(sums.particleCount, 0);
    EXPECT_EQ(sums.acceleration, (std::array<double, NDIM>{}));
    EXPECT_DOUBLE_EQ(sums.potential, 0.0);
    EXPECT_DOUBLE_EQ(sums.momentumSquared, 0.0);
    EXPECT_EQ(sums.maximumRung, 0);
}

TEST(Kick, DiagnosticPotentialAndStorageAreIndependentOptions) {
    PROFILE();
    KickParticles<2> particles;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::diagnostics;
    KickSums sums;
    particles.acceleration() = 17.0f;
    particles.potential() = 19.0f;
    DirectKickBackend{parameters, &sums}(0, {1.0f, 2.0f, 3.0f}, -5.0f, particles.flags(0), 1.0, particles.view());
    EXPECT_FLOAT_EQ(particles.acceleration(), 17.0f);
    EXPECT_FLOAT_EQ(particles.potential(), 19.0f);
    EXPECT_DOUBLE_EQ(sums.potential, 0.0);
    parameters.sumPotential = true;
    DirectKickBackend{parameters, &sums}(1, {-4.0f, 5.0f, -6.0f}, -7.0f, particles.flags(1), 1.0, particles.view(false, true));
    EXPECT_EQ(sums.acceleration, (std::array<double, NDIM>{-3.0, 7.0, -3.0}));
    EXPECT_DOUBLE_EQ(sums.potential, -7.0);
    EXPECT_EQ(sums.particleCount, 2);
    EXPECT_FLOAT_EQ(particles.potential(1), -7.0f);
    parameters.sumPotential = false;
    DirectKickBackend{parameters}(1, {9.0f, 8.0f, 7.0f}, -11.0f, particles.flags(1), 1.0, particles.view(true, false));
    EXPECT_FLOAT_EQ(particles.acceleration(0, 1), 9.0f);
    EXPECT_FLOAT_EQ(particles.potential(1), -7.0f);
}

TEST(Kick, DirectPathUsesFerrersPlummerEquivalentSofteningAndCompleteSums) {
    PROFILE();
    KickParticles<2> particles;
    particles.setPosition(0, {0.25, 0.5, 0.5});
    particles.setPosition(1, {0.5, 0.5, 0.5});
    auto const softening = cosmictiger::gravity::makeFerrersSoftening(0.125f);
    KickSums sums;
    DirectInteractionRequest request{{0, 2}, {0, 2}, particles.view(true, true), softening, 1.0f, 1.0, 0};
    request.kick = unitParameters();
    request.kick.eta = 0.125;
    request.kick.epsilon = 1.0e20;
    request.kick.sumPotential = true;
    request.sums = &sums;
    cosmictiger::gravity::directInteractionsCpu<DirectKickBackend, 0, true>(request);
    // F0 phi(0)=-3/(2q), hence epsilon=2q/3=1/12. With |g|=16,
    // allowed dt=sqrt(1/768), which rounds down to 1/32 (rung five).
    EXPECT_EQ(particles.flags(0).rung(), 5);
    EXPECT_EQ(particles.flags(1).rung(), 5);
    EXPECT_FLOAT_EQ(particles.momentum(0, 0), 0.25f);
    EXPECT_FLOAT_EQ(particles.momentum(0, 1), -0.25f);
    EXPECT_FLOAT_EQ(particles.acceleration(0, 0), 16.0f);
    EXPECT_FLOAT_EQ(particles.acceleration(0, 1), -16.0f);
    EXPECT_FLOAT_EQ(particles.potential(0), -4.0f);
    EXPECT_FLOAT_EQ(particles.potential(1), -4.0f);
    EXPECT_EQ(sums.acceleration, (std::array<double, NDIM>{}));
    EXPECT_DOUBLE_EQ(sums.potential, -8.0);
    EXPECT_EQ(sums.particleCount, 2);
}

TEST(Kick, DirectPathRejectsPotentialConsumersWhenPotentialIsDisabled) {
    PROFILE();
    KickParticles<1> particles;
    auto const softening = cosmictiger::gravity::makeFerrersSoftening(0.125f);
    DirectInteractionRequest request{{0, 1}, {0, 1}, particles.view(), softening, 1.0f, 1.0, 0};
    request.kick.sumPotential = true;
    EXPECT_THROW((cosmictiger::gravity::directInteractionsCpu<DirectKickBackend, 0, false>(request)), std::invalid_argument);
    request.kick.sumPotential = false;
    request.particles = particles.view(false, true);
    EXPECT_THROW((cosmictiger::gravity::directInteractionsCpu<DirectKickBackend, 0, false>(request)), std::invalid_argument);
    request.particles = particles.view(true, false);
    EXPECT_NO_THROW((cosmictiger::gravity::directInteractionsCpu<DirectKickBackend, 0, false>(request)));
}

TEST(Kick, HarmonicOscillatorHasSecondOrderConvergenceAndBoundedEnergyError) {
    PROFILE();
    auto const coarse = integrateOscillator(4);
    auto const fine = integrateOscillator(5);
    EXPECT_LT(coarse.maximumEnergyError, 5.0e-4);
    EXPECT_LT(fine.maximumEnergyError, coarse.maximumEnergyError / 3.5);
    EXPECT_LT(fine.error, coarse.error / 3.5);
    EXPECT_GT(fine.error, coarse.error / 4.5);
}

TEST(Kick, SnapshotCapturesInitialMomentumBeforeOutgoingHalfKick) {
    constexpr double kMomentum = 1.25;
    PROFILE();
    KickParticles<1> particles;
    particles.momentum() = kMomentum;
    auto parameters = unitParameters();
    double output[NDIM]{};
    parameters.synchronizedMomentumCount = 1;
    for (size_t axis = 0; axis < NDIM; axis++) {
        parameters.synchronizedMomenta[axis] = &output[axis];
    }
    DirectKickBackend{parameters}(0, {2.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(output[0], kMomentum);
    EXPECT_EQ(particles.flags().rung(), 1);
    EXPECT_FLOAT_EQ(particles.momentum(), 1.75f);
}

TEST(Kick, SnapshotCapturesIncomingRungBeforeCoarsening) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(2);
    particles.momentum() = 1.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::regular;
    parameters.momentaStaggered = true;
    parameters.timeTick = rungTicks(0);
    parameters.sumKinetic = true;
    double output[NDIM]{};
    parameters.synchronizedMomentumCount = 1;
    for (size_t axis = 0; axis < NDIM; axis++) {
        parameters.synchronizedMomenta[axis] = &output[axis];
    }
    KickSums sums;
    DirectKickBackend{parameters, &sums}(0, {0.125f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(output[0], 1.015625);
    EXPECT_DOUBLE_EQ(sums.momentumSquared, output[0] * output[0]);
    EXPECT_EQ(particles.flags().rung(), 0);
    EXPECT_FLOAT_EQ(particles.momentum(), 1.078125f);
}

TEST(Kick, SnapshotSynchronizesPartialEpochWithoutSumsOrParticleChanges) {
    PROFILE();
    KickParticles<1> particles;
    particles.flags().setRung(2);
    particles.momentum() = 1.0f;
    auto parameters = unitParameters();
    parameters.phase = KickPhase::diagnostics;
    parameters.momentaStaggered = true;
    parameters.conformalOffset = 0.0625;
    double output[NDIM]{};
    parameters.synchronizedMomentumCount = 1;
    for (size_t axis = 0; axis < NDIM; axis++) {
        parameters.synchronizedMomenta[axis] = &output[axis];
    }
    DirectKickBackend{parameters}(0, {2.0f, 0.0f, 0.0f}, 0.0f, particles.flags(), 1.0, particles.view());
    EXPECT_DOUBLE_EQ(output[0], 0.875);
    EXPECT_FLOAT_EQ(particles.momentum(), 1.0f);
    EXPECT_EQ(particles.flags().rung(), 2);
}
