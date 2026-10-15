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

#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/profiler.hpp>

#include <exception>
#include <string>
#include <utility>

namespace cosmictiger::gravity {

namespace detail {
    template <bool computePotential>
    void dispatchDirectKick(DirectInteractionRequest const& request, int order) {
        PROFILE();
        switch (order) {
        case 0:
            directInteractionsCpu<DirectKickBackend, 0, computePotential>(request);
            break;
        case 1:
            directInteractionsCpu<DirectKickBackend, 1, computePotential>(request);
            break;
        case 2:
            directInteractionsCpu<DirectKickBackend, 2, computePotential>(request);
            break;
        case 3:
            directInteractionsCpu<DirectKickBackend, 3, computePotential>(request);
            break;
        default:
            throw std::invalid_argument("kick Ferrers order must be between zero and three");
        }
    }
}    // namespace detail

KickSums kickDirectInteractionsCpu(DirectInteractionRequest request, int ferrersOrder, std::uint64_t operationId) {
    PROFILE();
    KickSums local;
    request.sums = &local;
    std::string localError;
    try {
        if (request.kick.sumPotential || request.particles.potentials != nullptr) {
            detail::dispatchDirectKick<true>(request, ferrersOrder);
        } else {
            detail::dispatchDirectKick<false>(request, ferrersOrder);
        }
    } catch (std::exception const& error) {
        localError = error.what();
    } catch (...) {
        localError = "unknown local direct kick failure";
    }
    return allReduceKickSums(local, request.kick.sumPotential, operationId, std::move(localError));
}

DirectInteractionRequest makeDirectKickRequest(ParticleStorage& particles, ParticleRange source, ParticleRange sink, SofteningLength softening,
    std::uint8_t activeRung, double scaleFactor, std::uint64_t timeTick, KickPhase phase, KickSums* sums, bool sumPotential) {
    PROFILE();
    auto const& constants = globalConstants().data();
    particles.configureGravityFields(constants.outputAcceleration, constants.outputPotential);
    softening.model = constants.softeningModel;
    DirectInteractionParticles view{};
    view.flags = particles.rungData();
    for (size_t axis = 0; axis < NDIM; axis++) {
        view.positions[axis] = particles.positionData(axis);
        view.velocities[axis] = particles.velocityData(axis);
        view.accelerations[axis] = particles.storesAcceleration() ? particles.accelerationData(axis) : nullptr;
    }
    view.potentials = particles.storesPotential() ? particles.potentialData() : nullptr;
    return {source, sink, view, softening, static_cast<float>(constants.gravityParticleMass), constants.rungZeroTimestep, activeRung,
        {constants.timestepEta, 0.0, scaleFactor, timeTick, phase, sumPotential}, sums};
}

}    // namespace cosmictiger::gravity
