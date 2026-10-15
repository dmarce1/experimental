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
#include <cosmictiger/global_constants.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_time.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/hpx.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>

#include <atomic>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger::detail {

static std::atomic<GlobalConstants const*> localClient{nullptr};
static std::unique_ptr<GlobalConstants> localClientOwner;
static std::atomic<bool> publicationStarted{false};

void installGlobalConstants(hpx::id_type componentId) {
    PROFILE();
    if (localClient.load(std::memory_order_acquire) != nullptr) {
        throw std::logic_error("GlobalConstants has already been published on this locality");
    }

    auto client = std::make_unique<GlobalConstants>(componentId);
    GlobalConstants const* expected = nullptr;
    if (!localClient.compare_exchange_strong(expected, client.get(), std::memory_order_release, std::memory_order_relaxed)) {
        throw std::logic_error("GlobalConstants has already been published on this locality");
    }
    localClientOwner = std::move(client);
    hpx::register_pre_shutdown_function(hpx::annotated_function(
        [] {
            PROFILE();
            localClient.store(nullptr, std::memory_order_release);
            localClientOwner.reset();
        },
        "globalConstants.register_pre_shutdown_function"));
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::installGlobalConstants, CosmictigerInstallGlobalConstantsAction)

namespace cosmictiger {

server::GlobalConstants::GlobalConstants(GlobalConstantsData data)
  : data_(std::move(data)) {
    PROFILE();
    if (hpx::get_locality_id() != 0) {
        throw std::logic_error("GlobalConstants must be created on the root locality");
    }
}

GlobalConstantsData server::GlobalConstants::getData() const {
    PROFILE();
    return data_;
}

GlobalConstants::GlobalConstants(hpx::id_type const& componentId)
  : hpx::components::client_base<GlobalConstants, server::GlobalConstants>(componentId)
  , data_(hpx::async<server::GlobalConstants::GetDataAction>(componentId).get())
  , unitScales_(data_.cosmology.hubble, data_.initializationOptions.boxLength) {
    PROFILE();
}

GlobalConstantsData const& GlobalConstants::data() const noexcept {
    PROFILE();
    return data_;
}

units::UnitScales const& GlobalConstants::unitScales() const noexcept {
    PROFILE();
    return unitScales_;
}

GlobalConstants const& globalConstants() {
    PROFILE();
    auto const* client = detail::localClient.load(std::memory_order_acquire);
    if (client == nullptr) {
        throw std::logic_error("GlobalConstants has not been published on this locality");
    }
    return *client;
}

void initializeSimulationConstants(GlobalConstantsData& data, double initialScaleFactor) {
    PROFILE();
    FpeGuard guard;
    SimulationBackground const background(data.cosmology);
    data.initialScaleFactor = expectPositive(initialScaleFactor);
    auto const initial = background.state(data.initialScaleFactor);
    data.initialPhysicalTime = -background.physicalInterval(data.initialScaleFactor, 1.0);
    initializeSimulationEndpoint(data);
    if (!(data.stopPhysicalTime > data.initialPhysicalTime)) {
        throw std::invalid_argument("simulation.stopTime must follow the initial-condition epoch");
    }
    data.rungZeroTimestep = background.advancePhysical(initial, data.stopPhysicalTime - data.initialPhysicalTime).conformalInterval;
    data.originalRungZeroTimestep = data.rungZeroTimestep;
    data.initialConformalDerivative = initial.conformalDerivative;
    initializeLightconeConstants(data);
    if (data.softeningModel != gravity::SofteningModel::ferrers && data.softeningModel != gravity::SofteningModel::plummer) {
        throw std::invalid_argument("unknown gravity softening model");
    }
    data.softeningLength = expectRange(0.0, expectPositive(data.softeningCellFraction), 1.0) / static_cast<double>(expectPositive(data.gridSideLength));
    expectRange(0, data.softeningKernelOrder, 3);
    double densityH2 = data.cosmology.omegaCdmH2;
    if (data.initializationOptions.component != MatterComponent::cdm) {
        densityH2 += data.cosmology.omegaBaryonH2;
    }
    if (data.initializationOptions.component == MatterComponent::totalMatter) {
        if (data.cosmology.omegaNeutrinoH2 != 0.0) {
            throw std::invalid_argument("thermal massive neutrinos cannot be represented by the cold particle population");
        }
    }
    double const h = expectPositive(data.cosmology.hubble) / 100.0;
    data.gravityParticleMass =
        3.0 * expectPositive(densityH2) / (8.0 * std::numbers::pi * h * h * static_cast<double>(expectPositive(data.globalParticleCount)));
}

double physicalTimeInYears(double physicalTime, CosmologyParameters const& cosmology) {
    PROFILE();
    FpeGuard guard;
    return (physicalTime * units::hubbleTime(cosmology.hubble) / units::kYear).value();
}

void initializeLightconeConstants(GlobalConstantsData& data) {
    constexpr double kLightSpeedKmPerSecond = 299792.458;
    constexpr double kHubbleUnit = 100.0;
    PROFILE();
    if (!data.lightcone.enabled) {
        data.lightconeObservationConformalTime = 0.0;
        data.lightconeSpeed = 0.0;
        return;
    }
    FpeGuard guard;
    SimulationBackground const background(data.cosmology);
    double const observationPhysicalTime = data.lightcone.observationTimeYears / physicalTimeInYears(1.0, data.cosmology);
    if (!(observationPhysicalTime > data.initialPhysicalTime)) {
        throw std::invalid_argument("lightcone.observationTime must follow the original initial-condition epoch");
    }
    data.lightconeObservationConformalTime =
        background.advancePhysical(background.state(data.initialScaleFactor), observationPhysicalTime - data.initialPhysicalTime).conformalInterval;
    data.lightconeSpeed = kLightSpeedKmPerSecond / (kHubbleUnit * expectPositive(data.initializationOptions.boxLength));
}

void initializeSimulationEndpoint(GlobalConstantsData& data) {
    PROFILE();
    if (data.stopAtEnd) {
        SimulationBackground const background(data.cosmology);
        auto const times = background.collapseTimes(data.minimumScaleFactor);
        data.stopPhysicalTime = times.stopPhysicalTime;
        data.endPhysicalTime = times.endPhysicalTime;
    } else {
        data.stopPhysicalTime = data.stopTimeYears / physicalTimeInYears(1.0, data.cosmology);
        data.endPhysicalTime = 0.0;
    }
}

void publishGlobalConstants(GlobalConstantsData data) {
    PROFILE();
    if (hpx::get_locality_id() != 0) {
        throw std::logic_error("only the root locality may publish GlobalConstants");
    }

    auto const localities = hpx::find_all_localities();
    expectEqual(data.localityCount, localities.size());

    bool expected = false;
    if (!detail::publicationStarted.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        throw std::logic_error("GlobalConstants may only be published once");
    }

    auto const componentId = hpx::new_<server::GlobalConstants>(hpx::find_here(), std::move(data)).get();
    std::vector<hpx::future<void>> installations;
    installations.reserve(localities.size());
    for (auto const& locality : localities) {
        installations.push_back(hpx::async<CosmictigerInstallGlobalConstantsAction>(locality, componentId));
    }
    hpx::wait_all(installations);
    for (auto& installation : installations) {
        installation.get();
    }
}

}    // namespace cosmictiger

using CosmictigerGlobalConstantsComponent = hpx::components::component<cosmictiger::server::GlobalConstants>;
HPX_REGISTER_COMPONENT(CosmictigerGlobalConstantsComponent, CosmictigerGlobalConstants)
HPX_REGISTER_ACTION(cosmictiger::server::GlobalConstants::GetDataAction, CosmictigerGlobalConstantsGetDataAction)
