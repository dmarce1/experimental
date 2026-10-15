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

#include <boost/units/physical_dimensions/specific_energy.hpp>
#include <boost/units/quantity.hpp>
#include <boost/units/systems/cgs/acceleration.hpp>
#include <boost/units/systems/cgs/energy.hpp>
#include <boost/units/systems/cgs/force.hpp>
#include <boost/units/systems/cgs/frequency.hpp>
#include <boost/units/systems/cgs/length.hpp>
#include <boost/units/systems/cgs/mass.hpp>
#include <boost/units/systems/cgs/mass_density.hpp>
#include <boost/units/systems/cgs/time.hpp>
#include <boost/units/systems/cgs/velocity.hpp>

namespace cosmictiger::units {

// Boost.Units checks physical dimensions. Semantic wrappers below additionally
// distinguish quantities with identical dimensions but different factors of a.
using Length = boost::units::quantity<boost::units::cgs::length, double>;
using Time = boost::units::quantity<boost::units::cgs::time, double>;
using Frequency = boost::units::quantity<boost::units::cgs::frequency, double>;
using Mass = boost::units::quantity<boost::units::cgs::mass, double>;
using Velocity = boost::units::quantity<boost::units::cgs::velocity, double>;
using Acceleration = boost::units::quantity<boost::units::cgs::acceleration, double>;
using Potential = boost::units::quantity<boost::units::unit<boost::units::specific_energy_dimension, boost::units::cgs::system>, double>;
using Energy = boost::units::quantity<boost::units::cgs::energy, double>;
using Force = boost::units::quantity<boost::units::cgs::force, double>;
using Density = boost::units::quantity<boost::units::cgs::mass_density, double>;
using GravitationalConstant = decltype(Length{} * Velocity{} * Velocity{} / Mass{});

// Preserve the Mpc and G values used by the cosmological background/CAMB
// interface. A year is a Julian year; solar masses use 1.98847e33 grams.
inline constexpr Length kMegaparsec = Length::from_value(3.085677581e24);
inline constexpr Length kKilometer = Length::from_value(1.0e5);
inline constexpr Time kSecond = Time::from_value(1.0);
inline constexpr Time kYear = Time::from_value(31557600.0);
inline constexpr Mass kSolarMass = Mass::from_value(1.98847e33);
inline constexpr GravitationalConstant kGravitationalConstant = GravitationalConstant::from_value(6.67430e-8);

inline Frequency hubbleRate(double hubbleKmPerSecondMpc) {
    PROFILE();
    FpeGuard guard;
    return expectPositive(hubbleKmPerSecondMpc) * (kKilometer / kSecond) / kMegaparsec;
}

inline Time hubbleTime(double hubbleKmPerSecondMpc) {
    PROFILE();
    FpeGuard guard;
    return 1.0 / hubbleRate(hubbleKmPerSecondMpc);
}

namespace detail {
    class ComovingPositionTag;
    class ProperPositionTag;
    class CanonicalMomentumPerMassTag;
    class PeculiarVelocityTag;
    class PhysicalTimeTag;
    class ConformalTimeTag;
}    // namespace detail

template <typename Quantity, typename Tag>
class CosmologicalQuantity {
public:
    explicit constexpr CosmologicalQuantity(Quantity quantity)
      : quantity_(quantity) {
        PROFILE();
    }

    constexpr Quantity const& quantity() const {
        PROFILE();
        return quantity_;
    }

    // Scalar extraction is always in CGS: cm, cm/s, or seconds. There is no
    // implicit conversion to another cosmological frame or to a raw scalar.
    constexpr double value() const {
        PROFILE();
        return quantity_.value();
    }

private:
    Quantity quantity_;
};

using ComovingPosition = CosmologicalQuantity<Length, detail::ComovingPositionTag>;
using ProperPosition = CosmologicalQuantity<Length, detail::ProperPositionTag>;
using CanonicalMomentumPerMass = CosmologicalQuantity<Velocity, detail::CanonicalMomentumPerMassTag>;
using PeculiarVelocity = CosmologicalQuantity<Velocity, detail::PeculiarVelocityTag>;
using PhysicalTime = CosmologicalQuantity<Time, detail::PhysicalTimeTag>;
using ConformalTime = CosmologicalQuantity<Time, detail::ConformalTimeTag>;

// Fixed run scales. The input box length retains its existing Mpc/h convention;
// all output quantities use ordinary Mpc/Msun or CGS, with no little-h factors.
// This host-only conversion layer does not change fixed-point particle storage
// or impose Boost.Units objects on CPU/CUDA numerical kernels.
class UnitScales {
public:
    UnitScales(double hubbleKmPerSecondMpc, double boxLengthMpcOverH) {
        constexpr double kReferenceHubble = 100.0;
        PROFILE();
        FpeGuard guard;
        double const hubble = expectPositive(hubbleKmPerSecondMpc);
        double const littleH = hubble / kReferenceHubble;
        length_ = (expectPositive(boxLengthMpcOverH) / littleH) * kMegaparsec;
        time_ = hubbleTime(hubble);
        velocity_ = length_ / time_;
        potential_ = velocity_ * velocity_;
        mass_ = potential_ * length_ / kGravitationalConstant;
        acceleration_ = velocity_ / time_;
        energy_ = mass_ * potential_;
        force_ = mass_ * acceleration_;
        density_ = mass_ / (length_ * length_ * length_);
    }

    double lengthMpc() const {
        PROFILE();
        return (length_ / kMegaparsec).value();
    }

    double lengthCgs() const {
        PROFILE();
        return length_.value();
    }

    double timeSeconds() const {
        PROFILE();
        return time_.value();
    }

    double velocityKmPerSecond() const {
        PROFILE();
        return (velocity_ / (kKilometer / kSecond)).value();
    }

    double velocityCgs() const {
        PROFILE();
        return velocity_.value();
    }

    double massSolar() const {
        PROFILE();
        return (mass_ / kSolarMass).value();
    }

    double massCgs() const {
        PROFILE();
        return mass_.value();
    }

    double accelerationCgs() const {
        PROFILE();
        return acceleration_.value();
    }

    double potentialCgs() const {
        PROFILE();
        return potential_.value();
    }

    double energyCgs() const {
        PROFILE();
        return energy_.value();
    }

    double forceCgs() const {
        PROFILE();
        return force_.value();
    }

    double densityCgs() const {
        PROFILE();
        return density_.value();
    }

    ComovingPosition comovingPosition(double coordinate) const {
        PROFILE();
        FpeGuard guard;
        return ComovingPosition(coordinate * length_);
    }

    ProperPosition properPosition(double coordinate, double scaleFactor) const {
        PROFILE();
        FpeGuard guard;
        return ProperPosition(expectPositive(scaleFactor) * coordinate * length_);
    }

    CanonicalMomentumPerMass canonicalMomentumPerMass(double momentum) const {
        PROFILE();
        FpeGuard guard;
        return CanonicalMomentumPerMass(momentum * velocity_);
    }

    // p = a*v_pec/(H0*L). The caller supplies p synchronized to the output epoch,
    // never the staggered restart momentum. No Hubble flow is added.
    PeculiarVelocity peculiarVelocity(double synchronizedMomentum, double scaleFactor) const {
        PROFILE();
        FpeGuard guard;
        return PeculiarVelocity((synchronizedMomentum / expectPositive(scaleFactor)) * velocity_);
    }

    // The native completed field is g0 = dp/d(H0*tau); physical peculiar
    // acceleration is H0^2*L*g0/a^2. This excludes the background acceleration.
    Acceleration peculiarAcceleration(double field, double scaleFactor) const {
        PROFILE();
        FpeGuard guard;
        double const inverseScaleFactor = 1.0 / expectPositive(scaleFactor);
        double const inverseScaleFactorSquared = inverseScaleFactor * inverseScaleFactor;
        return (field * inverseScaleFactorSquared) * acceleration_;
    }

    Potential peculiarPotential(double potential, double scaleFactor) const {
        PROFILE();
        FpeGuard guard;
        return (potential / expectPositive(scaleFactor)) * potential_;
    }

    PhysicalTime physicalTime(double codeTime) const {
        PROFILE();
        FpeGuard guard;
        return PhysicalTime(codeTime * time_);
    }

    ConformalTime conformalTime(double codeTime) const {
        PROFILE();
        FpeGuard guard;
        return ConformalTime(codeTime * time_);
    }

private:
    Length length_;
    Time time_;
    Velocity velocity_;
    Potential potential_;
    Mass mass_;
    Acceleration acceleration_;
    Energy energy_;
    Force force_;
    Density density_;
};

}    // namespace cosmictiger::units
