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

#include <cosmictiger/einstein_boltzmann.hpp>
#include <cosmictiger/friedmann.hpp>

#include <array>
#include <stddef.h>
#include <vector>

namespace cosmictiger {

struct SimulationBackgroundStep {
    FriedmannState state;
    double driftFactor;
    double conformalInterval;
    double physicalInterval;
};

struct SimulationCollapseTimes {
    double endPhysicalTime;
    double stopPhysicalTime;
};

// The simulation uses H0^-1 as its time unit: tau=H0*integral(dt/a).
// Radiation, curvature, thermal massive neutrinos and w(a)=w0+wa*(1-a)
// use the same density/species conventions as the spectrum interface.
class SimulationBackgroundTable;

class SimulationBackground {
public:
    explicit SimulationBackground(CosmologyParameters const& parameters);

    // da/dtau = a*a*H(a)/H0. This is also the multiplier converting
    // initial dx/dln(a) into canonical momentum p=a*dx/dtau.
    double conformalDerivative(double scaleFactor) const;
    double conformalInterval(double initialScaleFactor, double finalScaleFactor) const;

    // Signed derivatives evolve smoothly through a turnaround. All time
    // intervals use H0^-1 units and may be signed. Both clocks and the drift
    // integral are advanced together, including on the contracting branch.
    FriedmannState state(double scaleFactor, ExpansionBranch branch = ExpansionBranch::expanding) const;
    SimulationBackgroundStep advance(FriedmannState const& initialState, double elapsedConformalTime) const;
    SimulationBackgroundStep advancePhysical(FriedmannState const& initialState, double elapsedPhysicalTime) const;

    // Expanding-branch proper-time quadrature, with signed endpoint order.
    double physicalInterval(double initialScaleFactor, double finalScaleFactor) const;
    // Proper time from a Big Bang to the expanding a=1 epoch, in H0^-1.
    // Reject backgrounds without a finite Big Bang age on this branch.
    double presentAge() const;
    SimulationCollapseTimes collapseTimes(double minimumScaleFactor) const;

    // Integral d tau/a, multiplying canonical momentum in a position drift.
    // Equal endpoints return zero; reversed endpoints reverse the drift.
    double driftFactor(double initialScaleFactor, double finalScaleFactor) const;

private:
    friend class SimulationBackgroundTable;
    void evaluate(double scaleFactor, double& squaredDerivative, double& amplitude, double& derivative) const;
    SimulationBackgroundStep advanceInTime(
        FriedmannState const& initialState, double elapsedTime, CosmologicalTime time, SimulationBackgroundTable* table = nullptr) const;
    double omegaCold_;
    double omegaRadiation_;
    double omegaMassiveRelativistic_;
    double neutrinoMassRatio_;
    double omegaCurvature_;
    double omegaDarkEnergy_;
    double darkEnergyW_;
    double darkEnergyWa_;
};

// One forward integration in physical time, with quartic dense output from
// each accepted Dormand-Prince step. Queries use conformal time since initial;
// no background evaluation or quadrature occurs during interpolation. Signed
// da/dtau is retained through turnaround. Rebuild from the saved tick on restart.
class SimulationBackgroundTable {
public:
    SimulationBackgroundTable(SimulationBackground const& background, FriedmannState const& initial, double physicalDuration);
    SimulationBackgroundStep const& end() const;
    SimulationBackgroundStep at(double conformalTime) const;
    // Integrals over [begin,end], evaluated locally to preserve short drifts.
    SimulationBackgroundStep interval(double begin, double end) const;
    size_t segmentCount() const;

private:
    friend class SimulationBackground;
    static constexpr size_t kVariableCount_ = 4;
    static constexpr size_t kCoefficientCount_ = 5;
    static constexpr size_t kScaleFactor_ = 0;
    static constexpr size_t kDerivative_ = 1;
    static constexpr size_t kDrift_ = 2;
    static constexpr size_t kConformal_ = 3;
    using Polynomial = std::array<double, kCoefficientCount_>;
    struct Segment {
        double physicalBegin;
        double physicalWidth;
        double conformalBegin;
        double conformalEnd;
        double driftBegin;
        std::array<Polynomial, kVariableCount_> coefficients;
    };
    struct Location {
        size_t index;
        double fraction;
    };
    static double polynomial(Polynomial const& coefficients, double fraction);
    static double dividedDifference(Polynomial const& coefficients, double left, double right);
    Location locate(double conformalTime) const;
    FriedmannState initial_;
    SimulationBackgroundStep end_;
    std::vector<Segment> segments_;
};

}    // namespace cosmictiger
