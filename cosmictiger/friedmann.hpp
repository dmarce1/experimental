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

#include <stddef.h>

namespace cosmictiger {

enum class CosmologicalTime { physical, conformal };
enum class ExpansionBranch { expanding, contracting };

// Values at a=1. H0 is in inverse units of the time supplied to the solver.
// The four Omegas must sum to one; curvature and dark energy may be signed.
// Dark energy is a cosmological constant (equation of state w=-1).
struct BackgroundCosmologyParameters {
    double hubbleConstant = 1.0;
    double omegaRadiation = 0.0;
    double omegaMatter = 0.3;
    double omegaDarkEnergy = 0.7;
    double omegaCurvature = 0.0;
};

struct FriedmannOptions {
    double relativeTolerance = 1.0e-12;
    // Absolute tolerance for the expansion rate normalized by the local
    // component amplitude. The scale factor always uses relative tolerance.
    double absoluteTolerance = 1.0e-14;
    size_t maxSteps = 100000;
};

// Both derivatives are returned regardless of the integration coordinate.
// dt=a*deta, so conformalDerivative=scaleFactor*physicalDerivative.
struct FriedmannState {
    double scaleFactor;
    double physicalDerivative;
    double conformalDerivative;
};

// An immutable, double-precision FLRW background. Times are elapsed from the
// supplied initial state, with no assumption about a Big Bang time origin.
// advance() follows signed expansion through a turnaround into contraction.
class FriedmannCosmology {
public:
    explicit FriedmannCosmology(BackgroundCosmologyParameters parameters = {}, FriedmannOptions options = {});

    BackgroundCosmologyParameters const& parameters() const noexcept;
    FriedmannOptions const& options() const noexcept;

    FriedmannState state(double scaleFactor, ExpansionBranch branch = ExpansionBranch::expanding) const;
    FriedmannState advance(FriedmannState const& initialState, double elapsedTime, CosmologicalTime time = CosmologicalTime::physical) const;
    FriedmannState evolve(double initialScaleFactor, double elapsedTime, CosmologicalTime time = CosmologicalTime::physical,
        ExpansionBranch branch = ExpansionBranch::expanding) const;

private:
    BackgroundCosmologyParameters parameters_;
    FriedmannOptions options_;
};

}    // namespace cosmictiger
