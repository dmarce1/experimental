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

/** @file
 * @brief Runtime dispatch to generated compact STF Cartesian gravity operators.
 * @ingroup numerics
 */
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/profiler.hpp>
#include <memory>
#include <span>
#include <stddef.h>
#include <vector>

namespace cosmictiger::gravity::cartesian {

using Vector = std::array<double, NDIM>;
using Offset = std::array<int, NDIM>;
using Coefficients = std::vector<double>;
using Field = std::vector<Coefficients>;

inline constexpr int kMinOrder = stf::kMinOrder < 1 ? 1 : stf::kMinOrder;
inline constexpr int kMaxOrder = stf::kMaxOrder;

constexpr bool validOrder(int order) {
    PROFILE();
    return order >= kMinOrder && order <= kMaxOrder;
}

/// Independent raw STF tensor index. Rank n stores z=0 followed by z=1.
class Index {
public:
    constexpr Index(int x, int y, int z) noexcept
      : x_(x)
      , y_(y)
      , z_(z) {
        PROFILE();
    }

    constexpr int x() const noexcept {
        PROFILE();
        return x_;
    }

    constexpr int y() const noexcept {
        PROFILE();
        return y_;
    }

    constexpr int z() const noexcept {
        PROFILE();
        return z_;
    }

    constexpr int degree() const noexcept {
        PROFILE();
        return x_ + y_ + z_;
    }

private:
    int x_;
    int y_;
    int z_;
};

/// Storage includes all ranks zero through p, even with nonzero source dipoles.
int coefficientCount(int p);
std::vector<Index> const& indices(int p);

/// Raw STF moments of one source at offset, already normalized by source scale.
Coefficients point(double mass, Vector const& offset, int p);

/// offset=(old center-new center)/new scale; ratio=old scale/new scale.
/// A single input mass is accepted as a point at the old center.
Coefficients shiftMultipole(Coefficients const&, Vector const& offset, double ratio, int p);

/// offset=(new center-old center)/old scale; ratio=new scale/old scale.
/// An optional final scalar is the normalized local Laplacian, not a tensor entry.
Coefficients shiftLocal(Coefficients const&, Vector const& offset, double ratio, int p);

/// Reconstruct a local derivative by STF trace, including an optional Laplacian.
double derivative(Coefficients const&, int x, int y, int z);

/// Potential and normalized-coordinate derivatives of the represented polynomial.
std::array<double, NDIM + 1> evaluate(Coefficients const&, Vector const& offset, int p);

/// Physical separation and independent source/target normalization lengths.
void addM2L(
    Coefficients& local, Coefficients const& source, Vector const& separation, double sourceScale, double targetScale, int order, bool centered = false);

struct M2lInteraction {
    size_t sourceIndex;
    Vector separation;
    double sourceScale;
    bool centered = false;
};

struct M2lTarget {
    std::span<M2lInteraction const> interactions;
    double targetScale;
};

/// Add ordered interactions to one preallocated local expansion per target.
/// Source and local must be distinct Fields; local.size() must equal targets.size().
/// Sources and interaction spans remain immutable and live until this call returns.
/// Tasks own disjoint target rows, preserve the supplied interaction order, and
/// accumulate into existing coefficients without changing the trace. Empty lists
/// leave their rows unchanged. The minimum chunk of 16 targets is an untuned heuristic.
/// All launched tasks finish before return or rethrow; failures can leave partial sums.
/// Outside an HPX task or with one worker, evaluation is synchronous.
void addM2LTargets(Field& local, Field const& source, std::span<M2lTarget const> targets, int order);

/// A lightweight cached geometry plus runtime dispatch to generated STF M2L.
/// Both source and local order are p; only sourceRank+localRank<=p is retained.
/// Geometric expansion centers are supported: the source dipole is retained.
class Operator {
public:
    Operator(int p, Offset separation);

    /// separation=(target center-source center)/cellWidth. Both expansions use
    /// cellWidth as their scale. An extra source trace is irrelevant to the
    /// exterior Newtonian kernel; an extra local Laplacian is preserved.
    void add(Coefficients& local, Coefficients const& moment, double cellWidth) const;

private:
    using Apply = void (*)(Coefficients&, Coefficients const&, Vector const&, double, double);
    int count_;
    Vector direction_;
    double inverseLatticeRadius_;
    Apply apply_;
};

std::shared_ptr<Operator const> getOperator(int p, Offset separation);
size_t cachedOperatorCount();

/// Exact non-self P2P contribution to potential and first local derivatives only.
void addDirect(Coefficients& local, double mass, Offset separation, double cellWidth);

}    // namespace cosmictiger::gravity::cartesian
