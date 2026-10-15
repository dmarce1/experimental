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

#include <cosmictiger/gravity/stf/generated/generated_config.hpp>

#ifndef COSMICTIGER_MULTIPOLE_ORDER
#define COSMICTIGER_MULTIPOLE_ORDER 8
#endif
#ifndef COSMICTIGER_EXPANSION_ORDER
#define COSMICTIGER_EXPANSION_ORDER 9
#endif

namespace cosmictiger::gravity {

inline constexpr int kMultipoleOrder = COSMICTIGER_MULTIPOLE_ORDER;
inline constexpr int kExpansionOrder = COSMICTIGER_EXPANSION_ORDER;
// FMM order p uses source rank p-1 and local rank p. The periodic
// background requires quadratic locals; fixed cell arrays bound the range.
inline constexpr int kMinimumFmmOrder = 2;
inline constexpr int kDefaultFmmOrder = 6;
inline constexpr int kMaximumFmmOrder = kExpansionOrder < kMultipoleOrder + 1 ? kExpansionOrder : kMultipoleOrder + 1;
// Default only: runtime gravity settings select the active near-cell width.
inline constexpr int kPmmmNeighborWidth = 2;
inline constexpr int kPmmmMaximumNeighborWidth = 2;
// Tree nodes with at most this many particles are not split. CUDA FMM kernels
// size their per-bucket shared memory from it.
inline constexpr size_t kBucketSize = 128;
// The near region (the cells excluded from the PMMM far field) is evaluated
// either by direct P2P over every near cell or by a local FMM walk over the
// per-cell kd-trees with this default opening angle.
enum class NearFieldMethod { direct, tree };
inline constexpr double kDefaultOpeningAngle = 0.55;
static_assert(kMultipoleOrder >= stf::kMinOrder && kMultipoleOrder <= stf::kMaxOrder, "Cell multipole order is outside the generated range");
static_assert(kExpansionOrder >= stf::kMinOrder && kExpansionOrder <= stf::kMaxOrder, "Cell expansion order is outside the generated range");
static_assert(kMaximumFmmOrder == stf::kMaxOrder, "Every build must support all FMM orders through nine");

}    // namespace cosmictiger::gravity
