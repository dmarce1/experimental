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

#include <cosmictiger/dimensions.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace cosmictiger {

struct LightconeOptions {
    bool enabled = false;
    std::uint32_t nside = 64;
    double maximumRadius = 1.0;
    std::array<double, NDIM> observer{};
    std::string outputPrefix = "lightcone";
    double observationTimeYears = 0.0;
};

}    // namespace cosmictiger
