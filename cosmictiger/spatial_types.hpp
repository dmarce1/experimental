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

#include <cosmictiger/fixed.hpp>

#include <cstdint>

namespace cosmictiger {

using Coordinate = Fixed<std::uint32_t>;
static_assert(sizeof(Coordinate) == 4 && sizeof(float) == 4);

struct Position {
    Coordinate x;
    Coordinate y;
    Coordinate z;
};

struct Acceleration {
    float x;
    float y;
    float z;
};

}    // namespace cosmictiger
