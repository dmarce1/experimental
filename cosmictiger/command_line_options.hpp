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

#include <cosmictiger/global_constants.hpp>

#include <hpx/program_options.hpp>

#include <stddef.h>

namespace cosmictiger {

void addCommandLineOptions(hpx::program_options::options_description& options);
GlobalConstantsData readCommandLineOptions(hpx::program_options::variables_map const& options, size_t localityCount);
// Reject explicit CLI/INI changes to the saved FMM order, mesh, ghost width, and softening model.
void validateGravityRestartOptions(hpx::program_options::variables_map const& options, GlobalConstantsData const& data);
// Preserve saved lightcone settings except fields explicitly supplied in CLI/INI.
void applyLightconeRestartOptions(hpx::program_options::variables_map const& options, GlobalConstantsData& data);

}    // namespace cosmictiger
