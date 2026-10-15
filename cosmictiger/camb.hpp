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

#include <hpx/modules/filesystem.hpp>

namespace cosmictiger {

class CambSolver final : public EinsteinBoltzmannSolver {
public:
    explicit CambSolver(hpx::filesystem::path executable = defaultExecutable());

    static hpx::filesystem::path defaultExecutable();
    hpx::filesystem::path const& executable() const noexcept;

    // Call from an HPX task. File I/O and the separate CAMB process run on
    // HPX's I/O pool, leaving worker threads available. Each CAMB run uses
    // one OpenMP thread and a unique temporary directory. Failed runs retain
    // their input/output/log files; successful runs remove them.
    MatterPowerSpectrum compute(CosmologyParameters const& parameters, MatterPowerRequest const& request = {}) const override;

private:
    hpx::filesystem::path executable_;
};

}    // namespace cosmictiger
