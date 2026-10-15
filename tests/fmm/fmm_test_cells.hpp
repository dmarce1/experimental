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

// Shared particle-cell fixtures for the CPU and CUDA FMM interaction tests.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stddef.h>
#include <vector>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/spatial_types.hpp>

namespace cosmictiger::gravity::fmm::test {

inline constexpr double kSpacing = 1.0 / 16.0;
inline constexpr float kParticleMass = 0.375f;
inline constexpr std::uint32_t kTestSeed = 42;

inline double wrappedOffset(Coordinate position, Coordinate center) {
    constexpr double kCoordinateScale = 0x1p-32;
    PROFILE();
    std::uint32_t const wrapped = position.raw() - center.raw();
    return static_cast<double>(static_cast<std::int32_t>(wrapped)) * kCoordinateScale;
}

// A raw fixed-point center keeps edge centers such as 0 exactly representable.
inline Position centerOf(std::array<double, NDIM> const& center) {
    constexpr double kRawScale = 0x1p32;
    PROFILE();
    Position result{};
    Coordinate* const axes[NDIM] = {&result.x, &result.y, &result.z};
    for (size_t axis = 0; axis < NDIM; axis++) {
        double const wrapped = center[axis] - std::floor(center[axis]);
        *axes[axis] = Coordinate::fromRaw(static_cast<std::uint32_t>(static_cast<std::uint64_t>(wrapped * kRawScale)));
    }
    return result;
}

inline std::array<Coordinate, NDIM> axesOf(Position const& position) {
    PROFILE();
    return {position.x, position.y, position.z};
}

// Particles in structure-of-arrays form with the multipole of each cell.
class TestCells {
public:
    // Adds count particles within half a cell of center, wrapped periodically.
    ParticleRange addCell(std::array<double, NDIM> const& center, size_t count) {
        constexpr double kRawScale = 0x1p32;
        PROFILE();
        std::uniform_real_distribution<double> jitter(-0.45 * kSpacing, 0.45 * kSpacing);
        ParticleRange const range{static_cast<ParticleCount>(positions_[0].size()), static_cast<ParticleCount>(positions_[0].size() + count)};
        for (size_t particle = 0; particle < count; particle++) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                double const value = center[axis] + jitter(generator_);
                double const wrapped = value - std::floor(value);
                positions_[axis].push_back(Coordinate::fromRaw(static_cast<std::uint32_t>(static_cast<std::uint64_t>(wrapped * kRawScale))));
            }
        }
        ranges_.push_back(range);
        centers_.push_back(centerOf(center));
        CellMultipole multipole{};
        stf::Multipole<double, kMultipoleOrder> moments{};
        auto const reference = axesOf(centers_.back());
        for (ParticleCount particle = range.begin; particle < range.end; particle++) {
            std::array<double, NDIM> offset{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                offset[axis] = wrappedOffset(positions_[axis][particle], reference[axis]);
            }
            stf::addPoint<double, kMultipoleOrder>(moments, kParticleMass, offset, kSpacing);
        }
        for (size_t coefficient = 0; coefficient < moments.size(); coefficient++) {
            multipole[coefficient] = static_cast<float>(moments[coefficient]);
        }
        multipoles_.push_back(multipole);
        return range;
    }

    std::vector<Coordinate> const& positions(size_t axis) const {
        PROFILE();
        return positions_[axis];
    }

    ParticleRange range(size_t cell) const {
        PROFILE();
        return ranges_[cell];
    }

    Position center(size_t cell) const {
        PROFILE();
        return centers_[cell];
    }

    CellMultipole const& multipole(size_t cell) const {
        PROFILE();
        return multipoles_[cell];
    }

    size_t particleCount() const {
        PROFILE();
        return positions_[0].size();
    }

private:
    std::mt19937_64 generator_{kTestSeed};
    std::array<std::vector<Coordinate>, NDIM> positions_;
    std::vector<ParticleRange> ranges_;
    std::vector<Position> centers_;
    std::vector<CellMultipole> multipoles_;
};

}    // namespace cosmictiger::gravity::fmm::test
