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

#include <cosmictiger/box.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/stf/m2l.hpp>
#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/particle/particle_storage.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger {

// Half-open indices into a ParticleStorage, not pointers or particle payloads.
struct ParticleRange {
    ParticleCount begin = 0;
    ParticleCount end = 0;
};

using CellMultipole = gravity::stf::Multipole<float, gravity::kMultipoleOrder>;
using CellExpansion = gravity::stf::Expansion<float, gravity::kExpansionOrder, true>;
using FullCellMultipole = gravity::symmetric::Multipole<float, gravity::kMultipoleOrder>;
using FullCellExpansion = gravity::symmetric::Expansion<float, gravity::kExpansionOrder>;
using CellBox = Box<double>;

struct Cell {
    ParticleRange localParticles;
    ParticleRange ownerParticles;
    // Half-open ranges into local ParticleStorage, indexed by rung 0..63.
    // Valid after sortCellParticlesByRung() or sortParticlesByRung().
    std::array<ParticleRange, Rung::maximumRung() + 1> rungParticles{};
    bool rungRangesValid = true;
    CellBox physicalBox;
    // Raw STF moments / spacing^rank and potential derivatives * spacing^rank,
    // both about layout().cellCenter(), including the nonzero source dipole.
    CellMultipole multipole{};
    // Sum of mass * |particle offset / spacing|^2. The STF multipole omits
    // this trace, which the periodic background needs.
    double secondMomentTrace = 0.0;
    CellExpansion expansion{};
    // Full Cartesian tensors preserve the nonzero traces of Plummer gravity.
    FullCellMultipole fullMultipole{};
    FullCellExpansion fullExpansion{};
};

using GridIndices = std::array<std::int64_t, NDIM>;

struct GridCellOwner {
    size_t localityIndex;
    GridIndices coordinates;
};

// Geometry for the periodic unit cube [0,1)^3. Locality indices refer to the
// same ordered HPX locality list on every process. Both cells and locality
// blocks use x-fastest ordering. No global grid or locality table is allocated.
class CartesianGridLayout {
public:
    CartesianGridLayout(size_t cellExponent, size_t localityExponent, size_t localityIndex, size_t ghostWidth = 1)
      : globalSide_(powerOfEightSide(cellExponent))
      , localitySide_(powerOfEightSide(localityExponent))
      , rank_(localityIndex)
      , ghostWidth_(ghostWidth) {
        PROFILE();
        expectRange(0, localityExponent, cellExponent);
        expectIndex(rank_, localityCount());
        positionShift_ = Coordinate::fractionalBits() - cellExponent;
        blockShift_ = cellExponent - localityExponent;
        localSide_ = globalSide_ / localitySide_;
        if (ghostWidth_ == 0 || ghostWidth_ > globalSide_) {
            throw std::invalid_argument("grid ghost width must be between one and the global side length");
        }
        storedSide_ = localSide_ + 2 * ghostWidth_;
        storedCount_ = checkedCube(storedSide_);
        auto const block = localityCoordinates();
        for (size_t axis = 0; axis < NDIM; axis++) {
            origin_[axis] = block[axis] * static_cast<std::int64_t>(localSide_);
        }
    }

    size_t globalSideLength() const noexcept {
        PROFILE();
        return globalSide_;
    }

    size_t localitiesPerAxis() const noexcept {
        PROFILE();
        return localitySide_;
    }

    size_t localityIndex() const noexcept {
        PROFILE();
        return rank_;
    }

    size_t localityCount() const noexcept {
        PROFILE();
        return localitySide_ * localitySide_ * localitySide_;
    }

    size_t globalCellCount() const noexcept {
        PROFILE();
        return globalSide_ * globalSide_ * globalSide_;
    }

    size_t interiorSideLength() const noexcept {
        PROFILE();
        return localSide_;
    }

    size_t interiorCellCount() const noexcept {
        PROFILE();
        return localSide_ * localSide_ * localSide_;
    }

    size_t storedSideLength() const noexcept {
        PROFILE();
        return storedSide_;
    }

    size_t ghostWidth() const noexcept {
        PROFILE();
        return ghostWidth_;
    }

    size_t storedCellCount() const noexcept {
        PROFILE();
        return storedCount_;
    }

    double spacing() const {
        PROFILE();
        FpeGuard guard;
        return 1.0 / static_cast<double>(globalSide_);
    }

    GridIndices localityCoordinates() const noexcept {
        PROFILE();
        return unflatten(rank_, localitySide_);
    }

    GridIndices globalOrigin() const noexcept {
        PROFILE();
        return origin_;
    }

    // Coordinates are [-w,n+w) on each axis; [0,n) is the interior. The whole
    // surrounding shell is present, including all edges and corners.
    bool isInterior(GridIndices coordinates) const {
        PROFILE();
        for (auto const coordinate : coordinates) {
            expectRange(-static_cast<std::int64_t>(ghostWidth_), coordinate, static_cast<std::int64_t>(localSide_ + ghostWidth_ - 1));
            if (coordinate < 0 || coordinate >= static_cast<std::int64_t>(localSide_)) {
                return false;
            }
        }
        return true;
    }

    size_t cellIndex(GridIndices coordinates) const {
        PROFILE();
        size_t index = 0;
        size_t stride = 1;
        for (auto const coordinate : coordinates) {
            index +=
                static_cast<size_t>(expectRange(-static_cast<std::int64_t>(ghostWidth_), coordinate, static_cast<std::int64_t>(localSide_ + ghostWidth_ - 1)) +
                    static_cast<std::int64_t>(ghostWidth_)) *
                stride;
            stride *= storedSide_;
        }
        return index;
    }

    GridIndices cellCoordinates(size_t index) const {
        PROFILE();
        expectIndex(index, storedCount_);
        auto coordinates = unflatten(index, storedSide_);
        for (auto& coordinate : coordinates) {
            coordinate -= static_cast<std::int64_t>(ghostWidth_);
        }
        return coordinates;
    }

    GridIndices globalCoordinates(GridIndices coordinates) const {
        PROFILE();
        auto const side = static_cast<std::int64_t>(globalSide_);
        for (size_t axis = 0; axis < NDIM; axis++) {
            coordinates[axis] =
                expectRange(-static_cast<std::int64_t>(ghostWidth_), coordinates[axis], static_cast<std::int64_t>(localSide_ + ghostWidth_ - 1)) +
                origin_[axis];
            if (coordinates[axis] < 0) {
                coordinates[axis] += side;
            } else {
                if (coordinates[axis] >= side) {
                    coordinates[axis] -= side;
                }
            }
        }
        return coordinates;
    }

    // The returned coordinates always index an interior cell on its owner.
    // A periodic ghost cell can be owned by this locality (including p=0).
    GridCellOwner cellOwner(GridIndices coordinates) const {
        PROFILE();
        auto const global = globalCoordinates(coordinates);
        std::array<size_t, NDIM> block{};
        GridIndices interior{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            block[axis] = static_cast<size_t>(global[axis]) / localSide_;
            interior[axis] = global[axis] % static_cast<std::int64_t>(localSide_);
        }
        return {block[0] + localitySide_ * (block[1] + localitySide_ * block[2]), interior};
    }

    // Exact half-open cell membership, including raw zero and the last raw
    // position before the periodic endpoint. No floating-point conversion.
    GridCellOwner particleOwner(std::array<Coordinate, NDIM> const& position) const noexcept {
        PROFILE();
        std::array<size_t, NDIM> block{};
        GridIndices interior{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            size_t const global = static_cast<size_t>(std::uint64_t(position[axis].raw()) >> positionShift_);
            block[axis] = global >> blockShift_;
            interior[axis] = static_cast<std::int64_t>(global & (localSide_ - 1));
        }
        return {block[0] + localitySide_ * (block[1] + localitySide_ * block[2]), interior};
    }

    // Keep ghost bounds in the image adjacent to this locality. For example,
    // a left seam ghost occupies [-spacing,0), not [1-spacing,1).
    CellBox physicalBox(GridIndices coordinates) const {
        PROFILE();
        FpeGuard guard;
        CellBox::Point lower{};
        CellBox::Point upper{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            std::int64_t const global = origin_[axis] +
                expectRange(-static_cast<std::int64_t>(ghostWidth_), coordinates[axis], static_cast<std::int64_t>(localSide_ + ghostWidth_ - 1));
            lower[axis] = static_cast<double>(global) * spacing();
            upper[axis] = static_cast<double>(global + 1) * spacing();
        }
        return CellBox(lower, upper);
    }

    std::array<double, NDIM> cellCenter(GridIndices coordinates) const {
        PROFILE();
        FpeGuard guard;
        auto const global = globalCoordinates(coordinates);
        std::array<double, NDIM> center{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            center[axis] = (static_cast<double>(global[axis]) + 0.5) * spacing();
        }
        return center;
    }

private:
    size_t globalSide_;
    size_t localitySide_;
    size_t rank_;
    size_t ghostWidth_;
    size_t localSide_;
    size_t storedSide_;
    size_t storedCount_;
    size_t positionShift_;
    size_t blockShift_;
    GridIndices origin_{};

    static size_t powerOfEightSide(size_t exponent) {
        constexpr size_t kMaximumExponent = (std::numeric_limits<size_t>::digits - 1) / NDIM;
        PROFILE();
        // Check before multiplying or shifting. All counts, coordinates, and
        // flattening arithmetic are then representable, including 3m bits.
        if (exponent > kMaximumExponent) {
            throw std::length_error("grid power-of-eight count overflows size_t");
        }
        return size_t(1) << exponent;
    }

    static size_t checkedCube(size_t side) {
        constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
        PROFILE();
        if (side > kMaximum / side || side * side > kMaximum / side) {
            throw std::length_error("grid ghost zone cell count overflows size_t");
        }
        return side * side * side;
    }

    static GridIndices unflatten(size_t index, size_t side) noexcept {
        PROFILE();
        GridIndices coordinates{};
        for (auto& coordinate : coordinates) {
            coordinate = static_cast<std::int64_t>(index % side);
            index /= side;
        }
        return coordinates;
    }
};

// Construct one grid per locality. Cells also store float gravity tensors.
// Geometry computes ownership; no particles are allocated, sorted, copied, or
// exchanged. Mutation requires exclusive access and completed GPU work.
class CartesianGrid {
public:
    CartesianGrid(size_t cellExponent, size_t localityExponent, size_t localityIndex, size_t ghostWidth = 1)
      : geometry_(cellExponent, localityExponent, localityIndex, ghostWidth) {
        PROFILE();
        if (geometry_.storedCellCount() > cells_.max_size()) {
            throw std::length_error("grid ranges exceed addressable memory");
        }
        cells_.resize(geometry_.storedCellCount());
        for (size_t index = 0; index < cells_.size(); index++) {
            auto const coordinates = geometry_.cellCoordinates(index);
            cells_[index].physicalBox = geometry_.physicalBox(coordinates);
            if (!geometry_.isInterior(coordinates)) {
                ghostCells_.push_back(index);
            }
        }
    }

    CartesianGridLayout const& layout() const noexcept {
        PROFILE();
        return geometry_;
    }

    Cell const& cell(GridIndices coordinates) const {
        PROFILE();
        return cells_[geometry_.cellIndex(coordinates)];
    }

    // Mutable gravity access does not expose the particle-range invariants.
    CellExpansion& expansion(GridIndices coordinates) {
        PROFILE();
        return cells_[geometry_.cellIndex(coordinates)].expansion;
    }

    FullCellMultipole& fullMultipole(GridIndices coordinates) {
        PROFILE();
        return cells_[geometry_.cellIndex(coordinates)].fullMultipole;
    }

    FullCellExpansion& fullExpansion(GridIndices coordinates) {
        PROFILE();
        return cells_[geometry_.cellIndex(coordinates)].fullExpansion;
    }

    // Replace moments from localParticles, including ghost copies. A common
    // particle mass is used because ParticleStorage has no per-particle mass.
    // Positions are differenced in double before normalization and float storage.
    // Recomputing moments preserves the local expansion. The bulk operation
    // schedules independent cell chunks and finishes them before returning;
    // accumulation within each cell retains the single-cell particle order.
    void computeMultipole(GridIndices coordinates, ParticleStorage const& particles, float particleMass = 1.0f);
    void computeMultipoles(ParticleStorage const& particles, float particleMass = 1.0f, bool fullMoments = false);
#ifdef COSMICTIGER_ENABLE_CUDA
    // One batched GPU launch computes every stored cell from localParticles,
    // including local ghost copies. Completion and publication precede return.
    void computeMultipolesCuda(ParticleStorage const& particles, float particleMass = 1.0f, int deviceId = 0, bool fullMoments = false);

    void computeMultipolesCuda(float particleMass = 1.0f, int deviceId = 0) {
        PROFILE();
        computeMultipolesCuda(localParticleStorage(), particleMass, deviceId);
    }
#endif

    void computeMultipoles(float particleMass = 1.0f) {
        PROFILE();
        computeMultipoles(localParticleStorage(), particleMass);
    }

    // Interior cells refer twice to the same local ParticleStorage range.
    void setInteriorRange(GridIndices coordinates, ParticleRange range) {
        PROFILE();
        expect(!(!geometry_.isInterior(coordinates)));
        checkRange(range);
        auto& entry = cells_[geometry_.cellIndex(coordinates)];
        entry.localParticles = range;
        entry.ownerParticles = range;
        resetRungRanges(entry);
        entry.multipole = {};
        entry.secondMomentTrace = 0.0;
        entry.expansion = {};
        entry.fullMultipole = {};
        entry.fullExpansion = {};
    }

    // localParticles indexes this locality's ParticleStorage. ownerParticles
    // indexes the ParticleStorage on layout().cellOwner(coordinates).localityIndex.
    // Both ranges remain independent even when periodic wrapping has a local owner.
    void setGhostRanges(GridIndices coordinates, ParticleRange localParticles, ParticleRange ownerParticles) {
        PROFILE();
        expect(!(geometry_.isInterior(coordinates)));
        ghostRangesClear_ = false;
        checkRange(localParticles);
        checkRange(ownerParticles);
        auto& entry = cells_[geometry_.cellIndex(coordinates)];
        entry.localParticles = localParticles;
        entry.ownerParticles = ownerParticles;
        resetRungRanges(entry);
        entry.multipole = {};
        entry.secondMomentTrace = 0.0;
        entry.expansion = {};
        entry.fullMultipole = {};
        entry.fullExpansion = {};
    }

    // Interior ranges partition the owned prefix in x-fastest order. Its
    // endpoint is available directly, without scanning cells or particle flags.
    ParticleCount ownedParticleEnd() const {
        PROFILE();
        GridIndices last{};
        for (auto& coordinate : last) {
            coordinate = static_cast<std::int64_t>(geometry_.interiorSideLength() - 1);
        }
        return cell(last).localParticles.end;
    }

    // Particle removal itself only changes ParticleStorage's logical size.
    // Reset the boundary metadata once; never scan or rewrite interior cells.
    // Gravity tensors remain stale until the following field rebuild.
    void clearGhostRanges() {
        PROFILE();
        if (ghostRangesClear_) {
            return;
        }
        for (size_t const index : ghostCells_) {
            auto& entry = cells_[index];
            entry.localParticles = {};
            entry.ownerParticles = {};
            resetRungRanges(entry);
        }
        ghostRangesClear_ = true;
    }

    // Reorder all seven particle fields together within one cell. Empty rung
    // ranges are positioned at the boundary between neighboring rungs.
    void sortCellParticlesByRung(GridIndices coordinates, ParticleStorage& particles) {
        PROFILE();
        auto& entry = cells_[geometry_.cellIndex(coordinates)];
        sortCellParticlesByRung(entry, particles);
    }

    // Cell ranges must refer to disjoint portions of the local storage. Sort
    // independent cell chunks concurrently and finish them before returning.
    // Optional transient component arrays follow the exact particle permutation.
    // Each supplied array must cover the same index range as particle storage.
    void sortParticlesByRung(ParticleStorage& particles, double* const* companionFields = nullptr);

private:
    CartesianGridLayout geometry_;
    std::vector<Cell> cells_;
    std::vector<size_t> ghostCells_;
    bool ghostRangesClear_ = true;

    static void checkRange(ParticleRange range) {
        PROFILE();
        expectRange(0, range.begin, range.end);
    }

    static void resetRungRanges(Cell& entry) noexcept {
        PROFILE();
        for (auto& range : entry.rungParticles) {
            range = {entry.localParticles.begin, entry.localParticles.begin};
        }
        entry.rungRangesValid = entry.localParticles.begin == entry.localParticles.end;
    }

    static void sortCellParticlesByRung(Cell& entry, ParticleStorage& particles, double* const* companionFields = nullptr) {
        PROFILE();
        ParticleRange const range = entry.localParticles;
        expectRange(0, range.begin, range.end);
        expectRange(0, range.end, particles.size());
        std::array<ParticleCount, Rung::maximumRung() + 1> counts{};
        for (ParticleCount index = range.begin; index < range.end; index++) {
            counts[particles.flags(index).rung()]++;
        }
        std::array<ParticleRange, Rung::maximumRung() + 1> rungRanges{};
        std::array<ParticleCount, Rung::maximumRung() + 1> next{};
        ParticleCount cursor = range.begin;
        for (size_t rung = 0; rung < rungRanges.size(); rung++) {
            next[rung] = cursor;
            cursor += counts[rung];
            rungRanges[rung] = {next[rung], cursor};
        }
        for (size_t rung = 0; rung < rungRanges.size(); rung++) {
            while (next[rung] < rungRanges[rung].end) {
                ParticleCount const index = next[rung];
                size_t const destinationRung = particles.flags(index).rung();
                if (destinationRung == rung) {
                    next[rung]++;
                } else {
                    ParticleCount const destination = next[destinationRung]++;
                    particles.swapParticles(index, destination);
                    if (companionFields != nullptr) {
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            if (companionFields[axis] != nullptr) {
                                double const value = companionFields[axis][index];
                                companionFields[axis][index] = companionFields[axis][destination];
                                companionFields[axis][destination] = value;
                            }
                        }
                    }
                }
            }
        }
        entry.rungParticles = rungRanges;
        entry.rungRangesValid = true;
    }
};

}    // namespace cosmictiger
