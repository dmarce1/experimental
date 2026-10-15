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
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/spatial_types.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <atomic>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#endif

#include <hpx/parallel/algorithms/uninitialized_value_construct.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#define COSMICTIGER_RUNG_HOST_DEVICE __host__ __device__
#else
#define COSMICTIGER_RUNG_HOST_DEVICE
#endif

namespace cosmictiger {

using ParticleCount = std::uint64_t;

// A remote ghost is a copy whose owner is another locality. Ownership
// migration and periodic copies owned by this locality do not set isGhost.
class Rung {
public:
    COSMICTIGER_RUNG_HOST_DEVICE static constexpr std::uint8_t maximumRung() noexcept {
        PROFILE();
        return kMaximumRung_;
    }

    COSMICTIGER_RUNG_HOST_DEVICE Rung() = default;

    COSMICTIGER_RUNG_HOST_DEVICE Rung(std::uint64_t rungValue, bool ghost) {
        PROFILE();
        setRung(rungValue);
        setGhost(ghost);
    }

    COSMICTIGER_RUNG_HOST_DEVICE std::uint8_t rung() const noexcept {
        PROFILE();
        return bits_ & kRungMask_;
    }

    COSMICTIGER_RUNG_HOST_DEVICE void setRung(std::uint64_t rungValue) {
        PROFILE();
        expectRange(0, rungValue, kMaximumRung_);
        bits_ = static_cast<std::uint8_t>((bits_ & (kGhostMask_ | kReservedMask_)) | rungValue);
    }

    COSMICTIGER_RUNG_HOST_DEVICE bool isGhost() const noexcept {
        PROFILE();
        return (bits_ & kGhostMask_) != 0;
    }

    COSMICTIGER_RUNG_HOST_DEVICE void setGhost(bool ghost) noexcept {
        PROFILE();
        bits_ = static_cast<std::uint8_t>((bits_ & (kRungMask_ | kReservedMask_)) | (ghost ? kGhostMask_ : 0));
    }

private:
    static constexpr std::uint8_t kMaximumRung_ = 63;
    static constexpr std::uint8_t kRungMask_ = 0x3f;
    static constexpr std::uint8_t kGhostMask_ = 0x40;
    static constexpr std::uint8_t kReservedMask_ = 0x80;

    std::uint8_t bits_ = 0;
};
static_assert(sizeof(Rung) == 1 && std::is_trivially_copyable_v<Rung>);

#undef COSMICTIGER_RUNG_HOST_DEVICE

namespace detail {

    template <typename T>
    size_t particleAllocationBytes(ParticleCount count) {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
        PROFILE();
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::length_error("particle field exceeds addressable memory");
        }
        return static_cast<size_t>(count) * sizeof(T);
    }

    template <typename T>
    class ParticleField {
    public:
        ParticleField() = default;
        ParticleField(ParticleField const&) = delete;
        ParticleField& operator=(ParticleField const&) = delete;

        ~ParticleField() {
            PROFILE();
            release(values_);
        }

        T* data() noexcept {
            PROFILE();
            return values_;
        }

        T const* data() const noexcept {
            PROFILE();
            return values_;
        }

        void swap(ParticleField& other) noexcept {
            PROFILE();
            std::swap(values_, other.values_);
            std::swap(allocatedCapacity_, other.allocatedCapacity_);
        }

        void reserve(ParticleCount newCapacity, ParticleCount liveSize) {
            PROFILE();
            if (newCapacity <= allocatedCapacity_) {
                return;
            }
            size_t const bytes = particleAllocationBytes<T>(newCapacity);
            T* replacement = nullptr;
#ifdef COSMICTIGER_ENABLE_CUDA
            replacement = static_cast<T*>(cuda::mallocManaged(bytes).get());
#else
            replacement = static_cast<T*>(std::malloc(bytes));
            if (replacement == nullptr) {
                throw std::bad_alloc();
            }
#endif
            if (liveSize != 0) {
                // memcpy creates the trivially copyable objects in the new field.
                std::memcpy(replacement, values_, particleAllocationBytes<T>(liveSize));
            }
            release(values_);
            values_ = replacement;
            allocatedCapacity_ = newCapacity;
        }

        void initialize(ParticleCount first, ParticleCount count) noexcept {
            PROFILE();
            if (count != 0) {
                hpx::uninitialized_value_construct_n(values_ + static_cast<size_t>(first), static_cast<size_t>(count));
            }
        }

    private:
        T* values_ = nullptr;
        ParticleCount allocatedCapacity_ = 0;

        static void release(T* pointer) noexcept {
            PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
            if (pointer != nullptr) {
                // Destruction cannot report CUDA errors; callers must finish GPU
                // work before resizing, removing particles, or destroying storage.
                try {
                    cuda::free(pointer).get();
                } catch (...) {
                }
            }
#else
            std::free(pointer);
#endif
        }
    };

}    // namespace detail

// Local SoA owner. Mutation requires exclusive access and completed GPU work.
// Individual fields grow in sequence; failed growth preserves live particles
// and the previous reported capacity. Fields already grown are retained.
class ParticleStorage {
public:
    using Size = ParticleCount;
    using Coordinate = cosmictiger::Coordinate;

    static constexpr bool boundsChecking() noexcept {
        PROFILE();
        return validationChecksEnabled();
    }

    explicit ParticleStorage(Size initialSize = 0) {
        PROFILE();
        resize(initialSize);
    }

    ParticleStorage(ParticleStorage const&) = delete;
    ParticleStorage& operator=(ParticleStorage const&) = delete;

    ParticleStorage(ParticleStorage&& other) noexcept {
        PROFILE();
        swap(other);
    }

    ParticleStorage& operator=(ParticleStorage&& other) noexcept {
        PROFILE();
        if (this != &other) {
            ParticleStorage old(std::move(other));
            swap(old);
        }
        return *this;
    }

    Size size() const noexcept {
        PROFILE();
        return particleSize_;
    }

    Size capacity() const noexcept {
        PROFILE();
        return particleCapacity_;
    }

    bool storesAcceleration() const noexcept {
        PROFILE();
        return storeAcceleration_;
    }

    bool storesPotential() const noexcept {
        PROFILE();
        return storePotential_;
    }

    // Output fields are opt-in. Newly enabled fields start at zero; disabling
    // a field releases its allocation. Reconfiguration preserves enabled data.
    void configureGravityFields(bool acceleration, bool potential) {
        PROFILE();
        std::array<detail::ParticleField<float>, NDIM> accelerationFields;
        detail::ParticleField<float> potentialField;
        if (acceleration && !storeAcceleration_) {
            for (auto& field : accelerationFields) {
                field.reserve(particleCapacity_, 0);
                field.initialize(0, particleSize_);
            }
        }
        if (potential && !storePotential_) {
            potentialField.reserve(particleCapacity_, 0);
            potentialField.initialize(0, particleSize_);
        }
        if (acceleration != storeAcceleration_) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                accelerations_[axis].swap(accelerationFields[axis]);
            }
        }
        if (potential != storePotential_) {
            potentials_.swap(potentialField);
        }
        storeAcceleration_ = acceleration;
        storePotential_ = potential;
    }

    // Empty storage has zero capacity; every nonzero capacity is 2^m.
    static Size capacityForSize(Size requestedSize) {
        PROFILE();
        Size result = 1;
        if (requestedSize == 0) {
            return 0;
        }
        while (result < requestedSize) {
            if (result > std::numeric_limits<Size>::max() / 2) {
                throw std::length_error("particle power-of-two capacity overflows");
            }
            result *= 2;
        }
        return result;
    }

    void reserve(Size requestedCapacity) {
        PROFILE();
        if (requestedCapacity <= particleCapacity_) {
            return;
        }
        Size const newCapacity = capacityForSize(requestedCapacity);
        // Check every field before allocating or changing any pointers.
        (void) detail::particleAllocationBytes<Coordinate>(newCapacity);
        (void) detail::particleAllocationBytes<float>(newCapacity);
        (void) detail::particleAllocationBytes<Rung>(newCapacity);
        for (auto& field : positions_) {
            field.reserve(newCapacity, particleSize_);
        }
        for (auto& field : velocities_) {
            field.reserve(newCapacity, particleSize_);
        }
        rungs_.reserve(newCapacity, particleSize_);
        if (storeAcceleration_) {
            for (auto& field : accelerations_) {
                field.reserve(newCapacity, particleSize_);
            }
        }
        if (storePotential_) {
            potentials_.reserve(newCapacity, particleSize_);
        }
        particleCapacity_ = newCapacity;
    }

    void resize(Size newSize) {
        PROFILE();
        reserve(newSize);
        if (newSize > particleSize_) {
            for (auto& field : positions_) {
                field.initialize(particleSize_, newSize - particleSize_);
            }
            for (auto& field : velocities_) {
                field.initialize(particleSize_, newSize - particleSize_);
            }
            rungs_.initialize(particleSize_, newSize - particleSize_);
            if (storeAcceleration_) {
                for (auto& field : accelerations_) {
                    field.initialize(particleSize_, newSize - particleSize_);
                }
            }
            if (storePotential_) {
                potentials_.initialize(particleSize_, newSize - particleSize_);
            }
        }
        particleSize_ = newSize;
    }

    void clear() noexcept {
        PROFILE();
        particleSize_ = 0;
    }

    Size append(std::array<Coordinate, NDIM> positionValues, std::array<float, NDIM> velocityValues, std::uint64_t rungValue, bool isGhost = false) {
        PROFILE();
        expectRange(0, rungValue, Rung::maximumRung());
        if (particleSize_ == std::numeric_limits<Size>::max()) {
            throw std::length_error("particle size overflows");
        }
        Size const index = particleSize_;
        resize(index + 1);
        for (size_t axis = 0; axis < NDIM; axis++) {
            position(axis, index) = positionValues[axis];
            velocity(axis, index) = velocityValues[axis];
        }
        flags(index) = Rung{rungValue, isGhost};
        return index;
    }

    void swapParticles(Size first, Size second) {
        PROFILE();
        expectIndex(first, particleSize_);
        expectIndex(second, particleSize_);
        if (first == second) {
            return;
        }
        size_t const firstOffset = static_cast<size_t>(first);
        size_t const secondOffset = static_cast<size_t>(second);
        for (size_t axis = 0; axis < NDIM; axis++) {
            Coordinate const firstPosition = positions_[axis].data()[firstOffset];
            positions_[axis].data()[firstOffset] = positions_[axis].data()[secondOffset];
            positions_[axis].data()[secondOffset] = firstPosition;
            float const firstVelocity = velocities_[axis].data()[firstOffset];
            velocities_[axis].data()[firstOffset] = velocities_[axis].data()[secondOffset];
            velocities_[axis].data()[secondOffset] = firstVelocity;
            if (storeAcceleration_) {
                std::swap(accelerations_[axis].data()[firstOffset], accelerations_[axis].data()[secondOffset]);
            }
        }
        if (storePotential_) {
            std::swap(potentials_.data()[firstOffset], potentials_.data()[secondOffset]);
        }
        Rung const firstFlags = rungs_.data()[firstOffset];
        rungs_.data()[firstOffset] = rungs_.data()[secondOffset];
        rungs_.data()[secondOffset] = firstFlags;
    }

    // Unordered removal: replace the removed particle with the last particle.
    void remove(Size index) {
        PROFILE();
        checkIndex(index);
        Size const last = particleSize_ - 1;
        if (index != last) {
            for (size_t axis = 0; axis < NDIM; axis++) {
                positions_[axis].data()[static_cast<size_t>(index)] = positions_[axis].data()[static_cast<size_t>(last)];
                velocities_[axis].data()[static_cast<size_t>(index)] = velocities_[axis].data()[static_cast<size_t>(last)];
                if (storeAcceleration_) {
                    accelerations_[axis].data()[static_cast<size_t>(index)] = accelerations_[axis].data()[static_cast<size_t>(last)];
                }
            }
            if (storePotential_) {
                potentials_.data()[static_cast<size_t>(index)] = potentials_.data()[static_cast<size_t>(last)];
            }
            rungs_.data()[static_cast<size_t>(index)] = rungs_.data()[static_cast<size_t>(last)];
        }
        --particleSize_;
    }

    Coordinate& position(size_t axis, Size index) {
        PROFILE();
        checkIndex(index);
        return positionData(axis)[static_cast<size_t>(index)];
    }

    Coordinate const& position(size_t axis, Size index) const {
        PROFILE();
        checkIndex(index);
        return positionData(axis)[static_cast<size_t>(index)];
    }

    float& velocity(size_t axis, Size index) {
        PROFILE();
        checkIndex(index);
        return velocityData(axis)[static_cast<size_t>(index)];
    }

    float const& velocity(size_t axis, Size index) const {
        PROFILE();
        checkIndex(index);
        return velocityData(axis)[static_cast<size_t>(index)];
    }

    float& acceleration(size_t axis, Size index) {
        PROFILE();
        checkIndex(index);
        expect(storeAcceleration_);
        return accelerationData(axis)[static_cast<size_t>(index)];
    }

    float const& acceleration(size_t axis, Size index) const {
        PROFILE();
        checkIndex(index);
        expect(storeAcceleration_);
        return accelerationData(axis)[static_cast<size_t>(index)];
    }

    float& potential(Size index) {
        PROFILE();
        checkIndex(index);
        expect(storePotential_);
        return potentialData()[static_cast<size_t>(index)];
    }

    float const& potential(Size index) const {
        PROFILE();
        checkIndex(index);
        expect(storePotential_);
        return potentialData()[static_cast<size_t>(index)];
    }

    Rung& flags(Size index) {
        PROFILE();
        checkIndex(index);
        return rungs_.data()[static_cast<size_t>(index)];
    }

    Rung const& flags(Size index) const {
        PROFILE();
        checkIndex(index);
        return rungs_.data()[static_cast<size_t>(index)];
    }

    // Raw pointers for kernels/communication bypass index bounds checks and
    // must be reacquired after reserve/resize/append, including failed growth.
    Coordinate* positionData(size_t axis) {
        PROFILE();
        checkAxis(axis);
        return positions_[axis].data();
    }

    Coordinate const* positionData(size_t axis) const {
        PROFILE();
        checkAxis(axis);
        return positions_[axis].data();
    }

    float* velocityData(size_t axis) {
        PROFILE();
        checkAxis(axis);
        return velocities_[axis].data();
    }

    float const* velocityData(size_t axis) const {
        PROFILE();
        checkAxis(axis);
        return velocities_[axis].data();
    }

    float* accelerationData(size_t axis) {
        PROFILE();
        checkAxis(axis);
        return accelerations_[axis].data();
    }

    float const* accelerationData(size_t axis) const {
        PROFILE();
        checkAxis(axis);
        return accelerations_[axis].data();
    }

    float* potentialData() noexcept {
        PROFILE();
        return potentials_.data();
    }

    float const* potentialData() const noexcept {
        PROFILE();
        return potentials_.data();
    }

    Rung* rungData() noexcept {
        PROFILE();
        return rungs_.data();
    }

    Rung const* rungData() const noexcept {
        PROFILE();
        return rungs_.data();
    }

    void swap(ParticleStorage& other) noexcept {
        PROFILE();
        for (size_t axis = 0; axis < NDIM; axis++) {
            positions_[axis].swap(other.positions_[axis]);
            velocities_[axis].swap(other.velocities_[axis]);
            accelerations_[axis].swap(other.accelerations_[axis]);
        }
        potentials_.swap(other.potentials_);
        std::swap(storeAcceleration_, other.storeAcceleration_);
        std::swap(storePotential_, other.storePotential_);
        rungs_.swap(other.rungs_);
        std::swap(particleSize_, other.particleSize_);
        std::swap(particleCapacity_, other.particleCapacity_);
    }

private:
    std::array<detail::ParticleField<Coordinate>, NDIM> positions_;
    std::array<detail::ParticleField<float>, NDIM> velocities_;
    std::array<detail::ParticleField<float>, NDIM> accelerations_;
    detail::ParticleField<float> potentials_;
    detail::ParticleField<Rung> rungs_;
    bool storeAcceleration_ = false;
    bool storePotential_ = false;
    Size particleSize_ = 0;
    Size particleCapacity_ = 0;

    void checkIndex(Size index) const {
        PROFILE();
        expectIndex(index, particleSize_);
    }

    static void checkAxis(size_t axis) {
        PROFILE();
        expectIndex(axis, NDIM);
    }
};

// Each process hosting an HPX locality owns an independent storage instance.
// Initialize its particle count with resize() during locality startup.
inline ParticleStorage& localParticleStorage() {
    PROFILE();
    static ParticleStorage particles;
#ifdef COSMICTIGER_ENABLE_CUDA
    // Static storage outlives the runtime. Release its managed fields while HPX
    // can still execute the future-based CUDA cleanup. Registration occurs
    // outside static initialization and without an application lock.
    static std::atomic<bool> cleanupRegistered{false};
    if (!cleanupRegistered.exchange(true)) {
        try {
            hpx::register_pre_shutdown_function(hpx::annotated_function(
                [] {
                    PROFILE();
                    {
                        ParticleStorage empty;
                        particles.swap(empty);
                    }
                    cleanupRegistered.store(false);
                },
                "particleStorage.register_pre_shutdown_function"));
        } catch (...) {
            cleanupRegistered.store(false);
            throw;
        }
    }
#endif
    return particles;
}

}    // namespace cosmictiger
