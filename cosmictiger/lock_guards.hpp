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

#include <hpx/mutex.hpp>
#include <hpx/shared_mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <concepts>
#include <mutex>
#include <shared_mutex>

namespace cosmictiger {

template <typename Mutex>
concept HpxMutex = std::same_as<Mutex, hpx::mutex> || std::same_as<Mutex, hpx::shared_mutex> || std::same_as<Mutex, hpx::spinlock>;

template <HpxMutex Mutex>
using LockGuard = std::lock_guard<Mutex>;

template <HpxMutex Mutex>
using UniqueLock = std::unique_lock<Mutex>;

template <typename Mutex>
concept HpxSharedMutex = std::same_as<Mutex, hpx::shared_mutex>;

template <HpxSharedMutex Mutex>
using SharedLock = std::shared_lock<Mutex>;

}    // namespace cosmictiger
