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

#include <cosmictiger/profiler.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/collectives/channel_communicator.hpp>
#include <hpx/future.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stddef.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace cosmictiger {

// One instance represents one logical FFT transport endpoint. The same operationId and
// ordered locality list must be used on every participating locality. Distinct
// concurrent FFTs require distinct operation IDs. Submissions on one instance
// must be sequential, and partners must use matching sequence numbers. Async
// submissions reserve their tags before scheduling and may remain in flight
// together; futures may be retrieved in any order. Concurrent calls on the
// same instance are not supported.
// Each side may send and receive different numbers of values, including zero.
// Empty sends generate no HPX message; a nonempty send must match the peer's
// receive span/count. Both spans must remain live, and the send span unmodified,
// until exchange returns. exchangeAsync owns its buffers and channel handle;
// its future may outlive the endpoint and completes after both send and receive
// finish, including on a payload-size error. Missing peer submissions and
// runtime/network failures have no cancellation or recovery protocol.
// One HPX channel communicator is reused with a distinct tag for
// every exchange sequence; HPX retains tag state until the communicator dies.
// The raw payload requires identical T representation on all localities.
class FftHpxExchange {
public:
    FftHpxExchange(std::vector<hpx::id_type> localities, std::uint64_t operationId, size_t localIndex);

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    void exchange(size_t partnerId, std::span<Value const> sendValues, std::span<Value> receiveValues) {
        PROFILE();
        completeExchange(reserveExchange(partnerId), std::as_bytes(sendValues), std::as_writable_bytes(receiveValues));
    }

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    void operator()(size_t partnerId, std::span<Value const> sendValues, std::span<Value> receiveValues) {
        PROFILE();
        exchange(partnerId, sendValues, receiveValues);
    }

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    hpx::future<std::vector<Value>> exchangeAsync(size_t partnerId, std::vector<Value> outgoing, size_t receiveCount) {
        PROFILE();
        try {
            auto request = reserveExchange(partnerId);
            return hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [request = std::move(request), outgoing = std::move(outgoing), receiveCount]() mutable {
                        PROFILE();
                        std::vector<Value> incoming(receiveCount);
                        completeExchange(request, std::as_bytes(std::span<Value const>(outgoing)), std::as_writable_bytes(std::span<Value>(incoming)));
                        return incoming;
                    },
                    "FftHpxExchange complete owned asynchronous exchange"));
        } catch (...) {
            return hpx::make_exceptional_future<std::vector<Value>>(std::current_exception());
        }
    }

private:
    struct PendingExchange {
        hpx::collectives::channel_communicator channel;
        std::uint64_t operationId;
        size_t sequence;
        size_t partnerId;
        bool self;
        bool sameProcess;
    };

    PendingExchange reserveExchange(size_t partnerId);
    static void completeExchange(PendingExchange const& request, std::span<std::byte const> sendBytes, std::span<std::byte> receiveBytes);

    std::vector<hpx::id_type> localities_;
    std::uint64_t operationId_;
    size_t localIndex_;
    size_t sequence_ = 0;
    hpx::collectives::channel_communicator channel_;
};

}    // namespace cosmictiger
