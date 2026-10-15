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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/serialize_buffer.hpp>

#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmictiger {

namespace detail {
    using FftPayload = hpx::serialization::serialize_buffer<unsigned char>;
}

FftHpxExchange::FftHpxExchange(std::vector<hpx::id_type> localityIds, std::uint64_t fftOperationId, size_t thisLocalityIndex)
  : localities_(std::move(localityIds))
  , operationId_(fftOperationId)
  , localIndex_(thisLocalityIndex) {
    PROFILE();
    expectIndex(localIndex_, localities_.size());
    expect(localities_[localIndex_] == hpx::find_here());

    // Wait for this endpoint's registration while the name is still live.
    // The asynchronous factory leaves peer lookups pending until first use.
    std::string const name = "cosmictiger_fft_exchange_" + std::to_string(operationId_);
    channel_ = hpx::collectives::create_channel_communicator(
        name.c_str(), hpx::collectives::num_sites_arg(localities_.size()), hpx::collectives::this_site_arg(localIndex_))
                   .get();
}

FftHpxExchange::PendingExchange FftHpxExchange::reserveExchange(size_t partnerId) {
    PROFILE();
    expectIndex(partnerId, localities_.size());
    expectIndex(sequence_, std::numeric_limits<size_t>::max());
    size_t const currentSequence = sequence_++;
    return {channel_, operationId_, currentSequence, partnerId, partnerId == localIndex_, localities_[partnerId] == hpx::find_here()};
}

void FftHpxExchange::completeExchange(PendingExchange const& request, std::span<std::byte const> sendBytes, std::span<std::byte> receiveBytes) {
    PROFILE();
    auto sizeError = [&](size_t receivedBytes) {
        PROFILE();
        return std::runtime_error("FFT HPX exchange operation " + std::to_string(request.operationId) + ", sequence " + std::to_string(request.sequence) +
            " received " + std::to_string(receivedBytes) + " bytes from partner " + std::to_string(request.partnerId) + "; expected " +
            std::to_string(receiveBytes.size()));
    };
    if (request.self) {
        if (sendBytes.size() != receiveBytes.size()) {
            throw sizeError(sendBytes.size());
        }
        if (!sendBytes.empty()) {
            std::memmove(receiveBytes.data(), sendBytes.data(), sendBytes.size());
        }
        return;
    }

    auto const peer = hpx::collectives::that_site_arg(request.partnerId);
    auto const tag = hpx::collectives::tag_arg(request.sequence);

    // Register the receive and schedule the send before waiting. The tag
    // separates successive exchanges on this reusable communicator.
    std::optional<hpx::future<detail::FftPayload>> incoming;
    std::exception_ptr failure;
    try {
        if (!receiveBytes.empty()) {
            incoming.emplace(hpx::collectives::get<detail::FftPayload>(request.channel, peer, tag));
        }
    } catch (...) {
        failure = std::current_exception();
    }

    std::optional<hpx::future<void>> sent;
    try {
        if (!sendBytes.empty()) {
            // The send future keeps a remote referenced payload live until HPX
            // finishes serialization. Same-process channels retain their values
            // after this call returns, so those payloads must own their bytes.
            auto const* data = reinterpret_cast<unsigned char const*>(sendBytes.data());
            auto const mode = request.sameProcess ? detail::FftPayload::copy : detail::FftPayload::reference;
            detail::FftPayload outgoing(data, sendBytes.size(), mode);
            sent.emplace(hpx::collectives::set(request.channel, peer, std::move(outgoing), tag));
        }
    } catch (...) {
        if (!failure) {
            failure = std::current_exception();
        }
    }

    if (sent) {
        try {
            sent->get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (incoming) {
        try {
            auto received = incoming->get();
            if (received.size() != receiveBytes.size()) {
                throw sizeError(received.size());
            }
            std::memcpy(receiveBytes.data(), received.data(), received.size());
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}    // namespace cosmictiger
