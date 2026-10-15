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

#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmictiger::detail {
void runRemoteFftExchange(std::vector<hpx::id_type> localityIds, std::uint64_t operationId, size_t localIndex) {
    PROFILE();
    size_t const partner = localIndex ^ size_t(1);
    FftHpxExchange transport(std::move(localityIds), operationId, localIndex);
    using Batch = std::array<double, 2>;
    std::vector<Batch> send(2048);
    std::vector<Batch> receive(send.size());
    for (size_t round = 0; round != 8; round++) {
        for (size_t index = 0; index != send.size(); index++) {
            double const value = double((round * 2 + localIndex) * send.size() + index);
            send[index] = {value, -value};
        }
        transport.exchange<Batch>(partner, std::span<Batch const>(send), std::span<Batch>(receive));
        for (size_t index = 0; index != receive.size(); index++) {
            double const expected = double((round * 2 + partner) * receive.size() + index);
            if (receive[index] != Batch{expected, -expected}) {
                throw std::runtime_error("remote FFT exchange corrupted a reference payload");
            }
        }
    }

    std::vector<hpx::future<std::vector<Batch>>> pending;
    for (size_t round = 0; round != 8; round++) {
        std::vector<Batch> outgoing(2048);
        for (size_t index = 0; index != outgoing.size(); index++) {
            double const value = double((round * 2 + localIndex) * outgoing.size() + index);
            outgoing[index] = {value, -value};
        }
        pending.push_back(transport.exchangeAsync<Batch>(partner, std::move(outgoing), 2048));
    }
    std::exception_ptr failure;
    for (size_t round = pending.size(); round-- > 0;) {
        try {
            auto const received = pending[round].get();
            for (size_t index = 0; index != received.size(); index++) {
                double const expected = double((round * 2 + partner) * received.size() + index);
                if (received[index] != Batch{expected, -expected}) {
                    throw std::runtime_error("remote asynchronous FFT exchange corrupted an owned payload");
                }
            }
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
}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::runRemoteFftExchange, cosmictiger_run_remote_fft_exchange_action)

TEST(FftHpxExchange, ConcurrentEndpointsExchangeThroughHpxActions) {
    PROFILE();
    auto const here = hpx::find_here();
    cosmictiger::FftHpxExchange first({here, here}, 5, 0);
    cosmictiger::FftHpxExchange second({here, here}, 5, 1);
    auto runEndpoint = [](cosmictiger::FftHpxExchange& endpoint, size_t partner) {
        PROFILE();
        for (size_t iteration = 0; iteration < 16; iteration++) {
            std::array<size_t, 2> const send{iteration, partner};
            std::array<size_t, 2> receive{};
            endpoint.exchange<size_t>(partner, std::span<size_t const>(send), std::span<size_t>(receive));
            EXPECT_EQ(receive[0], iteration);
            EXPECT_EQ(receive[1], 1 - partner);
        }
    };
    auto firstTask = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                runEndpoint(first, 1);
            },
            "FFT exchange test first endpoint"));
    auto secondTask = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                runEndpoint(second, 0);
            },
            "FFT exchange test second endpoint"));
    firstTask.get();
    secondTask.get();
}

TEST(FftHpxExchange, ReceivesTaggedPayloadsAfterSeveralCompletedSends) {
    PROFILE();
    auto const here = hpx::find_here();
    cosmictiger::FftHpxExchange sender({here, here}, 8, 0);
    cosmictiger::FftHpxExchange receiver({here, here}, 8, 1);
    auto senderTask = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                for (size_t iteration = 0; iteration != 16; iteration++) {
                    std::array<size_t, 2> const send{iteration, iteration + 100};
                    sender.exchange<size_t>(1, std::span<size_t const>(send), {});
                }
            },
            "FFT exchange test send to empty receive"));
    senderTask.get();
    for (size_t iteration = 0; iteration != 16; iteration++) {
        std::array<size_t, 2> receive{};
        receiver.exchange<size_t>(0, {}, std::span<size_t>(receive));
        EXPECT_EQ(receive[0], iteration);
        EXPECT_EQ(receive[1], iteration + 100);
    }
}

TEST(FftHpxExchange, AsyncSubmissionsOwnBuffersAndOutliveEndpointBeforePeerArrives) {
    size_t constexpr kRounds = 12;
    PROFILE();

    auto const here = hpx::find_here();
    std::vector<hpx::future<std::vector<size_t>>> firstResults;
    firstResults.reserve(kRounds);
    {
        cosmictiger::FftHpxExchange first({here, here}, 13, 0);
        for (size_t round = 0; round < kRounds; round++) {
            std::vector<size_t> original{round, round + 100, round + 200};
            if (round % 2 == 0) {
                firstResults.push_back(first.exchangeAsync<size_t>(1, original, 2));
            } else {
                firstResults.push_back(first.exchangeAsync<size_t>(1, std::move(original), 2));
            }
            original.assign(3, size_t(-1));
        }
    }
    // All sends have been submitted and the endpoint/original buffers are
    // gone before the peer exists. Reserved tags must not follow task order.
    cosmictiger::FftHpxExchange second({here, here}, 13, 1);
    std::vector<hpx::future<std::vector<size_t>>> secondResults;
    secondResults.reserve(kRounds);
    for (size_t round = 0; round < kRounds; round++) {
        secondResults.push_back(second.exchangeAsync<size_t>(0, {round + 300, round + 400}, 3));
    }
    for (size_t round = kRounds; round-- > 0;) {
        std::vector<size_t> firstReceived;
        std::vector<size_t> secondReceived;
        EXPECT_NO_THROW(firstReceived = firstResults[round].get());
        EXPECT_NO_THROW(secondReceived = secondResults[round].get());
        EXPECT_EQ(firstReceived, (std::vector<size_t>{round + 300, round + 400}));
        EXPECT_EQ(secondReceived, (std::vector<size_t>{round, round + 100, round + 200}));
    }
}

TEST(FftHpxExchange, AsyncOneWayEmptyAndSelfSubmissionsComplete) {
    constexpr std::array<size_t, 2> kSend{88, 99};
    PROFILE();

    auto const here = hpx::find_here();
    cosmictiger::FftHpxExchange first({here, here}, 14, 0);
    cosmictiger::FftHpxExchange second({here, here}, 14, 1);
    auto firstSend = first.exchangeAsync<size_t>(1, {11, 22, 33}, 0);
    auto firstReceive = first.exchangeAsync<size_t>(1, {}, 2);
    auto firstEmpty = first.exchangeAsync<size_t>(1, {}, 0);
    auto secondReceive = second.exchangeAsync<size_t>(0, {}, 3);
    auto secondSend = second.exchangeAsync<size_t>(0, {44, 55}, 0);
    auto secondEmpty = second.exchangeAsync<size_t>(0, {}, 0);
    EXPECT_TRUE(firstEmpty.get().empty());
    EXPECT_TRUE(secondEmpty.get().empty());
    EXPECT_EQ(firstReceive.get(), (std::vector<size_t>{44, 55}));
    EXPECT_EQ(secondReceive.get(), (std::vector<size_t>{11, 22, 33}));
    EXPECT_TRUE(firstSend.get().empty());
    EXPECT_TRUE(secondSend.get().empty());

    cosmictiger::FftHpxExchange self({here}, 15, 0);
    auto selfValues = self.exchangeAsync<size_t>(0, {66, 77}, 2);
    auto selfEmpty = self.exchangeAsync<size_t>(0, {}, 0);
    EXPECT_TRUE(selfEmpty.get().empty());
    EXPECT_EQ(selfValues.get(), (std::vector<size_t>{66, 77}));
    std::array<size_t, 2> received{};
    self(0, std::span<size_t const>(kSend), std::span<size_t>(received));
    EXPECT_EQ(received, kSend);
}

TEST(FftHpxExchange, AsyncPayloadErrorsStillCompleteSendsAndLaterTags) {
    PROFILE();
    auto const here = hpx::find_here();
    cosmictiger::FftHpxExchange first({here, here}, 16, 0);
    cosmictiger::FftHpxExchange second({here, here}, 16, 1);
    auto badReceive = first.exchangeAsync<size_t>(1, {11, 22}, 1);
    auto goodReceive = second.exchangeAsync<size_t>(0, {33, 44}, 2);
    auto nextFirst = first.exchangeAsync<size_t>(1, {55}, 1);
    auto nextSecond = second.exchangeAsync<size_t>(0, {66}, 1);
    EXPECT_THROW(badReceive.get(), std::runtime_error);
    EXPECT_EQ(goodReceive.get(), (std::vector<size_t>{11, 22}));
    EXPECT_EQ(nextSecond.get(), (std::vector<size_t>{55}));
    EXPECT_EQ(nextFirst.get(), (std::vector<size_t>{66}));

    cosmictiger::FftHpxExchange self({here}, 17, 0);
    auto mismatchedSelf = self.exchangeAsync<size_t>(0, {77}, 2);
    auto nextSelf = self.exchangeAsync<size_t>(0, {88}, 1);
    EXPECT_THROW(mismatchedSelf.get(), std::runtime_error);
    EXPECT_EQ(nextSelf.get(), (std::vector<size_t>{88}));
}

TEST(FftHpxExchange, ConcurrentOperationIdsKeepPayloadsSeparate) {
    PROFILE();
    auto const here = hpx::find_here();
    std::vector<hpx::future<void>> tasks;
    for (std::uint64_t operationId = 9; operationId != 11; operationId++) {
        for (size_t localIndex = 0; localIndex != 2; localIndex++) {
            tasks.emplace_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [here, operationId, localIndex] {
                        PROFILE();
                        cosmictiger::FftHpxExchange endpoint({here, here}, operationId, localIndex);
                        size_t const partner = localIndex ^ size_t(1);
                        std::vector<hpx::future<std::vector<std::uint64_t>>> pending;
                        for (size_t iteration = 0; iteration != 4; iteration++) {
                            pending.push_back(endpoint.exchangeAsync<std::uint64_t>(partner, {operationId, localIndex, iteration}, 3));
                        }
                        for (size_t iteration = pending.size(); iteration-- > 0;) {
                            auto const receive = pending[iteration].get();
                            EXPECT_EQ(receive.size(), size_t(3));
                            EXPECT_EQ(receive[0], operationId);
                            EXPECT_EQ(receive[1], partner);
                            EXPECT_EQ(receive[2], iteration);
                        }
                    },
                    "FFT exchange test concurrent operation locality")));
        }
    }
    for (auto& task : tasks) {
        task.get();
    }
}

TEST(FftHpxExchange, MultipleLogicalPairsSharePhysicalLocality) {
    PROFILE();
    auto const here = hpx::find_here();
    std::vector<hpx::future<void>> tasks;
    for (size_t localIndex = 0; localIndex != 4; localIndex++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [here, localIndex] {
                    PROFILE();
                    cosmictiger::FftHpxExchange endpoint(std::vector<hpx::id_type>(4, here), 11, localIndex);
                    size_t const partner = localIndex ^ size_t(1);
                    std::array<size_t, 2> const send{localIndex, localIndex + 100};
                    std::array<size_t, 2> receive{};
                    endpoint.exchange<size_t>(partner, std::span<size_t const>(send), std::span<size_t>(receive));
                    EXPECT_EQ(receive[0], partner);
                    EXPECT_EQ(receive[1], partner + 100);
                },
                "FFT exchange test zero length locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
}

TEST(FftHpxExchange, ReusesOperationIdAfterSelfOnlyEndpointDestruction) {
    PROFILE();
    auto const here = hpx::find_here();
    for (size_t iteration = 0; iteration != 8; iteration++) {
        cosmictiger::FftHpxExchange endpoint({here}, 12, 0);
        std::array<size_t, 1> const send{iteration};
        std::array<size_t, 1> receive{};
        endpoint.exchange<size_t>(0, std::span<size_t const>(send), std::span<size_t>(receive));
        EXPECT_EQ(receive, send);
    }
}

TEST(FftHpxExchange, OneWayTransferBetweenLogicalEndpoints) {
    constexpr std::array<size_t, NDIM> kSend{11, 22, 33};
    PROFILE();

    auto const here = hpx::find_here();
    cosmictiger::FftHpxExchange sender({here, here}, 6, 0);
    cosmictiger::FftHpxExchange receiver({here, here}, 6, 1);
    std::array<size_t, NDIM> receive{};
    auto sendTask = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                sender.exchange<size_t>(1, std::span<size_t const>(kSend), {});
            },
            "FFT exchange test mismatched sender"));
    auto receiveTask = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                receiver.exchange<size_t>(0, {}, std::span<size_t>(receive));
            },
            "FFT exchange test mismatched receiver"));
    sendTask.get();
    receiveTask.get();
    EXPECT_EQ(receive, kSend);
}

TEST(FftHpxExchange, RemoteReferencePayload) {
    PROFILE();
    auto const localityIds = hpx::find_all_localities();
    if (localityIds.size() != 2) {
        GTEST_SKIP() << "requires two HPX localities";
    }
    auto const here = hpx::find_here();
    auto const local = std::find(localityIds.begin(), localityIds.end(), here);
    ASSERT_NE(local, localityIds.end());
    size_t const localIndex = static_cast<size_t>(local - localityIds.begin());
    size_t const partner = localIndex ^ size_t(1);
    auto remote = hpx::async<cosmictiger_run_remote_fft_exchange_action>(localityIds[partner], localityIds, std::uint64_t(7), partner);
    cosmictiger::detail::runRemoteFftExchange(localityIds, 7, localIndex);
    remote.get();
}
