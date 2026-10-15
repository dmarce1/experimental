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

#include <cosmictiger/fft/fft.hpp>
#include <cosmictiger/fft/fft_communication_plan.hpp>
#include <cosmictiger/fft/fft_fftw.hpp>
#include <cosmictiger/fft/fft_filters.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/fft/fft_packed_vector_radix.hpp>
#include <cosmictiger/fft/fft_pencil_slab.hpp>
#include <cosmictiger/fft/fft_redistribution.hpp>
#include <cosmictiger/fft/fft_vector_radix.hpp>
#include <cosmictiger/lock_guards.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/mutex.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/type_support/bit_cast.hpp>
#include <memory>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <numbers>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using Batch = std::array<double, 2>;

class SimulatedFftExchange {
public:
    explicit SimulatedFftExchange(size_t localityCount)
      : localityCount_(localityCount) {
        PROFILE();
    }

    template <typename Value>
    void exchange(size_t locality, size_t sequence, size_t partner, std::span<Value const> send, std::span<Value> receive) {
        PROFILE();
        cosmictiger::expectIndex(partner, localityCount_);
        if (partner == locality) {
            cosmictiger::expectEqual(send.size_bytes(), receive.size_bytes());
            if (!send.empty()) {
                std::memcpy(receive.data(), send.data(), send.size_bytes());
            }
            return;
        }
        size_t const first = std::min(locality, partner);
        size_t const second = std::max(locality, partner);
        bool const isFirst = locality == first;
        ExchangeKey const key{sequence, first, second};
        Payload outgoing(send.size_bytes());
        if (!send.empty()) {
            std::memcpy(outgoing.data(), send.data(), send.size_bytes());
        }
        auto packet = findPacket(key);
        auto incoming = packet->promises[isFirst ? 1 : 0].get_future();
        packet->promises[isFirst ? 0 : 1].set_value(std::move(outgoing));
        // Suspend this HPX task until its peer delivers; CTest bounds the test runtime.
        auto const inbound = incoming.get();
        if (inbound.size() != receive.size_bytes()) {
            throw std::runtime_error("simulated FFT exchange payload size mismatch");
        }
        if (!receive.empty()) {
            std::memcpy(receive.data(), inbound.data(), receive.size_bytes());
        }
        {
            cosmictiger::LockGuard lock(mutex_);
            if (++packet->readers == 2) {
                // packet retains ownership so its HPX members die after unlock.
                packets_.erase(key);
            }
        }
    }

private:
    using Payload = std::vector<unsigned char>;
    using ExchangeKey = std::tuple<size_t, size_t, size_t>;

    struct Packet {
        std::array<hpx::promise<Payload>, 2> promises;
        size_t readers = 0;
    };

    std::shared_ptr<Packet> findPacket(ExchangeKey const& key) {
        PROFILE();
        {
            cosmictiger::LockGuard lock(mutex_);
            auto const found = packets_.find(key);
            if (found != packets_.end()) {
                return found->second;
            }
        }
        auto candidate = std::make_shared<Packet>();
        std::shared_ptr<Packet> packet;
        {
            cosmictiger::LockGuard lock(mutex_);
            packet = packets_.try_emplace(key, candidate).first->second;
        }
        return packet;
    }

    size_t localityCount_;
    hpx::mutex mutex_;
    std::map<ExchangeKey, std::shared_ptr<Packet>> packets_;
};

struct FftCaseResult {
    std::vector<Batch> input;
    std::vector<Batch> output;
};

// A peer window cannot finish until all of its submissions have arrived.
// Fulfill in reverse order to exercise completion independently of tag order.
class WindowedFftExchange {
public:
    template <typename Value>
    hpx::future<std::vector<Value>> exchangeAsync(size_t peer, std::vector<Value> outgoing, size_t receiveCount) {
        static_assert(std::is_same_v<Value, size_t>);
        PROFILE();
        EXPECT_EQ(receiveCount, 1);
        EXPECT_EQ(outgoing, std::vector<size_t>({peer * 8}));
        peers_.push_back(peer);
        Pending packet{peer, hpx::promise<std::vector<size_t>>()};
        auto received = packet.result.get_future();
        pending_.push_back(std::move(packet));
        maximumWindow_ = std::max(maximumWindow_, pending_.size());
        if (peer == 4 || peer == 7) {
            while (!pending_.empty()) {
                auto next = std::move(pending_.back());
                pending_.pop_back();
                next.result.set_value({next.peer});
            }
        }
        return received;
    }

    std::vector<size_t> const& peers() const {
        PROFILE();
        return peers_;
    }

    size_t maximumWindow() const {
        PROFILE();
        return maximumWindow_;
    }

private:
    struct Pending {
        size_t peer;
        hpx::promise<std::vector<size_t>> result;
    };

    std::vector<Pending> pending_;
    std::vector<size_t> peers_;
    size_t maximumWindow_ = 0;
};

TEST(FftRedistribution, SubmitsBoundedWindowsBeforeWaiting) {
    PROFILE();
    cosmictiger::detail::FftBitLayout const from(6, {0, 1, 2});
    cosmictiger::detail::FftBitLayout const to(6, {3, 4, 5});
    std::vector<size_t> const input{0, 8, 16, 24, 32, 40, 48, 56};
    WindowedFftExchange exchange;
    auto const output = cosmictiger::detail::redistributeFftBits(std::span<size_t const>(input), 0, from, to, 0, 3, exchange);
    EXPECT_EQ(output, std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(exchange.peers(), std::vector<size_t>({1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(exchange.maximumWindow(), 4);
}

TEST(FftwMemoization, ReusesPlansAcrossTransformLifetimesAndSeparatesKeys) {
    constexpr size_t kLength = 8;

    using cosmictiger::detail::FftwTransformKind;
    using cosmictiger::detail::memoizedFftwPlan;
    PROFILE();
    auto const forward = memoizedFftwPlan(kLength, FftwTransformKind::complexForward);
    std::array<fftw_plan, 5> const plans{forward, memoizedFftwPlan(kLength, FftwTransformKind::complexBackward),
        memoizedFftwPlan(kLength, FftwTransformKind::realToComplex), memoizedFftwPlan(kLength, FftwTransformKind::complexToReal),
        memoizedFftwPlan(kLength + 1, FftwTransformKind::complexForward)};
    for (size_t first = 0; first < plans.size(); first++) {
        ASSERT_NE(plans[first], nullptr);
        for (size_t second = first + 1; second < plans.size(); second++) {
            EXPECT_NE(plans[first], plans[second]);
        }
    }
    for (size_t repetition = 0; repetition < 2; repetition++) {
        {
            cosmictiger::FftVectorRadix<1> transform(kLength, 1, 0);
            EXPECT_EQ(transform.localSideLength(), kLength);
            EXPECT_EQ(memoizedFftwPlan(kLength, FftwTransformKind::complexForward), forward);
        }
        EXPECT_EQ(memoizedFftwPlan(kLength, FftwTransformKind::complexForward), forward);
        std::array<std::complex<double>, kLength> values{};
        values[0] = {1.0, -2.0};
        auto* data = reinterpret_cast<fftw_complex*>(values.data());
        fftw_execute_dft(forward, data, data);
        for (auto const& value : values) {
            EXPECT_NEAR(value.real(), 1.0, 1e-12);
            EXPECT_NEAR(value.imag(), -2.0, 1e-12);
        }
    }
}

TEST(FftwMemoization, ConcurrentColdLookupsSharePlansAndExecuteIndependentArrays) {
    constexpr size_t kLength = 11;
    constexpr size_t kTaskCount = 16;

    using cosmictiger::detail::FftwTransformKind;
    using cosmictiger::detail::memoizedFftwPlan;
    PROFILE();
    // Other FFT tests use power-of-two lengths, leaving this key cold.
    struct ExecutionResult {
        fftw_plan forward;
        fftw_plan backward;
        std::array<std::complex<double>, kLength> spectrum;
        std::array<std::complex<double>, kLength> restored;
    };
    hpx::promise<void> start;
    auto const ready = start.get_future().share();
    std::vector<hpx::future<ExecutionResult>> tasks;
    tasks.reserve(kTaskCount);
    for (size_t taskIndex = 0; taskIndex < kTaskCount; taskIndex++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [ready, taskIndex] {
                    PROFILE();
                    ready.get();
                    auto const forward = memoizedFftwPlan(kLength, FftwTransformKind::complexForward);
                    auto const backward = memoizedFftwPlan(kLength, FftwTransformKind::complexBackward);
                    std::array<std::complex<double>, kLength> values{};
                    values[taskIndex % kLength] = {static_cast<double>(taskIndex + 1), -0.25};
                    auto* data = reinterpret_cast<fftw_complex*>(values.data());
                    fftw_execute_dft(forward, data, data);
                    auto const spectrum = values;
                    fftw_execute_dft(backward, data, data);
                    return ExecutionResult{forward, backward, spectrum, values};
                },
                "FFT test concurrent memoized plan lookup")));
    }
    start.set_value();
    std::vector<ExecutionResult> results;
    results.reserve(kTaskCount);
    std::exception_ptr firstError;
    for (auto& task : tasks) {
        try {
            results.push_back(task.get());
        } catch (...) {
            if (!firstError) {
                firstError = std::current_exception();
            }
        }
    }
    if (firstError) {
        std::rethrow_exception(firstError);
    }
    for (size_t taskIndex = 0; taskIndex < kTaskCount; taskIndex++) {
        auto const& result = results[taskIndex];
        EXPECT_EQ(result.forward, results.front().forward);
        EXPECT_EQ(result.backward, results.front().backward);
        std::complex<double> const amplitude{static_cast<double>(taskIndex + 1), -0.25};
        for (size_t frequency = 0; frequency < kLength; frequency++) {
            double const angle = -2.0 * std::numbers::pi * static_cast<double>(frequency * (taskIndex % kLength)) / static_cast<double>(kLength);
            auto const expectedSpectrum = amplitude * std::complex<double>{std::cos(angle), std::sin(angle)};
            auto const expectedRestored = frequency == taskIndex % kLength ? amplitude : std::complex<double>{};
            EXPECT_NEAR(std::abs(result.spectrum[frequency] - expectedSpectrum), 0.0, 1e-11);
            EXPECT_NEAR(std::abs(result.restored[frequency] / static_cast<double>(kLength) - expectedRestored), 0.0, 1e-12);
        }
    }
}

TEST(FftwMemoization, RealPlansRoundTripOddEvenAndUnitLengths) {
    using cosmictiger::detail::FftwTransformKind;
    using cosmictiger::detail::memoizedFftwPlan;
    PROFILE();
    for (size_t length : {size_t{1}, size_t{7}, size_t{8}}) {
        auto const forward = memoizedFftwPlan(length, FftwTransformKind::realToComplex);
        auto const backward = memoizedFftwPlan(length, FftwTransformKind::complexToReal);
        std::vector<double> input(length);
        for (size_t index = 0; index < length; index++) {
            input[index] = static_cast<double>((index * 7 + 3) % 13) - 4.0;
        }
        auto const original = input;
        std::vector<std::complex<double>> spectrum(length / 2 + 1);
        std::vector<double> output(length);
        auto* spectrumData = reinterpret_cast<fftw_complex*>(spectrum.data());
        fftw_execute_dft_r2c(forward, input.data(), spectrumData);
        for (size_t frequency = 0; frequency < spectrum.size(); frequency++) {
            std::complex<double> expected{};
            for (size_t index = 0; index < length; index++) {
                double const angle = -2.0 * std::numbers::pi * static_cast<double>(frequency * index) / static_cast<double>(length);
                expected += original[index] * std::complex<double>{std::cos(angle), std::sin(angle)};
            }
            EXPECT_NEAR(std::abs(spectrum[frequency] - expected), 0.0, 1e-11);
        }
        fftw_execute_dft_c2r(backward, spectrumData, output.data());
        for (size_t index = 0; index < length; index++) {
            EXPECT_NEAR(output[index] / static_cast<double>(length), original[index], 1e-12);
        }
        EXPECT_EQ(memoizedFftwPlan(length, FftwTransformKind::realToComplex), forward);
        EXPECT_EQ(memoizedFftwPlan(length, FftwTransformKind::complexToReal), backward);
    }
}

TEST(FftwMemoization, RejectsInvalidLengthAndTransformKind) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    using cosmictiger::detail::FftwTransformKind;
    using cosmictiger::detail::memoizedFftwPlan;
    EXPECT_THROW(memoizedFftwPlan(0, FftwTransformKind::complexForward), std::runtime_error);
    EXPECT_THROW(memoizedFftwPlan(static_cast<size_t>(INT_MAX) + 1, FftwTransformKind::complexForward), std::runtime_error);
    EXPECT_THROW(memoizedFftwPlan(std::numeric_limits<size_t>::max(), FftwTransformKind::realToComplex), std::runtime_error);
    EXPECT_THROW(memoizedFftwPlan(8, static_cast<FftwTransformKind>(-1)), std::runtime_error);
}

TEST(FftExchange, MultiplePeersSuspendWhileWaiting) {
    PROFILE();
    SimulatedFftExchange transport(8);
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality < 8; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality] {
                    PROFILE();
                    for (size_t sequence = 0; sequence < 16; sequence++) {
                        std::array<size_t, 1> const outgoing{locality};
                        std::array<size_t, 1> incoming{};
                        transport.exchange(locality, sequence, locality ^ 1, std::span<size_t const>(outgoing), std::span<size_t>(incoming));
                        EXPECT_EQ(incoming[0], locality ^ 1);
                    }
                },
                "FFT test paired locality exchange")));
    }
    for (auto& task : tasks) {
        task.get();
    }
}

FftCaseResult runPencilConvolution(cosmictiger::FftCommunicationPlan plan, bool asynchronous = false) {
    constexpr size_t kSide = 16;
    constexpr size_t kLocalities = 8;
    constexpr size_t kLocalitySide = 2;
    constexpr size_t kBlockSide = kSide / kLocalitySide;
    constexpr size_t kValuesPerLocality = kBlockSide * kBlockSide * kBlockSide;
    constexpr size_t kTotalValues = kSide * kSide * kSide;
    PROFILE();

    SimulatedFftExchange transport(kLocalities);
    std::vector<Batch> input(kTotalValues);
    std::vector<Batch> output(kTotalValues);
    std::array<std::exception_ptr, kLocalities> errors{};
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    try {
                        cosmictiger::FftPencilSlab<2> fft(kSide, kLocalities, locality, plan);
                        std::vector<Batch> block(kValuesPerLocality);
                        std::vector<Batch> kernel(kValuesPerLocality);
                        size_t const blockX = locality >> 2;
                        size_t const blockY = (locality >> 1) & 1;
                        size_t const blockZ = locality & 1;
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalX = blockX * kBlockSide + x;
                                    size_t const globalY = blockY * kBlockSide + y;
                                    size_t const globalZ = blockZ * kBlockSide + z;
                                    size_t const globalIndex = (globalZ * kSide + globalY) * kSide + globalX;
                                    block[localIndex] = {double((globalIndex * 17 + 3) % 101) / 101.0, double((globalIndex * 13 + 7) % 97) / 97.0};
                                    kernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                    input[globalIndex] = block[localIndex];
                                }
                            }
                        }
                        size_t sequence = 0;
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(locality, sequence++, peer, send, receive);
                            },
                            "FFT test pencil slab synchronous exchange");
                        auto run = [&](auto& endpoint) {
                            PROFILE();
                            auto const spectrum = fft.forward(block, endpoint);
                            auto const multiplier = fft.forward(kernel, endpoint);
                            auto const product = fft.multiply(spectrum, multiplier);
                            return fft.inverse(product, endpoint);
                        };
                        auto const result = [&] {
                            PROFILE();
                            if (asynchronous) {
                                cosmictiger::FftHpxExchange endpoint(std::vector<hpx::id_type>(kLocalities, hpx::find_here()), 130, locality);
                                return run(endpoint);
                            }
                            return run(exchange);
                        }();
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalIndex =
                                        ((blockZ * kBlockSide + z) * kSide + (blockY * kBlockSide + y)) * kSide + blockX * kBlockSide + x;
                                    output[globalIndex] = result[localIndex];
                                }
                            }
                        }
                    } catch (...) {
                        errors[locality] = std::current_exception();
                    }
                },
                "FFT test pencil slab transform locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return {std::move(input), std::move(output)};
}

FftCaseResult runVectorConvolution() {
    constexpr size_t kSide = 16;
    constexpr size_t kLocalities = 8;
    constexpr size_t kBlockSide = kSide / 2;
    constexpr size_t kValuesPerLocality = kBlockSide * kBlockSide * kBlockSide;
    constexpr size_t kTotalValues = kSide * kSide * kSide;
    PROFILE();

    SimulatedFftExchange transport(kLocalities);
    std::vector<Batch> input(kTotalValues);
    std::vector<Batch> output(kTotalValues);
    std::array<std::exception_ptr, kLocalities> errors{};
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    try {
                        cosmictiger::FftVectorRadix<2> fft(kSide, kLocalities, locality);
                        std::vector<Batch> block(kValuesPerLocality);
                        std::vector<Batch> kernel(kValuesPerLocality);
                        size_t const blockX = locality >> 2;
                        size_t const blockY = (locality >> 1) & 1;
                        size_t const blockZ = locality & 1;
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalX = blockX * kBlockSide + x;
                                    size_t const globalY = blockY * kBlockSide + y;
                                    size_t const globalZ = blockZ * kBlockSide + z;
                                    size_t const globalIndex = (globalZ * kSide + globalY) * kSide + globalX;
                                    block[localIndex] = {double((globalIndex * 17 + 3) % 101) / 101.0, double((globalIndex * 13 + 7) % 97) / 97.0};
                                    kernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                    input[globalIndex] = block[localIndex];
                                }
                            }
                        }
                        size_t sequence = 0;
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(locality, sequence++, peer, send, receive);
                            },
                            "FFT test full spectrum synchronous exchange");
                        auto const spectrum = fft.forward(block, exchange);
                        auto const multiplier = fft.forward(kernel, exchange);
                        std::vector<cosmictiger::FftVectorRadix<2>::ComplexBatch> product(spectrum.size());
                        for (size_t index = 0; index != spectrum.size(); index++) {
                            for (size_t lane = 0; lane != 2; lane++) {
                                product[index][lane] = spectrum[index][lane] * multiplier[index][lane];
                            }
                        }
                        auto const result = fft.inverse(product, exchange);
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalIndex =
                                        ((blockZ * kBlockSide + z) * kSide + (blockY * kBlockSide + y)) * kSide + blockX * kBlockSide + x;
                                    output[globalIndex] = result[localIndex];
                                }
                            }
                        }
                    } catch (...) {
                        errors[locality] = std::current_exception();
                    }
                },
                "FFT test full spectrum convolution locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return {std::move(input), std::move(output)};
}

FftCaseResult runPackedConvolution(bool prepositionedMultiplier) {
    constexpr size_t kSide = 16;
    constexpr size_t kLocalities = 8;
    constexpr size_t kBlockSide = kSide / 2;
    constexpr size_t kValuesPerLocality = kBlockSide * kBlockSide * kBlockSide;
    constexpr size_t kTotalValues = kSide * kSide * kSide;
    PROFILE();

    SimulatedFftExchange transport(kLocalities);
    std::vector<Batch> input(kTotalValues);
    std::vector<Batch> output(kTotalValues);
    std::array<std::exception_ptr, kLocalities> errors{};
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    try {
                        cosmictiger::FftPackedVectorRadix<2> fft(kSide, kLocalities, locality);
                        std::vector<Batch> block(kValuesPerLocality);
                        std::vector<Batch> kernel(kValuesPerLocality);
                        std::vector<cosmictiger::FftPackedVectorRadix<2>::ComplexBatch> multiplier(kValuesPerLocality);
                        size_t const blockX = locality >> 2;
                        size_t const blockY = (locality >> 1) & 1;
                        size_t const blockZ = locality & 1;
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalX = blockX * kBlockSide + x;
                                    size_t const globalY = blockY * kBlockSide + y;
                                    size_t const globalZ = blockZ * kBlockSide + z;
                                    size_t const globalIndex = (globalZ * kSide + globalY) * kSide + globalX;
                                    block[localIndex] = {double((globalIndex * 17 + 3) % 101) / 101.0, double((globalIndex * 13 + 7) % 97) / 97.0};
                                    kernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                    input[globalIndex] = block[localIndex];
                                    size_t const frequencyX = blockX + 2 * (x % (kBlockSide / 2)) + (x >= kBlockSide / 2 ? kSide / 2 : 0);
                                    size_t const frequencyY = blockY + 2 * y;
                                    size_t const frequencyZ = blockZ + 2 * z;
                                    double const angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(frequencyX + 2 * frequencyY + 3 * frequencyZ) /
                                        static_cast<double>(kSide);
                                    multiplier[localIndex] = {std::complex<double>(1.0, 0.0), std::complex<double>(std::cos(angle), std::sin(angle))};
                                }
                            }
                        }
                        size_t sequence = 0;
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(locality, sequence++, peer, send, receive);
                            },
                            "FFT test packed spectrum synchronous exchange");
                        auto const spectrum = fft.forward(block, exchange);
                        auto const product = [&]() {
                            PROFILE();
                            if (prepositionedMultiplier) {
                                return fft.multiplyWithPrepositionedMultiplier(spectrum, multiplier, exchange);
                            }
                            auto const kernelSpectrum = fft.forward(kernel, exchange);
                            return fft.multiply(spectrum, kernelSpectrum, exchange);
                        }();
                        auto const result = fft.inverse(product, exchange);
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalIndex =
                                        ((blockZ * kBlockSide + z) * kSide + (blockY * kBlockSide + y)) * kSide + blockX * kBlockSide + x;
                                    output[globalIndex] = result[localIndex];
                                }
                            }
                        }
                    } catch (...) {
                        errors[locality] = std::current_exception();
                    }
                },
                "FFT test packed spectrum convolution locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return {std::move(input), std::move(output)};
}

void expectConvolution(FftCaseResult const& result) {
    constexpr size_t kSide = 16;
    PROFILE();

    double maximumFirstError = 0.0;
    double maximumSecondError = 0.0;
    for (size_t z = 0; z != kSide; z++) {
        for (size_t y = 0; y != kSide; y++) {
            for (size_t x = 0; x != kSide; x++) {
                size_t const index = (z * kSide + y) * kSide + x;
                size_t const shifted = (((z + kSide - 3) % kSide) * kSide + (y + kSide - 2) % kSide) * kSide + (x + kSide - 1) % kSide;
                maximumFirstError = std::max(maximumFirstError, std::abs(result.output[index][0] - result.input[index][0]));
                maximumSecondError = std::max(maximumSecondError, std::abs(result.output[index][1] - result.input[shifted][1]));
            }
        }
    }
    EXPECT_LT(maximumFirstError, 2e-11);
    EXPECT_LT(maximumSecondError, 2e-11);
}

struct FftFilterCaseResult {
    size_t side;
    std::array<std::vector<Batch>, 2> inputs;
    std::array<std::vector<Batch>, 3> outputs;
    std::array<size_t, 8> filterValueCounts{};
    std::array<bool, 8> planUnavailableBeforeCreation{};
    std::array<bool, 8> weightsFrozen{};
    std::array<bool, 8> unknownFilterRejected{};
};

FftFilterCaseResult runStoredFilterConvolutions(
    size_t side, double weightPerByte, double weightPerMessage, cosmictiger::FftCommunicationStrategy expectedStrategy) {
    constexpr size_t kLocalities = 8;
    PROFILE();

    size_t const blockSide = side / 2;
    size_t const valuesPerLocality = blockSide * blockSide * blockSide;
    FftFilterCaseResult result{};
    result.side = side;
    for (auto& input : result.inputs) {
        input.resize(side * side * side);
    }
    for (auto& output : result.outputs) {
        output.resize(side * side * side);
    }
    SimulatedFftExchange transport(kLocalities);
    std::array<std::exception_ptr, kLocalities> errors{};
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    try {
                        cosmictiger::FftFilterBank<2> filters(side, kLocalities, locality, 0.0, 1.0);
                        if (cosmictiger::validationChecksEnabled()) {
                            try {
                                (void) filters.plan();
                            } catch (std::runtime_error const&) {
                                result.planUnavailableBeforeCreation[locality] = true;
                            }
                        }
                        filters.setCommunicationWeights(weightPerByte, weightPerMessage);

                        std::vector<Batch> firstInput(valuesPerLocality);
                        std::vector<Batch> secondInput(valuesPerLocality);
                        std::vector<Batch> shiftedKernel(valuesPerLocality);
                        std::vector<Batch> identityKernel(valuesPerLocality);
                        size_t const blockX = locality >> 2;
                        size_t const blockY = (locality >> 1) & 1;
                        size_t const blockZ = locality & 1;
                        for (size_t z = 0; z != blockSide; z++) {
                            for (size_t y = 0; y != blockSide; y++) {
                                for (size_t x = 0; x != blockSide; x++) {
                                    size_t const localIndex = (z * blockSide + y) * blockSide + x;
                                    size_t const globalX = blockX * blockSide + x;
                                    size_t const globalY = blockY * blockSide + y;
                                    size_t const globalZ = blockZ * blockSide + z;
                                    size_t const globalIndex = (globalZ * side + globalY) * side + globalX;
                                    firstInput[localIndex] = {double((globalIndex * 17 + 3) % 101) / 101.0, double((globalIndex * 13 + 7) % 97) / 97.0};
                                    secondInput[localIndex] = {double((globalIndex * 31 + 5) % 109) / 109.0, double((globalIndex * 19 + 11) % 89) / 89.0};
                                    shiftedKernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                    identityKernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 0 && globalY == 0 && globalZ == 0)};
                                    result.inputs[0][globalIndex] = firstInput[localIndex];
                                    result.inputs[1][globalIndex] = secondInput[localIndex];
                                }
                            }
                        }

                        size_t sequence = 0;
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(locality, sequence++, peer, send, receive);
                            },
                            "FFT test filter bank synchronous exchange");
                        filters.createFilter("shift", shiftedKernel, exchange);
                        filters.createFilter("identity", identityKernel, exchange);
                        std::fill(shiftedKernel.begin(), shiftedKernel.end(), Batch{});
                        std::fill(identityKernel.begin(), identityKernel.end(), Batch{});
                        if (filters.plan().strategy != expectedStrategy) {
                            throw std::runtime_error("filter bank selected an unexpected FFT strategy");
                        }
                        result.filterValueCounts[locality] = filters.filterValueCount("shift");
                        if (filters.filterValueCount("identity") != result.filterValueCounts[locality]) {
                            throw std::runtime_error("named filters have different spectral layouts");
                        }
                        if (cosmictiger::validationChecksEnabled()) {
                            try {
                                (void) filters.filterValueCount("missing");
                            } catch (std::runtime_error const&) {
                                result.unknownFilterRejected[locality] = true;
                            }
                        }
                        filters.setCommunicationWeights(weightPerByte, weightPerMessage);
                        if (cosmictiger::validationChecksEnabled()) {
                            try {
                                filters.setCommunicationWeights(weightPerMessage, weightPerByte);
                            } catch (std::runtime_error const&) {
                                result.weightsFrozen[locality] = true;
                            }
                        }
                        if (filters.plan().strategy != expectedStrategy) {
                            throw std::runtime_error("filter bank changed its FFT strategy after creation");
                        }

                        auto const shiftedFirst = filters.convolve(firstInput, "shift", exchange);
                        auto const identitySecond = filters.convolve(secondInput, "identity", exchange);
                        auto const shiftedSecond = filters.convolve(secondInput, "shift", exchange);
                        for (size_t z = 0; z != blockSide; z++) {
                            for (size_t y = 0; y != blockSide; y++) {
                                for (size_t x = 0; x != blockSide; x++) {
                                    size_t const localIndex = (z * blockSide + y) * blockSide + x;
                                    size_t const globalIndex = ((blockZ * blockSide + z) * side + blockY * blockSide + y) * side + blockX * blockSide + x;
                                    result.outputs[0][globalIndex] = shiftedFirst[localIndex];
                                    result.outputs[1][globalIndex] = identitySecond[localIndex];
                                    result.outputs[2][globalIndex] = shiftedSecond[localIndex];
                                }
                            }
                        }
                    } catch (...) {
                        errors[locality] = std::current_exception();
                    }
                },
                "FFT test filter bank locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return result;
}

void expectStoredFilterConvolutions(FftFilterCaseResult const& result, size_t fullFilterValues) {
    PROFILE();
    for (size_t locality = 0; locality != 8; locality++) {
        if (cosmictiger::validationChecksEnabled()) {
            EXPECT_TRUE(result.planUnavailableBeforeCreation[locality]);
            EXPECT_TRUE(result.weightsFrozen[locality]);
            EXPECT_TRUE(result.unknownFilterRejected[locality]);
        }
        EXPECT_EQ(result.filterValueCounts[locality], result.side == 8 && (locality & 1) != 0 ? 0 : fullFilterValues);
    }
    for (size_t z = 0; z != result.side; z++) {
        for (size_t y = 0; y != result.side; y++) {
            for (size_t x = 0; x != result.side; x++) {
                size_t const index = (z * result.side + y) * result.side + x;
                size_t const shifted = (((z + result.side - 3) % result.side) * result.side + (y + result.side - 2) % result.side) * result.side +
                    (x + result.side - 1) % result.side;
                EXPECT_NEAR(result.outputs[0][index][0], result.inputs[0][index][0], 2e-11);
                EXPECT_NEAR(result.outputs[0][index][1], result.inputs[0][shifted][1], 2e-11);
                EXPECT_NEAR(result.outputs[1][index][0], result.inputs[1][index][0], 2e-11);
                EXPECT_NEAR(result.outputs[1][index][1], result.inputs[1][index][1], 2e-11);
                EXPECT_NEAR(result.outputs[2][index][0], result.inputs[1][index][0], 2e-11);
                EXPECT_NEAR(result.outputs[2][index][1], result.inputs[1][shifted][1], 2e-11);
            }
        }
    }
}

TEST(FftFilterBank, ReusesNamedFiltersInPencilSlabLayout) {
    PROFILE();
    expectStoredFilterConvolutions(runStoredFilterConvolutions(16, 1.0, 0.0, cosmictiger::FftCommunicationStrategy::pencilSlab), 256);
}

TEST(FftFilterBank, ReusesNamedFiltersInPackedVectorLayout) {
    PROFILE();
    expectStoredFilterConvolutions(runStoredFilterConvolutions(16, 0.0, 1.0, cosmictiger::FftCommunicationStrategy::packedVectorRadix), 512);
}

TEST(FftFilterBank, StoresBoundaryFiltersOnlyOnActiveLocalities) {
    PROFILE();
    expectStoredFilterConvolutions(runStoredFilterConvolutions(8, 1.0, 0.0, cosmictiger::FftCommunicationStrategy::pencilSlab), 64);
}

TEST(FftPencilSlab, BatchedCircularConvolutionDirectRoute) {
    PROFILE();
    auto const plan = cosmictiger::findOptimalFftCommunicationPlan<double, 2>(16 * 16 * 16, 8, 1.0, 0.0);
    ASSERT_EQ(plan.strategy, cosmictiger::FftCommunicationStrategy::pencilSlab);
    expectConvolution(runPencilConvolution(plan));
}

TEST(FftPencilSlab, BatchedCircularConvolutionStagedRoute) {
    PROFILE();
    auto plan = cosmictiger::findOptimalFftCommunicationPlan<double, 2>(16 * 16 * 16, 8, 1.0, 0.0);
    plan.pencilToSlab = {{2, 2}, {2, 2}, {2, 2}};
    expectConvolution(runPencilConvolution(plan));
}

TEST(FftPencilSlab, AsyncHpxDirectAndStagedRoutes) {
    PROFILE();
    auto plan = cosmictiger::findOptimalFftCommunicationPlan<double, 2>(4096, 8, 1.0, 0.0);
    ASSERT_EQ(plan.strategy, cosmictiger::FftCommunicationStrategy::pencilSlab);
    expectConvolution(runPencilConvolution(plan, true));
    plan.blockToPencil = {{2, 2}};
    plan.pencilToSlab = {{2, 2}, {2, 2}, {2, 2}};
    expectConvolution(runPencilConvolution(plan, true));
}

TEST(FftVectorRadix, BatchedCircularConvolution) {
    PROFILE();
    expectConvolution(runVectorConvolution());
}

TEST(FftPackedVectorRadix, BatchedCircularConvolutionWithPrepositionedMultiplier) {
    PROFILE();
    expectConvolution(runPackedConvolution(true));
}

TEST(FftPackedVectorRadix, BatchedCircularConvolutionWithTwoForwardInputs) {
    PROFILE();
    expectConvolution(runPackedConvolution(false));
}

TEST(FftDistributed, DispatchesEveryPlannedStrategy) {
    PROFILE();
    cosmictiger::FftDistributed<2> const pencil(16, 8, 0, 1.0, 0.0);
    EXPECT_EQ(pencil.plan().strategy, cosmictiger::FftCommunicationStrategy::pencilSlab);
    EXPECT_EQ(pencil.spectralValueCount(), 256);

    cosmictiger::FftDistributed<2> const packed(16, 8, 0, 0.0, 1.0);
    EXPECT_EQ(packed.plan().strategy, cosmictiger::FftCommunicationStrategy::packedVectorRadix);
    EXPECT_EQ(packed.spectralValueCount(), 256);

    cosmictiger::FftDistributed<2> const full(64, 64, 0, 0.0, 1.0);
    EXPECT_EQ(full.plan().strategy, cosmictiger::FftCommunicationStrategy::fullSpectrumVectorRadix);
    EXPECT_EQ(full.spectralValueCount(), 4096);

    cosmictiger::FftDistributed<2> const boundaryActive(8, 8, 0, 1.0, 0.0);
    cosmictiger::FftDistributed<2> const boundaryInactive(8, 8, 1, 1.0, 0.0);
    EXPECT_EQ(boundaryActive.plan().strategy, cosmictiger::FftCommunicationStrategy::pencilSlab);
    EXPECT_EQ(boundaryActive.spectralValueCount(), 64);
    EXPECT_EQ(boundaryInactive.spectralValueCount(), 0);
}

TEST(FftDistributed, BatchedPencilBoundaryConvolution) {
    constexpr size_t kSide = 8;
    constexpr size_t kLocalities = 8;
    constexpr size_t kBlockSide = 4;
    constexpr size_t kValuesPerLocality = kBlockSide * kBlockSide * kBlockSide;
    PROFILE();

    SimulatedFftExchange transport(kLocalities);
    std::vector<Batch> input(kSide * kSide * kSide);
    std::vector<Batch> output(input.size());
    std::array<std::exception_ptr, kLocalities> errors{};
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    try {
                        cosmictiger::FftDistributed<2> fft(kSide, kLocalities, locality, 1.0, 0.0, cosmictiger::FftCommunicationWorkload::twoForwardInputs);
                        if (fft.plan().strategy != cosmictiger::FftCommunicationStrategy::pencilSlab) {
                            throw std::runtime_error("expected pencil/slab boundary plan");
                        }
                        std::vector<Batch> block(kValuesPerLocality);
                        std::vector<Batch> kernel(kValuesPerLocality);
                        size_t const blockX = locality >> 2;
                        size_t const blockY = (locality >> 1) & 1;
                        size_t const blockZ = locality & 1;
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalX = blockX * kBlockSide + x;
                                    size_t const globalY = blockY * kBlockSide + y;
                                    size_t const globalZ = blockZ * kBlockSide + z;
                                    size_t const globalIndex = (globalZ * kSide + globalY) * kSide + globalX;
                                    block[localIndex] = {double((globalIndex * 17 + 3) % 101) / 101.0, double((globalIndex * 13 + 7) % 97) / 97.0};
                                    kernel[localIndex] = {
                                        double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                    input[globalIndex] = block[localIndex];
                                }
                            }
                        }
                        size_t sequence = 0;
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(locality, sequence++, peer, send, receive);
                            },
                            "FFT test distributed convolution synchronous exchange");
                        auto const result = fft.convolve(block, kernel, exchange);
                        for (size_t z = 0; z != kBlockSide; z++) {
                            for (size_t y = 0; y != kBlockSide; y++) {
                                for (size_t x = 0; x != kBlockSide; x++) {
                                    size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                    size_t const globalIndex = ((blockZ * kBlockSide + z) * kSide + blockY * kBlockSide + y) * kSide + blockX * kBlockSide + x;
                                    output[globalIndex] = result[localIndex];
                                }
                            }
                        }
                    } catch (...) {
                        errors[locality] = std::current_exception();
                    }
                },
                "FFT test distributed convolution locality")));
    }
    for (auto& task : tasks) {
        task.get();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    for (size_t z = 0; z != kSide; z++) {
        for (size_t y = 0; y != kSide; y++) {
            for (size_t x = 0; x != kSide; x++) {
                size_t const index = (z * kSide + y) * kSide + x;
                size_t const shifted = (((z + kSide - 3) % kSide) * kSide + (y + kSide - 2) % kSide) * kSide + (x + kSide - 1) % kSide;
                EXPECT_NEAR(output[index][0], input[index][0], 2e-11);
                EXPECT_NEAR(output[index][1], input[shifted][1], 2e-11);
            }
        }
    }
}

TEST(FftVectorRadix, ParallelLocalFftwCallsOnHpxWorkers) {
    constexpr size_t kSide = 16;
    PROFILE();

    cosmictiger::FftVectorRadix<2> fft(kSide, 1, 0);
    std::vector<Batch> input(kSide * kSide * kSide);
    for (size_t index = 0; index != input.size(); index++) {
        input[index] = {double(index % 53) / 53.0, double(index % 97) / 97.0};
    }
    auto noExchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("single locality FFT exchanged data");
        },
        "FFT test reject unexpected single locality exchange");
    auto const spectrum = fft.forward(input, noExchange);
    auto const output = fft.inverse(spectrum, noExchange);
    for (size_t index = 0; index != input.size(); index++) {
        for (size_t lane = 0; lane != 2; lane++) {
            EXPECT_NEAR(output[index][lane], input[index][lane], 2e-11);
        }
    }
}

TEST(FftLocalTransforms, ConcurrentCallsKeepScratchPrivateAcrossLanesAndAxes) {
    constexpr size_t kLanes = 3;

    using Values = std::vector<std::array<double, kLanes>>;
    PROFILE();
    for (size_t const side : {size_t(2), size_t(16), size_t(32)}) {
        cosmictiger::FftVectorRadix<kLanes> full(side, 1, 0);
        cosmictiger::FftPackedVectorRadix<kLanes> packed(side, 1, 0);
        auto verify = [&](auto& fft) {
            PROFILE();
            std::array<Values, 2> inputs;
            for (size_t run = 0; run < inputs.size(); run++) {
                inputs[run].resize(side * side * side);
                for (size_t index = 0; index < inputs[run].size(); index++) {
                    for (size_t lane = 0; lane < kLanes; lane++) {
                        inputs[run][index][lane] = double((index * (17 + lane) + 31 * run + lane) % 101) / 101.0;
                    }
                }
            }
            auto noExchange = hpx::annotated_function(
                [](size_t, auto, auto) {
                    PROFILE();
                    throw std::logic_error("single locality FFT exchanged data");
                },
                "FFT test reject concurrent single locality exchange");
            std::array<Values, 2> expected;
            for (size_t run = 0; run < inputs.size(); run++) {
                expected[run] = fft.inverse(fft.forward(inputs[run], noExchange), noExchange);
            }
            std::vector<hpx::future<Values>> tasks;
            for (size_t run = 0; run < inputs.size(); run++) {
                tasks.push_back(hpx::async(hpx::launch::async,
                    hpx::annotated_function(
                        [&, run] {
                            PROFILE();
                            return fft.inverse(fft.forward(inputs[run], noExchange), noExchange);
                        },
                        "FFT test simultaneous local round trip")));
            }
            hpx::wait_all(tasks);
            for (size_t run = 0; run < inputs.size(); run++) {
                auto const output = tasks[run].get();
                for (size_t index = 0; index < output.size(); index++) {
                    for (size_t lane = 0; lane < kLanes; lane++) {
                        EXPECT_EQ(hpx::bit_cast<std::uint64_t>(output[index][lane]), hpx::bit_cast<std::uint64_t>(expected[run][index][lane]));
                        EXPECT_NEAR(output[index][lane], inputs[run][index][lane], 2e-11);
                    }
                }
            }
        };
        verify(full);
        verify(packed);
    }
}

TEST(FftDistributed, PencilBoundaryThroughHpxExchange) {
    constexpr size_t kSide = 8;
    constexpr size_t kLocalities = 8;
    constexpr size_t kBlockSide = 4;
    PROFILE();

    std::vector<hpx::id_type> localityIds(kLocalities, hpx::find_here());
    std::vector<Batch> input(kSide * kSide * kSide);
    std::vector<Batch> output(input.size());
    std::vector<hpx::future<void>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    cosmictiger::FftDistributed<2> fft(kSide, kLocalities, locality, 1.0, 0.0, cosmictiger::FftCommunicationWorkload::twoForwardInputs);
                    cosmictiger::FftHpxExchange transport(localityIds, 77, locality);
                    std::vector<Batch> block(kBlockSide * kBlockSide * kBlockSide);
                    std::vector<Batch> kernel(block.size());
                    size_t const blockX = locality >> 2;
                    size_t const blockY = (locality >> 1) & 1;
                    size_t const blockZ = locality & 1;
                    for (size_t z = 0; z != kBlockSide; z++) {
                        for (size_t y = 0; y != kBlockSide; y++) {
                            for (size_t x = 0; x != kBlockSide; x++) {
                                size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                size_t const globalX = blockX * kBlockSide + x;
                                size_t const globalY = blockY * kBlockSide + y;
                                size_t const globalZ = blockZ * kBlockSide + z;
                                size_t const globalIndex = (globalZ * kSide + globalY) * kSide + globalX;
                                block[localIndex] = {double(globalIndex % 37) / 37.0, double(globalIndex % 43) / 43.0};
                                kernel[localIndex] = {
                                    double(globalX == 0 && globalY == 0 && globalZ == 0), double(globalX == 1 && globalY == 2 && globalZ == 3)};
                                input[globalIndex] = block[localIndex];
                            }
                        }
                    }
                    auto const result = fft.convolve(block, kernel, transport);
                    for (size_t z = 0; z != kBlockSide; z++) {
                        for (size_t y = 0; y != kBlockSide; y++) {
                            for (size_t x = 0; x != kBlockSide; x++) {
                                size_t const localIndex = (z * kBlockSide + y) * kBlockSide + x;
                                size_t const globalIndex = ((blockZ * kBlockSide + z) * kSide + blockY * kBlockSide + y) * kSide + blockX * kBlockSide + x;
                                output[globalIndex] = result[localIndex];
                            }
                        }
                    }
                },
                "FFT test boundary convolution locality")));
    }
    std::exception_ptr failure;
    for (auto& task : tasks) {
        try {
            task.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    for (size_t z = 0; z != kSide; z++) {
        for (size_t y = 0; y != kSide; y++) {
            for (size_t x = 0; x != kSide; x++) {
                size_t const index = (z * kSide + y) * kSide + x;
                size_t const shifted = (((z + kSide - 3) % kSide) * kSide + (y + kSide - 2) % kSide) * kSide + (x + kSide - 1) % kSide;
                EXPECT_NEAR(output[index][0], input[index][0], 2e-11);
                EXPECT_NEAR(output[index][1], input[shifted][1], 2e-11);
            }
        }
    }
}

TEST(FftPencilBoundary, StagedBothRedistributionsAtMinimumStorageThroughHpx) {
    constexpr size_t kSide = 64;
    constexpr size_t kLocalities = 64;
    constexpr size_t kBlockSide = 16;
    PROFILE();

    auto plan = cosmictiger::findOptimalFftCommunicationPlan<double, 2>(kSide * kSide * kSide, kLocalities, 1.0, 0.0);
    ASSERT_EQ(plan.strategy, cosmictiger::FftCommunicationStrategy::pencilSlab);
    plan.blockToPencil = {{2, 2}, {2, 2}, {2, 2}};
    plan.pencilToSlab = {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}};
    std::vector<hpx::future<double>> tasks;
    for (size_t locality = 0; locality != kLocalities; locality++) {
        tasks.emplace_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(
                [&, locality]() {
                    PROFILE();
                    cosmictiger::FftPencilBoundary<2> fft(kSide, kLocalities, locality, plan);
                    cosmictiger::FftHpxExchange transport(std::vector<hpx::id_type>(kLocalities, hpx::find_here()), 78, locality);
                    std::vector<Batch> block(kBlockSide * kBlockSide * kBlockSide);
                    for (size_t index = 0; index != block.size(); index++) {
                        size_t const globalSeed = locality * block.size() + index;
                        block[index] = {double(globalSeed % 101) / 101.0, double(globalSeed % 103) / 103.0};
                    }
                    auto const spectrum = fft.forward(block, transport);
                    auto const result = fft.inverse(spectrum, transport);
                    double maximumError = 0.0;
                    cosmictiger::FpeGuard guard;
                    for (size_t index = 0; index != block.size(); index++) {
                        for (size_t lane = 0; lane != 2; lane++) {
                            double const error = std::abs(result[index][lane] - block[index][lane]);
                            if (!(error <= 2e-11)) {
                                throw std::runtime_error("boundary FFT round trip differs from its input");
                            }
                            maximumError = std::max(maximumError, error);
                        }
                    }
                    return maximumError;
                },
                "FFT test boundary round trip locality")));
    }
    std::exception_ptr failure;
    for (auto& task : tasks) {
        try {
            EXPECT_LT(task.get(), 2e-11);
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

}    // namespace
