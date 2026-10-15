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

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/fft/fft_hpx_exchange.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_convolution.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_runtime_convolution.hpp>
#include <cosmictiger/lock_guards.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#ifdef COSMICTIGER_ENABLE_CUDA
#include <cosmictiger/cuda/cuda_runtime.hpp>
#endif

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/mutex.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace gravity = cosmictiger::gravity;
using cosmictiger::CartesianGrid;
using cosmictiger::Coordinate;
using cosmictiger::GridIndices;
using cosmictiger::ParticleStorage;

constexpr size_t kSide = 8;
constexpr size_t kLocalitiesPerAxis = 2;
constexpr size_t kLocalityCount = kLocalitiesPerAxis * kLocalitiesPerAxis * kLocalitiesPerAxis;
constexpr size_t kBlockSide = kSide / kLocalitiesPerAxis;
constexpr int kImageCutoff = 2;
constexpr int kWaveCutoff = 2;
constexpr double kAlphaBox = 1.5;
constexpr float kUntouchedExpansion = -17.0f;

class SimulatedPmmmExchange {
public:
    template <typename Value>
    void exchange(size_t locality, size_t sequence, size_t partner, std::span<Value const> send, std::span<Value> receive) {
        PROFILE();
        cosmictiger::expectIndex(locality, kLocalityCount);
        cosmictiger::expectIndex(partner, kLocalityCount);
        if (partner == locality) {
            cosmictiger::expectEqual(send.size_bytes(), receive.size_bytes());
            if (!send.empty()) {
                std::memcpy(receive.data(), send.data(), send.size_bytes());
            }
            return;
        }
        bool const isFirst = locality < partner;
        ExchangeKey const key{sequence, std::min(locality, partner), std::max(locality, partner)};
        Payload outgoing(send.size_bytes());
        if (!send.empty()) {
            std::memcpy(outgoing.data(), send.data(), send.size_bytes());
        }
        auto packet = findPacket(key);
        auto incoming = packet->promises[isFirst ? 1 : 0].get_future();
        packet->promises[isFirst ? 0 : 1].set_value(std::move(outgoing));
        auto const inbound = incoming.get();
        if (inbound.size() != receive.size_bytes()) {
            throw std::runtime_error("simulated PMMM exchange payload size mismatch");
        }
        if (!receive.empty()) {
            std::memcpy(receive.data(), inbound.data(), receive.size_bytes());
        }
        {
            cosmictiger::LockGuard lock(mutex_);
            if (++packet->readers == 2) {
                // The local shared pointer keeps HPX members alive past the lock.
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

    hpx::mutex mutex_;
    std::map<ExchangeKey, std::shared_ptr<Packet>> packets_;
};

// Offsets are in eighths of a cell, making all source moments exactly
// representable in fixed-point positions, float, and double.
struct Source {
    GridIndices global;
    std::array<int, NDIM> offsetEighths;
    float mass;
};

size_t compactCoefficientIndex(int nx, int ny, int nz) {
    PROFILE();
    int const rank = nx + ny + nz;
    return static_cast<size_t>(rank * rank + (nz == 0 ? nx : rank + 1 + nx));
}

// A harmonic tensor has G_(i,j,k+2) = -G_(i+2,j,k) - G_(i,j+2,k).
// This oracle deliberately expands unstored z components without using the
// PMMM contraction table or its STF recovery routine.
template <size_t coefficientCount>
double greenComponent(std::array<double, coefficientCount> const& green, int nx, int ny, int nz) {
    PROFILE();
    if (nz >= 2) {
        return -greenComponent(green, nx + 2, ny, nz - 2) - greenComponent(green, nx, ny + 2, nz - 2);
    }
    return green[compactCoefficientIndex(nx, ny, nz)];
}

size_t sourceLocality(Source const& source) {
    PROFILE();
    size_t const blockX = static_cast<size_t>(source.global[0]) / kBlockSide;
    size_t const blockY = static_cast<size_t>(source.global[1]) / kBlockSide;
    size_t const blockZ = static_cast<size_t>(source.global[2]) / kBlockSide;
    return blockX + kLocalitiesPerAxis * (blockY + kLocalitiesPerAxis * blockZ);
}

GridIndices sourceLocalCoordinates(Source const& source) {
    PROFILE();
    return {source.global[0] % static_cast<std::int64_t>(kBlockSide), source.global[1] % static_cast<std::int64_t>(kBlockSide),
        source.global[2] % static_cast<std::int64_t>(kBlockSide)};
}

void addSource(CartesianGrid& grid, ParticleStorage& particles, Source const& source) {
    std::uint64_t constexpr kEighthCellRaw = std::uint64_t(1) << (Coordinate::fractionalBits() - 6);

    static_assert(Coordinate::fractionalBits() >= 6);
    PROFILE();
    std::array<Coordinate, NDIM> position{};
    for (size_t axis = 0; axis < NDIM; axis++) {
        auto const rawUnits = static_cast<std::uint64_t>(8 * source.global[axis] + 4 + source.offsetEighths[axis]);
        position[axis] = Coordinate::fromRaw(static_cast<std::uint32_t>(rawUnits * kEighthCellRaw));
    }
    auto const particleIndex = particles.append(position, {0.0f, 0.0f, 0.0f}, 0);
    auto local = source.global;
    for (auto& coordinate : local) {
        coordinate %= static_cast<std::int64_t>(grid.layout().interiorSideLength());
    }
    grid.setInteriorRange(local, {particleIndex, particleIndex + 1});
    grid.computeMultipole(local, particles, source.mass);
}

// This point-source oracle keeps the dipole signs and nonharmonic trace
// correction independent of the coefficient batching and contraction tables.
std::array<double, 10> directExpansion(GridIndices const& target, std::span<Source const> sources, int ghostWidth = gravity::kPmmmNeighborWidth) {
    PROFILE();
    cosmictiger::FpeGuard guard;
    double totalMass = 0.0;
    std::array<double, NDIM> totalDipole{};
    double totalSecondMomentTrace = 0.0;
    for (auto const& source : sources) {
        totalMass += source.mass;
        double squaredOffset = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const offset = static_cast<double>(source.offsetEighths[axis]) / 8.0;
            totalDipole[axis] += source.mass * offset;
            squaredOffset += offset * offset;
        }
        totalSecondMomentTrace += source.mass * squaredOffset;
    }
    double const background = gravity::pmmmPeriodicBackgroundLaplacian(kSide, 1.0 / kSide);

    std::array<double, 10> expected{};
    for (auto const& source : sources) {
        std::array<int, NDIM> displacement{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            displacement[axis] = static_cast<int>((target[axis] - source.global[axis] + static_cast<std::int64_t>(kSide)) % static_cast<std::int64_t>(kSide));
        }
        auto const green = ghostWidth == 1 ? gravity::pmmmPeriodicGreen<3, 1>(displacement, kSide, 1.0 / kSide, kImageCutoff, kWaveCutoff, kAlphaBox) :
                                             gravity::pmmmPeriodicGreen<3, 2>(displacement, kSide, 1.0 / kSide, kImageCutoff, kWaveCutoff, kAlphaBox);
        std::array<double, NDIM> dipole{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            dipole[axis] = static_cast<double>(source.mass) * source.offsetEighths[axis] / 8.0;
        }
        for (int rank = 0; rank <= 2; rank++) {
            for (int nz = 0; nz <= 1 && nz <= rank; nz++) {
                for (int nx = 0; nx <= rank - nz; nx++) {
                    int const ny = rank - nx - nz;
                    size_t const coefficient = compactCoefficientIndex(nx, ny, nz);
                    expected[coefficient] += static_cast<double>(source.mass) * greenComponent(green, nx, ny, nz) -
                        dipole[0] * greenComponent(green, nx + 1, ny, nz) - dipole[1] * greenComponent(green, nx, ny + 1, nz) -
                        dipole[2] * greenComponent(green, nx, ny, nz + 1);
                }
            }
        }
    }
    expected[0] += background * totalSecondMomentTrace / (2.0 * NDIM);
    expected[1] -= background * totalDipole[1] / NDIM;    // y
    expected[2] -= background * totalDipole[0] / NDIM;    // x
    expected[3] -= background * totalDipole[2] / NDIM;    // z
    expected[4] += background * totalMass / NDIM;         // yy
    expected[6] += background * totalMass / NDIM;         // xx

    expected.back() = background * totalMass;
    return expected;
}

constexpr std::array<Source, 2> kDisplacedSources{{{{1, 2, 5}, {2, -1, 0}, 1.5f}, {{6, 5, 1}, {-2, 2, 1}, 2.25f}}};

void seedConvolutionGrid(CartesianGrid& grid, ParticleStorage& particles) {
    PROFILE();
    for (auto const& source : kDisplacedSources) {
        if (grid.layout().localityCount() == 1 || sourceLocality(source) == grid.layout().localityIndex()) {
            addSource(grid, particles, source);
        }
    }
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        grid.expansion(grid.layout().cellCoordinates(index)).fill(kUntouchedExpansion);
    }
}

using ExpansionSnapshot = std::vector<cosmictiger::CellExpansion>;

ExpansionSnapshot expansionSnapshot(CartesianGrid const& grid) {
    PROFILE();
    ExpansionSnapshot result;
    result.reserve(grid.layout().storedCellCount());
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        result.push_back(grid.cell(grid.layout().cellCoordinates(index)).expansion);
    }
    return result;
}

void requireMatchingExpansions(ExpansionSnapshot const& actual, ExpansionSnapshot const& expected) {
    PROFILE();
    if (actual.size() != expected.size()) {
        throw std::runtime_error("PMMM batch comparison changed the number of cells");
    }
    for (size_t cell = 0; cell < actual.size(); cell++) {
        cosmictiger::FpeGuard guard;
        for (size_t coefficient = 0; coefficient < actual[cell].size(); coefficient++) {
            double const tolerance = 3e-5 * std::max(1.0, std::abs(static_cast<double>(expected[cell][coefficient])));
            if (!(std::abs(static_cast<double>(actual[cell][coefficient]) - expected[cell][coefficient]) <= tolerance)) {
                throw std::runtime_error("PMMM batch comparison differs at cell " + std::to_string(cell) + " coefficient " + std::to_string(coefficient));
            }
        }
    }
}

void requireDirectExpansions(CartesianGrid const& grid, ExpansionSnapshot const& values) {
    PROFILE();
    auto const& layout = grid.layout();
    for (size_t index = 0; index < values.size(); index++) {
        auto const local = layout.cellCoordinates(index);
        auto const& actual = values[index];
        if (!layout.isInterior(local)) {
            for (float coefficient : actual) {
                if (coefficient != kUntouchedExpansion) {
                    throw std::runtime_error("PMMM convolution modified a ghost expansion");
                }
            }
            continue;
        }
        auto const expected = directExpansion(layout.globalCoordinates(local), kDisplacedSources);
        cosmictiger::FpeGuard guard;
        for (size_t coefficient = 0; coefficient < actual.size(); coefficient++) {
            double const value = coefficient + 1 == actual.size() ? expected.back() : (coefficient + 1 < expected.size() ? expected[coefficient] : 0.0);
            double const tolerance = 3e-5 * std::max(1.0, std::abs(value));
            if (!(std::abs(static_cast<double>(actual[coefficient]) - value) <= tolerance)) {
                throw std::runtime_error(
                    "PMMM convolution differs from direct periodic sum at cell " + std::to_string(index) + " coefficient " + std::to_string(coefficient));
            }
        }
    }
}

void drainTasks(std::vector<hpx::future<void>>& tasks) {
    PROFILE();
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
}

class PmmmConvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
        int deviceCount = 0;
        try {
            deviceCount = cosmictiger::cuda::deviceCount().get();
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << "CUDA device unavailable: " << error.what();
            }
            throw;
        }
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        int const device = cosmictiger::cuda::device().get();
        auto const properties = cosmictiger::cuda::deviceProperties(device).get();
        if (!properties.managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
#endif
    }
};

#if COSMICTIGER_EXPANSION_ORDER >= 2
TEST(PmmmConvolution, ConfiguredOrdersRunOnOneLocality) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    auto noExchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("Single-locality PMMM attempted an FFT exchange");
        },
        "PMMM test reject configured single locality exchange");
    gravity::computePmmmExpansions(grid, noExchange, 0, 0);
    for (size_t index = 0; index < grid.layout().interiorCellCount(); index++) {
        size_t const x = index % kSide;
        size_t const y = (index / kSide) % kSide;
        size_t const z = index / (kSide * kSide);
        for (float coefficient : grid.cell({static_cast<std::int64_t>(x), static_cast<std::int64_t>(y), static_cast<std::int64_t>(z)}).expansion) {
            EXPECT_FLOAT_EQ(coefficient, 0.0f);
        }
    }
}
#endif

#if COSMICTIGER_MULTIPOLE_ORDER >= 1 && COSMICTIGER_EXPANSION_ORDER >= 2
TEST_F(PmmmConvolutionTest, MonopoleAndDipoleFftMatchDirectPeriodicSumAndBackground) {
    constexpr std::array<Source, 2> kSources{{{{1, 2, 5}, {2, -1, 0}, 1.5f}, {{6, 5, 1}, {-2, 2, 1}, 2.25f}}};
    PROFILE();

    // Grid IDs are x-fastest; FFT block IDs are z-fastest. These two source
    // blocks therefore exchange IDs 4->1 and 3->6 across the layouts.
    ASSERT_EQ(sourceLocality(kSources[0]), 4);
    ASSERT_EQ(sourceLocality(kSources[1]), 3);

    std::vector<std::unique_ptr<CartesianGrid>> grids;
    grids.reserve(kLocalityCount);
    std::vector<ParticleStorage> particles(kLocalityCount);
    for (size_t locality = 0; locality < kLocalityCount; locality++) {
        grids.push_back(std::make_unique<CartesianGrid>(3, 1, locality));
    }
    for (auto const& source : kSources) {
        size_t const locality = sourceLocality(source);
        addSource(*grids[locality], particles[locality], source);
        auto const& cell = grids[locality]->cell(sourceLocalCoordinates(source));
        EXPECT_FLOAT_EQ(cell.multipole[0], source.mass);
        double expectedTrace = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const offset = static_cast<double>(source.offsetEighths[axis]) / 8.0;
            expectedTrace += static_cast<double>(source.mass) * offset * offset;
        }
        EXPECT_FLOAT_EQ(cell.multipole[1], source.mass * source.offsetEighths[1] / 8.0f);    // y
        EXPECT_FLOAT_EQ(cell.multipole[2], source.mass * source.offsetEighths[0] / 8.0f);    // x
        EXPECT_FLOAT_EQ(cell.multipole[3], source.mass * source.offsetEighths[2] / 8.0f);    // z
        EXPECT_DOUBLE_EQ(cell.secondMomentTrace, expectedTrace);
    }
    for (auto& grid : grids) {
        auto const& layout = grid->layout();
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            grid->expansion(layout.cellCoordinates(index)).fill(kUntouchedExpansion);
        }
    }

    using Convolution = gravity::PmmmPeriodicConvolution<1, 2>;
    std::vector<std::unique_ptr<Convolution>> convolutions;
    convolutions.reserve(kLocalityCount);
    for (auto const& grid : grids) {
        auto const& layout = grid->layout();
        convolutions.push_back(std::make_unique<Convolution>(layout, kImageCutoff, kWaveCutoff, kAlphaBox));
        auto const block = layout.localityCoordinates();
        size_t const expectedFftLocality =
            (static_cast<size_t>(block[0]) * kLocalitiesPerAxis + static_cast<size_t>(block[1])) * kLocalitiesPerAxis + static_cast<size_t>(block[2]);
        EXPECT_EQ(convolutions.back()->fftLocalityIndex(), expectedFftLocality);
    }

    SimulatedPmmmExchange transport;
    for (int repetition = 0; repetition < 2; repetition++) {
        // The second call reuses the cached Green spectra on every locality.
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(kLocalityCount);
        for (size_t locality = 0; locality < kLocalityCount; locality++) {
            tasks.push_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, locality] {
                        PROFILE();
                        size_t sequence = 0;
                        size_t const fftLocality = convolutions[locality]->fftLocalityIndex();
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(fftLocality, sequence++, peer, send, receive);
                            },
                            "PMMM test periodic synchronous exchange");
                        convolutions[locality]->computeExpansions(*grids[locality], exchange);
                    },
                    "PMMM test periodic convolution locality")));
        }
        drainTasks(tasks);
    }

    for (auto const& grid : grids) {
        auto const& layout = grid->layout();
        for (size_t index = 0; index < layout.storedCellCount(); index++) {
            auto const local = layout.cellCoordinates(index);
            auto const& actual = grid->cell(local).expansion;
            if (!layout.isInterior(local)) {
                for (float coefficient : actual) {
                    EXPECT_EQ(coefficient, kUntouchedExpansion) << "ghost at locality " << layout.localityIndex();
                }
                continue;
            }

            auto const target = layout.globalCoordinates(local);
            auto const expected = directExpansion(target, kSources);

            for (size_t coefficient = 0; coefficient + 1 < expected.size(); coefficient++) {
                double const tolerance = 3e-5 * std::max(1.0, std::abs(expected[coefficient]));
                EXPECT_NEAR(actual[coefficient], expected[coefficient], tolerance) << "locality=" << layout.localityIndex() << " target=(" << target[0] << ","
                                                                                   << target[1] << "," << target[2] << ") coefficient=" << coefficient;
            }
            for (size_t coefficient = expected.size() - 1; coefficient + 1 < actual.size(); coefficient++) {
                EXPECT_EQ(actual[coefficient], 0.0f) << "unused coefficient=" << coefficient;
            }
            EXPECT_NEAR(actual.back(), expected.back(), 3e-5)
                << "locality=" << layout.localityIndex() << " target=(" << target[0] << "," << target[1] << "," << target[2] << ")";
        }
    }
}
#endif

#if COSMICTIGER_MULTIPOLE_ORDER >= 2 && COSMICTIGER_EXPANSION_ORDER >= 2
TEST_F(PmmmConvolutionTest, RuntimeGhostWidthSelectsThePeriodicNearExclusion) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedConvolutionGrid(grid, particles);
    auto noExchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("Single-locality PMMM attempted an FFT exchange");
        },
        "PMMM test runtime width single locality exchange");
    for (int ghostWidth : {1, 2}) {
        gravity::PmmmPeriodicConvolution<1, 2> convolution(grid.layout(), kImageCutoff, kWaveCutoff, kAlphaBox, ghostWidth);
        convolution.computeExpansions(grid, noExchange);
        SCOPED_TRACE(ghostWidth);
        for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
            auto const coordinates = grid.layout().cellCoordinates(index);
            if (!grid.layout().isInterior(coordinates)) {
                continue;
            }
            auto const expected = directExpansion(coordinates, kDisplacedSources, ghostWidth);
            auto const& actual = grid.cell(coordinates).expansion;
            for (size_t coefficient = 0; coefficient + 1 < expected.size(); coefficient++) {
                EXPECT_NEAR(actual[coefficient], expected[coefficient], 3.0e-5 * std::max(1.0, std::abs(expected[coefficient])))
                    << "cell=" << index << " coefficient=" << coefficient;
            }
            EXPECT_NEAR(actual.back(), expected.back(), 3.0e-5);
        }
    }
}

TEST_F(PmmmConvolutionTest, RuntimeGhostWidthRejectsUnsupportedMasksAndAliasedNearCells) {
    PROFILE();
    CartesianGrid grid(3, 0, 0);
    for (int ghostWidth : {0, 3}) {
        EXPECT_THROW((gravity::PmmmPeriodicConvolution<1, 2>(grid.layout(), 0, 0, kAlphaBox, ghostWidth)), std::invalid_argument);
    }
    CartesianGrid small(2, 0, 0);
    EXPECT_NO_THROW((gravity::PmmmPeriodicConvolution<1, 2>(small.layout(), 0, 0, kAlphaBox, 1)));
    EXPECT_THROW((gravity::PmmmPeriodicConvolution<1, 2>(small.layout(), 0, 0, kAlphaBox, 2)), std::invalid_argument);
}

TEST_F(PmmmConvolutionTest, CoefficientBatchesMatchScalarAcrossEightLogicalLocalities) {
    // Order 2 has nine source/local coefficients and 25 Green coefficients.
    // The additional source trace, constant filter, and five background fields
    // make every transform family exercise a partially occupied batch.
    using Scalar = gravity::PmmmPeriodicConvolution<2, 2, 1>;
    using Batched = gravity::PmmmPeriodicConvolution<2, 2>;
    PROFILE();
    std::vector<std::unique_ptr<CartesianGrid>> grids;
    std::vector<ParticleStorage> particles(kLocalityCount);
    std::vector<std::unique_ptr<Scalar>> scalar;
    std::vector<std::unique_ptr<Batched>> batched;
    for (size_t locality = 0; locality < kLocalityCount; locality++) {
        grids.push_back(std::make_unique<CartesianGrid>(3, 1, locality));
        seedConvolutionGrid(*grids.back(), particles[locality]);
        scalar.push_back(std::make_unique<Scalar>(grids.back()->layout(), kImageCutoff, kWaveCutoff, kAlphaBox));
        batched.push_back(std::make_unique<Batched>(grids.back()->layout(), kImageCutoff, kWaveCutoff, kAlphaBox));
    }
    auto run = [&](auto& convolutions) {
        PROFILE();
        SimulatedPmmmExchange transport;
        std::array<size_t, kLocalityCount> calls{};
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(kLocalityCount);
        for (size_t locality = 0; locality < kLocalityCount; locality++) {
            tasks.push_back(hpx::async(hpx::launch::async,
                hpx::annotated_function(
                    [&, locality] {
                        PROFILE();
                        size_t const fftLocality = convolutions[locality]->fftLocalityIndex();
                        auto exchange = hpx::annotated_function(
                            [&](size_t peer, auto send, auto receive) {
                                PROFILE();
                                transport.exchange(fftLocality, calls[locality]++, peer, send, receive);
                            },
                            "PMMM test coefficient batch synchronous exchange");
                        convolutions[locality]->computeExpansions(*grids[locality], exchange);
                    },
                    "PMMM test coefficient batch locality")));
        }
        drainTasks(tasks);
        return calls;
    };
    std::vector<ExpansionSnapshot> coldScalar;
    for (size_t repetition = 0; repetition < 2; repetition++) {
        auto const scalarCalls = run(scalar);
        std::vector<ExpansionSnapshot> expected;
        for (auto const& grid : grids) {
            expected.push_back(expansionSnapshot(*grid));
        }
        auto const batchedCalls = run(batched);
        for (size_t locality = 0; locality < kLocalityCount; locality++) {
            SCOPED_TRACE(::testing::Message() << "repetition=" << repetition << " locality=" << locality);
            EXPECT_NO_THROW(requireMatchingExpansions(expansionSnapshot(*grids[locality]), expected[locality]));
            // Every distributed FFT on this decomposition exchanges once per
            // axis. Cold calls create filters; cached calls omit that traffic.
            EXPECT_EQ(scalarCalls[locality], repetition == 0 ? 150U : 72U);
            EXPECT_EQ(batchedCalls[locality], repetition == 0 ? 45U : 24U);
            if (repetition != 0) {
                EXPECT_NO_THROW(requireMatchingExpansions(expected[locality], coldScalar[locality]));
            }
        }
        if (repetition == 0) {
            coldScalar = std::move(expected);
        }
    }
}
#endif

#if COSMICTIGER_EXPANSION_ORDER >= 2
TEST_F(PmmmConvolutionTest, ConfiguredCoefficientBatchesMatchScalarOnOneLocality) {
    PROFILE();
    // Exercise the configured high-order contractions and all final lanes
    // without the cost of a large periodic image/wave sum.
    CartesianGrid grid(3, 0, 0);
    ParticleStorage particles;
    seedConvolutionGrid(grid, particles);
    gravity::PmmmPeriodicConvolution<gravity::kMultipoleOrder, gravity::kExpansionOrder, 1> scalar(grid.layout(), 0, 0, kAlphaBox);
    gravity::PmmmPeriodicConvolution<> batched(grid.layout(), 0, 0, kAlphaBox);
    auto noExchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("Single-locality PMMM attempted an FFT exchange");
        },
        "PMMM test reject batched single locality exchange");
    ExpansionSnapshot coldScalar;
    for (size_t repetition = 0; repetition < 2; repetition++) {
        scalar.computeExpansions(grid, noExchange);
        auto expected = expansionSnapshot(grid);
        batched.computeExpansions(grid, noExchange);
        EXPECT_NO_THROW(requireMatchingExpansions(expansionSnapshot(grid), expected));
        if (repetition == 0) {
            coldScalar = std::move(expected);
        } else {
            EXPECT_NO_THROW(requireMatchingExpansions(expected, coldScalar));
        }
    }
}
#endif

template <int order>
void checkRuntimeOrder(gravity::SofteningModel model) {
    constexpr size_t kGridExponent = 3;
    constexpr size_t kPassCount = 2;
    constexpr std::uint64_t kExchangeOperation = 0x464D4D4F52444552ULL;
    constexpr int kImageCutoff = 0;
    constexpr int kWaveCutoff = 0;
    constexpr double kEpsilon = 0.0625;
    constexpr float kUnusedSource = 19.0f;
    constexpr size_t kLocalCount = gravity::stf::coefficientCount(order);
    constexpr size_t kFullSourceCount = gravity::symmetric::coefficientCount(order - 1);
    constexpr size_t kFullLocalCount = gravity::symmetric::coefficientCount(order);
    constexpr double kTolerance = 3e-5;
    PROFILE();
    CartesianGrid grid(kGridExponent, 0, 0);
    ParticleStorage particles;
    seedConvolutionGrid(grid, particles);
    if (model == gravity::SofteningModel::plummer) {
        grid.computeMultipoles(particles, 1.0f, true);
    }
    gravity::PmmmPeriodicConvolution<order - 1, order> reference(
        grid.layout(), kImageCutoff, kWaveCutoff, kAlphaBox, gravity::kPmmmNeighborWidth, model, kEpsilon);
    auto noExchange = hpx::annotated_function(
        [](size_t, auto, auto) {
            PROFILE();
            throw std::logic_error("Single-locality PMMM attempted an FFT exchange");
        },
        "PMMM test runtime order reference exchange");
    reference.computeExpansions(grid, noExchange);
    auto const expected = expansionSnapshot(grid);
    std::vector<cosmictiger::FullCellExpansion> expectedFull;
    for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
        auto const coordinates = grid.layout().cellCoordinates(index);
        expectedFull.push_back(grid.cell(coordinates).fullExpansion);
        if (grid.layout().isInterior(coordinates)) {
            // Stale coefficients outside the selected source rank must not
            // contribute, and stale locals must be replaced by padded zeros.
            auto& fullMoments = grid.fullMultipole(coordinates);
            for (size_t coefficient = kFullSourceCount; coefficient < fullMoments.size(); coefficient++) {
                fullMoments[coefficient] = kUnusedSource;
            }
            grid.expansion(coordinates).fill(kUntouchedExpansion);
            grid.fullExpansion(coordinates).fill(kUntouchedExpansion);
        }
    }
    gravity::PmmmRuntimeConvolution runtime(grid.layout(), order, kImageCutoff, kWaveCutoff, kAlphaBox, gravity::kPmmmNeighborWidth, model, kEpsilon);
    EXPECT_EQ(runtime.fftLocalityIndex(), 0U);
    cosmictiger::FftHpxExchange exchange(hpx::find_all_localities(), kExchangeOperation, runtime.fftLocalityIndex());
    for (size_t repetition = 0; repetition < kPassCount; repetition++) {
        runtime.computeExpansions(grid, exchange);
        if (model == gravity::SofteningModel::ferrers) {
            EXPECT_NO_THROW(requireMatchingExpansions(expansionSnapshot(grid), expected));
        }
        for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
            auto const coordinates = grid.layout().cellCoordinates(index);
            if (!grid.layout().isInterior(coordinates)) {
                continue;
            }
            auto const& cell = grid.cell(coordinates);
            if (model == gravity::SofteningModel::plummer) {
                for (size_t coefficient = 0; coefficient < cell.fullExpansion.size(); coefficient++) {
                    EXPECT_NEAR(cell.fullExpansion[coefficient], expectedFull[index][coefficient],
                        kTolerance * std::max(1.0, std::abs(double(expectedFull[index][coefficient]))));
                }
                for (size_t coefficient = kFullLocalCount; coefficient < cell.fullExpansion.size(); coefficient++) {
                    EXPECT_FLOAT_EQ(cell.fullExpansion[coefficient], 0.0f);
                }
            } else {
                for (size_t coefficient = kLocalCount; coefficient + 1 < cell.expansion.size(); coefficient++) {
                    EXPECT_FLOAT_EQ(cell.expansion[coefficient], 0.0f);
                }
            }
        }
    }
}

TEST_F(PmmmConvolutionTest, RuntimeOrderSelectsFerrersAndPlummerRanksAndClearsUnusedLocals) {
    PROFILE();
    for (auto const model : {gravity::SofteningModel::ferrers, gravity::SofteningModel::plummer}) {
        checkRuntimeOrder<gravity::kMinimumFmmOrder>(model);
        if constexpr (gravity::kMaximumFmmOrder > gravity::kMinimumFmmOrder) {
            checkRuntimeOrder<gravity::kMaximumFmmOrder>(model);
        }
    }
}

TEST_F(PmmmConvolutionTest, RuntimeOrderRejectsRanksOutsideCompiledCapacity) {
    constexpr size_t kGridExponent = 3;
    constexpr int kUnsupportedHighOrder = gravity::kMaximumFmmOrder + 1;
    PROFILE();
    CartesianGrid grid(kGridExponent, 0, 0);
    for (int const order : {-1, 0, 1, kUnsupportedHighOrder}) {
        EXPECT_THROW(gravity::PmmmRuntimeConvolution(grid.layout(), order), std::invalid_argument);
    }
}

TEST_F(PmmmConvolutionTest, EverySupportedRuntimeOrderComputesExpansions) {
    constexpr size_t kGridExponent = 2;
    constexpr int kGhostWidth = 1;
    constexpr int kImageCutoff = 0;
    constexpr int kWaveCutoff = 0;
    constexpr std::uint64_t kExchangeOperation = 0x464D4D414C4C3039ULL;
    PROFILE();
    CartesianGrid grid(kGridExponent, 0, 0, kGhostWidth);
    cosmictiger::FftHpxExchange exchange(hpx::find_all_localities(), kExchangeOperation, 0);
    for (int order = gravity::kMinimumFmmOrder; order <= gravity::kMaximumFmmOrder; order++) {
        for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
            grid.expansion(grid.layout().cellCoordinates(index)).fill(kUntouchedExpansion);
        }
        gravity::PmmmRuntimeConvolution runtime(grid.layout(), order, kImageCutoff, kWaveCutoff, kAlphaBox, kGhostWidth);
        runtime.computeExpansions(grid, exchange);
        for (size_t index = 0; index < grid.layout().storedCellCount(); index++) {
            auto const coordinates = grid.layout().cellCoordinates(index);
            if (grid.layout().isInterior(coordinates)) {
                for (float const coefficient : grid.cell(coordinates).expansion) {
                    EXPECT_FLOAT_EQ(coefficient, 0.0f);
                }
            }
        }
    }
}

}    // namespace

#if COSMICTIGER_MULTIPOLE_ORDER >= 1 && COSMICTIGER_EXPANSION_ORDER >= 2
namespace cosmictiger::detail {

std::vector<size_t> runRemotePmmmCoefficientBatches(std::vector<hpx::id_type> fftLocalities, std::uint64_t operationId, size_t gridLocality) {
    PROFILE();
    CartesianGrid grid(3, 1, gridLocality);
    ParticleStorage particles;
    seedConvolutionGrid(grid, particles);
    gravity::PmmmPeriodicConvolution<1, 2, 1> scalar(grid.layout(), kImageCutoff, kWaveCutoff, kAlphaBox);
    gravity::PmmmPeriodicConvolution<1, 2> batched(grid.layout(), kImageCutoff, kWaveCutoff, kAlphaBox);
    std::vector<size_t> counts;
    std::vector<ExpansionSnapshot> results;
    auto run = [&](auto& convolution, std::uint64_t transportId) {
        PROFILE();
        FftHpxExchange transport(fftLocalities, transportId, convolution.fftLocalityIndex());
        for (size_t repetition = 0; repetition < 2; repetition++) {
            size_t calls = 0;
            auto exchange = hpx::annotated_function(
                [&](size_t peer, auto send, auto receive) {
                    PROFILE();
                    calls++;
                    transport.exchange(peer, send, receive);
                },
                "PMMM test remote coefficient batch synchronous exchange");
            convolution.computeExpansions(grid, exchange);
            counts.push_back(calls);
            results.push_back(expansionSnapshot(grid));
        }
    };
    run(scalar, operationId);
    run(batched, operationId + 1);
    // Defer numerical checks until all exchanges finish so an oracle failure
    // cannot strand peers in a later transform. Exceptions reach the caller.
    requireMatchingExpansions(results[0], results[1]);
    requireMatchingExpansions(results[0], results[2]);
    requireMatchingExpansions(results[1], results[3]);
    requireDirectExpansions(grid, results[0]);
    requireDirectExpansions(grid, results[3]);
    return counts;
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::runRemotePmmmCoefficientBatches, cosmictiger_run_remote_pmmm_coefficient_batches_action)

TEST_F(PmmmConvolutionTest, RemoteCoefficientBatches) {
    constexpr std::uint64_t kOperationId = 0x504d4d4d00000000ULL;
    PROFILE();

    auto const physicalLocalities = hpx::find_all_localities();
    if (physicalLocalities.size() != 2 && physicalLocalities.size() != kLocalityCount) {
        GTEST_SKIP() << "requires two or eight HPX localities";
    }
    // The transport is indexed in FFT (z-fastest) order. Multiple logical
    // endpoints can share a process while still sending across physical ranks.
    std::vector<hpx::id_type> fftLocalities;
    for (size_t locality = 0; locality < kLocalityCount; locality++) {
        fftLocalities.push_back(physicalLocalities[locality % physicalLocalities.size()]);
    }
    std::vector<hpx::future<std::vector<size_t>>> tasks;
    tasks.reserve(kLocalityCount);
    for (size_t gridLocality = 0; gridLocality < kLocalityCount; gridLocality++) {
        size_t fftLocality = 0;
        size_t remainder = gridLocality;
        for (size_t axis = 0; axis < NDIM; axis++) {
            fftLocality = fftLocality * kLocalitiesPerAxis + remainder % kLocalitiesPerAxis;
            remainder /= kLocalitiesPerAxis;
        }
        tasks.push_back(
            hpx::async<cosmictiger_run_remote_pmmm_coefficient_batches_action>(fftLocalities[fftLocality], fftLocalities, kOperationId, gridLocality));
    }
    for (auto& task : tasks) {
        std::vector<size_t> counts;
        EXPECT_NO_THROW(counts = task.get());
        EXPECT_EQ(counts, (std::vector<size_t>{108, 57, 36, 21}));
    }
}
#endif
