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

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <new>
#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/cartesian_grid.hpp>
#include <cosmictiger/cuda/cuda_runtime.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/profiler.hpp>

#include "fmm_test_cells.hpp"

namespace {

namespace fmm = cosmictiger::gravity::fmm;
namespace test = cosmictiger::gravity::fmm::test;

using cosmictiger::CellExpansion;
using cosmictiger::CellMultipole;
using cosmictiger::Coordinate;
using cosmictiger::ParticleCount;
using cosmictiger::gravity::kMaximumFmmOrder;
using cosmictiger::gravity::kMinimumFmmOrder;
using Vector = std::array<double, NDIM>;

// Single precision on the device against the double-precision CPU path.
constexpr double kDeviceTolerance = 2e-5;
// The generated rank-eight and rank-nine polynomials cancel terms with odd
// factorial coefficients up to 34459425, which costs about one more digit in
// float. Their field contribution is suppressed by (offset/spacing)^rank.
constexpr int kHighRank = 8;
constexpr double kHighRankToleranceFactor = 10.0;
constexpr size_t kSourceCells = 5;
constexpr size_t kTargetCells = 9;
constexpr size_t kParticlesPerCell = 45;

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(size_t count)
      : count_(count) {
        PROFILE();
        if (count_ != 0) {
            values_ = static_cast<T*>(cosmictiger::cuda::mallocManaged(count_ * sizeof(T)).get());
            for (size_t index = 0; index < count_; index++) {
                ::new (static_cast<void*>(values_ + index)) T{};
            }
        }
    }

    ManagedArray(ManagedArray const&) = delete;
    ManagedArray& operator=(ManagedArray const&) = delete;

    ~ManagedArray() {
        PROFILE();
        if (values_ != nullptr) {
            try {
                cosmictiger::cuda::free(values_).get();
            } catch (std::exception const& error) {
                ADD_FAILURE() << "managed FMM buffer release failed: " << error.what();
            }
        }
    }

    T* data() noexcept {
        PROFILE();
        return values_;
    }

    T& operator[](size_t index) noexcept {
        PROFILE();
        return values_[index];
    }

    size_t size() const noexcept {
        PROFILE();
        return count_;
    }

private:
    T* values_ = nullptr;
    size_t count_ = 0;
};

template <typename Function, int... orders>
void forEachOrder(Function&& function, std::integer_sequence<int, orders...>) {
    PROFILE();
    (function(std::integral_constant<int, kMinimumFmmOrder + orders>{}), ...);
}

template <typename Function>
void forEachOrder(Function&& function) {
    PROFILE();
    forEachOrder(std::forward<Function>(function), std::make_integer_sequence<int, kMaximumFmmOrder - kMinimumFmmOrder + 1>{});
}

// Several target cells, each receiving every source cell, with all buffers
// in managed memory so the same request runs on both backends.
class DeviceScene {
public:
    DeviceScene()
      : positions_{ManagedArray<Coordinate>(kTotalParticles_), ManagedArray<Coordinate>(kTotalParticles_), ManagedArray<Coordinate>(kTotalParticles_)}
      , multipoles_(kSourceCells)
      , multipoleSources_(kTargetCells * kSourceCells)
      , particleSources_(kTargetCells * kSourceCells)
      , expansionTargets_(kTargetCells)
      , particleTargets_(kTargetCells) {
        PROFILE();
        std::array<Vector, kTargetCells> targetCenters{};
        for (size_t target = 0; target < kTargetCells; target++) {
            targetCenters[target] = {(2.5 + 2.0 * static_cast<double>(target % 3)) * test::kSpacing,
                (3.5 + 2.0 * static_cast<double>(target / 3)) * test::kSpacing, 7.5 * test::kSpacing};
            cells_.addCell(targetCenters[target], kParticlesPerCell);
        }
        std::array<Vector, kSourceCells> sourceCenters{};
        for (size_t source = 0; source < kSourceCells; source++) {
            sourceCenters[source] = {
                (1.5 + 3.0 * static_cast<double>(source)) * test::kSpacing, 12.5 * test::kSpacing, (source % 2 == 0 ? 2.5 : 12.5) * test::kSpacing};
            cells_.addCell(sourceCenters[source], kParticlesPerCell);
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            for (size_t particle = 0; particle < kTotalParticles_; particle++) {
                positions_[axis][particle] = cells_.positions(axis)[particle];
            }
        }
        for (size_t source = 0; source < kSourceCells; source++) {
            multipoles_[source] = cells_.multipole(kTargetCells + source);
        }
        for (size_t target = 0; target < kTargetCells; target++) {
            for (size_t source = 0; source < kSourceCells; source++) {
                size_t const entry = target * kSourceCells + source;
                multipoleSources_[entry].multipole = multipoles_[source].data();
                particleSources_[entry].particles = cells_.range(kTargetCells + source);
                particleSources_[entry].center = cells_.center(kTargetCells + source);
                for (size_t axis = 0; axis < NDIM; axis++) {
                    multipoleSources_[entry].separation[axis] = targetCenters[target][axis] - sourceCenters[source][axis];
                    particleSources_[entry].separation[axis] = multipoleSources_[entry].separation[axis];
                }
            }
            particleTargets_[target] = {cells_.range(target), cells_.center(target), target * kSourceCells, (target + 1) * kSourceCells};
        }
    }

    fmm::FmmM2lRequest m2l(ManagedArray<CellExpansion>& expansions, int order) {
        PROFILE();
        bindExpansions(expansions);
        return {expansionTargets_.data(), kTargetCells, multipoleSources_.data(), multipoleSources_.size(), test::kSpacing, order};
    }

    fmm::FmmP2lRequest p2l(ManagedArray<CellExpansion>& expansions, int order) {
        PROFILE();
        bindExpansions(expansions);
        fmm::FmmP2lRequest request{
            expansionTargets_.data(), kTargetCells, particleSources_.data(), particleSources_.size(), {}, test::kParticleMass, test::kSpacing, order};
        for (size_t axis = 0; axis < NDIM; axis++) {
            request.positions[axis] = positions_[axis].data();
        }
        return request;
    }

    fmm::FmmM2pRequest m2p(std::array<ManagedArray<float>, NDIM + 1>& field, int order) {
        PROFILE();
        fmm::FmmM2pRequest request{
            particleTargets_.data(), kTargetCells, multipoleSources_.data(), multipoleSources_.size(), {}, {}, field[0].data(), test::kSpacing, order};
        for (size_t axis = 0; axis < NDIM; axis++) {
            request.positions[axis] = positions_[axis].data();
            request.accelerations[axis] = field[axis + 1].data();
        }
        return request;
    }

    static constexpr size_t particleCount() noexcept {
        PROFILE();
        return kTotalParticles_;
    }

private:
    static constexpr size_t kTotalParticles_ = (kTargetCells + kSourceCells) * kParticlesPerCell;

    void bindExpansions(ManagedArray<CellExpansion>& expansions) {
        PROFILE();
        for (size_t target = 0; target < kTargetCells; target++) {
            expansionTargets_[target] = {expansions[target].data(), target * kSourceCells, (target + 1) * kSourceCells};
        }
    }

    test::TestCells cells_;
    std::array<ManagedArray<Coordinate>, NDIM> positions_;
    ManagedArray<CellMultipole> multipoles_;
    ManagedArray<fmm::FmmMultipoleSource> multipoleSources_;
    ManagedArray<fmm::FmmParticleSource> particleSources_;
    ManagedArray<fmm::FmmExpansionTarget> expansionTargets_;
    ManagedArray<fmm::FmmParticleTarget> particleTargets_;
};

// Targets whose sink and source counts straddle the M2P warp width, source
// tile, and bucket size, including empty lists and a multi-pass target.
class TiledM2pScene {
public:
    TiledM2pScene()
      : positions_{ManagedArray<Coordinate>(totalParticles()), ManagedArray<Coordinate>(totalParticles()), ManagedArray<Coordinate>(totalParticles())}
      , multipoles_(kSourceCells_)
      , sources_(totalSources())
      , targets_(kTargets_) {
        PROFILE();
        std::array<Vector, kTargets_> targetCenters{};
        for (size_t target = 0; target < kTargets_; target++) {
            targetCenters[target] = {(1.5 + 2.0 * static_cast<double>(target)) * test::kSpacing, 3.5 * test::kSpacing, 2.5 * test::kSpacing};
            cells_.addCell(targetCenters[target], kSinkCounts_[target]);
        }
        std::array<Vector, kSourceCells_> sourceCenters{};
        for (size_t source = 0; source < kSourceCells_; source++) {
            sourceCenters[source] = {(1.5 + 2.0 * static_cast<double>(source)) * test::kSpacing, 12.5 * test::kSpacing, 12.5 * test::kSpacing};
            cells_.addCell(sourceCenters[source], kSourceParticles_);
            multipoles_[source] = cells_.multipole(kTargets_ + source);
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            for (size_t particle = 0; particle < totalParticles(); particle++) {
                positions_[axis][particle] = cells_.positions(axis)[particle];
            }
        }
        size_t entry = 0;
        for (size_t target = 0; target < kTargets_; target++) {
            size_t const begin = entry;
            for (size_t listed = 0; listed < kSourceCounts_[target]; listed++) {
                size_t const source = (target + listed) % kSourceCells_;
                sources_[entry].multipole = multipoles_[source].data();
                for (size_t axis = 0; axis < NDIM; axis++) {
                    sources_[entry].separation[axis] = targetCenters[target][axis] - sourceCenters[source][axis];
                }
                entry++;
            }
            targets_[target] = {cells_.range(target), cells_.center(target), begin, entry};
        }
    }

    fmm::FmmM2pRequest m2p(std::array<ManagedArray<float>, NDIM + 1>& field, int order) {
        PROFILE();
        fmm::FmmM2pRequest request{targets_.data(), kTargets_, sources_.data(), sources_.size(), {}, {}, field[0].data(), test::kSpacing, order};
        for (size_t axis = 0; axis < NDIM; axis++) {
            request.positions[axis] = positions_[axis].data();
            request.accelerations[axis] = field[axis + 1].data();
        }
        return request;
    }

    static constexpr size_t particleCount() noexcept {
        PROFILE();
        return totalParticles();
    }

private:
    static constexpr size_t kTargets_ = 8;
    static constexpr size_t kSourceCells_ = 7;
    static constexpr size_t kSourceParticles_ = 12;
    static constexpr std::array<size_t, kTargets_> kSinkCounts_{
        0, 1, 31, 32, 33, cosmictiger::gravity::kBucketSize, cosmictiger::gravity::kBucketSize + 1, 300};
    static constexpr std::array<size_t, kTargets_> kSourceCounts_{5, 0, 31, 32, 33, 70, 1, 64};

    static constexpr size_t totalParticles() noexcept {
        PROFILE();
        size_t total = kSourceCells_ * kSourceParticles_;
        for (size_t count : kSinkCounts_) {
            total += count;
        }
        return total;
    }

    static constexpr size_t totalSources() noexcept {
        PROFILE();
        size_t total = 0;
        for (size_t count : kSourceCounts_) {
            total += count;
        }
        return total;
    }

    test::TestCells cells_;
    std::array<ManagedArray<Coordinate>, NDIM> positions_;
    ManagedArray<CellMultipole> multipoles_;
    ManagedArray<fmm::FmmMultipoleSource> sources_;
    ManagedArray<fmm::FmmParticleTarget> targets_;
};

void expectExpansionsNear(ManagedArray<CellExpansion>& device, ManagedArray<CellExpansion>& host, int order, char const* label) {
    PROFILE();
    for (size_t target = 0; target < kTargetCells; target++) {
        for (int rank = 0; rank <= order; rank++) {
            double const rankTolerance = rank >= kHighRank ? kDeviceTolerance * kHighRankToleranceFactor : kDeviceTolerance;
            double rankScale = 0.0;
            for (int coefficient = rank * rank; coefficient < (rank + 1) * (rank + 1); coefficient++) {
                rankScale = std::max(rankScale, static_cast<double>(std::abs(host[target][coefficient])));
            }
            for (int coefficient = rank * rank; coefficient < (rank + 1) * (rank + 1); coefficient++) {
                EXPECT_NEAR(device[target][coefficient], host[target][coefficient], rankTolerance * rankScale + 1e-30)
                    << label << " order " << order << " target " << target << " coefficient " << coefficient;
            }
        }
        for (size_t coefficient = static_cast<size_t>((order + 1) * (order + 1)); coefficient < host[target].size(); coefficient++) {
            EXPECT_EQ(device[target][coefficient], 0.0f) << label << " inactive coefficient " << coefficient;
        }
    }
}

class FmmInteractionsCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cosmictiger::cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable";
            }
        } catch (cosmictiger::cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        if (!cosmictiger::cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
        stream_ = cosmictiger::cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cosmictiger::cuda::streamDestroy(stream_).get());
        }
    }

    cudaStream_t stream() const noexcept {
        PROFILE();
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

TEST_F(FmmInteractionsCudaTest, M2lMatchesCpuAtEveryOrder) {
    PROFILE();
    DeviceScene scene;
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        ManagedArray<CellExpansion> device(kTargetCells);
        ManagedArray<CellExpansion> host(kTargetCells);
        fmm::launchM2lCuda(scene.m2l(device, kOrder), stream()).get();
        fmm::addM2lCpu(scene.m2l(host, kOrder));
        expectExpansionsNear(device, host, kOrder, "M2L");
    });
}

TEST_F(FmmInteractionsCudaTest, P2lMatchesCpuAtEveryOrder) {
    PROFILE();
    DeviceScene scene;
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        ManagedArray<CellExpansion> device(kTargetCells);
        ManagedArray<CellExpansion> host(kTargetCells);
        fmm::launchP2lCuda(scene.p2l(device, kOrder), stream()).get();
        fmm::addP2lCpu(scene.p2l(host, kOrder));
        expectExpansionsNear(device, host, kOrder, "P2L");
    });
}

TEST_F(FmmInteractionsCudaTest, M2pMatchesCpuAtEveryOrder) {
    PROFILE();
    DeviceScene scene;
    size_t const count = DeviceScene::particleCount();
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        std::array<ManagedArray<float>, NDIM + 1> device{
            ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count)};
        std::array<ManagedArray<float>, NDIM + 1> host{
            ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count)};
        fmm::launchM2pCuda(scene.m2p(device, kOrder), stream()).get();
        fmm::addM2pCpu(scene.m2p(host, kOrder));
        for (size_t sink = 0; sink < count; sink++) {
            double const accelerationScale = std::sqrt(static_cast<double>(host[1][sink]) * host[1][sink] + static_cast<double>(host[2][sink]) * host[2][sink] +
                static_cast<double>(host[3][sink]) * host[3][sink]);
            EXPECT_NEAR(device[0][sink], host[0][sink], kDeviceTolerance * std::abs(host[0][sink])) << "order " << kOrder << " sink " << sink;
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(device[axis + 1][sink], host[axis + 1][sink], kDeviceTolerance * accelerationScale)
                    << "order " << kOrder << " sink " << sink << " axis " << axis;
            }
        }
    });
}

TEST_F(FmmInteractionsCudaTest, M2pTilesSourcesAndPassesOverLargeBuckets) {
    PROFILE();
    TiledM2pScene scene;
    size_t const count = TiledM2pScene::particleCount();
    forEachOrder([&](auto orderConstant) {
        constexpr int kOrder = decltype(orderConstant)::value;
        PROFILE();
        std::array<ManagedArray<float>, NDIM + 1> device{
            ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count)};
        std::array<ManagedArray<float>, NDIM + 1> repeat{
            ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count)};
        std::array<ManagedArray<float>, NDIM + 1> host{
            ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count), ManagedArray<float>(count)};
        fmm::launchM2pCuda(scene.m2p(device, kOrder), stream()).get();
        fmm::launchM2pCuda(scene.m2p(repeat, kOrder), stream()).get();
        fmm::addM2pCpu(scene.m2p(host, kOrder));
        for (size_t sink = 0; sink < count; sink++) {
            double const accelerationScale = std::sqrt(static_cast<double>(host[1][sink]) * host[1][sink] + static_cast<double>(host[2][sink]) * host[2][sink] +
                static_cast<double>(host[3][sink]) * host[3][sink]);
            EXPECT_NEAR(device[0][sink], host[0][sink], kDeviceTolerance * std::abs(host[0][sink])) << "order " << kOrder << " sink " << sink;
            for (size_t component = 0; component < NDIM + 1; component++) {
                EXPECT_EQ(device[component][sink], repeat[component][sink]) << "order " << kOrder << " sink " << sink << " component " << component;
            }
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(device[axis + 1][sink], host[axis + 1][sink], kDeviceTolerance * accelerationScale)
                    << "order " << kOrder << " sink " << sink << " axis " << axis;
            }
        }
    });
}

TEST_F(FmmInteractionsCudaTest, AccumulatesIntoExistingOutputs) {
    constexpr int kOrder = 5;
    constexpr float kSentinel = 0.25f;
    PROFILE();
    DeviceScene scene;
    ManagedArray<CellExpansion> fresh(kTargetCells);
    ManagedArray<CellExpansion> preset(kTargetCells);
    for (size_t target = 0; target < kTargetCells; target++) {
        preset[target].fill(kSentinel);
    }
    fmm::launchM2lCuda(scene.m2l(fresh, kOrder), stream()).get();
    fmm::launchM2lCuda(scene.m2l(preset, kOrder), stream()).get();
    for (size_t target = 0; target < kTargetCells; target++) {
        for (size_t coefficient = 0; coefficient < preset[target].size(); coefficient++) {
            float const expected = coefficient < static_cast<size_t>((kOrder + 1) * (kOrder + 1)) ? kSentinel + fresh[target][coefficient] : kSentinel;
            EXPECT_FLOAT_EQ(preset[target][coefficient], expected) << target << ' ' << coefficient;
        }
    }
}

TEST_F(FmmInteractionsCudaTest, ReportsInvalidRequestsThroughTheFuture) {
    PROFILE();
    DeviceScene scene;
    ManagedArray<CellExpansion> expansions(kTargetCells);
    EXPECT_THROW(fmm::launchM2lCuda(scene.m2l(expansions, kMaximumFmmOrder + 1), stream()).get(), std::invalid_argument);
    auto missingTargets = scene.p2l(expansions, kMinimumFmmOrder);
    missingTargets.targets = nullptr;
    EXPECT_THROW(fmm::launchP2lCuda(missingTargets, stream()).get(), std::invalid_argument);
    std::array<ManagedArray<float>, NDIM + 1> field{ManagedArray<float>(1), ManagedArray<float>(1), ManagedArray<float>(1), ManagedArray<float>(1)};
    auto missingOutputs = scene.m2p(field, kMinimumFmmOrder);
    missingOutputs.accelerations[2] = nullptr;
    EXPECT_THROW(fmm::launchM2pCuda(missingOutputs, stream()).get(), std::invalid_argument);
    EXPECT_NO_THROW(fmm::launchM2lCuda(fmm::FmmM2lRequest{nullptr, 0, nullptr, 0, test::kSpacing, kMinimumFmmOrder}, stream()).get());
}

}    // namespace
