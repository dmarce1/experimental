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

#include <cosmictiger/gravity/stf/m2p_cuda.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <new>
#include <stddef.h>
#include <stdexcept>

namespace {

namespace stf = cosmictiger::gravity::stf;

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(size_t count)
      : count_(count) {
        PROFILE();
        values_ = static_cast<T*>(cosmictiger::cuda::mallocManaged(count_ * sizeof(T)).get());
        for (size_t index = 0; index < count_; index++) {
            ::new (static_cast<void*>(values_ + index)) T{};
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
                ADD_FAILURE() << "managed M2P buffer release failed: " << error.what();
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

    T const& operator[](size_t index) const noexcept {
        PROFILE();
        return values_[index];
    }

private:
    T* values_ = nullptr;
    size_t count_ = 0;
};

class M2pCudaTest : public ::testing::Test {
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

TEST_F(M2pCudaTest, BatchedFieldsMatchCpuForCenteredAndUncenteredMultipoles) {
    constexpr int kOrder = COSMICTIGER_MULTIPOLE_ORDER;
    constexpr size_t kCoefficientCount = stf::coefficientCount(kOrder);
    constexpr size_t kSampleCount = 517;
    PROFILE();
    stf::Multipole<double, kOrder> mixed{};
    stf::Multipole<double, kOrder> highestRank{};
    for (size_t index = 0; index < kCoefficientCount; index++) {
        mixed[index] = (index % 2 == 0 ? 1.0 : -1.0) * 0.2 / static_cast<double>(index + 1);
    }
    mixed[0] = 1.75;
    highestRank[static_cast<size_t>(kOrder * kOrder)] = 0.375;

    ManagedArray<double> sources(2 * kCoefficientCount);
    ManagedArray<stf::M2pCudaSample> samples(kSampleCount);
    ManagedArray<stf::M2pCudaField> fields(kSampleCount);
    for (size_t index = 0; index < kCoefficientCount; index++) {
        sources[index] = mixed[index];
        sources[kCoefficientCount + index] = highestRank[index];
    }
    for (size_t index = 0; index < kSampleCount; index++) {
        samples[index].source = sources.data() + (index % 2) * kCoefficientCount;
        samples[index].separation[0] = 1.1 + 0.03 * static_cast<double>(index % 17);
        samples[index].separation[1] = -0.8 + 0.02 * static_cast<double>(index % 13);
        samples[index].separation[2] = 0.5 + 0.01 * static_cast<double>(index % 19);
        samples[index].sourceScale = 0.35 + 0.04 * static_cast<double>(index % 7);
    }

    stf::M2pCudaRequest const request{samples.data(), fields.data(), kSampleCount};
    for (bool centered : {false, true}) {
        for (size_t index = 0; index < kSampleCount; index++) {
            fields[index].potential = 12345.0;
            for (size_t axis = 0; axis < NDIM; axis++) {
                fields[index].acceleration[axis] = 12345.0;
            }
        }
        auto completion = stf::launchM2pCuda(request, centered, stream());
        ASSERT_TRUE(completion.valid());
        ASSERT_NO_THROW(completion.get());
        for (size_t index = 0; index < kSampleCount; index++) {
            SCOPED_TRACE(index);
            auto const& source = index % 2 == 0 ? mixed : highestRank;
            std::array<double, NDIM> const separation{samples[index].separation[0], samples[index].separation[1], samples[index].separation[2]};
            auto const expected = centered ? stf::evaluateM2P<double, kOrder, true>(source, separation, samples[index].sourceScale) :
                                             stf::evaluateM2P<double, kOrder, false>(source, separation, samples[index].sourceScale);
            EXPECT_NEAR(fields[index].potential, expected.potential, 5.0e-11 * (1.0 + std::abs(expected.potential)));
            for (size_t axis = 0; axis < NDIM; axis++) {
                EXPECT_NEAR(fields[index].acceleration[axis], expected.acceleration[axis], 5.0e-11 * (1.0 + std::abs(expected.acceleration[axis])));
            }
        }
    }
}

TEST(M2pCudaRequests, EmptyAndMissingBuffersCompleteOrFailBeforeSubmission) {
    PROFILE();
    auto empty = stf::launchM2pCuda({});
    ASSERT_TRUE(empty.is_ready());
    EXPECT_NO_THROW(empty.get());

    auto missing = stf::launchM2pCuda({nullptr, nullptr, 1});
    ASSERT_TRUE(missing.valid());
    EXPECT_THROW(missing.get(), std::invalid_argument);
}

}    // namespace
