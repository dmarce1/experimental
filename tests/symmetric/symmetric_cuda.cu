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

#include <cosmictiger/gravity/config.hpp>
#include <cosmictiger/gravity/symmetric/m2p_cuda.hpp>
#include <cosmictiger/gravity/symmetric/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <exception>
#include <new>
#include <stddef.h>
#include <stdexcept>

namespace {

namespace symmetric = cosmictiger::gravity::symmetric;
namespace cuda = cosmictiger::cuda;

inline constexpr int kSourceOrder = 3;
inline constexpr int kLocalOrder = 4;

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(size_t count) {
        PROFILE();
        values_ = static_cast<T*>(cuda::mallocManaged(count * sizeof(T)).get());
        for (size_t element = 0; element < count; element++) {
            ::new (static_cast<void*>(values_ + element)) T{};
        }
    }

    ManagedArray(ManagedArray const&) = delete;
    ManagedArray& operator=(ManagedArray const&) = delete;

    ~ManagedArray() {
        PROFILE();
        if (values_ != nullptr) {
            try {
                cuda::free(values_).get();
            } catch (std::exception const& error) {
                ADD_FAILURE() << "Full Cartesian CUDA buffer release failed: " << error.what();
            }
        }
    }

    T* data() noexcept {
        PROFILE();
        return values_;
    }

    T& operator[](size_t element) noexcept {
        PROFILE();
        return values_[element];
    }

private:
    T* values_ = nullptr;
};

class SymmetricCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        PROFILE();
        try {
            if (cuda::deviceCount().get() == 0) {
                GTEST_SKIP() << "CUDA device unavailable";
            }
        } catch (cuda::CudaError const& error) {
            if (error.code() == cudaErrorNoDevice || error.code() == cudaErrorInsufficientDriver) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        if (!cuda::deviceProperties(0).get().managedMemory) {
            GTEST_SKIP() << "CUDA device does not support managed memory";
        }
        stream_ = cuda::streamCreate().get();
    }

    void TearDown() override {
        PROFILE();
        if (stream_ != nullptr) {
            EXPECT_NO_THROW(cuda::streamDestroy(stream_).get());
        }
    }

    cudaStream_t stream() const noexcept {
        PROFILE();
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

struct OperatorOutputs {
    symmetric::Multipole<double, kSourceOrder> moments;
    symmetric::Multipole<double, kSourceOrder> shiftedMoments;
    symmetric::Expansion<double, kLocalOrder> pointLocal;
    symmetric::Expansion<double, kLocalOrder> momentLocal;
    symmetric::Expansion<double, kLocalOrder> shiftedLocal;
    symmetric::Expansion<double, kSourceOrder + kLocalOrder> green;
    symmetric::Field<double> p2p;
    symmetric::Field<double> m2p;
    symmetric::Field<double> l2p;
    double radiusSquared;
};

__host__ __device__ OperatorOutputs exerciseOperators(double softening) {
    constexpr double kSourceScale = 0.2;
    constexpr double kTargetScale = 0.4;
    constexpr double kTranslatedScale = 0.3;
    constexpr double kMass = 1.7;
    constexpr std::array<double, NDIM> kSourceOffset{0.021, -0.017, 0.013};
    constexpr std::array<double, NDIM> kSecondOffset{-0.031, 0.024, -0.005};
    constexpr std::array<double, NDIM> kSeparation{0.8, -0.7, 0.9};
    constexpr std::array<double, NDIM> kShift{0.12, 0.03, -0.08};
    constexpr std::array<double, NDIM> kTargetOffset{-0.02, 0.04, 0.015};
    PROFILE();
    OperatorOutputs result{};
    result.radiusSquared = symmetric::addPoint<double, kSourceOrder>(result.moments, kMass, kSourceOffset, kSourceScale);
    symmetric::addPoint<double, kSourceOrder>(result.moments, kMass, kSecondOffset, kSourceScale);
    symmetric::addM2M<double, kSourceOrder>(result.shiftedMoments, result.moments, kShift, kSourceScale, kTranslatedScale);
    symmetric::addP2L<double, kLocalOrder>(result.pointLocal, kMass, kSeparation, kTargetScale, softening);
    symmetric::addM2L<double, kSourceOrder, kLocalOrder>(result.momentLocal, result.moments, kSeparation, kSourceScale, kTargetScale, softening);
    symmetric::addL2L<double, kLocalOrder>(result.shiftedLocal, result.momentLocal, kShift, kTargetScale, kTranslatedScale);
    result.green = symmetric::greenDerivatives<double, kSourceOrder + kLocalOrder>(kSeparation, softening, kTargetScale);
    result.p2p = symmetric::evaluateP2P(kMass, kSeparation, softening);
    result.m2p = symmetric::evaluateM2P<double, kSourceOrder>(result.moments, kSeparation, kSourceScale, softening);
    result.l2p = symmetric::evaluateL2P<double, kLocalOrder>(result.momentLocal, kTargetOffset, kTargetScale);
    return result;
}

__global__ void exerciseOperatorsKernel(OperatorOutputs* outputs) {
    constexpr unsigned kScenarios = 2;
    constexpr double kSoftening = 0.3;
    PROFILE();
    if (threadIdx.x < kScenarios) {
        outputs[threadIdx.x] = exerciseOperators(threadIdx.x == 0 ? 0.0 : kSoftening);
    }
}

template <size_t count>
void expectArrayNear(std::array<double, count> const& actual, std::array<double, count> const& expected) {
    constexpr double kTolerance = 2e-11;
    PROFILE();
    for (size_t coefficient = 0; coefficient < count; coefficient++) {
        EXPECT_NEAR(actual[coefficient], expected[coefficient], kTolerance * (1.0 + std::abs(expected[coefficient]))) << "coefficient=" << coefficient;
    }
}

void expectFieldNear(symmetric::Field<double> const& actual, symmetric::Field<double> const& expected) {
    constexpr double kTolerance = 2e-11;
    PROFILE();
    EXPECT_NEAR(actual.potential, expected.potential, kTolerance * (1.0 + std::abs(expected.potential)));
    expectArrayNear(actual.acceleration, expected.acceleration);
}

TEST_F(SymmetricCudaTest, EveryOperatorMatchesHostForNewtonianAndPlummerKernels) {
    constexpr size_t kScenarios = 2;
    constexpr unsigned kThreads = 32;
    constexpr double kSoftening = 0.3;
    constexpr double kRadiusTolerance = 2e-15;
    PROFILE();
    ManagedArray<OperatorOutputs> outputs(kScenarios);
    auto completion = cuda::launch(hpx::annotated_function(
                                       [pointer = outputs.data()](cudaStream_t launchStream) {
                                           PROFILE();
                                           exerciseOperatorsKernel<<<1, kThreads, 0, launchStream>>>(pointer);
                                       },
                                       "symmetric::testAllCudaOperators"),
        stream());
    ASSERT_NO_THROW(completion.get());
    for (size_t scenario = 0; scenario < kScenarios; scenario++) {
        SCOPED_TRACE(scenario);
        auto const expected = exerciseOperators(scenario == 0 ? 0.0 : kSoftening);
        auto const& actual = outputs[scenario];
        expectArrayNear(actual.moments, expected.moments);
        expectArrayNear(actual.shiftedMoments, expected.shiftedMoments);
        expectArrayNear(actual.pointLocal, expected.pointLocal);
        expectArrayNear(actual.momentLocal, expected.momentLocal);
        expectArrayNear(actual.shiftedLocal, expected.shiftedLocal);
        expectArrayNear(actual.green, expected.green);
        expectFieldNear(actual.p2p, expected.p2p);
        expectFieldNear(actual.m2p, expected.m2p);
        expectFieldNear(actual.l2p, expected.l2p);
        EXPECT_NEAR(actual.radiusSquared, expected.radiusSquared, kRadiusTolerance);
    }
}

TEST_F(SymmetricCudaTest, BatchedM2pPreservesFullTraceMomentsAndSelectedSoftening) {
    constexpr int kOrder = cosmictiger::gravity::kMultipoleOrder;
    constexpr size_t kCoefficientCount = symmetric::coefficientCount(kOrder);
    constexpr size_t kSampleCount = 7;
    constexpr double kSourceScale = 0.25;
    constexpr double kSoftening = 0.3;
    constexpr double kTolerance = 2e-11;
    PROFILE();
    symmetric::Multipole<double, kOrder> source{};
    for (size_t coefficient = 0; coefficient < kCoefficientCount; coefficient++) {
        source[coefficient] = 0.02 * double(int(coefficient % 7) - 3) / double(coefficient + 1);
    }
    source[0] = 1.7;
    source[symmetric::index(0, 0, kOrder)] = 0.05;
    ManagedArray<double> sources(kCoefficientCount);
    ManagedArray<symmetric::M2pCudaSample> samples(kSampleCount);
    ManagedArray<symmetric::M2pCudaField> fields(kSampleCount);
    for (size_t coefficient = 0; coefficient < kCoefficientCount; coefficient++) {
        sources[coefficient] = source[coefficient];
    }
    for (size_t sample = 0; sample < kSampleCount; sample++) {
        samples[sample].source = sources.data();
        samples[sample].separation[0] = 0.8 + 0.03 * double(sample);
        samples[sample].separation[1] = -0.6 + 0.02 * double(sample);
        samples[sample].separation[2] = 0.9 - 0.01 * double(sample);
        samples[sample].sourceScale = kSourceScale;
        samples[sample].softening = sample % 2 == 0 ? 0.0 : kSoftening;
    }
    auto completion = symmetric::launchM2pCuda({samples.data(), fields.data(), kSampleCount}, stream());
    ASSERT_TRUE(completion.valid());
    ASSERT_NO_THROW(completion.get());
    for (size_t sample = 0; sample < kSampleCount; sample++) {
        SCOPED_TRACE(sample);
        std::array<double, NDIM> const separation{samples[sample].separation[0], samples[sample].separation[1], samples[sample].separation[2]};
        auto const expected = symmetric::evaluateM2P<double, kOrder>(source, separation, kSourceScale, samples[sample].softening);
        EXPECT_NEAR(fields[sample].potential, expected.potential, kTolerance * (1.0 + std::abs(expected.potential)));
        for (int d = 0; d < NDIM; d++) {
            EXPECT_NEAR(fields[sample].acceleration[d], expected.acceleration[d], kTolerance * (1.0 + std::abs(expected.acceleration[d])));
        }
    }
}

TEST(SymmetricCudaRequests, EmptyAndMissingBuffersCompleteWithoutSubmission) {
    PROFILE();
    auto empty = symmetric::launchM2pCuda({});
    ASSERT_TRUE(empty.is_ready());
    EXPECT_NO_THROW(empty.get());
    auto missing = symmetric::launchM2pCuda({nullptr, nullptr, 1});
    EXPECT_THROW(missing.get(), std::invalid_argument);
}

}    // namespace
