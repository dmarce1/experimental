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
#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/gravity/direct/direct_solve.hpp>
#include <cosmictiger/gravity/direct/direct_solve_kernel.hpp>
#include <cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <stdexcept>

namespace {

using cosmictiger::Coordinate;
using cosmictiger::Position;
using cosmictiger::gravity::compareDirectFields;
using cosmictiger::gravity::DirectEwaldParameters;
using cosmictiger::gravity::directSolveCpu;
using cosmictiger::gravity::DirectSolveField;
using cosmictiger::gravity::DirectSolveRequest;
using cosmictiger::gravity::sampleDirectSinks;

constexpr double kPeriodicSelfPotential = 2.837297479480619;

Position directTestPosition(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    PROFILE();
    return {Coordinate::fromRaw(x), Coordinate::fromRaw(y), Coordinate::fromRaw(z)};
}

}    // namespace

TEST(DirectSolve, SamplingUsesExactDistinctCountAndIndependentReproducibleSeed) {
    constexpr size_t kSourceCount = 1000;
    constexpr double kFraction = 0.125;
    constexpr size_t kSinkCount = 125;
    constexpr std::uint64_t kDefaultSeed = 42;
    constexpr std::uint64_t kOtherSeed = 43;
    PROFILE();
    auto const first = sampleDirectSinks(kSourceCount, kFraction);
    EXPECT_EQ(first, sampleDirectSinks(kSourceCount, kFraction, kDefaultSeed));
    EXPECT_NE(first, sampleDirectSinks(kSourceCount, kFraction, kOtherSeed));
    ASSERT_EQ(first.size(), kSinkCount);
    for (size_t index = 0; index < first.size(); index++) {
        EXPECT_LT(first[index], kSourceCount);
        if (index != 0) {
            EXPECT_LT(first[index - 1], first[index]);
        }
    }
    EXPECT_EQ(sampleDirectSinks(13, 0.1).size(), size_t(2));
    EXPECT_EQ(sampleDirectSinks(13, 0x1p-40).size(), size_t(1));
    EXPECT_TRUE(sampleDirectSinks(kSourceCount, 0.0).empty());
    EXPECT_TRUE(sampleDirectSinks(0, 1.0).empty());
    auto const all = sampleDirectSinks(kSourceCount, 1.0);
    ASSERT_EQ(all.size(), kSourceCount);
    for (size_t index = 0; index < all.size(); index++) {
        EXPECT_EQ(all[index], index);
    }
    EXPECT_THROW(sampleDirectSinks(kSourceCount, -0.01), std::invalid_argument);
    EXPECT_THROW(sampleDirectSinks(kSourceCount, 1.01), std::invalid_argument);
}

TEST(DirectSolve, PrimarySelfIsExcludedButPeriodicSelfAndZeroMeanGaugeRemain) {
    constexpr size_t kSink = 0;
    constexpr double kParticleMass = 0.25;
    constexpr double kTolerance = 2e-13;
    PROFILE();
    Position const source = directTestPosition(0x12345678U, 0x87654321U, 0xabcdef01U);
    DirectSolveField field;
    DirectSolveRequest const request{&source, 1, &kSink, 1, &field, kParticleMass, 0.03125, 2};
    directSolveCpu(request);
    EXPECT_NEAR(field.potential, kParticleMass * kPeriodicSelfPotential, kTolerance);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_DOUBLE_EQ(field.acceleration[axis], 0.0);
    }
}

TEST(DirectSolve, AgreesWithPmmmPeriodicGreenAtNearAndFarCellCenters) {
    constexpr size_t kSide = 8;
    constexpr double kSpacing = 0.125;
    constexpr std::int64_t kCellRawSize = std::int64_t(1) << 29;
    constexpr size_t kSink = 0;
    constexpr double kTolerance = 8e-13;
    constexpr std::array<size_t, NDIM> kGradientIndices = {2, 1, 3};
    constexpr std::array<std::array<int, NDIM>, 3> kOffsets = {{{1, 0, 0}, {-2, 1, 0}, {3, 2, -1}}};
    constexpr DirectEwaldParameters kParameters{};
    PROFILE();
    for (auto const& offset : kOffsets) {
        std::array<Position, 2> const sources = {directTestPosition(0, 0, 0),
            directTestPosition(static_cast<std::uint32_t>(-offset[0] * kCellRawSize), static_cast<std::uint32_t>(-offset[1] * kCellRawSize),
                static_cast<std::uint32_t>(-offset[2] * kCellRawSize))};
        DirectSolveField field;
        DirectSolveRequest const request{sources.data(), sources.size(), &kSink, 1, &field, 1.0, 0.03125, 2};
        directSolveCpu(request, kParameters);
        auto green = cosmictiger::gravity::pmmmPeriodicGreen<1, 2>(offset, kSide, kSpacing, kParameters.imageCutoff, kParameters.waveCutoff, kParameters.alpha);
        if (cosmictiger::gravity::pmmmNear<2>(offset)) {
            std::array<double, NDIM> separation{};
            for (size_t axis = 0; axis < NDIM; axis++) {
                separation[axis] = kSpacing * offset[axis];
            }
            auto const primary = cosmictiger::gravity::pmmmFreeSpaceGreen<1>(separation, kSpacing);
            for (size_t coefficient = 0; coefficient < green.size(); coefficient++) {
                green[coefficient] += primary[coefficient];
            }
        }
        EXPECT_NEAR(field.potential - kPeriodicSelfPotential, green[0], kTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(field.acceleration[axis], -green[kGradientIndices[axis]] / kSpacing, kTolerance);
        }
    }
}

TEST(DirectSolve, EwaldSplitAndPeriodicTranslationLeaveTheSolutionUnchanged) {
    constexpr size_t kSourceCount = 4;
    constexpr double kTolerance = 2e-12;
    constexpr DirectEwaldParameters kFirstParameters{1.5, 5, 5};
    constexpr DirectEwaldParameters kSecondParameters{2.5, 5, 5};
    constexpr std::array<std::uint32_t, NDIM> kTranslation = {0x80000000U, 0x654321feU, 0x23456789U};
    constexpr std::array<size_t, kSourceCount> kSinks = {0, 1, 2, 3};
    PROFILE();
    std::array<Position, kSourceCount> sources = {directTestPosition(0x11111111U, 0x22222222U, 0x33333333U),
        directTestPosition(0x44444444U, 0x55555555U, 0x66666666U), directTestPosition(0x99887766U, 0x11223344U, 0x55667788U),
        directTestPosition(0x01234567U, 0x89abcdefU, 0xfedcba98U)};
    std::array<DirectSolveField, kSourceCount> first;
    std::array<DirectSolveField, kSourceCount> second;
    std::array<DirectSolveField, kSourceCount> translated;
    DirectSolveRequest request{sources.data(), sources.size(), kSinks.data(), kSinks.size(), first.data(), 0.25, 0.015625, 3};
    directSolveCpu(request, kFirstParameters);
    request.fields = second.data();
    directSolveCpu(request, kSecondParameters);
    for (Position& source : sources) {
        source = directTestPosition(source.x.raw() + kTranslation[0], source.y.raw() + kTranslation[1], source.z.raw() + kTranslation[2]);
    }
    request.fields = translated.data();
    directSolveCpu(request, kFirstParameters);
    std::array<double, NDIM> totalAcceleration{};
    for (size_t index = 0; index < kSourceCount; index++) {
        EXPECT_NEAR(first[index].potential, second[index].potential, kTolerance);
        EXPECT_DOUBLE_EQ(first[index].potential, translated[index].potential);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(first[index].acceleration[axis], second[index].acceleration[axis], kTolerance);
            EXPECT_DOUBLE_EQ(first[index].acceleration[axis], translated[index].acceleration[axis]);
            totalAcceleration[axis] += first[index].acceleration[axis];
        }
    }
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_NEAR(totalAcceleration[axis], 0.0, kTolerance);
    }
}

TEST(DirectSolve, DistinctCoincidentParticlesRemainSoftenedAndNearZeroForceIsStable) {
    constexpr size_t kSink = 0;
    constexpr double kSoftening = 0.0625;
    constexpr double kTolerance = 3e-13;
    constexpr std::uint32_t kSeparationRaw = 32;
    constexpr double kSeparation = 0x1p-27;
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr std::array<double, 4> kPotentialAtOrigin = {1.5, 1.875, 2.1875, 2.4609375};
    constexpr std::array<double, 4> kForceAtOrigin = {1.0, 2.5, 4.375, 6.5625};
    PROFILE();
    std::array<Position, 2> sources = {directTestPosition(0, 0, 0), directTestPosition(0, 0, 0)};
    DirectSolveField field;
    DirectSolveRequest request{sources.data(), sources.size(), &kSink, 1, &field, 1.0, kSoftening, 0};
    for (int order = 0; order < static_cast<int>(kPotentialAtOrigin.size()); order++) {
        request.ferrersOrder = order;
        sources[1] = directTestPosition(0, 0, 0);
        directSolveCpu(request);
        EXPECT_NEAR(field.potential, 2.0 * kPeriodicSelfPotential - kPotentialAtOrigin[order] / kSoftening, kTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_DOUBLE_EQ(field.acceleration[axis], 0.0);
        }
        sources[1] = directTestPosition(kSeparationRaw, 0, 0);
        directSolveCpu(request);
        double const expectedForce = (kForceAtOrigin[order] / (kSoftening * kSoftening * kSoftening) - 4.0 * kPi / NDIM) * kSeparation;
        EXPECT_NEAR(field.acceleration[0], expectedForce, kTolerance);
        EXPECT_NEAR(field.acceleration[1], 0.0, kTolerance);
        EXPECT_NEAR(field.acceleration[2], 0.0, kTolerance);
    }
}

TEST(DirectSolve, EmptyRequestsDoNothingAndInvalidRequestsReportErrors) {
    constexpr size_t kInvalidSink = 1;
    PROFILE();
    Position const source = directTestPosition(0, 0, 0);
    DirectSolveField field;
    EXPECT_NO_THROW(directSolveCpu({}));
    EXPECT_THROW(directSolveCpu({nullptr, 1, &kInvalidSink, 1, &field}), std::invalid_argument);
    if (cosmictiger::validationChecksEnabled()) {
        EXPECT_THROW(directSolveCpu({&source, 1, &kInvalidSink, 1, &field}), std::runtime_error);
    }
}

TEST(DirectSolve, ComparisonNormsNormalizeVectorAndPotentialErrorsByCorrespondingReferenceNorms) {
    constexpr double kTolerance = 2e-15;
    constexpr std::array<DirectSolveField, 2> kReference = {DirectSolveField{{3.0, 4.0, 0.0}, -3.0}, DirectSolveField{{0.0, 0.0, 12.0}, 4.0}};
    constexpr std::array<DirectSolveField, 2> kApproximation = {DirectSolveField{{6.0, 8.0, 0.0}, -6.0}, DirectSolveField{{0.0, 0.0, 10.0}, 2.0}};
    PROFILE();
    auto const result = compareDirectFields(kReference, kApproximation);
    EXPECT_EQ(result.sinkCount, kReference.size());
    EXPECT_DOUBLE_EQ(result.acceleration.l1, 3.5);
    EXPECT_NEAR(result.acceleration.l2, std::sqrt(14.5), kTolerance);
    EXPECT_DOUBLE_EQ(result.acceleration.linf, 5.0);
    EXPECT_DOUBLE_EQ(result.acceleration.referenceL1, 8.5);
    EXPECT_NEAR(result.acceleration.referenceL2, std::sqrt(84.5), kTolerance);
    EXPECT_DOUBLE_EQ(result.acceleration.referenceLinf, 12.0);
    EXPECT_NEAR(result.acceleration.relativeL1, 7.0 / 17.0, kTolerance);
    EXPECT_NEAR(result.acceleration.relativeL2, std::sqrt(29.0) / 13.0, kTolerance);
    EXPECT_NEAR(result.acceleration.relativeLinf, 5.0 / 12.0, kTolerance);
    EXPECT_NEAR(result.potential.relativeL1, 5.0 / 7.0, kTolerance);
    EXPECT_NEAR(result.potential.relativeL2, std::sqrt(13.0) / 5.0, kTolerance);
    EXPECT_NEAR(result.potential.relativeLinf, 3.0 / 4.0, kTolerance);
}

TEST(DirectSolve, ComparisonHandlesZeroReferenceAndEmptySamplesWithoutDividingByZero) {
    constexpr std::array<DirectSolveField, 1> kZero = {};
    constexpr std::array<DirectSolveField, 1> kNonzero = {DirectSolveField{{1.0, 0.0, 0.0}, 2.0}};
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    PROFILE();
    auto const exact = compareDirectFields(kZero, kZero);
    EXPECT_DOUBLE_EQ(exact.acceleration.relativeL1, 0.0);
    EXPECT_DOUBLE_EQ(exact.acceleration.relativeL2, 0.0);
    EXPECT_DOUBLE_EQ(exact.acceleration.relativeLinf, 0.0);
    EXPECT_DOUBLE_EQ(exact.potential.relativeL1, 0.0);
    auto const nonzero = compareDirectFields(kZero, kNonzero);
    EXPECT_EQ(nonzero.acceleration.relativeL1, kInfinity);
    EXPECT_EQ(nonzero.acceleration.relativeL2, kInfinity);
    EXPECT_EQ(nonzero.acceleration.relativeLinf, kInfinity);
    EXPECT_EQ(nonzero.potential.relativeL1, kInfinity);
    auto const empty = compareDirectFields({}, {});
    EXPECT_EQ(empty.sinkCount, size_t(0));
    EXPECT_DOUBLE_EQ(empty.acceleration.relativeL1, 0.0);
    EXPECT_DOUBLE_EQ(empty.potential.relativeL2, 0.0);
    EXPECT_THROW(compareDirectFields(kZero, {}), std::invalid_argument);
}

TEST(DirectSolve, PlummerEwaldAgreesWithIndependentBesselFourierSeries) {
    constexpr int kWaveCutoff = 24;
    constexpr size_t kMaximumWaveSquared = NDIM * kWaveCutoff * kWaveCutoff;
    constexpr double kEpsilon = 0.25;
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kTolerance = 2.0e-11;
    constexpr std::array<double, NDIM> kDisplacement{0.125, 0.1875, -0.25};
    constexpr cosmictiger::gravity::SofteningModel kModel = cosmictiger::gravity::SofteningModel::plummer;
    constexpr std::array<DirectEwaldParameters, 2> kParameters{{{1.5, 5, 5}, {2.5, 5, 5}}};
    PROFILE();
    // The full Plummer Fourier transform is -4*pi*epsilon*K1(epsilon*k)/k.
    // This independent, unsplit reference has no reciprocal zero mode.
    std::array<double, kMaximumWaveSquared + 1> coefficients{};
    for (size_t squared = 1; squared <= kMaximumWaveSquared; squared++) {
        double const magnitude = std::sqrt(double(squared));
        coefficients[squared] = -2.0 * kEpsilon * std::cyl_bessel_k(1.0, 2.0 * kPi * kEpsilon * magnitude) / magnitude;
    }
    DirectSolveField reference;
    double selfPotential = 1.0 / kEpsilon;
    for (int z = -kWaveCutoff; z <= kWaveCutoff; z++) {
        for (int y = -kWaveCutoff; y <= kWaveCutoff; y++) {
            for (int x = -kWaveCutoff; x <= kWaveCutoff; x++) {
                int const squared = x * x + y * y + z * z;
                if (squared == 0) {
                    continue;
                }
                std::array<int, NDIM> const wave{x, y, z};
                double phase = 0.0;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    phase += double(wave[axis]) * kDisplacement[axis];
                }
                phase *= 2.0 * kPi;
                double const coefficient = coefficients[squared];
                reference.potential += coefficient * std::cos(phase);
                selfPotential += coefficient;
                for (size_t axis = 0; axis < NDIM; axis++) {
                    reference.acceleration[axis] += coefficient * 2.0 * kPi * double(wave[axis]) * std::sin(phase);
                }
            }
        }
    }
    Position const source = directTestPosition(0, 0, 0);
    Position const target = directTestPosition(0x20000000U, 0x30000000U, 0xc0000000U);
    for (auto const& parameters : kParameters) {
        auto const actual = cosmictiger::gravity::detail::directPeriodicPair(target, source, false, kEpsilon, 0, parameters, kModel);
        EXPECT_NEAR(actual.potential, reference.potential, kTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(actual.acceleration[axis], reference.acceleration[axis], kTolerance);
        }
        auto const self = cosmictiger::gravity::detail::directPeriodicPair(source, source, true, kEpsilon, 0, parameters, kModel);
        EXPECT_NEAR(self.potential, selfPotential, kTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_DOUBLE_EQ(self.acceleration[axis], 0.0);
        }
    }
}

TEST(DirectSolve, PlummerCachedWeightsPreserveTheSplitAndCoincidentPair) {
    constexpr size_t kCount = 2;
    constexpr std::array<size_t, kCount> kSinks{0, 1};
    constexpr double kTolerance = 3.0e-12;
    constexpr double kEpsilon = 0.03125;
    constexpr DirectEwaldParameters kFirst{1.5, 5, 5};
    constexpr DirectEwaldParameters kSecond{2.5, 5, 5};
    PROFILE();
    std::array<Position, kCount> sources{directTestPosition(0, 0, 0), directTestPosition(0x40000000U, 0x20000000U, 0)};
    std::array<DirectSolveField, kCount> first{};
    std::array<DirectSolveField, kCount> second{};
    DirectSolveRequest request{sources.data(), kCount, kSinks.data(), kCount, first.data(), 1.0, kEpsilon, 2, cosmictiger::gravity::SofteningModel::plummer};
    directSolveCpu(request, kFirst);
    request.fields = second.data();
    directSolveCpu(request, kSecond);
    for (size_t sink = 0; sink < kCount; sink++) {
        EXPECT_NEAR(first[sink].potential, second[sink].potential, kTolerance);
        for (size_t axis = 0; axis < NDIM; axis++) {
            EXPECT_NEAR(first[sink].acceleration[axis], second[sink].acceleration[axis], kTolerance);
            EXPECT_NEAR(first[0].acceleration[axis], -first[1].acceleration[axis], kTolerance);
        }
    }
    auto const self = cosmictiger::gravity::detail::directPeriodicPair(sources[0], sources[0], true, kEpsilon, 2, kFirst, request.softeningModel);
    sources[1] = sources[0];
    request.fields = first.data();
    directSolveCpu(request, kFirst);
    EXPECT_NEAR(first[0].potential, 2.0 * self.potential - 1.0 / kEpsilon, kTolerance);
    for (size_t axis = 0; axis < NDIM; axis++) {
        EXPECT_DOUBLE_EQ(first[0].acceleration[axis], 0.0);
    }
}
