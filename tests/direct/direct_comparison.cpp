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

#include <cosmictiger/gravity/direct/direct_comparison.hpp>
#include <cosmictiger/profiler.hpp>

#include <gtest/gtest.h>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/serialization/vector.hpp>

#include <array>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <vector>

namespace cosmictiger::detail {

void appendDirectComparisonNorms(std::vector<double>& values, gravity::DirectErrorNorms const& norms) {
    PROFILE();
    values.push_back(norms.l1);
    values.push_back(norms.l2);
    values.push_back(norms.linf);
    values.push_back(norms.referenceL1);
    values.push_back(norms.referenceL2);
    values.push_back(norms.referenceLinf);
    values.push_back(norms.relativeL1);
    values.push_back(norms.relativeL2);
    values.push_back(norms.relativeLinf);
}

void appendDirectComparison(std::vector<double>& values, gravity::DirectComparison const& comparison) {
    PROFILE();
    values.push_back(static_cast<double>(comparison.sourceCount));
    values.push_back(static_cast<double>(comparison.sinkCount));
    appendDirectComparisonNorms(values, comparison.acceleration);
    appendDirectComparisonNorms(values, comparison.potential);
}

// Each physical locality owns one synthetic interior range. The reference
// deliberately uses all source positions, including those stored remotely.
std::vector<double> runOwnedDirectComparison(size_t localityIndex, size_t particlesPerLocality, std::uint64_t operationId) {
    constexpr size_t kMeshExponent = 3;
    constexpr double kCoordinateScale = 1.0 / 128.0;
    constexpr double kMass = 1.0 / 64.0;
    constexpr double kSoftening = 0.02;
    constexpr double kSinkFraction = 0.5;
    constexpr double kAccelerationErrorScale = 0.25;
    constexpr double kPotentialErrorScale = 0.125;
    constexpr float kGhostField = 10000.0f;
    constexpr int kFerrersOrder = 2;
    constexpr std::uint64_t kSeed = 42;
    PROFILE();

    auto const localities = hpx::find_all_localities();
    size_t const sourceCount = particlesPerLocality * localities.size();
    size_t const sourceOffset = localityIndex * particlesPerLocality;
    std::vector<Position> sources(sourceCount);
    std::vector<size_t> allSinks(sourceCount);
    for (size_t source = 0; source < sourceCount; source++) {
        allSinks[source] = source;
        std::array<Coordinate, NDIM> coordinates{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            coordinates[axis] = Coordinate(static_cast<double>((source + 1) * (axis + 1)) * kCoordinateScale);
        }
        sources[source] = {coordinates[0], coordinates[1], coordinates[2]};
    }
    std::vector<gravity::DirectSolveField> reference(sourceCount);
    gravity::directSolveCpu({sources.data(), sources.size(), allSinks.data(), allSinks.size(), reference.data(), kMass, kSoftening, kFerrersOrder});
    std::vector<gravity::DirectSolveField> approximation = reference;
    for (size_t source = 0; source < sourceCount; source++) {
        for (size_t axis = 0; axis < NDIM; axis++) {
            approximation[source].acceleration[axis] =
                static_cast<float>(reference[source].acceleration[axis] + kAccelerationErrorScale * static_cast<double>((source + 1) * (axis + 1)));
        }
        approximation[source].potential = static_cast<float>(reference[source].potential - kPotentialErrorScale * static_cast<double>(source + 1));
    }

    CartesianGrid grid(kMeshExponent, 0, 0);
    ParticleStorage particles;
    particles.configureGravityFields(true, true);
    for (size_t particle = 0; particle < particlesPerLocality; particle++) {
        size_t const source = sourceOffset + particle;
        particles.append({sources[source].x, sources[source].y, sources[source].z}, {}, 0);
        for (size_t axis = 0; axis < NDIM; axis++) {
            particles.acceleration(axis, particle) = static_cast<float>(approximation[source].acceleration[axis]);
        }
        particles.potential(particle) = static_cast<float>(approximation[source].potential);
    }
    grid.setInteriorRange({0, 0, 0}, {0, particlesPerLocality});
    // Periodic copies can have their ghost flag clear. Only the grid's owned
    // ranges identify sources and sinks; poison copied fields to expose leaks.
    particles.append({sources[sourceOffset].x, sources[sourceOffset].y, sources[sourceOffset].z}, {}, 0);
    if (particles.flags(particlesPerLocality).isGhost()) {
        throw std::runtime_error("test requires an unflagged periodic ghost copy");
    }
    for (size_t axis = 0; axis < NDIM; axis++) {
        particles.acceleration(axis, particlesPerLocality) = kGhostField;
    }
    particles.potential(particlesPerLocality) = kGhostField;
    grid.setGhostRanges({-1, 0, 0}, {particlesPerLocality, particlesPerLocality + 1}, {0, 1});

    GlobalConstantsData constants;
    constants.globalParticleCount = sourceCount;
    constants.gravityParticleMass = kMass;
    constants.softeningLength = kSoftening;
    constants.softeningKernelOrder = kFerrersOrder;
    constants.directSinkFraction = kSinkFraction;
    constants.directSeed = kSeed;
    auto const actual = gravity::compareDirectGravity(particles, grid, constants, operationId);

    std::vector<gravity::DirectSolveField> selectedReference;
    std::vector<gravity::DirectSolveField> selectedApproximation;
    auto const sampledSinks = gravity::sampleDirectSinks(sourceCount, kSinkFraction, kSeed);
    size_t localSinkCount = 0;
    for (size_t sink : sampledSinks) {
        selectedReference.push_back(reference[sink]);
        selectedApproximation.push_back(approximation[sink]);
        if (sink >= sourceOffset && sink - sourceOffset < particlesPerLocality) {
            localSinkCount++;
        }
    }
    auto expected = gravity::compareDirectFields(selectedReference, selectedApproximation);
    expected.sourceCount = sourceCount;
    std::vector<double> result;
    appendDirectComparison(result, actual);
    appendDirectComparison(result, expected);
    result.push_back(static_cast<double>(localSinkCount));
    return result;
}

}    // namespace cosmictiger::detail

HPX_PLAIN_ACTION(cosmictiger::detail::runOwnedDirectComparison, CosmictigerRunOwnedDirectComparisonAction)

namespace {

constexpr size_t kComparisonSize = 20;
constexpr size_t kResultSize = 2 * kComparisonSize + 1;

void expectComparisonMatches(std::vector<double> const& values, size_t sourceCount, size_t sinkCount) {
    PROFILE();
    ASSERT_EQ(values.size(), kResultSize);
    EXPECT_EQ(values[0], static_cast<double>(sourceCount));
    EXPECT_EQ(values[1], static_cast<double>(sinkCount));
    for (size_t index = 0; index < kComparisonSize; index++) {
        EXPECT_DOUBLE_EQ(values[index], values[kComparisonSize + index]) << "comparison field " << index;
    }
}

}    // namespace

TEST(DirectComparison, OwnedSourcesExcludeUnflaggedGhostCopies) {
    constexpr std::uint64_t kOperation = 0x4449524543540001ULL;
    constexpr size_t kSourceCount = 3;
    constexpr size_t kSinkCount = 2;
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU ownership fixture avoids CUDA-backed particle storage";
#endif
    if (hpx::find_all_localities().size() != 1) {
        GTEST_SKIP() << "requires one physical HPX locality";
    }
    auto const values = cosmictiger::detail::runOwnedDirectComparison(0, kSourceCount, kOperation);
    expectComparisonMatches(values, kSourceCount, kSinkCount);
}

TEST(DirectComparison, RemoteOwnedSources) {
    constexpr std::uint64_t kOperation = 0x4449524543540002ULL;
    constexpr size_t kSourceCount = 2;
    constexpr size_t kSinkCount = 1;
    constexpr size_t kParticlesPerLocality = 1;
    PROFILE();
#ifdef COSMICTIGER_ENABLE_CUDA
    GTEST_SKIP() << "CPU ownership fixture avoids CUDA-backed particle storage";
#endif
    auto const localities = hpx::find_all_localities();
    if (localities.size() != kSourceCount) {
        GTEST_SKIP() << "requires two physical HPX localities";
    }
    auto const here = hpx::find_here();
    size_t localIndex = 0;
    while (localIndex < localities.size() && localities[localIndex] != here) {
        localIndex++;
    }
    ASSERT_LT(localIndex, localities.size());
    size_t const partner = localIndex ^ size_t(1);
    auto remote = hpx::async<CosmictigerRunOwnedDirectComparisonAction>(localities[partner], partner, kParticlesPerLocality, kOperation);
    auto const localValues = cosmictiger::detail::runOwnedDirectComparison(localIndex, kParticlesPerLocality, kOperation);
    auto const remoteValues = remote.get();
    expectComparisonMatches(localValues, kSourceCount, kSinkCount);
    expectComparisonMatches(remoteValues, kSourceCount, kSinkCount);
    ASSERT_EQ(localValues.size(), kResultSize);
    ASSERT_EQ(remoteValues.size(), kResultSize);
    EXPECT_EQ(localValues.back() + remoteValues.back(), 1.0);
    EXPECT_TRUE(localValues.back() == 0.0 || remoteValues.back() == 0.0);
    for (size_t index = 0; index < kComparisonSize; index++) {
        EXPECT_DOUBLE_EQ(localValues[index], remoteValues[index]);
    }
}
