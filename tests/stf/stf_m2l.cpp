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
#include <cosmictiger/profiler.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "cosmictiger/gravity/stf/m2l.hpp"

namespace {
namespace stf = cosmictiger::gravity::stf;
using Wide = double;
using Index = std::array<int, NDIM>;
using Polynomial = std::map<Index, Wide>;
struct Particle {
    Wide mass;
    std::array<Wide, NDIM> offset;
};

int degree(Index const& a) {
    PROFILE();
    return a[0] + a[1] + a[2];
}
Index sum(Index a, Index const& b) {
    PROFILE();
    for (int d = 0; d < NDIM; d++) {
        a[d] += b[d];
    }
    return a;
}
Wide factorial(int n) {
    PROFILE();
    Wide v = 1;
    for (int i = 2; i <= n; i++) {
        v *= i;
    }
    return v;
}
Wide factorial(Index const& a) {
    PROFILE();
    return factorial(a[0]) * factorial(a[1]) * factorial(a[2]);
}
Wide monomial(std::array<Wide, NDIM> const& x, Index const& a) {
    PROFILE();
    return std::pow(x[0], a[0]) * std::pow(x[1], a[1]) * std::pow(x[2], a[2]);
}
std::vector<Index> indices(int n) {
    PROFILE();
    std::vector<Index> out;
    for (int x = 0; x <= n; x++) {
        for (int y = 0; y <= n - x; y++) {
            out.push_back({x, y, n - x - y});
        }
    }
    return out;
}
Polynomial multiply(Polynomial const& a, Polynomial const& b, int maximum) {
    PROFILE();
    Polynomial out;
    for (auto const& [ia, va] : a) {
        for (auto const& [ib, vb] : b) {
            auto const k = sum(ia, ib);
            if (degree(k) <= maximum) {
                out[k] += va * vb;
            }
        }
    }
    return out;
}
Polynomial laplacian(Polynomial const& a) {
    PROFILE();
    Polynomial out;
    for (auto const& [i, v] : a) {
        for (int d = 0; d < NDIM; d++) {
            if (i[d] >= 2) {
                auto j = i;
                j[d] -= 2;
                out[j] += v * i[d] * (i[d] - 1);
            }
        }
    }
    return out;
}

// Independent source projection: harmonic projection of (s.t)^n using repeated
// polynomial Laplacians. This does not use the library's component-pair formula.
template <int sourceOrder>
stf::Multipole<double, sourceOrder> projectedMoments(std::vector<Particle> const& particles, Wide scale) {
    PROFILE();
    stf::Multipole<double, sourceOrder> out{};
    Polynomial const radius{{{2, 0, 0}, 1}, {{0, 2, 0}, 1}, {{0, 0, 2}, 1}};
    for (auto const& particle : particles) {
        Polynomial const linear{{{1, 0, 0}, particle.offset[0]}, {{0, 1, 0}, particle.offset[1]}, {{0, 0, 1}, particle.offset[2]}};
        Polynomial raw{{{0, 0, 0}, 1}};
        for (int n = 0; n <= sourceOrder; n++) {
            Polynomial harmonic, trace = raw, radial{{{0, 0, 0}, 1}};
            Wide coefficient = 1;
            for (int k = 0; k <= n / 2; k++) {
                for (auto const& [i, v] : multiply(radial, trace, n)) {
                    harmonic[i] += coefficient * v;
                }
                trace = laplacian(trace);
                radial = multiply(radial, radius, n);
                coefficient /= -Wide(2 * (k + 1) * (2 * n - 2 * k - 1));
            }
            for (auto const& i : indices(n)) {
                if (i[2] <= 1) {
                    out[stf::index(i[0], i[1], i[2])] += double(particle.mass * harmonic[i] * factorial(i) / factorial(n) / std::pow(scale, n));
                }
            }
            raw = multiply(raw, linear, sourceOrder);
        }
    }
    return out;
}

// Independent Cartesian oracle: expand -1/|R+h| with a scalar binomial series.
// The coefficients are converted to derivatives by multi-index factorials.
Polynomial greenDerivatives(std::array<Wide, NDIM> const& separation, int order) {
    PROFILE();
    Wide const r2 = separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2];
    Polynomial q{{{1, 0, 0}, 2 * separation[0] / r2}, {{0, 1, 0}, 2 * separation[1] / r2}, {{0, 0, 1}, 2 * separation[2] / r2}, {{2, 0, 0}, 1 / r2},
        {{0, 2, 0}, 1 / r2}, {{0, 0, 2}, 1 / r2}};
    Polynomial power{{{0, 0, 0}, 1}}, result;
    Wide binomial = 1;
    for (int k = 0; k <= order; k++) {
        for (auto const& [i, v] : power) {
            result[i] -= binomial * v / std::sqrt(r2);
        }
        power = multiply(power, q, order);
        binomial *= (-Wide(.5) - k) / (k + 1);
    }
    for (auto& [i, v] : result) {
        v *= factorial(i);
    }
    return result;
}

template <int sourceOrder, int expansionOrder, bool centered>
Polynomial cartesianOracle(std::vector<Particle> const& particles, std::array<double, NDIM> const& separation, Wide targetScale) {
    PROFILE();
    auto const derivatives = greenDerivatives({separation[0], separation[1], separation[2]}, expansionOrder);
    Polynomial raw;
    for (int n = 0; n <= sourceOrder; n++) {
        if (!centered || n != 1) {
            for (auto const& b : indices(n)) {
                for (auto const& p : particles) {
                    raw[b] += p.mass * monomial(p.offset, b);
                }
            }
        }
    }
    Polynomial result;
    for (int k = 0; k <= expansionOrder; k++) {
        for (auto const& a : indices(k)) {
            Wide value = 0;
            for (auto const& [b, m] : raw) {
                if (k + degree(b) <= expansionOrder) {
                    value += (degree(b) % 2 ? -1 : 1) * m * derivatives.at(sum(a, b)) / factorial(b);
                }
            }
            result[a] = value * std::pow(targetScale, k);
        }
    }
    return result;
}

std::vector<Particle> particleCloud(bool centered) {
    PROFILE();
    std::mt19937_64 rng(160908621);
    std::uniform_real_distribution<double> dist(-.22, .22), mass(.25, 1.4);
    std::vector<Particle> out;
    Wide total = 0;
    std::array<Wide, NDIM> com{};
    for (int i = 0; i < 7; i++) {
        Particle p{mass(rng), {dist(rng) + .07, dist(rng) - .03, dist(rng) + .02}};
        total += p.mass;
        for (int d = 0; d < NDIM; d++) {
            com[d] += p.mass * p.offset[d];
        }
        out.push_back(p);
    }
    if (centered) {
        for (auto& p : out) {
            for (int d = 0; d < NDIM; d++) {
                p.offset[d] -= com[d] / total;
            }
        }
    }
    return out;
}

template <int sourceOrder, int expansionOrder, bool centered>
Wide checkOracle() {
    constexpr std::array<std::array<double, NDIM>, 7> kDirections{
        {{1.3, -.9, 1.7}, {-1.3, .9, -1.7}, {2.1, 0, 0}, {0, -2.1, 0}, {0, 0, 2.1}, {.001, 2., -.03}, {-1.7, -1.1, .7}}};
    PROFILE();

    Wide worst = 0;
    auto const particles = particleCloud(centered);
    for (size_t test = 0; test < kDirections.size(); test++) {
        double const u = .47 + .13 * test, v = .63 + .11 * test;
        auto const moment = projectedMoments<sourceOrder>(particles, u);
        stf::Expansion<double, expansionOrder> actual{};
        stf::addM2L<double, sourceOrder, expansionOrder, centered>(actual, moment, kDirections[test], u, v);
        auto const expected = cartesianOracle<sourceOrder, expansionOrder, centered>(particles, kDirections[test], v);
        for (int k = 0; k <= expansionOrder; k++) {
            Wide scale = 0;
            for (auto const& i : indices(k)) {
                scale = std::max(scale, std::abs(expected.at(i)));
            }
            for (auto const& i : indices(k)) {
                SCOPED_TRACE(::testing::Message() << "source=" << sourceOrder << " expansion=" << expansionOrder << " centered=" << centered
                                                  << " direction=" << test << " index=" << i[0] << "," << i[1] << "," << i[2]);
                double const value = stf::component<double, expansionOrder>(actual, i[0], i[1], i[2]);
                worst = std::max(worst, std::abs(Wide(value) - expected.at(i)) / std::max(scale, Wide(1e-16)));
                EXPECT_NEAR(value, double(expected.at(i)), double(2e-11 * std::max(scale, Wide(1e-16))));
            }
        }
    }
    return worst;
}

template <int expansionOrder>
void checkOrders() {
    PROFILE();
    Wide worst = std::max(checkOracle<expansionOrder, expansionOrder, false>(), checkOracle<expansionOrder, expansionOrder, true>());
    if constexpr (expansionOrder > 0) {
        worst = std::max({worst, checkOracle<expansionOrder - 1, expansionOrder, false>(), checkOracle<expansionOrder - 1, expansionOrder, true>()});
    }
    std::ostringstream value;
    value << std::setprecision(18) << worst;
    ::testing::Test::RecordProperty("max_normalized_coefficient_error", value.str());
}

#define STF_ORDER_TEST(order)                   \
    TEST(StfM2l, CartesianOracleOrder##order) { \
        PROFILE();                              \
        checkOrders<order>();                   \
    }
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 0 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 0
STF_ORDER_TEST(0)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 1 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 1
STF_ORDER_TEST(1)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 2 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 2
STF_ORDER_TEST(2)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 3 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 3
STF_ORDER_TEST(3)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 4 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 4
STF_ORDER_TEST(4)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 5 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 5
STF_ORDER_TEST(5)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 6 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 6
STF_ORDER_TEST(6)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 7 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 7
STF_ORDER_TEST(7)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 8 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 8
STF_ORDER_TEST(8)
#endif
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
STF_ORDER_TEST(9)
#endif
#undef STF_ORDER_TEST

#if COSMICTIGER_STF_GENERATED_MAX_ORDER > 9
template <int order>
void checkExtendedOrders() {
    PROFILE();
    checkOrders<order>();
    if constexpr (order < stf::kMaxOrder) {
        checkExtendedOrders<order + 1>();
    }
}
TEST(StfM2l, CartesianOracleExtendedConfiguredOrders) {
    constexpr int kFirst = stf::kMinOrder > 9 ? stf::kMinOrder : 10;
    PROFILE();

    checkExtendedOrders<kFirst>();
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, PointFormationMatchesIndependentHarmonicProjectionThroughOrderNine) {
    PROFILE();
    auto const particles = particleCloud(false);
    for (double u : {.4, 1., 1.7}) {
        auto const expected = projectedMoments<9>(particles, u);
        stf::Multipole<double, 9> actual{};
        for (auto const& p : particles) {
            stf::addPoint<double, 9>(actual, double(p.mass), std::array<double, NDIM>{double(p.offset[0]), double(p.offset[1]), double(p.offset[2])}, u);
        }
        for (int n = 0; n <= 9; n++) {
            double scale = 0;
            for (size_t j = n * n; j < size_t((n + 1) * (n + 1)); j++) {
                scale = std::max(scale, std::abs(expected[j]));
            }
            for (size_t j = n * n; j < size_t((n + 1) * (n + 1)); j++) {
                EXPECT_NEAR(actual[j], expected[j], 3e-13 * scale) << "rank=" << n << " index=" << j;
            }
        }
    }
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, RankNineSourceChangesPotentialOnlyAtExpansionOrderNine) {
    constexpr std::array<double, NDIM> kSeparation{.9, -.8, .7};
    PROFILE();

    auto const particles = particleCloud(false);
    auto const high = projectedMoments<9>(particles, .8);
    stf::Multipole<double, 8> low{};
    std::copy_n(high.begin(), low.size(), low.begin());
    stf::Expansion<double, 9> full{}, omitted{};
    stf::addM2L<double, 9, 9>(full, high, kSeparation, .8, .9);
    stf::addM2L<double, 8, 9>(omitted, low, kSeparation, .8, .9);
    for (size_t i = 1; i < full.size(); i++) {
        EXPECT_EQ(full[i], omitted[i]);
    }
    auto const fullOracle = cartesianOracle<9, 9, false>(particles, kSeparation, .9);
    auto const lowOracle = cartesianOracle<8, 9, false>(particles, kSeparation, .9);
    Wide const correction = fullOracle.at({0, 0, 0}) - lowOracle.at({0, 0, 0});
    EXPECT_GT(std::abs(correction), 1e-12);
    EXPECT_NEAR(full[0] - omitted[0], double(correction), 4e-15);
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 1 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 1
TEST(StfM2l, DefaultIncludesDipoleAndCenteredModeExplicitlyOmitsIt) {
    constexpr std::array<double, NDIM> kSeparation{2, 0, 0};
    PROFILE();

    stf::Multipole<double, 1> source{};
    source[stf::index(1, 0, 0)] = 3;
    stf::Expansion<double, 1> full{}, centered{};
    stf::addM2L<double, 1>(full, source, kSeparation);
    stf::addM2L<double, 1, 1, true>(centered, source, kSeparation);
    EXPECT_DOUBLE_EQ(full[0], -.75);
    for (double value : centered) {
        EXPECT_EQ(value, 0);
    }
    for (size_t i = 1; i < full.size(); i++) {
        EXPECT_EQ(full[i], 0);
    }
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, AddsToExistingLocalsAndSuperposesSources) {
    constexpr std::array<double, NDIM> kSeparation{1.1, -1.8, .7};
    PROFILE();

    auto const first = projectedMoments<9>(particleCloud(false), .7);
    auto const second = projectedMoments<9>(particleCloud(true), .7);
    auto combined = first;
    for (size_t i = 0; i < combined.size(); i++) {
        combined[i] += second[i];
    }
    stf::Expansion<double, 9> delta{}, sequential{}, together{}, initial{};
    for (size_t i = 0; i < initial.size(); i++) {
        initial[i] = .013 * (i + 1);
    }
    sequential = initial;
    together = initial;
    stf::addM2L<double, 9>(sequential, first, kSeparation, .7, .8);
    stf::addM2L<double, 9>(sequential, second, kSeparation, .7, .8);
    stf::addM2L<double, 9>(together, combined, kSeparation, .7, .8);
    stf::addM2L<double, 9>(delta, combined, kSeparation, .7, .8);
    for (size_t i = 0; i < initial.size(); i++) {
        EXPECT_NEAR(sequential[i], together[i], 2e-12 * std::max(1., std::abs(together[i])));
        EXPECT_NEAR(together[i] - initial[i], delta[i], 2e-12 * std::max(1., std::abs(delta[i])));
    }
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, ConstantBackgroundTraceSurvivesNewtonianTranslation) {
    constexpr std::array<double, NDIM> kSeparation{1.4, -1.2, 1.7};

    static_assert(stf::coefficientCount(9) == 100);
    static_assert(stf::localCoefficientCount(9, true) == 101);
    PROFILE();
    auto const source = projectedMoments<9>(particleCloud(false), .8);
    stf::Expansion<double, 9, true> local{};
    for (size_t i = 0; i < stf::coefficientCount(9); i++) {
        local[i] = .017 * (i + 1);
    }
    stf::setLaplacian<double, 9>(local, -3.75);
    auto const before = local;
    EXPECT_DOUBLE_EQ((stf::laplacian<double, 9>(local)), -3.75);
    stf::Expansion<double, 9> harmonic{};
    stf::addM2L<double, 9>(harmonic, source, kSeparation, .8, .6);
    stf::addM2L<double, 9, 9, false, true>(local, source, kSeparation, .8, .6);
    EXPECT_DOUBLE_EQ((stf::laplacian<double, 9>(local)), -3.75);
    for (int n = 0; n <= 9; n++) {
        for (auto const& i : indices(n)) {
            double const expected = stf::component<double, 9, true>(before, i[0], i[1], i[2]) + stf::component<double, 9>(harmonic, i[0], i[1], i[2]);
            EXPECT_NEAR((stf::component<double, 9, true>(local, i[0], i[1], i[2])), expected, 5e-12 * std::max(1., std::abs(expected)));
        }
    }
    for (int n = 0; n <= 7; n++) {
        for (auto const& i : indices(n)) {
            double const trace = stf::component<double, 9, true>(local, i[0] + 2, i[1], i[2]) + stf::component<double, 9, true>(local, i[0], i[1] + 2, i[2]) +
                stf::component<double, 9, true>(local, i[0], i[1], i[2] + 2);
            EXPECT_NEAR(trace, n == 0 ? -3.75 : 0., 5e-12);
        }
    }
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 3 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 3
TEST(StfM2l, RejectsInvalidGeometryScalesAndAliasingBeforeMutation) {
    constexpr std::array<double, NDIM> kSeparation{2, 1, -1};
    PROFILE();

    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    stf::Multipole<double, 3> source{};
    source[0] = 1;
    stf::Expansion<double, 3> initial{};
    initial.fill(.25);
    for (auto const& invalid : std::vector<std::array<double, NDIM>>{{0, 0, 0}}) {
        auto local = initial;
        EXPECT_THROW((stf::addM2L<double, 3>(local, source, invalid)), std::runtime_error);
        EXPECT_EQ(local, initial);
    }
    for (double invalid : {0., -1.}) {
        auto local = initial;
        EXPECT_THROW((stf::addM2L<double, 3>(local, source, kSeparation, invalid, 1.)), std::runtime_error);
        EXPECT_EQ(local, initial);
        EXPECT_THROW((stf::addM2L<double, 3>(local, source, kSeparation, 1., invalid)), std::runtime_error);
        EXPECT_EQ(local, initial);
    }
    {
        auto aliased = source;
        EXPECT_THROW((stf::addM2L<double, 3>(aliased, aliased, kSeparation)), std::runtime_error);
        EXPECT_EQ(aliased, source);
    }
    EXPECT_THROW((stf::component<double, 3>(source, -1, 0, 0)), std::runtime_error);
    EXPECT_THROW((stf::component<double, 3>(source, 4, 0, 0)), std::runtime_error);
}
#endif

#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, PointFormationRejectsInvalidInputsBeforeMutation) {
    constexpr std::array<double, NDIM> kOffset{.1, .2, .3};
    PROFILE();

    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "requires validation checks";
    }
    stf::Multipole<double, 9> initial{};
    initial.fill(.125);
    for (double scale : {0., -1.}) {
        auto source = initial;
        EXPECT_THROW((stf::addPoint<double, 9>(source, 1., kOffset, scale)), std::runtime_error);
        EXPECT_EQ(source, initial);
    }
}
#endif

// The force error for a collinear point source has an exact geometric-series
// remainder, independently specifying the claimed total-order truncation.
#if COSMICTIGER_STF_GENERATED_MIN_ORDER <= 9 && COSMICTIGER_STF_GENERATED_MAX_ORDER >= 9
TEST(StfM2l, OrderNineForceAndPotentialRecoverAnalyticGeometricRemainders) {
    PROFILE();
    for (Wide q : {.3, .15}) {
        std::vector<Particle> const particles{{1, {2 * q, 0, 0}}};
        auto const source = projectedMoments<9>(particles, 1);
        stf::Expansion<double, 9> local{};
        stf::addM2L<double, 9>(local, source, std::array<double, NDIM>{2, 0, 0});
        Wide const exactPotential = -1 / (2 * (1 - q));
        Wide const exactGradient = 1 / (4 * (1 - q) * (1 - q));
        Wide const potentialRemainder = -std::pow(q, 10) / (2 * (1 - q));
        Wide const gradientRemainder = std::pow(q, 9) * (10 - 9 * q) / (4 * (1 - q) * (1 - q));
        EXPECT_NEAR(double(exactPotential - local[0]), double(potentialRemainder), 2e-15);
        EXPECT_NEAR(double(exactGradient - local[stf::index(1, 0, 0)]), double(gradientRemainder), 2e-15);
        EXPECT_EQ(local[stf::index(0, 1, 0)], 0);
        EXPECT_EQ(local[stf::index(0, 0, 1)], 0);
    }
}
#endif

}    // namespace
