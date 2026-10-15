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

// Distributed under the Boost Software License, Version 1.0.
// Independent scientific checks for the standalone STF operator family.
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/profiler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

int failures = 0;
Wide worstCoefficientError = 0;

void require(bool condition, std::string const& message) {
    PROFILE();
    if (!condition) {
        ++failures;
        std::cerr << message << '\n';
    }
}

void near(Wide actual, Wide expected, Wide scale, std::string const& message, Wide tolerance = 5e-11, bool record = true) {
    PROFILE();
    scale = std::max(std::abs(scale), Wide(1e-24));
    Wide const error = std::abs(actual - expected) / scale;
    if (record) {
        worstCoefficientError = std::max(worstCoefficientError, error);
    }
    if (!(error <= tolerance)) {
        ++failures;
        std::cerr << message << ": actual=" << double(actual) << " expected=" << double(expected) << " normalized_error=" << double(error) << '\n';
    }
}

template <typename Function>
void rejects(Function&& function, std::string const& message) {
    PROFILE();
    bool caught = false;
    try {
        function();
    } catch (std::runtime_error const&) {
        caught = true;
    }
    require(caught, message);
}

template <typename Array, int order>
void compareStored(Array const& actual, Array const& expected, std::string const& message) {
    PROFILE();
    for (int n = 0; n <= order; n++) {
        Wide scale = 0;
        for (size_t i = n * n; i < size_t((n + 1) * (n + 1)); i++) {
            scale = std::max(scale, std::abs(Wide(expected[i])));
        }
        for (size_t i = n * n; i < size_t((n + 1) * (n + 1)); i++) {
            near(actual[i], expected[i], scale, message + " rank=" + std::to_string(n));
        }
    }
}

std::vector<Particle> shifted(std::vector<Particle> particles, std::array<double, NDIM> const& shift) {
    PROFILE();
    for (auto& particle : particles) {
        for (int d = 0; d < NDIM; d++) {
            particle.offset[d] += shift[d];
        }
    }
    return particles;
}

std::array<double, NDIM> addVectors(std::array<double, NDIM> a, std::array<double, NDIM> const& b) {
    PROFILE();
    for (int d = 0; d < NDIM; d++) {
        a[d] += b[d];
    }
    return a;
}

template <int order>
void checkM2M() {
    constexpr std::array<double, NDIM> kD1{.13, -.17, .09};
    constexpr std::array<double, NDIM> kD2{-.04, .06, .11};
    constexpr std::array<double, NDIM> kZero{};
    PROFILE();

    auto const particles = particleCloud(false);
    for (auto const& scales : std::array<std::array<double, 2>, 2>{{{.6, 1.3}, {1.4, .7}}}) {
        double const oldScale = scales[0], newScale = scales[1];
        auto const source = projectedMoments<order>(particles, oldScale);
        auto const expected = projectedMoments<order>(shifted(particles, kD1), newScale);
        stf::Multipole<double, order> actual{};
        stf::addM2M<double, order>(actual, source, kD1, oldScale, newScale);
        compareStored<decltype(actual), order>(actual, expected, "M2M point re-expansion");
        stf::addM2M<double, order>(actual, source, kD1, oldScale, newScale);
        auto doubled = expected;
        for (auto& x : doubled) {
            x *= 2;
        }
        compareStored<decltype(actual), order>(actual, doubled, "M2M accumulation");

        stf::Multipole<double, order> first{}, composed{}, direct{}, rescaled{};
        stf::addM2M<double, order>(first, source, kD1, oldScale, .85);
        stf::addM2M<double, order>(composed, first, kD2, .85, newScale);
        stf::addM2M<double, order>(direct, source, addVectors(kD1, kD2), oldScale, newScale);
        compareStored<decltype(composed), order>(composed, direct, "M2M shift composition");
        stf::addM2M<double, order>(rescaled, source, kZero, oldScale, newScale);
        compareStored<decltype(rescaled), order>(rescaled, projectedMoments<order>(particles, newScale), "M2M zero shift rescaling");
    }
}

// Construct an explicit harmonic polynomial via repeated Laplacians, then add
// an optional quadratic background with a known constant physical Laplacian.
Polynomial potentialPolynomial(int order, Wide background) {
    PROFILE();
    Polynomial result;
    Polynomial const radius{{{2, 0, 0}, 1}, {{0, 2, 0}, 1}, {{0, 0, 2}, 1}};
    Polynomial const linear{{{1, 0, 0}, .31}, {{0, 1, 0}, -.27}, {{0, 0, 1}, .43}};
    Polynomial raw{{{0, 0, 0}, 1}};
    for (int n = 0; n <= order; n++) {
        Polynomial trace = raw, radial{{{0, 0, 0}, 1}};
        Wide coefficient = (n % 2 ? -.4 : .4) / factorial(n);
        for (int k = 0; k <= n / 2; k++) {
            for (auto const& [i, v] : multiply(radial, trace, n)) {
                result[i] += coefficient * v;
            }
            trace = laplacian(trace);
            radial = multiply(radial, radius, n);
            coefficient /= -Wide(2 * (k + 1) * (2 * n - 2 * k - 1));
        }
        raw = multiply(raw, linear, order);
    }
    if (order >= 2) {
        for (auto const& [i, v] : radius) {
            result[i] += background * v / 6;
        }
    }
    return result;
}

// Direct complete Cartesian polynomial differentiation/evaluation; no compact
// local reconstruction or library translation is used by this oracle.
Wide polynomialDerivative(Polynomial const& polynomial, Index const& derivative, std::array<Wide, NDIM> const& position) {
    PROFILE();
    Wide result = 0;
    for (auto const& [power, value] : polynomial) {
        Index remainder{};
        bool contains = true;
        for (int d = 0; d < NDIM; d++) {
            remainder[d] = power[d] - derivative[d];
            contains = contains && remainder[d] >= 0;
        }
        if (contains) {
            result += value * factorial(power) / factorial(remainder) * monomial(position, remainder);
        }
    }
    return result;
}

template <int order, bool withTrace>
stf::Expansion<double, order, withTrace> localAt(Polynomial const& polynomial, std::array<double, NDIM> const& position, double scale, Wide background) {
    PROFILE();
    stf::Expansion<double, order, withTrace> local{};
    std::array<Wide, NDIM> const widePosition{position[0], position[1], position[2]};
    for (int n = 0; n <= order; n++) {
        for (auto const& a : indices(n)) {
            if (a[2] < 2) {
                local[stf::index(a[0], a[1], a[2])] = double(polynomialDerivative(polynomial, a, widePosition) * std::pow(Wide(scale), n));
            }
        }
    }
    if constexpr (withTrace) {
        stf::setLaplacian<double, order>(local, double(background * scale * scale));
    }
    return local;
}

template <int order, bool withTrace>
void checkL2L() {
    constexpr std::array<double, NDIM> kZero{};
    constexpr std::array<double, NDIM> kD1{.13, -.17, .09};
    constexpr std::array<double, NDIM> kD2{-.04, .06, .11};
    PROFILE();

    Wide const background = withTrace ? -1.7 : 0;
    auto const polynomial = potentialPolynomial(order, background);
    for (auto const& scales : std::array<std::array<double, 2>, 2>{{{.6, 1.3}, {1.4, .7}}}) {
        double const oldScale = scales[0], newScale = scales[1];
        auto const source = localAt<order, withTrace>(polynomial, kZero, oldScale, background);
        auto const expected = localAt<order, withTrace>(polynomial, kD1, newScale, background);
        stf::Expansion<double, order, withTrace> actual{};
        stf::addL2L<double, order, withTrace>(actual, source, kD1, oldScale, newScale);
        compareStored<decltype(actual), order>(actual, expected, "L2L Cartesian polynomial translation");
        for (int n = 0; n <= order; n++) {
            Wide scale = 0;
            for (auto const& i : indices(n)) {
                scale = std::max(scale, std::abs(polynomialDerivative(polynomial, i, {kD1[0], kD1[1], kD1[2]}) * std::pow(Wide(newScale), n)));
            }
            for (auto const& i : indices(n)) {
                near(stf::component<double, order, withTrace>(actual, i[0], i[1], i[2]),
                    polynomialDerivative(polynomial, i, {kD1[0], kD1[1], kD1[2]}) * std::pow(Wide(newScale), n), scale, "L2L complete tensor reconstruction");
            }
        }
        if constexpr (withTrace) {
            near(stf::laplacian<double, order>(actual), background * newScale * newScale, std::abs(background), "L2L background rescaling");
        }
        stf::addL2L<double, order, withTrace>(actual, source, kD1, oldScale, newScale);
        auto doubled = expected;
        for (auto& x : doubled) {
            x *= 2;
        }
        compareStored<decltype(actual), order>(actual, doubled, "L2L accumulation");
        if constexpr (withTrace) {
            near(stf::laplacian<double, order>(actual), 2 * background * newScale * newScale, std::abs(background), "L2L trace accumulation");
        }

        stf::Expansion<double, order, withTrace> first{}, composed{}, direct{}, rescaled{};
        stf::addL2L<double, order, withTrace>(first, source, kD1, oldScale, .85);
        stf::addL2L<double, order, withTrace>(composed, first, kD2, .85, newScale);
        stf::addL2L<double, order, withTrace>(direct, source, addVectors(kD1, kD2), oldScale, newScale);
        compareStored<decltype(composed), order>(composed, direct, "L2L shift composition");
        stf::addL2L<double, order, withTrace>(rescaled, source, kZero, oldScale, newScale);
        compareStored<decltype(rescaled), order>(rescaled, localAt<order, withTrace>(polynomial, kZero, newScale, background), "L2L zero shift rescaling");
    }
}

template <int order, bool withTrace>
void checkP2L() {
    constexpr std::array<std::array<double, NDIM>, 4> kDirections{{{1.3, -.9, 1.7}, {2.1, 0, 0}, {0, 0, -2.1}, {-1.3, .9, -1.7}}};
    constexpr double kMass = 1.7;
    PROFILE();

    for (size_t test = 0; test < kDirections.size(); test++) {
        double const scale = .63 + .11 * test;
        auto const derivative = greenDerivatives({kDirections[test][0], kDirections[test][1], kDirections[test][2]}, order);
        stf::Expansion<double, order, withTrace> local{}, initial{};
        for (size_t i = 0; i < initial.size(); i++) {
            initial[i] = .003 * (i + 1);
        }
        if constexpr (withTrace) {
            stf::setLaplacian<double, order>(initial, -3.25);
        }
        local = initial;
        stf::addP2L<double, order, withTrace>(local, kMass, kDirections[test], scale);
        for (int n = 0; n <= order; n++) {
            Wide rankScale = 0;
            for (auto const& i : indices(n)) {
                rankScale = std::max(rankScale, std::abs(kMass * derivative.at(i) * std::pow(Wide(scale), n)));
            }
            for (auto const& i : indices(n)) {
                near(stf::component<double, order, withTrace>(local, i[0], i[1], i[2]) - stf::component<double, order, withTrace>(initial, i[0], i[1], i[2]),
                    kMass * derivative.at(i) * std::pow(Wide(scale), n), rankScale, "P2L binomial derivative oracle");
            }
        }
        if constexpr (withTrace) {
            require(stf::laplacian<double, order>(local) == -3.25, "P2L preserves background trace");
        }
    }
}

std::array<Wide, NDIM + 1> particleFieldOracle(
    std::vector<Particle> const& particles, std::array<double, NDIM> const& separation, int order, bool centered, int onlyRank = -1) {
    PROFILE();
    auto const derivative = greenDerivatives({separation[0], separation[1], separation[2]}, order + 1);
    std::array<Wide, NDIM + 1> field{};
    for (int n = 0; n <= order; n++) {
        if ((centered && n == 1) || (onlyRank >= 0 && n != onlyRank)) {
            continue;
        }
        for (auto const& b : indices(n)) {
            Wide moment = 0;
            for (auto const& p : particles) {
                moment += p.mass * monomial(p.offset, b);
            }
            Wide const coefficient = (n % 2 ? -1 : 1) * moment / factorial(b);
            field[0] += coefficient * derivative.at(b);
            for (int d = 0; d < NDIM; d++) {
                Index a = b;
                a[d]++;
                field[d + 1] -= coefficient * derivative.at(a);
            }
        }
    }
    return field;
}

template <int order, bool centered>
void checkM2P() {
    constexpr double kH = 1e-4;
    constexpr std::array<std::array<double, NDIM>, 4> kDirections{{{1.3, -.9, 1.7}, {2.1, 0, 0}, {0, 0, -2.1}, {-1.3, .9, -1.7}}};
    constexpr std::array<double, NDIM> kSeparation{1.3, -.9, 1.7};
    PROFILE();

    auto const particles = particleCloud(centered);
    for (size_t test = 0; test < kDirections.size(); test++) {
        double const scale = .47 + .13 * test;
        auto const source = projectedMoments<order>(particles, scale);
        auto const expected = particleFieldOracle(particles, kDirections[test], order, centered);
        auto const actual = stf::evaluateM2P<double, order, centered>(source, kDirections[test], scale);
        near(actual.potential, expected[0], std::abs(expected[0]), "M2P potential Cartesian oracle");
        Wide forceScale = 0;
        for (int d = 0; d < NDIM; d++) {
            forceScale = std::max(forceScale, std::abs(expected[d + 1]));
        }
        for (int d = 0; d < NDIM; d++) {
            near(actual.acceleration[d], expected[d + 1], forceScale, "M2P force Cartesian oracle");
        }
        if (test == 0) {
            for (int d = 0; d < NDIM; d++) {
                auto rm2 = kDirections[test], rm1 = rm2, rp1 = rm2, rp2 = rm2;
                rm2[d] -= 2 * kH;
                rm1[d] -= kH;
                rp1[d] += kH;
                rp2[d] += 2 * kH;
                Wide const derivative = (stf::evaluateM2P<double, order, centered>(source, rm2, scale).potential -
                                            8 * Wide(stf::evaluateM2P<double, order, centered>(source, rm1, scale).potential) +
                                            8 * Wide(stf::evaluateM2P<double, order, centered>(source, rp1, scale).potential) -
                                            stf::evaluateM2P<double, order, centered>(source, rp2, scale).potential) /
                    (12 * kH);
                near(actual.acceleration[d], -derivative, forceScale, "M2P force is negative potential gradient", 2e-8, false);
            }
        }
    }
    if constexpr (order > 0) {
        auto const noncenteredParticles = particleCloud(false);
        auto highestOnly = projectedMoments<order>(noncenteredParticles, .8);
        std::fill_n(highestOnly.begin(), order * order, 0.);
        auto const expected = particleFieldOracle(noncenteredParticles, kSeparation, order, false, order);
        auto const actual = stf::evaluateM2P<double, order, false>(highestOnly, kSeparation, .8);
        Wide scale = 0;
        for (auto x : expected) {
            scale = std::max(scale, std::abs(x));
        }
        near(actual.potential, expected[0], scale, "M2P highest source rank potential");
        for (int d = 0; d < NDIM; d++) {
            near(actual.acceleration[d], expected[d + 1], scale, "M2P highest source rank force");
        }
    }
}

template <int order>
void checkParticleRemainders() {
    PROFILE();
    for (Wide q : {.3, .15}) {
        std::vector<Particle> const particles{{1, {2 * q, 0, 0}}};
        auto const source = projectedMoments<order>(particles, 1);
        auto const field = stf::evaluateM2P<double, order>(source, std::array<double, NDIM>{2, 0, 0});
        Wide const exactPotential = -1 / (2 * (1 - q));
        Wide const exactAcceleration = -1 / (4 * (1 - q) * (1 - q));
        Wide const potentialRemainder = -std::pow(q, order + 1) / (2 * (1 - q));
        Wide const accelerationRemainder = -std::pow(q, order + 1) * (order + 2 - (order + 1) * q) / (4 * (1 - q) * (1 - q));
        near(exactPotential - field.potential, potentialRemainder, 1, "M2P analytic potential remainder", 3e-14);
        near(exactAcceleration - field.acceleration[0], accelerationRemainder, 1, "M2P analytic force remainder", 3e-14);
        require(field.acceleration[1] == 0 && field.acceleration[2] == 0, "M2P collinear force has no transverse component");
    }
}

template <int order>
void checkInvalidInputs() {
    constexpr std::array<double, NDIM> kZero{};
    constexpr std::array<double, NDIM> kSeparation{1, 2, 3};
    PROFILE();

    if (!cosmictiger::validationChecksEnabled()) {
        return;
    }
    auto const source = projectedMoments<order>(particleCloud(false), 1);
    stf::Expansion<double, order> initial{};
    initial.fill(.125);
    {
        auto target = source;
        rejects(
            [&] {
                PROFILE();
                stf::addM2M<double, order>(target, target, kZero);
            },
            "M2M rejects aliasing");
        require(target == source, "M2M alias rejection is transactional");
        auto local = initial;
        rejects(
            [&] {
                PROFILE();
                stf::addL2L<double, order>(local, local, kZero);
            },
            "L2L rejects aliasing");
        require(local == initial, "L2L alias rejection is transactional");
        rejects(
            [&] {
                PROFILE();
                stf::addP2L<double, order>(local, 1., kZero);
            },
            "P2L rejects zero separation");
        require(local == initial, "P2L geometry rejection is transactional");
        rejects(
            [&] {
                PROFILE();
                (void) stf::evaluateM2P<double, order>(source, kZero);
            },
            "M2P rejects zero separation");
    }
    for (double badScale : {0., -1.}) {
        auto target = source;
        rejects(
            [&] {
                PROFILE();
                stf::addM2M<double, order>(target, source, kZero, badScale, 1.);
            },
            "M2M rejects source scale");
        rejects(
            [&] {
                PROFILE();
                stf::addM2M<double, order>(target, source, kZero, 1., badScale);
            },
            "M2M rejects target scale");
        require(target == source, "M2M scale rejection is transactional");
        auto local = initial;
        rejects(
            [&] {
                PROFILE();
                stf::addL2L<double, order>(local, initial, kZero, badScale, 1.);
            },
            "L2L rejects source scale");
        rejects(
            [&] {
                PROFILE();
                stf::addL2L<double, order>(local, initial, kZero, 1., badScale);
            },
            "L2L rejects target scale");
        rejects(
            [&] {
                PROFILE();
                stf::addP2L<double, order>(local, 1., kSeparation, badScale);
            },
            "P2L rejects target scale");
        require(local == initial, "Expansion scale rejection is transactional");
        rejects(
            [&] {
                PROFILE();
                (void) stf::evaluateM2P<double, order>(source, kSeparation, badScale);
            },
            "M2P rejects source scale");
    }
}

template <int order>
void checkOrders() {
    PROFILE();
    int const before = failures;
    checkM2M<order>();
    checkL2L<order, false>();
    checkP2L<order, false>();
    if constexpr (order >= 2) {
        checkL2L<order, true>();
        checkP2L<order, true>();
    }
    checkM2P<order, false>();
    checkM2P<order, true>();
    checkParticleRemainders<order>();
    checkInvalidInputs<order>();
    std::cout << "Order " << order << ": " << (failures == before ? "passed" : "FAILED") << '\n';
    if constexpr (order < stf::kMaxOrder) {
        checkOrders<order + 1>();
    }
}
}    // namespace

int main() {
    PROFILE();
    checkOrders<stf::kMinOrder>();
    std::cout << "Maximum normalized coefficient error: " << double(worstCoefficientError) << '\n';
    return failures == 0 ? 0 : 1;
}
