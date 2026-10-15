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
#include "cosmictiger/gravity/cartesian/fmm.hpp"
#include <cosmictiger/debug.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/lock_guards.hpp>
#include <cosmictiger/parallel.hpp>

#include <hpx/synchronization/shared_mutex.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cosmictiger::gravity::cartesian {

namespace detail {

    template <typename Function, int p>
    auto dispatch(int order, Function&& function) -> decltype(function(std::integral_constant<int, p>{})) {
        PROFILE();
        if (order == p) {
            return function(std::integral_constant<int, p>{});
        }
        if constexpr (p < kMaxOrder) {
            return dispatch<Function, p + 1>(order, std::forward<Function>(function));
        }
        expectEqual(order, p);
        return function(std::integral_constant<int, p>{});
    }

    std::pair<int, bool> localShape(Coefficients const& local) {
        PROFILE();
        FpeGuard fpeGuard;
        size_t const side = size_t(std::sqrt(double(local.size())));
        expect(!(side == 0 || side > size_t(kMaxOrder + 1)));
        size_t const count = side * side;
        expect(!(local.size() != count && local.size() != count + 1));
        return {int(side) - 1, local.size() == count + 1};
    }

    double recover(Coefficients const& local, int x, int y, int z, bool withTrace) {
        PROFILE();
        FpeGuard fpeGuard;
        if (z < 2) {
            return local[stf::index(x, y, z)];
        }
        double const harmonic = -recover(local, x + 2, y, z - 2, withTrace) - recover(local, x, y + 2, z - 2, withTrace);
        return harmonic + (withTrace && x == 0 && y == 0 && z == 2 ? local.back() : 0);
    }

    template <int p>
    void translate(Coefficients& local, Coefficients const& source, Vector const& direction, double inverseRadius, double ratio) {
        constexpr size_t kCount = stf::coefficientCount(p);
        PROFILE();
        auto add = [&](auto& destination) {
            PROFILE();
            if (source.size() == 1) {
                stf::detail::P2lKernel<p>::add(destination, source[0], direction[0], direction[1], direction[2], inverseRadius, ratio);
            } else {
                stf::detail::M2lKernel<p, false>::template add<double, std::remove_cvref_t<decltype(destination)>, Coefficients, p>(
                    destination, source, direction[0], direction[1], direction[2], inverseRadius, ratio, ratio);
            }
        };
        if (local.size() >= kCount) {
            add(local);
        } else {
            // Compatibility for a leaf that retains only potential and gradient.
            std::array<double, kCount> increment{};
            add(increment);
            for (size_t i = 0; i < 4; i++) {
                local[i] += increment[i];
            }
        }
    }

    std::array<std::vector<Index>, kMaxOrder + 1> const indexTables = [] {
        PROFILE();
        std::array<std::vector<Index>, kMaxOrder + 1> result;
        for (int p = 0; p <= kMaxOrder; p++) {
            for (int n = 0; n <= p; n++) {
                for (int x = 0; x <= n; x++) {
                    result[p].push_back({x, n - x, 0});
                }
                for (int x = 0; x < n; x++) {
                    result[p].push_back({x, n - 1 - x, 1});
                }
            }
        }
        return result;
    }();

    std::map<std::array<int, NDIM + 1>, std::shared_ptr<Operator const>> cache;
    hpx::shared_mutex cacheMutex;

}    // namespace detail

int coefficientCount(int p) {
    PROFILE();
    expectRange(0, p, kMaxOrder);
    return int(stf::coefficientCount(p));
}

std::vector<Index> const& indices(int p) {
    PROFILE();
    expectRange(0, p, kMaxOrder);
    return detail::indexTables[p];
}

Coefficients point(double mass, Vector const& offset, int p) {
    PROFILE();
    expectRange(0, p, kMaxOrder);
    auto calculate = [&](auto order) {
        constexpr int kP = decltype(order)::value;
        PROFILE();
        stf::Multipole<double, kP> result{};
        stf::addPoint<double, kP>(result, mass, offset);
        return Coefficients(result.begin(), result.end());
    };
    return detail::dispatch<decltype(calculate), 0>(p, std::move(calculate));
}

Coefficients shiftMultipole(Coefficients const& source, Vector const& offset, double ratio, int p) {
    PROFILE();
    FpeGuard fpeGuard;
    int const count = coefficientCount(p);
    expect(!(source.size() != 1 && source.size() != size_t(count)));
    expectPositive(ratio);
    auto calculate = [&](auto order) {
        constexpr int kP = decltype(order)::value;
        PROFILE();
        Coefficients result(count, 0);
        if (source.size() == 1) {
            stf::Multipole<double, kP> padded{};
            padded[0] = source[0];
            stf::detail::M2mKernel<kP>::add(result, padded, offset[0], offset[1], offset[2], ratio);
        } else {
            stf::detail::M2mKernel<kP>::add(result, source, offset[0], offset[1], offset[2], ratio);
        }
        return result;
    };
    return detail::dispatch<decltype(calculate), stf::kMinOrder>(p, std::move(calculate));
}

Coefficients shiftLocal(Coefficients const& source, Vector const& offset, double ratio, int p) {
    PROFILE();
    FpeGuard fpeGuard;
    int const count = coefficientCount(p);
    bool const withTrace = source.size() == size_t(count + 1);
    expect(!(source.size() != size_t(count) && !withTrace));
    expect(!(withTrace && p < 2 && source.back() != 0));
    expectPositive(ratio);
    auto calculate = [&](auto order) {
        constexpr int kP = decltype(order)::value;
        PROFILE();
        Coefficients result(source.size(), 0);
        if constexpr (kP >= 2) {
            if (withTrace) {
                stf::detail::L2lKernel<kP, true>::add(result, source, offset[0], offset[1], offset[2], ratio);
                return result;
            }
        }
        stf::detail::L2lKernel<kP, false>::add(result, source, offset[0], offset[1], offset[2], ratio);
        return result;
    };
    return detail::dispatch<decltype(calculate), stf::kMinOrder>(p, std::move(calculate));
}

double derivative(Coefficients const& local, int x, int y, int z) {
    PROFILE();
    auto const [p, withTrace] = detail::localShape(local);
    expect(!(x < 0 || y < 0 || z < 0 || x > p || y > p || z > p || x + y + z > p));
    return detail::recover(local, x, y, z, withTrace);
}

std::array<double, NDIM + 1> evaluate(Coefficients const& local, Vector const& offset, int p) {
    PROFILE();
    auto const shifted = shiftLocal(local, offset, 1, p);
    if (p == 0) {
        return {shifted[0], 0, 0, 0};
    }
    return {shifted[0], derivative(shifted, 1, 0, 0), derivative(shifted, 0, 1, 0), derivative(shifted, 0, 0, 1)};
}

void addM2L(Coefficients& local, Coefficients const& source, Vector const& r, double sourceScale, double targetScale, int p, bool centered) {
    PROFILE();
    FpeGuard fpeGuard;
    double const radius = std::hypot(r[0], r[1], r[2]);
    if constexpr (validationChecksEnabled()) {
        [[maybe_unused]] int const count = coefficientCount(p);
        expect(!(!validOrder(p) || !(radius > 0) || !(sourceScale > 0) || !(targetScale > 0) ||
            (source.size() != 1 && source.size() != size_t(count) && source.size() != size_t(count + 1)) ||
            (local.size() != size_t(count) && local.size() != size_t(count + 1)) || local.data() == source.data()));
    }
    double const inverse = 1 / radius;
    auto translate = [&](auto order) {
        constexpr int kP = decltype(order)::value;
        PROFILE();
        if (source.size() == 1) {
            stf::detail::P2lKernel<kP>::add(local, source[0], r[0] * inverse, r[1] * inverse, r[2] * inverse, inverse, targetScale * inverse);
        } else {
            if (centered) {
                stf::detail::M2lKernel<kP, true>::template add<double, Coefficients, Coefficients, kP>(
                    local, source, r[0] * inverse, r[1] * inverse, r[2] * inverse, inverse, sourceScale * inverse, targetScale * inverse);
            } else {
                stf::detail::M2lKernel<kP, false>::template add<double, Coefficients, Coefficients, kP>(
                    local, source, r[0] * inverse, r[1] * inverse, r[2] * inverse, inverse, sourceScale * inverse, targetScale * inverse);
            }
        }
    };
    detail::dispatch<decltype(translate), kMinOrder>(p, std::move(translate));
}

void addM2LTargets(Field& local, Field const& source, std::span<M2lTarget const> targets, int order) {
    PROFILE();
    expect(&local != &source);
    expectEqual(local.size(), targets.size());
    expectRange(kMinOrder, order, kMaxOrder);
    // A target retains its complete source order, including additions to a
    // preexisting expansion. Sixteen targets is a conservative scheduling
    // heuristic, not a measured optimum; no interaction-count prepass is needed.
    ::cosmictiger::detail::ParallelChunkPlan const chunks(targets.size(), 16);
    ::cosmictiger::detail::parallelForChunks(chunks,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t targetIndex = begin; targetIndex < end; targetIndex++) {
                    auto const& target = targets[targetIndex];
                    for (auto const& interaction : target.interactions) {
                        expectIndex(interaction.sourceIndex, source.size());
                        addM2L(local[targetIndex], source[interaction.sourceIndex], interaction.separation, interaction.sourceScale, target.targetScale, order,
                            interaction.centered);
                    }
                }
            },
            "fmm.parallelForChunks"));
}

Operator::Operator(int p, Offset separation)
  : count_(coefficientCount(p)) {
    PROFILE();
    FpeGuard fpeGuard;
    expect(validOrder(p));
    double const radius = std::hypot(double(separation[0]), double(separation[1]), double(separation[2]));
    expectPositive(radius);
    inverseLatticeRadius_ = 1 / radius;
    for (int d = 0; d < NDIM; d++) {
        direction_[d] = double(separation[d]) * inverseLatticeRadius_;
    }
    auto selectApply = [](auto order) -> Apply {
        PROFILE();
        return &detail::translate<decltype(order)::value>;
    };
    apply_ = detail::dispatch<decltype(selectApply), kMinOrder>(p, std::move(selectApply));
}

void Operator::add(Coefficients& local, Coefficients const& moment, double cellWidth) const {
    PROFILE();
    FpeGuard fpeGuard;
    if constexpr (validationChecksEnabled()) {
        [[maybe_unused]] size_t const coefficientSize = size_t(count_);
        expect(!(!(cellWidth > 0) || (moment.size() != 1 && moment.size() != coefficientSize && moment.size() != coefficientSize + 1) ||
            (local.size() != 4 && local.size() != 5 && local.size() != coefficientSize && local.size() != coefficientSize + 1) ||
            local.data() == moment.data()));
    }
    double const inverseRadius = inverseLatticeRadius_ / cellWidth;
    expectPositive(inverseRadius);
    auto apply = hpx::annotated_function(apply_, "fmm.translate");
    apply(local, moment, direction_, inverseRadius, inverseLatticeRadius_);
}

std::shared_ptr<Operator const> getOperator(int p, Offset separation) {
    PROFILE();
    std::array<int, NDIM + 1> const key{p, separation[0], separation[1], separation[2]};
    {
        SharedLock lock(detail::cacheMutex);
        auto const found = detail::cache.find(key);
        if (found != detail::cache.end()) {
            return found->second;
        }
    }
    // Build outside the lock; locked scopes only inspect or change local state.
    auto candidate = std::make_shared<Operator const>(p, separation);
    UniqueLock lock(detail::cacheMutex);
    return detail::cache.emplace(key, std::move(candidate)).first->second;
}

size_t cachedOperatorCount() {
    PROFILE();
    SharedLock lock(detail::cacheMutex);
    return detail::cache.size();
}

void addDirect(Coefficients& local, double mass, Offset separation, double cellWidth) {
    PROFILE();
    FpeGuard fpeGuard;
    double const radius = std::hypot(double(separation[0]), double(separation[1]), double(separation[2]));
    expect(!(local.size() < 4 || !(radius > 0) || !(cellWidth > 0)));
    double const inverseRadius = (1 / radius) / cellWidth;
    expectPositive(inverseRadius);
    local[0] -= mass * inverseRadius;
    double const factor = mass * inverseRadius / (radius * radius);
    local[stf::index(1, 0, 0)] += factor * separation[0];
    local[stf::index(0, 1, 0)] += factor * separation[1];
    local[stf::index(0, 0, 1)] += factor * separation[2];
}

}    // namespace cosmictiger::gravity::cartesian
