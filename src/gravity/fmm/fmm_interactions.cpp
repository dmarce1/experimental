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

#include <stddef.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/dimensions.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/gravity/fmm/fmm_kernels.hpp>
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::gravity::fmm {

namespace detail {

    // Unbenchmarked scheduling grain, as in the batched Cartesian M2L.
    inline constexpr size_t kMinimumTargetsPerTask = 16;

    template <int fmmOrder, typename Function>
    void dispatchOrder(int order, Function&& function) {
        PROFILE();
        if (order == fmmOrder) {
            function(std::integral_constant<int, fmmOrder>{});
            return;
        }
        if constexpr (fmmOrder < kMaximumFmmOrder) {
            dispatchOrder<fmmOrder + 1>(order, std::forward<Function>(function));
        }
    }

    template <typename Target>
    void expectSourceRange(Target const& target, size_t sourceCount) {
        PROFILE();
        expect(target.sourceBegin <= target.sourceEnd);
        expect(target.sourceEnd <= sourceCount);
    }

    template <int fmmOrder>
    void addLocal(float* expansion, double const (&local)[localCount<fmmOrder>()]) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        FpeGuard fpeGuard;
        for (size_t coefficient = 0; coefficient < kCount; coefficient++) {
            expansion[coefficient] += static_cast<float>(local[coefficient]);
        }
    }

    template <typename Function>
    void forEachTarget(size_t targetCount, char const* label, Function&& function) {
        PROFILE();
        ::cosmictiger::detail::ParallelChunkPlan const plan(targetCount, kMinimumTargetsPerTask);
        ::cosmictiger::detail::parallelForChunks(plan,
            hpx::annotated_function(
                [&](size_t, size_t begin, size_t end) {
                    PROFILE();
                    for (size_t target = begin; target < end; target++) {
                        function(target);
                    }
                },
                label));
    }

    template <int fmmOrder>
    void addM2lOrder(FmmM2lRequest const& request) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        forEachTarget(request.targetCount, "fmm.addM2lCpu", [&](size_t targetIndex) {
            PROFILE();
            FmmExpansionTarget const& target = request.targets[targetIndex];
            expectSourceRange(target, request.sourceCount);
            double local[kCount]{};
            {
                FpeGuard fpeGuard;
                for (size_t sourceIndex = target.sourceBegin; sourceIndex < target.sourceEnd; sourceIndex++) {
                    FmmMultipoleSource const& source = request.sources[sourceIndex];
                    double const separation[NDIM] = {source.separation[0], source.separation[1], source.separation[2]};
                    addM2l<double, fmmOrder>(local, expectNonZero(source.multipole), separation, request.spacing);
                }
            }
            addLocal<fmmOrder>(expectNonZero(target.expansion), local);
        });
    }

    template <int fmmOrder>
    void addP2lOrder(FmmP2lRequest const& request) {
        constexpr size_t kCount = localCount<fmmOrder>();
        PROFILE();
        double const mass = request.particleMass;
        forEachTarget(request.targetCount, "fmm.addP2lCpu", [&](size_t targetIndex) {
            PROFILE();
            FmmExpansionTarget const& target = request.targets[targetIndex];
            expectSourceRange(target, request.sourceCount);
            double local[kCount]{};
            {
                FpeGuard fpeGuard;
                for (size_t sourceIndex = target.sourceBegin; sourceIndex < target.sourceEnd; sourceIndex++) {
                    FmmParticleSource const& source = request.sources[sourceIndex];
                    Coordinate const center[NDIM] = {source.center.x, source.center.y, source.center.z};
                    expect(source.particles.begin <= source.particles.end);
                    for (ParticleCount particle = source.particles.begin; particle < source.particles.end; particle++) {
                        double separation[NDIM];
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            separation[axis] = source.separation[axis] - minimumImageOffset<double>(request.positions[axis][particle], center[axis]);
                        }
                        addP2l<double, fmmOrder>(local, mass, separation, request.spacing);
                    }
                }
            }
            addLocal<fmmOrder>(expectNonZero(target.expansion), local);
        });
    }

    template <int fmmOrder>
    void addM2pOrder(FmmM2pRequest const& request) {
        PROFILE();
        forEachTarget(request.targetCount, "fmm.addM2pCpu", [&](size_t targetIndex) {
            PROFILE();
            FmmParticleTarget const& target = request.targets[targetIndex];
            expectSourceRange(target, request.sourceCount);
            expect(target.sinks.begin <= target.sinks.end);
            Coordinate const center[NDIM] = {target.center.x, target.center.y, target.center.z};
            FpeGuard fpeGuard;
            for (ParticleCount sink = target.sinks.begin; sink < target.sinks.end; sink++) {
                double offset[NDIM];
                for (size_t axis = 0; axis < NDIM; axis++) {
                    offset[axis] = minimumImageOffset<double>(request.positions[axis][sink], center[axis]);
                }
                double field[NDIM + 1]{};
                for (size_t sourceIndex = target.sourceBegin; sourceIndex < target.sourceEnd; sourceIndex++) {
                    FmmMultipoleSource const& source = request.sources[sourceIndex];
                    double separation[NDIM];
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        separation[axis] = offset[axis] + source.separation[axis];
                    }
                    addM2p<double, fmmOrder>(field, expectNonZero(source.multipole), separation, request.spacing);
                }
                for (size_t axis = 0; axis < NDIM; axis++) {
                    request.accelerations[axis][sink] += static_cast<float>(field[axis + 1]);
                }
                if (request.potentials != nullptr) {
                    request.potentials[sink] += static_cast<float>(field[0]);
                }
            }
        });
    }

}    // namespace detail

void addM2lCpu(FmmM2lRequest const& request) {
    PROFILE();
    detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
    if (request.targetCount == 0) {
        return;
    }
    detail::requireTables(request);
    detail::dispatchOrder<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
        PROFILE();
        detail::addM2lOrder<decltype(order)::value>(request);
    });
}

void addP2lCpu(FmmP2lRequest const& request) {
    PROFILE();
    detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
    if (request.targetCount == 0) {
        return;
    }
    detail::requireTables(request);
    detail::requirePositions(request);
    detail::dispatchOrder<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
        PROFILE();
        detail::addP2lOrder<decltype(order)::value>(request);
    });
}

void addM2pCpu(FmmM2pRequest const& request) {
    PROFILE();
    detail::requireOrderAndSpacing(request.fmmOrder, request.spacing);
    if (request.targetCount == 0) {
        return;
    }
    detail::requireTables(request);
    detail::requirePositions(request);
    for (size_t axis = 0; axis < NDIM; axis++) {
        if (request.accelerations[axis] == nullptr) {
            throw std::invalid_argument("FMM M2P acceleration outputs are required for a nonempty batch");
        }
    }
    detail::dispatchOrder<kMinimumFmmOrder>(request.fmmOrder, [&](auto order) {
        PROFILE();
        detail::addM2pOrder<decltype(order)::value>(request);
    });
}

}    // namespace cosmictiger::gravity::fmm
