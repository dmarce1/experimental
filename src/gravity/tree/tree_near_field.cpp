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
#include <bit>
#include <cmath>
#include <cstdint>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/debug.hpp>
#include <cosmictiger/gravity/direct/direct_interactions.hpp>
#include <cosmictiger/gravity/fmm/fmm_interactions.hpp>
#include <cosmictiger/gravity/stf/operators.hpp>
#include <cosmictiger/gravity/tree/tree_near_field.hpp>
#include <cosmictiger/parallel.hpp>
#include <cosmictiger/profiler.hpp>

namespace cosmictiger::gravity::tree {

namespace detail {

    using SignedPoint = std::array<std::int64_t, NDIM>;
    using RawPoint = std::array<std::uint32_t, NDIM>;
    using DoubleExpansion = stf::Expansion<double, kExpansionOrder, true>;

    constexpr double kRawScale = 0x1p-32;
    constexpr size_t kMaximumNearCells = [] {
        PROFILE();
        size_t count = 1;
        for (size_t axis = 0; axis < NDIM; axis++) {
            count *= 2 * kPmmmMaximumNeighborWidth + 1;
        }
        return count;
    }();

    // The cell-relative frame of one source cell seen from a sink cell: its
    // lower corner sits at origin in the sink cell's raw offset frame.
    struct SourceFrame {
        SignedPoint origin;
        RawPoint lower;
        std::array<std::int64_t, NDIM> cellOffset;
        bool selfCell;
    };

    // A source node in the checklist, with its center in the sink frame.
    struct Entry {
        size_t node;
        std::uint32_t frame;
        SignedPoint position;
    };

    struct DirectSource {
        ParticleRange range;
        bool selfCell;
    };

    // Interaction lists of one sink cell. Targets reference cell-local
    // expansions, which stay at fixed addresses while lists are built.
    struct CellLists {
        std::vector<fmm::FmmExpansionTarget> m2lTargets;
        std::vector<fmm::FmmMultipoleSource> m2lSources;
        std::vector<fmm::FmmExpansionTarget> p2lTargets;
        std::vector<fmm::FmmParticleSource> p2lSources;
        std::vector<fmm::FmmParticleTarget> m2pTargets;
        std::vector<fmm::FmmMultipoleSource> m2pSources;
        std::vector<std::pair<size_t, std::pair<size_t, size_t>>> p2pLeaves;
        std::vector<DirectSource> p2pSources;
    };

    RawPoint cellLowerRaw(CartesianGridLayout const& layout, GridIndices coordinates, int positionShift) {
        PROFILE();
        auto const global = layout.globalCoordinates(coordinates);
        RawPoint lower{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            lower[axis] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(global[axis]) << positionShift);
        }
        return lower;
    }

    SignedPoint centerInFrame(KdNode const& node, SourceFrame const& frame) {
        PROFILE();
        Coordinate const axes[NDIM] = {node.center.x, node.center.y, node.center.z};
        SignedPoint result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = frame.origin[axis] + static_cast<std::int64_t>(static_cast<std::uint32_t>(axes[axis].raw() - frame.lower[axis]));
        }
        return result;
    }

    double squaredDistance(SignedPoint const& first, SignedPoint const& second) {
        PROFILE();
        double result = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            double const delta = static_cast<double>(first[axis] - second[axis]) * kRawScale;
            result += delta * delta;
        }
        return result;
    }

    // Squared distance from point to the closed particle box of a node whose
    // frame origin is boxOrigin, both in one sink frame.
    double squaredBoxDistance(KdBox const& box, SignedPoint const& boxOrigin, SignedPoint const& point) {
        PROFILE();
        double result = 0.0;
        for (size_t axis = 0; axis < NDIM; axis++) {
            std::int64_t const lower = boxOrigin[axis] + static_cast<std::int64_t>(box.min(axis).raw());
            std::int64_t const upper = boxOrigin[axis] + static_cast<std::int64_t>(box.max(axis).raw()) - 1;
            std::int64_t const gap = point[axis] < lower ? lower - point[axis] : (point[axis] > upper ? point[axis] - upper : 0);
            double const delta = static_cast<double>(gap) * kRawScale;
            result += delta * delta;
        }
        return result;
    }

    std::array<double, NDIM> separation(SignedPoint const& target, SignedPoint const& source) {
        PROFILE();
        std::array<double, NDIM> result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = static_cast<double>(target[axis] - source[axis]) * kRawScale;
        }
        return result;
    }

    ParticleCount nodeCount(KdNode const& node) {
        PROFILE();
        return node.particles.end - node.particles.begin;
    }

    bool isLeaf(KdNode const& node) {
        PROFILE();
        return node.leftChild == kNoChild;
    }

    struct CellWalk {
        TreeNearFieldRequest const& request;
        CellKdTrees const& trees;
        double thetaInverse;
        double support;
        std::vector<SourceFrame> frames;
        size_t sinkRoot;
        size_t sinkEnd;
    };

    // Old-tree-code opening tests, symmetric in the two nodes. Every compared
    // quantity is computed identically from the mirrored pair.
    TreeInteractionType classify(
        CellWalk const& walk, KdNode const& sink, SignedPoint const& sinkPosition, KdNode const& source, Entry const& entry, bool& accepted) {
        PROFILE();
        double const rs = sink.radius;
        double const ro = source.radius;
        double const r2 = squaredDistance(sinkPosition, entry.position);
        double const minimum = rs + ro + walk.support;
        double const cc = std::max((rs + ro) * walk.thetaInverse, minimum);
        accepted = true;
        if (r2 > cc * cc) {
            return TreeInteractionType::m2l;
        }
        if (!isLeaf(sink) || !isLeaf(source)) {
            accepted = false;
            return TreeInteractionType::p2p;
        }
        double const pc = std::max(rs + ro * walk.thetaInverse, minimum);
        double const cp = std::max(rs * walk.thetaInverse + ro, minimum);
        double const pcBall = std::max(ro * walk.thetaInverse, ro + walk.support);
        double const cpBall = std::max(rs * walk.thetaInverse, rs + walk.support);
        SignedPoint const zero{};
        bool const pcPass = r2 > pc * pc || squaredBoxDistance(sink.particleBox, zero, entry.position) > pcBall * pcBall;
        bool const cpPass = r2 > cp * cp || squaredBoxDistance(source.particleBox, walk.frames[entry.frame].origin, sinkPosition) > cpBall * cpBall;
        ParticleCount const sinkCount = nodeCount(sink);
        ParticleCount const sourceCount = nodeCount(source);
        if (pcPass && cpPass) {
            if (sinkCount < sourceCount) {
                return TreeInteractionType::m2p;
            }
            if (sinkCount > sourceCount) {
                return TreeInteractionType::p2l;
            }
            return TreeInteractionType::p2p;
        }
        if (pcPass) {
            return TreeInteractionType::m2p;
        }
        if (cpPass) {
            return TreeInteractionType::p2l;
        }
        return TreeInteractionType::p2p;
    }

    Position absoluteCenter(KdNode const& node) {
        PROFILE();
        return node.center;
    }

    // Build the cell's lists, descending sink and source trees together.
    // filled marks every sink node receiving an M2L or P2L contribution.
    void walkCell(CellWalk const& walk, std::vector<Entry> rootEntries, std::vector<CellExpansion>& expansions, std::vector<bool>& filled, CellLists& lists,
        std::vector<TreeInteraction>* interactions) {
        PROFILE();
        auto const& trees = walk.trees;
        struct Pending {
            size_t sink;
            std::vector<Entry> checklist;
        };
        std::vector<Pending> pending;
        pending.push_back({walk.sinkRoot, std::move(rootEntries)});
        SourceFrame const& sinkFrame = walk.frames.front();
        std::vector<Entry> next;
        std::vector<Entry> carried;
        while (!pending.empty()) {
            Pending item = std::move(pending.back());
            pending.pop_back();
            KdNode const& sink = trees.node(item.sink);
            if (nodeCount(sink) == 0) {
                continue;
            }
            SignedPoint const sinkPosition = centerInFrame(sink, sinkFrame);
            bool const sinkLeaf = isLeaf(sink);
            size_t const m2lBegin = lists.m2lSources.size();
            size_t const p2lBegin = lists.p2lSources.size();
            size_t const m2pBegin = lists.m2pSources.size();
            size_t const p2pBegin = lists.p2pSources.size();
            carried.clear();
            auto& checklist = item.checklist;
            do {
                next.clear();
                for (Entry const& entry : checklist) {
                    KdNode const& source = trees.node(entry.node);
                    bool accepted = false;
                    auto const type = classify(walk, sink, sinkPosition, source, entry, accepted);
                    if (accepted) {
                        auto const& frame = walk.frames[entry.frame];
                        auto const offset = separation(sinkPosition, entry.position);
                        switch (type) {
                        case TreeInteractionType::m2l:
                            lists.m2lSources.push_back({trees.multipole(entry.node).data(), {offset[0], offset[1], offset[2]}});
                            break;
                        case TreeInteractionType::m2p:
                            lists.m2pSources.push_back({trees.multipole(entry.node).data(), {offset[0], offset[1], offset[2]}});
                            break;
                        case TreeInteractionType::p2l:
                            lists.p2lSources.push_back({source.particles, absoluteCenter(source), {offset[0], offset[1], offset[2]}});
                            break;
                        case TreeInteractionType::p2p:
                            lists.p2pSources.push_back({source.particles, frame.selfCell});
                            break;
                        }
                        if (interactions != nullptr) {
                            interactions->push_back({item.sink, entry.node, frame.cellOffset, type});
                        }
                    } else if (isLeaf(source)) {
                        carried.push_back(entry);
                    } else {
                        for (size_t const child : {source.leftChild, source.rightChild}) {
                            KdNode const& childNode = trees.node(child);
                            if (nodeCount(childNode) != 0) {
                                next.push_back({child, entry.frame, centerInFrame(childNode, walk.frames[entry.frame])});
                            }
                        }
                    }
                }
                std::swap(checklist, next);
            } while (sinkLeaf && !checklist.empty());
            size_t const local = item.sink - walk.sinkRoot;
            float* const expansion = expansions[local].data();
            if (lists.m2lSources.size() != m2lBegin) {
                lists.m2lTargets.push_back({expansion, m2lBegin, lists.m2lSources.size()});
                filled[local] = true;
            }
            if (lists.p2lSources.size() != p2lBegin) {
                lists.p2lTargets.push_back({expansion, p2lBegin, lists.p2lSources.size()});
                filled[local] = true;
            }
            if (sinkLeaf) {
                expect(carried.empty());
                if (lists.m2pSources.size() != m2pBegin) {
                    lists.m2pTargets.push_back({sink.particles, sink.center, m2pBegin, lists.m2pSources.size()});
                }
                if (lists.p2pSources.size() != p2pBegin) {
                    lists.p2pLeaves.push_back({item.sink, {p2pBegin, lists.p2pSources.size()}});
                }
            } else {
                checklist.insert(checklist.end(), carried.begin(), carried.end());
                pending.push_back({sink.rightChild, checklist});
                pending.push_back({sink.leftChild, std::move(checklist)});
            }
        }
    }

    std::array<double, NDIM> minimumImage(Position const& particle, Position const& center) {
        PROFILE();
        Coordinate const p[NDIM] = {particle.x, particle.y, particle.z};
        Coordinate const c[NDIM] = {center.x, center.y, center.z};
        std::array<double, NDIM> result{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            result[axis] = static_cast<double>(static_cast<std::int32_t>(p[axis].raw() - c[axis].raw())) * kRawScale;
        }
        return result;
    }

    bool isSink(ParticleStorage const& particles, ParticleCount index, std::uint8_t activeRung) {
        PROFILE();
        auto const flags = particles.flags(index);
        return !flags.isGhost() && flags.rung() >= activeRung;
    }

    template <int order, bool computePotential>
    void evaluateDirect(TreeNearFieldRequest const& request, CellLists const& lists) {
        PROFILE();
        auto const& particles = *request.particles;
        auto const& trees = *request.trees;
        DirectInteractionParticles view{};
        for (size_t axis = 0; axis < NDIM; axis++) {
            view.positions[axis] = particles.positionData(axis);
        }
        gravity::DirectInteractionRequest const direct{{}, {}, view, request.softening, 1.0f, 1.0, request.activeRung};
        for (auto const& [leaf, sources] : lists.p2pLeaves) {
            auto const range = trees.node(leaf).particles;
            for (ParticleCount sink = range.begin; sink < range.end; sink++) {
                if (!isSink(particles, sink, request.activeRung)) {
                    continue;
                }
                Position const position{particles.position(0, sink), particles.position(1, sink), particles.position(2, sink)};
                gravity::detail::DirectInteractionLanes lanes;
                for (size_t index = sources.first; index < sources.second; index++) {
                    auto const& source = lists.p2pSources[index];
                    if (source.selfCell && sink >= source.range.begin && sink < source.range.end) {
                        gravity::detail::accumulateDirectRange<order, computePotential>(position, source.range.begin, sink, direct, lanes);
                        gravity::detail::accumulateDirectRange<order, computePotential>(position, sink + 1, source.range.end, direct, lanes);
                    } else {
                        gravity::detail::accumulateDirectRange<order, computePotential>(position, source.range.begin, source.range.end, direct, lanes);
                    }
                }
                FpeGuard guard;
                for (size_t lane = 0; lane < gravity::detail::kDirectFloatLanes; lane++) {
                    request.accelerations[0][sink] += lanes.x[lane];
                    request.accelerations[1][sink] += lanes.y[lane];
                    request.accelerations[2][sink] += lanes.z[lane];
                    if constexpr (computePotential) {
                        request.potentials[sink] += lanes.potential[lane];
                    }
                }
            }
        }
    }

    template <int order>
    void evaluateDirect(TreeNearFieldRequest const& request, CellLists const& lists) {
        PROFILE();
        if (request.computePotential) {
            evaluateDirect<order, true>(request, lists);
        } else {
            evaluateDirect<order, false>(request, lists);
        }
    }

    // Translate expansions down the sink tree, then evaluate them at the
    // leaves' sinks. Nodes are in depth-first order, parents first.
    void evaluateLocal(TreeNearFieldRequest const& request, CellWalk const& walk, std::vector<CellExpansion>& totals, std::vector<bool>& present) {
        PROFILE();
        auto const& particles = *request.particles;
        auto const& trees = walk.trees;
        double const spacing = request.grid->layout().spacing();
        for (size_t index = walk.sinkRoot; index < walk.sinkEnd; index++) {
            size_t const local = index - walk.sinkRoot;
            if (!present[local]) {
                continue;
            }
            KdNode const& node = trees.node(index);
            if (isLeaf(node)) {
                DoubleExpansion source{};
                for (size_t coefficient = 0; coefficient < source.size(); coefficient++) {
                    source[coefficient] = totals[local][coefficient];
                }
                for (ParticleCount sink = node.particles.begin; sink < node.particles.end; sink++) {
                    if (!isSink(particles, sink, request.activeRung)) {
                        continue;
                    }
                    Position const position{particles.position(0, sink), particles.position(1, sink), particles.position(2, sink)};
                    auto const displacement = minimumImage(position, node.center);
                    DoubleExpansion point{};
                    stf::addL2L<double, kExpansionOrder, true>(point, source, displacement, spacing, 1.0);
                    FpeGuard guard;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        request.accelerations[axis][sink] -= static_cast<float>(point[stf::index(axis == 0, axis == 1, axis == 2)]);
                    }
                    if (request.computePotential) {
                        request.potentials[sink] += static_cast<float>(point[0]);
                    }
                }
                continue;
            }
            DoubleExpansion source{};
            for (size_t coefficient = 0; coefficient < source.size(); coefficient++) {
                source[coefficient] = totals[local][coefficient];
            }
            for (size_t const child : {node.leftChild, node.rightChild}) {
                KdNode const& childNode = trees.node(child);
                if (nodeCount(childNode) == 0) {
                    continue;
                }
                size_t const childLocal = child - walk.sinkRoot;
                std::array<double, NDIM> const displacement = minimumImage(childNode.center, node.center);
                DoubleExpansion target{};
                for (size_t coefficient = 0; coefficient < target.size(); coefficient++) {
                    target[coefficient] = totals[childLocal][coefficient];
                }
                stf::addL2L<double, kExpansionOrder, true>(target, source, displacement, spacing, spacing);
                FpeGuard guard;
                for (size_t coefficient = 0; coefficient < target.size(); coefficient++) {
                    totals[childLocal][coefficient] = static_cast<float>(target[coefficient]);
                }
                present[childLocal] = true;
            }
        }
    }

    void validate(TreeNearFieldRequest const& request) {
        PROFILE();
        if (request.grid == nullptr || request.particles == nullptr || request.trees == nullptr) {
            throw std::invalid_argument("tree near field requires a grid, particles, and trees");
        }
        if (request.softening.model != SofteningModel::ferrers) {
            throw std::invalid_argument("tree near field requires Ferrers softening");
        }
        if (request.ferrersOrder < 0 || request.ferrersOrder > 3) {
            throw std::invalid_argument("tree near field Ferrers order must be between zero and three");
        }
        if (!(request.openingAngle > 0.0 && request.openingAngle < 1.0)) {
            throw std::invalid_argument("tree near field opening angle must lie in (0,1)");
        }
        if (request.fmmOrder < kMinimumFmmOrder || request.fmmOrder > kMaximumFmmOrder) {
            throw std::invalid_argument("tree near field FMM order is unsupported");
        }
        auto const& layout = request.grid->layout();
        if (request.ghostWidth < 1 || request.ghostWidth > kPmmmMaximumNeighborWidth ||
            layout.globalSideLength() < static_cast<size_t>(2 * request.ghostWidth + 1)) {
            throw std::invalid_argument("tree near field requires distinct near cells of width one or two");
        }
        if (layout.localityCount() > 1 && layout.ghostWidth() < static_cast<size_t>(request.ghostWidth)) {
            throw std::invalid_argument("tree near field requires a halo at least as wide as its near-cell mask");
        }
        if (request.trees->cellCount() != layout.storedCellCount()) {
            throw std::invalid_argument("tree near field trees do not cover the grid");
        }
        for (size_t axis = 0; axis < NDIM; axis++) {
            if (request.accelerations[axis] == nullptr) {
                throw std::invalid_argument("tree near field requires acceleration outputs");
            }
        }
        if (request.computePotential && request.potentials == nullptr) {
            throw std::invalid_argument("tree near field requires a potential output");
        }
    }

}    // namespace detail

void evaluateTreeNearField(TreeNearFieldRequest const& request) {
    PROFILE();
    detail::validate(request);
    auto const& grid = *request.grid;
    auto const& layout = grid.layout();
    auto const& trees = *request.trees;
    auto const& particles = *request.particles;
    int const cellExponent = std::countr_zero(layout.globalSideLength());
    int const positionShift = Coordinate::fractionalBits() - cellExponent;
    std::int64_t const cellWidth = std::int64_t(1) << positionShift;
    bool const thresholdTrees = trees.ordering() == KdOrdering::rungThreshold;
    double support = 0.0;
    double thetaInverse = 0.0;
    {
        // A small margin keeps float rounding of the Ferrers support from
        // admitting a pair the softened kernel would still soften.
        constexpr double kSupportMargin = 1.0 + 1.0e-6;
        FpeGuard guard;
        support = std::sqrt(static_cast<double>(request.softening.qSquared)) * kSupportMargin;
        thetaInverse = 1.0 / request.openingAngle;
    }
    size_t const nearSide = static_cast<size_t>(2 * request.ghostWidth + 1);
    size_t nearCount = 1;
    for (size_t axis = 0; axis < NDIM; axis++) {
        nearCount *= nearSide;
    }
    size_t const interiorCount = layout.interiorCellCount();
    std::vector<std::vector<TreeInteraction>> cellInteractions(request.interactions != nullptr ? interiorCount : 0);
    ::cosmictiger::detail::ParallelChunkPlan const plan(interiorCount, 1);
    ::cosmictiger::detail::parallelForChunks(plan,
        hpx::annotated_function(
            [&](size_t, size_t begin, size_t end) {
                PROFILE();
                for (size_t interior = begin; interior < end; interior++) {
                    GridIndices sinkCell{};
                    size_t remainder = interior;
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        sinkCell[axis] = static_cast<std::int64_t>(remainder % layout.interiorSideLength());
                        remainder /= layout.interiorSideLength();
                    }
                    size_t const sinkIndex = layout.cellIndex(sinkCell);
                    size_t const sinkRoot = thresholdTrees ? trees.atOrAboveThresholdRoot(sinkIndex) : trees.positionRoot(sinkIndex);
                    if (detail::nodeCount(trees.node(sinkRoot)) == 0) {
                        continue;
                    }
                    size_t sinkEnd = trees.nodeCount();
                    if (sinkIndex + 1 < trees.cellCount()) {
                        sinkEnd = thresholdTrees ? trees.belowThresholdRoot(sinkIndex + 1) : trees.positionRoot(sinkIndex + 1);
                    }
                    detail::CellWalk walk{request, trees, thetaInverse, support, {}, sinkRoot, sinkEnd};
                    walk.frames.reserve(detail::kMaximumNearCells + 1);
                    auto const sinkLower = detail::cellLowerRaw(layout, sinkCell, positionShift);
                    // Frame zero is the sink cell's own frame.
                    walk.frames.push_back({{}, sinkLower, {}, true});
                    std::vector<detail::Entry> entries;
                    for (size_t neighbor = 0; neighbor < nearCount; neighbor++) {
                        size_t offsetIndex = neighbor;
                        GridIndices coordinates = sinkCell;
                        std::array<std::int64_t, NDIM> cellOffset{};
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            cellOffset[axis] = static_cast<std::int64_t>(offsetIndex % nearSide) - request.ghostWidth;
                            offsetIndex /= nearSide;
                            coordinates[axis] += cellOffset[axis];
                            if (layout.localityCount() == 1) {
                                auto const side = static_cast<std::int64_t>(layout.globalSideLength());
                                coordinates[axis] = (coordinates[axis] + side) % side;
                            }
                        }
                        // Same-locality neighbors use the owner's tree, as the
                        // direct near field uses the owner's exact range.
                        auto const owner = layout.cellOwner(coordinates);
                        GridIndices const stored = owner.localityIndex == layout.localityIndex() ? owner.coordinates : coordinates;
                        size_t const storedIndex = layout.cellIndex(stored);
                        bool const selfCell = storedIndex == sinkIndex;
                        detail::SourceFrame frame{};
                        for (size_t axis = 0; axis < NDIM; axis++) {
                            frame.origin[axis] = cellOffset[axis] * cellWidth;
                        }
                        frame.lower = detail::cellLowerRaw(layout, stored, positionShift);
                        frame.cellOffset = cellOffset;
                        frame.selfCell = selfCell;
                        auto const frameIndex = static_cast<std::uint32_t>(walk.frames.size());
                        walk.frames.push_back(frame);
                        std::array<size_t, 2> roots{};
                        size_t rootCount = 1;
                        if (thresholdTrees) {
                            roots = {trees.belowThresholdRoot(storedIndex), trees.atOrAboveThresholdRoot(storedIndex)};
                            rootCount = 2;
                        } else {
                            roots[0] = trees.positionRoot(storedIndex);
                        }
                        for (size_t root = 0; root < rootCount; root++) {
                            KdNode const& node = trees.node(roots[root]);
                            if (detail::nodeCount(node) != 0) {
                                entries.push_back({roots[root], frameIndex, detail::centerInFrame(node, frame)});
                            }
                        }
                    }
                    std::vector<CellExpansion> expansions(sinkEnd - sinkRoot);
                    std::vector<bool> filled(expansions.size(), false);
                    detail::CellLists lists;
                    detail::walkCell(
                        walk, std::move(entries), expansions, filled, lists, request.interactions != nullptr ? &cellInteractions[interior] : nullptr);
                    double const spacing = layout.spacing();
                    fmm::FmmM2lRequest const m2l{
                        lists.m2lTargets.data(), lists.m2lTargets.size(), lists.m2lSources.data(), lists.m2lSources.size(), spacing, request.fmmOrder};
                    fmm::addM2lCpu(m2l);
                    fmm::FmmP2lRequest p2l{lists.p2lTargets.data(), lists.p2lTargets.size(), lists.p2lSources.data(), lists.p2lSources.size(), {}, 1.0f,
                        spacing, request.fmmOrder};
                    fmm::FmmM2pRequest m2p{lists.m2pTargets.data(), lists.m2pTargets.size(), lists.m2pSources.data(), lists.m2pSources.size(), {}, {}, nullptr,
                        spacing, request.fmmOrder};
                    for (size_t axis = 0; axis < NDIM; axis++) {
                        p2l.positions[axis] = particles.positionData(axis);
                        m2p.positions[axis] = particles.positionData(axis);
                        m2p.accelerations[axis] = request.accelerations[axis];
                    }
                    m2p.potentials = request.computePotential ? request.potentials : nullptr;
                    fmm::addP2lCpu(p2l);
                    detail::evaluateLocal(request, walk, expansions, filled);
                    fmm::addM2pCpu(m2p);
                    switch (request.ferrersOrder) {
                    case 0:
                        detail::evaluateDirect<0>(request, lists);
                        break;
                    case 1:
                        detail::evaluateDirect<1>(request, lists);
                        break;
                    case 2:
                        detail::evaluateDirect<2>(request, lists);
                        break;
                    default:
                        detail::evaluateDirect<3>(request, lists);
                        break;
                    }
                }
            },
            "treeNearField.cells"));
    if (request.interactions != nullptr) {
        request.interactions->clear();
        for (auto const& cell : cellInteractions) {
            request.interactions->insert(request.interactions->end(), cell.begin(), cell.end());
        }
    }
}

}    // namespace cosmictiger::gravity::tree
