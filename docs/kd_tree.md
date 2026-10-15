# Per-cell kd-trees

`cosmictiger/gravity/tree/kd_tree.hpp` builds a binary kd-tree in every stored
mesh cell (interior and ghost) for local FMM solves. **The trees are not yet
used by the kick or the evolution path**; no tree walk exists yet.

## Structure

- The root of each tree is the mesh cell. Every node has exactly two children
  or none. Empty children are kept, so the tree stays strictly binary.
- A split partitions the node's particle range in place with
  `ParticleStorage::swapParticles`, so every node owns one contiguous
  `ParticleRange` and the children tile the parent's range and box. Optional
  companion arrays (for example synchronized momenta) follow the permutation,
  as in `CartesianGrid::sortParticlesByRung`.
- Node boxes and split planes are raw 32-bit fixed-point offsets from the
  cell's lower corner. Wrapping subtraction makes periodic ghost copies work
  unchanged, and geometric midpoints of the dyadic boxes are exact.
- Nodes are stored in one flat array per grid, in depth-first order with
  each left child directly after its parent. Each subtree is a contiguous block
  of nodes. `KdNode` holds no pointers, so it can be copied to a device as is.
- Building is iterative (HPX thread stacks are small) and runs cells as HPX
  tasks. A single very crowded cell is built by one task.

## Splitting criterion

`KdSplitter` is the replaceable criterion. The default
`GeometricMidpointSplitter` splits at the geometric center of the node box on
axis `depth mod NDIM` (x, y, z, x, ...). A node is a leaf when it has at most
`leafSize` particles (default `kDefaultLeafSize = 100`) or when the splitter
declines. The geometric splitter declines at fixed-point resolution, so the
maximum depth is `NDIM * (32 - gravity.meshExponent)`. A custom splitter
must choose a plane strictly inside the node box on the chosen axis; the build
throws `std::logic_error` otherwise, which also guarantees termination.

## Orderings

There are three per-cell particle orderings in the code:

| Ordering | Entry | Used by |
| --- | --- | --- |
| Rung ranges (ascending rung) | `CartesianGrid::sortParticlesByRung` | PM-FMM kick (unchanged) |
| Position only | `CellKdTrees::buildByPosition` | Tree path, first integrator |
| Rung < threshold, then rung >= threshold, each by position | `CellKdTrees::buildByRungThreshold` | Implemented and tested; not yet used |

The rung-threshold ordering builds two trees per cell, both rooted at the full
cell box: `belowThresholdRoot` and `atOrAboveThresholdRoot`. It is intended
to be rebuilt each subtimestep.

A kd build overwrites the PM-FMM rung order, so it calls
`CartesianGrid::invalidateRungRanges()` first. The kick still works then but
scans whole cells for active sinks. `sortParticlesByRung` restores the rung
ranges. Ghost copies are reordered independently of their owners, so after a
build ghost index order does not match owner index order.

## Node moments

Each node stores its mass (count times the common particle mass), an
expansion center, a radius bound, and a `CellMultipole` with the cell spacing
as scale, the layout used by the cell-level FMM interactions.

- `MultipoleCenter::centerOfMass` (default): the center is the center of mass,
  computed from exact integer offset sums and rounded to the fixed-point grid.
  The dipole about it vanishes and is stored as zero.
- `MultipoleCenter::geometricCenter`: the center of the node box, with the
  computed dipole retained.

Empty nodes use the geometric center, zero moments, and zero radius. Leaves
accumulate P2M in double. Internal nodes combine their children with the
generated M2M operator in double, and the result is rounded once to float.
The radius is an upper bound on the distance from the center to any of the
node's particles. For an internal node it is the smaller of the children's
bound and the distance to the farthest box corner.

Leaf P2M uses `stf::addPoint`, which evaluates the STF projection explicitly.
It dominates build time. A generated P2M kernel would be needed before
rebuilding trees every step.

## Tests

`cosmictiger.kd_tree` (`tests/tree/kd_tree.cpp`) builds a 4^3 grid with a
one-cell periodic ghost shell. Its cells are uniform, clustered, coincident,
single-particle, or empty. The tests check:

- Particles, rungs, and companions are preserved, and each stays in its cell.
- Box containment, x/y/z alternation, exact midpoint planes, and range and box
  tiling.
- Leaf-size and fixed-point depth limits.
- Center of mass, radius bounds, and every multipole coefficient against a
  direct double P2M.
- Rung-threshold separation.
- The geometric-center dipole, a custom splitter, and rejected requests.
