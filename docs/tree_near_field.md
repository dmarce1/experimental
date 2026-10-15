# Tree near field

`gravity.nearField=tree` replaces the direct P2P sum over the PMMM near
region with a local FMM solve over the per-cell kd-trees. The near region is
unchanged: the `(2w+1)^NDIM` cells around each sink cell, `w =
gravity.ghostWidth`, chosen by the same rule the direct near field uses
(periodic same-locality neighbors resolve to their owner cell). PMMM still
evaluates everything outside it, so the two parts remain disjoint.

| Option | Values | Default |
| --- | --- | --- |
| `gravity.nearField` | `direct`, `tree` | `direct` |
| `gravity.openingAngle` | `0 < theta < 1` | `0.55` |

The tree near field requires Ferrers softening: it accepts multipole
interactions only between particles farther apart than the Ferrers support
`h`, where the softened kernel is exactly Newtonian, and a Plummer kernel is
never Newtonian. It currently runs on the CPU kick only;
`execution.cuda.kick=on` together with `tree` is rejected. Both options are
solver choices: a restart may select different values. They are recorded in
checkpoint schema 10; older checkpoints restart with the requested values.

## Code

| File | Contents |
| --- | --- |
| `cosmictiger/gravity/tree/tree_near_field.hpp`, `src/gravity/tree/tree_near_field.cpp` | The walk, list building, and evaluation |
| `cosmictiger/gravity/tree/kd_tree.hpp` | Trees, node centers and radii ([kd_tree.md](kd_tree.md)) |
| `cosmictiger/gravity/fmm/fmm_interactions.hpp` | Batched M2L, P2L, M2P ([fmm_interactions.md](fmm_interactions.md)) |
| `PmmmNearField::prepare` in `src/gravity/kick/pmmm_kick.cpp` | Builds the trees and selects the solver |

`PmmmNearField::prepare` runs in the near branch of the overlapped kick, so
the tree build and walk overlap the FFT far field and are counted in `G%`.
It builds rung-threshold trees with `threshold = activeRung`, centers chosen by
`MultipoleCenter::minimumRadius`, and unit particle mass. The build reorders
particles inside each cell (with any snapshot momenta) and invalidates the
rung ranges, so the evolution turns grid rung sorting off in tree mode. The
walk writes unit-mass fields per sink; each sink's total goes into the first
lane of the direct near-field accumulator, and the existing CPU kick applies
the particle mass, adds the PMMM far field, chooses the rung, and kicks once.

## Walk

For each interior sink cell, the at-or-above-threshold tree is the sink tree
and both trees of every near cell are sources. The initial checklist holds the
nonempty source roots. Sink and source trees are descended together, as in the
old CosmicTiger walk:

- A pair is accepted as cc (M2L into the sink node's expansion) when
  `R > max((rs+ro)/theta, rs+ro+h)`.
- When both nodes are leaves, pc (M2P of the source multipole at the sink
  particles) passes when `R > max(rs+ro/theta, rs+ro+h)`, or when every sink
  particle lies farther than `max(ro/theta, ro+h)` from the source center,
  using the sink's particle bounding box. cp (P2L of the source particles into
  the sink expansion) is the mirror test. If both pass, the one that loops
  over fewer particles is used (pc when the sink has fewer particles, cp when
  the source has fewer); equal counts fall back to pp.
- Otherwise two leaves interact by Ferrers P2P, excluding the sink itself.
- A non-accepted source that is a leaf is carried down to the sink's
  children; any other non-accepted source is opened. A sink leaf keeps
  opening sources until its checklist is empty; an internal sink passes its
  remaining checklist to both children.

Here `R` is the distance between node centers, `rs` and `ro` the radius
bounds, and `h` the Ferrers support (with a `1e-6` relative margin). Every
accepted multipole interaction is therefore between particle pairs farther
apart than `h`.

The old CPU box term was inverted (it accepted pc when the source sphere
*overlapped* the sink box), and the old CUDA box radius `max(r/theta, h)` did
not guarantee separation from every source particle; the corrected radius is
`max(r/theta, r+h)`. The old Ewald-checklist clamp of `R` is not used: it
belonged only to the old Ewald correction, which PMMM replaces.

After the walk, each sink cell evaluates its lists in batches: M2L and P2L
into float node expansions, L2L translation down the sink tree in depth-first
order, leaf L2P at every sink, M2P, then P2P. Each list target is one sink node
or leaf, so outputs never alias.

## Momentum conservation

With equal masses the near-region forces sum to zero up to rounding when every
particle is a sink:

- M2L keeps source rank `n` and local rank `k` with `n + k <= p`; the force on
  the sink uses `k = a + 1`, so the retained terms `a + n <= p - 1` are
  symmetric in the two nodes. P2L in one direction and M2P in the other keep
  the same terms. L2L translation of the finite polynomial is exact, so this
  holds for any expansion centers.
- Every opening test is a symmetric expression in the two nodes, so the pair
  (A, B) receives mirrored interactions from both sides: cc/cc, pc/cp, pp/pp.
- Separations and box distances are exact integer fixed-point differences in
  the sink cell's frame, including the periodic shift, so both sides compare
  bitwise-identical values.
- Tree geometry (boxes, centers, radii) depends only on the set of particles
  in a node, so a ghost copy of a cell builds the same tree as its owner.

During a substep only active particles are sinks, so the reaction on inactive
particles is not applied; this is inherent to block timesteps.

## Tests

`cosmictiger.tree_near_field` (`tests/tree/tree_near_field.cpp`) uses an 8^3
grid with uniform particles and a dense cluster across a cell corner and the
periodic seam:

- every sink sees every other particle of its near region exactly once,
  through exactly one accepted node pair, for widths 1 and 2 and several theta;
- the interaction lists are exactly mirror symmetric;
- near fields converge to the direct near field as theta decreases.

`cosmictiger.force_symmetry` (`tests/gravity/kick/force_symmetry.cpp`) measures
`|sum a| / sum |a|` for complete diagnostic evaluations with direct and tree
near fields. Measured on October 15, 2026 (RelWithDebInfo, CPU):

| Near field | Width | Order | Complete | Near only |
| --- | --- | --- | --- | --- |
| direct | 1 | 6 | 1.5e-9 | 1.7e-9 |
| direct | 2 | 6 | 2.5e-9 | 2.0e-9 |
| tree, theta 0.55 | 1 | 6 | 1.9e-9 | 1.8e-9 |
| tree, theta 0.55 | 2 | 6 | 1.5e-9 | 1.4e-9 |

The tree truncation error is far larger (below), so the agreement is evidence
of symmetry rather than of small errors. The CUDA direct kick measured
3e-10 to 8e-10.

Relative RMS near-field error of the tree against the direct near field
(width 1; width 2 is similar):

| theta | Order 4 acceleration | Order 6 acceleration | Order 6 potential |
| --- | --- | --- | --- |
| 0.80 | 5.6e-3 | 1.8e-3 | 1.2e-4 |
| 0.55 | 3.9e-4 | 5.3e-5 | 3.4e-6 |
| 0.30 | 1.3e-5 | 6.1e-7 | 1.4e-7 |

## Runs

Release CPU build, October 15, 2026:

| Check | Result |
| --- | --- |
| Opt-in benchmark (`TreeNearFieldBenchmark`), 262144 particles, half in a cluster, one near-only diagnostic kick, 16 HPX threads | 8^3 cells: direct 45.5 s, tree 3.7 s; 16^3 cells: direct 51.9 s, tree 3.3 s |
| Evolution, 32^3 particles on 8^3 cells, a = 0.0097 to 0.222, 6 HPX threads | Identical step and rung sequence; final K and U agree to 3e-7 relative; energy drift 3.4493424e-7 (tree) vs 3.4493453e-7 (direct); global force sums both about 1e-11. Wall time 70.4 s (tree) vs 63.9 s (direct): at 64 particles per cell the tree build costs more than it saves |
| 8 MPI processes, 64^3 particles on 16^3 cells, initial gravity output | Global force sums 5.8e-14, -4.4e-14, 2.1e-16 (direct) and 5.9e-14, -4.1e-14, 1.0e-14 (tree) in code units; potential sums both 6.75085 |

The tree pays off when cells are crowded; with few particles per cell the
direct near field is as fast or faster.

## Limitations

- CPU only. A CUDA tree near field would build lists on the CPU and launch the
  batched CUDA M2L/P2L/M2P kernels with a fused L2P+P2P kick.
- Trees are rebuilt every substep, and leaf P2M (`stf::addPoint`) dominates
  their build time; a generated P2M kernel would reduce it.
- Interaction lists are built and evaluated per sink cell; batches are not
  merged across cells.
