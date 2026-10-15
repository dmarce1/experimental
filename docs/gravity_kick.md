# Gravity kick and conformal timesteps

The simulation coordinate is `tau = H0 integral(dt/a)` and particle velocity
storage holds canonical momentum `p = a d(x/L)/d tau`. Initial-condition
construction still returns `d(x/L)/d ln(a)`; executable initialization converts
it once by multiplying by `a^2 H(a)/H0`. SILO initial-condition metadata records
the convention and time unit, and accepts the previous convention on read.

Root initializes and publishes `GlobalConstants` only after the actual
initial-condition scale factor is known. Rung zero spans the nominal interval
from that epoch to the proper-time endpoint `simulation.stopTime` (years from
the present, default zero). The coupled background supplies the conformal interval:

```
dtau0 = H0 integral dt/a
dtau(rung) = dtau0 * 2^(-rung)
```

The background interval includes radiation, curvature, thermal massive
neutrinos and the configured `w0,wa` dark-energy density. Times use `H0^-1`;
lengths use the comoving box side. The mass passed to both the near kernel and
cell multipoles is `3 Omega_selected/(8 pi N)` with `G=1` in these units.

## Options

All values can be supplied on the command line or in the file named by
`--configurationFile=run.ini`. Explicit command-line values override INI
values, and INI values override defaults. For example:

```ini
[timestep]
eta = 0.025
[simulation]
initialScaleFactor = 0.01
stopTime = 0
[initialConditions]
meshExponent = 6
[gravity]
meshExponent = 5
ghostWidth = 2
[gravity.softening]
cellFraction = 0.05
kernelOrder = 2
[output]
acceleration = true
potential = false
```

Generated initial conditions replace `simulation.initialScaleFactor` with the
selected initial epoch. `initialConditions.meshExponent=m` sets the original
particle lattice and initialization FFT to `2^m` elements per axis, producing
`2^(NDIM*m)` particles. `gravity.meshExponent` independently sets the evolution
grid and gravity FFT resolution; it defaults to the initialization exponent
when omitted. The example above uses 64³ particles and initialization FFT
elements with a 32³ gravity mesh. Initial gravity output uses that same
selected gravity mesh after particle redistribution.

`gravity.ghostWidth` accepts one or two cells and defaults to one. It selects
the ghost layers, direct-neighbor range, and matching periodic Green-function
exclusion together. Width one covers 27 cells and width two covers 125 cells,
including the sink cell. The mesh side must exceed twice the width: the
minimum gravity exponent is two for width one and three for width two.
These options accept command-line and INI values without recompiling.
Checkpoints retain the original particle lattice, selected gravity mesh, and
ghost width. Explicit restart mesh or width settings must match the checkpoint.

The softening radius `q` is `cellFraction` times the evolution gravity cell
width. The fraction is in `(0,1]`, and orders F0 through F3 are supported.
The Plummer-equivalent length is defined by matching the central potential:
`epsilon = q / {1.5, 1.875, 2.1875, 2.4609375}` for F0 through F3.

## Rung changes and kick phases

The force callback receives the **complete** comoving field `g0 = dp/dtau`.
With physical softening `a epsilon` and physical acceleration proportional to
`g0/a^2`, the acceleration criterion becomes

```
allowedDtau = sqrt(2 eta a epsilon / |g0|)
```

Zero acceleration permits rung zero. The allowed step is rounded downward to
the dyadic hierarchy. Refinement can cross any number of rungs immediately.
Coarsening crosses a boundary only when the next coarser rung is synchronized.
Every crossed boundary is checked, so multiple coarsenings at a common
synchronization time are permitted. Each nominal interval has `2^63` integer
ticks; a separate epoch counter supports arbitrarily many nominal intervals.
Synchronization never uses floating-point time.
A step requiring a rung beyond 63 raises an error (or traps on CUDA).

All particles initially have rung zero. `KickPhase::initial` selects their
first rung and applies only the new half-step. `regular` closes the previous
half-step using the old rung, then selects and assigns the new rung, then
opens the new half-step using that rung. These are separate momentum updates;
each half-step rounds to the stored momentum precision. `final` closes only
the old half-step. `diagnostics` evaluates fields without modifying
momentum or rungs. Drift must use the same canonical-momentum convention.
The evolution loop is a separate caller of these routines. The local drift
driver is described in [particle_drift.md](particle_drift.md).
The executable driver preserves open staggered momenta at its stopping time,
including partial drifts, and resumes them directly on restart. It does not
apply a full `final` half-kick at an arbitrary off-lattice endpoint.
An explicit restart rung-zero extension multiplies the nominal interval by
`2^n`, adds `n` to all particle rungs, and remaps the event clock exactly.

## Complete fields, reductions and output

`DirectKickBackend` is the CPU/CUDA sink callback. Use it once after every
source contribution for that sink has been combined. Calling it separately
for partial near-cell source lists gives an incorrect timestep decision.

`kickPmmmParticlesCpu` combines each owned sink's cell expansion and all direct
near cells, then calls the backend once. Same-locality periodic neighbors use
owned ranges for exact self exclusion. The caller must prepare current
multipoles, periodic expansions and remote ghosts first. Evolution grids use
the runtime `gravity.ghostWidth` setting, defaulting to one; distributed PMMM
kicks require at least that many ghost layers. The standalone grid constructor
still defaults to one layer. Sorting by rung occurs only after all
force reads finish. Refresh ghosts before reusing changed particle data.

`kickPmmmParticlesOverlappedCpu` starts a read-only `PmmmNearField` calculation
with `hpx::async`, then runs a caller-supplied far-field callback. The executable
uses this path to compute near interactions while PMMM builds/uses its Green
spectra and runs FFT exchanges. The near future is joined before applying one
combined kick. `kickPmmmParticlesOverlapped` then enters the existing global
force/potential/error reduction on every physical locality.

Near scratch retains the direct SIMD lanes, so the completed far polynomial
still receives each mass-weighted lane in its original order before one float
rounding and backend call. It covers owned candidates, using cached active rung
suffixes when available, and excludes appended ghosts. Lane storage costs
`16 * kDirectFloatLanes` bytes per candidate (128 for AVX2, 256 for AVX512), plus
cell-range/prefix metadata; it is temporary even when retained output fields
are disabled. The existing fused kick remains available when expansions are
already computed and no overlap scratch is wanted.

During overlap, freeze particle storage, positions, flags, output configuration,
and cell ranges. Far work may read moments/trace and write expansions. Near
preparation writes only its scratch and does not update momentum, rungs,
outputs, or diagnostic sums. A near preparation/launch failure still enters
the far callback to preserve distributed FFT participation. Both branches are
drained before a local error escapes or the global reduction begins. A local
branch failure skips its kick; another rank's completed update is not rolled
back if the reduction reports that failure. Existing distributed FFT failures
still require runtime-level recovery. Arithmetic guards end before scheduling
and future waits.

`KickSums` accumulates double-precision `sum(g0)`, optional `sum(phi0)`, and the
number of evaluated owned particles. It also supports synchronized momentum
square sums and the maximum post-kick rung. It excludes ghost sinks. A kick on an
active subset reports that subset. Use a diagnostics evaluation at active
rung zero for a complete snapshot/global diagnostic. The force sum is
`particleMass * sums.acceleration`; the potential energy in the corresponding
comoving normalization is `0.5 * particleMass * sums.potential`. Physical
potential energy carries the additional cosmological `1/a` factor.
The evolution driver evaluates all owned sinks, using `minimumKickRung` to
update only the active subset. It uses temporary synchronized momenta for
`K=m sum(|p_sync|^2)/(2a^2)`, and reports physical force `m sum(g0)/a^2`.
It retains neither synchronization scratch nor particle gravity fields unless
requested. Layzer-Irvine expansion work and the conserved `K+U+W` sum follow the
[evolution description](../README.md#evolution-physical-stopping-times-and-restart).

`allReduceKickSums` returns global sums on every participating HPX locality.
CUDA accumulator/output pointers must be device-accessible and remain live
through completion. Concurrent CPU and GPU work uses separate accumulators;
combine their values only after both complete. Call the collective after local
work and GPU completion futures finish, with a fresh operation
ID and identical potential selection everywhere. Empty localities still join.
`kickDirectInteractionsCpu` combines a complete direct interaction call and
the global reduction; `kickPmmmParticles` does the same for PMMM plus near
forces. A local kick failure is propagated through the collective to every
participant; partial field sums are not reported as successful diagnostics.
`makeDirectKickRequest` fills its constants and optional
storage pointers from `GlobalConstants`.

Acceleration and potential storage are independently optional. Disabled fields
allocate no persistent particle arrays. Enabled fields survive resize, sorting,
removal, migration, ghost exchange and SILO restart. Potential evaluation is
required if either potential output or potential sums are requested. Before a
snapshot, evaluate all owned particles so inactive particles do not retain old
fields. The initial-condition executable does this with a diagnostics kick
when either output flag is enabled. Default-width PMMM output requires a gravity
mesh side of at least eight (`gravity.meshExponent >= 3`), independently of
the particle-lattice size. Selecting `gravity.ghostWidth=1` permits a gravity
mesh side of four.

SILO stores `acceleration` and `potential` as floats in the same normalized
comoving gravity units as the kick callback. No acceleration/potential arrays
are required for the global reduction: only scalar totals remain afterward.

## Validation

Validated locally on 2026-10-05 with the Release HPX installation:

- 104 affected CPU tests passed, including kick phases, exact synchronization,
  PMMM fields, optional storage, sorting, migration and SILO round trips.
  Seventeen validation-only or distributed cases were skipped in that run.
- All 13 kick contract tests passed with validation and IEEE traps enabled;
  all seven background conformal-time tests passed.
- Five CUDA tests passed on the GPU, including direct CPU/GPU comparisons of
  momentum, rungs, retained fields and managed-memory diagnostic sums. One
  validation-only CUDA test was skipped in Release.
- Two real local MPI processes passed force/potential reduction, mismatched
  potential selection, constants publication, startup, ghost exchange and
  PMMM coefficient-transfer checks.
- Executable initialization and SILO verification passed for 64 particles
  without retained fields and 512 particles with both fields. The latter kept
  every initial rung at zero. CLI rejection of a zero-density particle
  population also passed.

The distributed PMMM kick itself was checked with eight logical localities
inside one process. A complete eight-process PMMM initialization and a full
cosmological evolution run were not exercised.

The subsequent near/PMMM overlap follow-up built the application and passed
114 Release tests and 124 validation-enabled tests per run with both one and
two HPX workers. This includes bitwise fused-versus-overlapped particle/rung
comparisons, a crowded-cell case, near/far cancellation, completion gates,
failure/retry and once-only application, real cold/cached PMMM convolution,
logical ghost inputs, and a single-physical-locality diagnostic reduction.
Four tests requiring additional physical localities were skipped in validation;
Release additionally skipped ten validation-only cases. The validation harness
uses Release HPX with strict floating-point flags and validation contracts,
not native Debug/APEX. Multi-process MPI, CUDA and full executable output were
not rerun for this change. See the [parallelism audit](thread_parallelism.md#near-field-and-pmmm-overlap-verification-2026-10-05)
for the full scope and limits.
