# Particle drift

`cosmictiger/drift/drift.hpp` supplies the local drift driver. Each call selects
one processor for the entire update: CPU or CUDA. The position-update
implementations are separate. There is no mixed CPU/GPU partition.

The stored velocity is canonical momentum `p = a dq/dtau`, with `q = x/L` and
`tau = H0 integral(dt/a)`. A drift holds momentum constant:

```
q1 = q0 + p D
D = integral(d tau/a) = integral da / (a da/dtau)
```

`SimulationBackground::driftFactor(a0, a1)` evaluates `D` using the same
cosmological background as conformal time, including radiation, curvature,
thermal massive neutrinos, and configured dark energy. Equal epochs return
zero; reversed positive endpoints return a negative factor. Compute the
factor once for the interval, then pass it to the selected backend:

```cpp
SimulationBackground const background(constants.cosmology);
double const factor = background.driftFactor(initialScaleFactor, finalScaleFactor);
driftParticles(particles, grid, factor, DriftExecution::cpu);
// Alternatively, choose CUDA for the entire call:
// driftParticles(particles, grid, factor, DriftExecution::cuda, deviceId);
```

The driver first clears the appended ghost suffix using interior ownership
ranges, then advances every owned particle regardless of rung. Same-locality
periodic copies have an unset ghost flag, so ownership cannot be inferred from
that flag alone. A zero factor leaves particles and ghost metadata unchanged.
Momentum, rung flags and optional retained gravity fields are preserved.
Positions wrap in the periodic unit cube using raw fixed-point integer
addition. Each displacement is reduced modulo one and quantized toward zero
at the coordinate resolution; existing coordinate low bits are retained.
No `long double` arithmetic is introduced.

CPU work uses bounded HPX chunks and completes before return. CUDA work goes
through `cuda::launch` and the driver waits for the GPU completion future.
CUDA selection in a CPU-only build raises an error before storage changes.

For callers that already manage ownership and completion:

```cpp
auto const request = makeDriftRequest(particles, ownedRange, factor);
driftParticlesCpu(request);
// Or, in a CUDA-enabled build:
// auto completion = launchDriftParticlesCuda(request, stream, deviceId);
// completion.get();
```

These lower-level entry points operate on the supplied half-open range and
skip remote ghost flags. The range must exclude same-locality periodic copies.
All raw arrays must cover the range. CUDA pointers must be device-accessible;
`ParticleStorage` provides managed fields in CUDA-enabled builds. Keep storage
and caller-owned streams alive until completion, and do not resize, remove,
reconfigure, or access positions while CUDA work is pending. Order dependent
submissions through their futures.

All entry points require exclusive access and completed previous GPU work.
The local driver performs no communication or ownership migration. After
movement, interior ranges identify the previous particle ordering; cell
membership, multipoles, expansions and retained gravity fields are stale.
Call `loadBalanceParticles`, then refresh ghosts and gravity data before using
them. A failed arithmetic contract or CUDA execution can leave a partial drift;
callers must not continue evolution as if it succeeded.

The evolution command line selects this driver with
`--execution.cuda.drift=on`; the independent `--execution.cuda.kick=on` switch
selects the CUDA particle kick, and `--execution.cuda.multipole=on` selects
CUDA cell P2M construction. All three switches default to `off`.

When CUDA kick feeds CPU drift, evolution batches a managed-memory prefetch of
owned positions and momenta to the host before accessing them. This avoids
per-page CPU faults after the GPU writes momenta. The prefetch completion is
awaited through the HPX CUDA interface and its full wall time remains in `D%`.
CUDA builds with CPU kick incur no prefetch submission; devices without
concurrent managed access retain demand migration. This changes data placement,
not the selected drift backend or the position arithmetic.

The opt-in handoff benchmark writes momenta on the GPU before each CPU drift,
compares demand migration with batch prefetch (including transfer and future
completion), and separately reports a warm CPU drift. Run from the repository
root after configuring a CUDA-enabled Release build with `BUILD_TESTING=ON`
and building `cosmictiger_drift_tests`:

```sh
./release/cosmictiger_drift_tests --gtest_also_run_disabled_tests \
  --gtest_filter=DriftCudaTest.DISABLED_ManagedCpuHandoffBenchmark \
  --hpx:threads=2 --hpx:bind=none --hpx:ini=hpx.parcel.mpi.enable=0 \
  --hpx:ini=hpx.threadpools.io_pool_size=1 \
  --hpx:ini=hpx.threadpools.timer_pool_size=1 \
  --hpx:ini=hpx.threadpools.parcel_pool_size=1
```

On the local RTX 3090 on 2026-10-08, 24 repeats with two HPX workers measured
2.34 ms versus 0.382 ms at 32,768 particles, and 61.2 ms versus 7.05 ms at
1,048,576 particles. These are synthetic CUDA-to-CPU handoff measurements,
not whole-simulation speedups; concurrent machine activity can affect timings.

## Validation

The managed-memory handoff change was validated locally on 2026-10-08 with
12 applicable ordinary CPU/CUDA drift tests, all 15 checked drift tests,
eight CUDA runtime tests (including prefetch completion and partial failure),
and both CUDA evolution tests, including CUDA kick with CPU drift and no
initial snapshot. CUDA Release and CPU-only evolution compilation passed.
The checked build ran the three contract tests skipped by ordinary Release.
No full production rerun or multi-locality run was used for the speedup above.

Validated locally on 2026-10-05:

- CPU-only Release and CUDA-enabled Release libraries and executable built.
- All nine CPU drift tests passed with numerical checks and IEEE traps enabled;
  the ordinary CPU-only Release suite passed its seven functional tests.
- With local GPU access, all fourteen CPU/CUDA drift tests passed in the
  checks-enabled build. The ordinary CUDA-enabled Release suite passed eleven
  tests and skipped the three validation-only cases, which the checked build
  executed successfully.
- All eleven background-time tests passed, including four drift-factor tests
  against analytic integrals, additivity, reversal and equal epochs.

Coverage includes positive/negative and multi-box wrapping, sub-coordinate
resolution shifts, preservation of raw position low bits, every rung,
unchanged momentum/flags/optional fields, disjoint CPU chunks, external CUDA
streams, GPU completion, empty work, invalid requests and same-owner periodic
ghost cleanup. The sandbox hid CUDA devices; GPU execution was verified with
permitted local GPU access. No multi-process drift/evolution run was performed;
the drift driver itself does not communicate.
