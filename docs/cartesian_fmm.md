# Cartesian FMM operators

The Cartesian STF (symmetric trace-free) operators were imported from the local
`/home/dmarce1/workspace/OctotigerII` working tree on 2026-10-04:

- `lib/stf/include/octotigerII/gravity/stf/`: generated kernels and fixed-size API.
- `octotigerII/gravity/cartesian/fmm.hpp` and `src/gravity/cartesian/fmm.cpp`:
  runtime dispatch and cached lattice geometry.
- `tools/generate_stf_m2l.py` and its two helper modules: exact rational generation.
- `lib/stf/tests/` and `tests/stfM2lChecks.cpp`: independent numerical oracles.

The source checkout HEAD was `aa1b712c1a8084162fc1ab631cf6d1cf96447996`.
These operator files were **untracked** there, so that commit identifies the
checkout context, not a committed version of the imported implementation.
The source files were copied, leaving the OctotigerII working tree unchanged.
The original Boost Software License notice and full license in
`cosmictiger/gravity/stf/LICENSE` are retained alongside the CosmicTiger notice.

## Interfaces

`<cosmictiger/gravity/stf/operators.hpp>` supplies the header-only fixed-size
API in `cosmictiger::gravity::stf`, for generated orders 0 through 9:

- `Multipole<order, Real>` and `Expansion<order, Real, withTrace>`.
- `addPoint`, `addM2M`, `addM2L`, `addL2L`, `addP2L`, and `evaluateM2P`.
- `component`, `setLaplacian`, and `laplacian`.

`<cosmictiger/gravity/cartesian/fmm.hpp>` supplies the vector-based runtime
API in `cosmictiger::gravity::cartesian`: `point`, `shiftMultipole`,
`shiftLocal`, `derivative`, `evaluate`, `addM2L`, `addM2LTargets`, `addDirect`,
and `Operator`/`getOperator`. Storage and translations accept order zero;
M2L and cached geometry accept orders 1 through 9. `getOperator` caches by order
and lattice offset using an HPX shared mutex. Construction happens outside the
lock; locked scopes only inspect or change the cache. Cache calls require an
active HPX runtime. Scalar numerical operators execute on the calling task.

`addM2LTargets(local, source, targets, order)` places bounded HPX tasks above
scalar M2L. Each `M2lTarget` supplies an immutable span of `M2lInteraction`
entries and its target scale. Each interaction supplies a source row index,
physical separation, source scale, and optional centered-source flag. The local
field must already have one correctly sized expansion per target; the source
and local fields must be distinct. Interaction storage, source rows, and local
rows must remain live without external mutation until the call finishes.

One task owns each target row and consumes its entire interaction list in the
supplied order, adding to existing coefficients and retaining its Laplacian.
Empty lists leave rows unchanged. A minimum chunk of 16 targets and a cap of
four chunks per worker bound scheduling; the cutoff is an unbenchmarked
heuristic. One-worker and non-HPX callers run synchronously. All launched tasks
finish before return or rethrow; errors can leave partial sums. Checks occur as
interactions are consumed, and scalar arithmetic guards end before task waits.

The CMake targets are `cosmictiger_stf` (header-only) and `cosmictiger_fmm`
(the runtime interface, linked with HPX). The application links the latter.
This import provides numerical operators; particle-tree traversal, interaction
lists, periodic/Ewald kernels, and a simulation gravity driver are separate work.

## Numerical conventions

The gravitational constant is one. There are `(order+1)^2` independent tensor
components. Rank `n` begins at `n*n`, storing `z=0` then `z=1`, with `x`
increasing within each plane. Other components are recovered by the trace
constraint. An optional last local scalar stores a constant background Laplacian.

Source moments are raw STF tensors divided by `sourceScale^rank`. Local entries
are derivatives of the negative Newtonian potential multiplied by
`targetScale^rank`; they do not include factorial normalization. Runtime
`evaluate` returns potential and derivatives with respect to normalized
coordinates. Physical acceleration is the negative gradient divided by the
local scale; `evaluateM2P` returns physical acceleration directly.

M2L retains `sourceRank + localRank <= order`; the highest source rank changes
only the constant local potential. M2P differentiates the complete retained
source potential, including its highest rank. A centered M2L explicitly drops
the dipole and is appropriate only when the source center is its center of mass.

The import uses `double` in place of the source's extended-precision constants
and oracle arithmetic. Generated rational constants are rounded to double.
Validation builds check sign/range contracts as values are consumed and use
local host `FpeGuard` scopes for arithmetic exceptions. Release omits these
checks; callers must supply valid, representable inputs. Validation builds
preserve IEEE trapping semantics even in optimized modes. Independent
numerical oracles also use `-fno-fast-math`.

## Generation and validation

The default 0..9 kernels are present in the source tree; building does not require Python or
clang-format. Regeneration requires Python 3.8+ and clang-format compatible with
the repository `.clang-format`:

```sh
python3 tools/generate_stf_m2l.py --check
python3 tools/generate_stf_m2l.py
```

The generator performs exact rational common-subexpression elimination, then
emits and formats C++ with the project names, notice, and class declarations.
`cosmictiger/gravity/stf/generated/operation_counts.json` retains the static arithmetic
report; it is not a measured performance benchmark.

Run the imported numerical checks and runtime-interface tests with:

```sh
cmake --build build --parallel 2
ctest --test-dir build -R 'CartesianFmm|StfM2l|cosmictiger.stf_' --parallel 2 --output-on-failure
```

The independent checks cover M2L coefficients and harmonic projection through
order nine, M2M re-expansion, local polynomial translations and traces, P2L/M2P,
analytic truncation remainders. Runtime tests cover every order, normalized scaling, direct
monopoles, compact leaf locals, input rejection, and concurrent HPX cache reuse.
Target-batch checks compare scalar and parallel coefficients bit for bit,
including cancellation-sensitive source order, traces, shared sources, empty
lists, alias rejection, and drainage after another target fails.

Validation after removing the angular momentum correction on 2026-10-04: the
Release FMM library and test targets built, and all 26 focused FMM checks
passed (24 GoogleTests and two standalone STF executables).
`python3 tools/generate_stf_m2l.py --check` confirmed the generated files are
current, and `git diff --check` passed.
