# CosmicTiger project handoff to Claude

Prepared for Dominic C. Marcello on October 14, 2026, America/Chicago.

CosmicTiger is a C++20 cosmological N-body code built around HPX tasks and
distributed components, MPI transport, particle-mesh multipole gravity
(PMMM), and optional CUDA particle work. Continue from the actual source tree
and preserve its numerical and restart contracts. The project has a running
evolution path, initialization through CAMB, periodic gravity, checkpoints,
diagnostics, and lightcone output; it is substantially beyond its original
scaffold.

Read this handoff and [AGENTS.md](../AGENTS.md) before implementation. The full
project standing orders are reproduced at the end for portability. Additional
user policies below apply even when the repository is moved to another host.
Current explicit user instructions take precedence over older documentation.

## Takeover instructions

Use the following as the initial task for Claude:

> Take over CosmicTiger from this checkout. Read CLAUDE.md, AGENTS.md, and
> docs/claude_handoff.md completely. Inspect Git status and the relevant source
> before editing. Preserve all existing work and scientific contracts. Report
> which revision and build configuration you are using. Distinguish source
> inspection, successful compilation, executed tests, and production validation.
> The built-in gravity.ghostWidth default is 1; preserve explicit width 2 and
> historical checkpoint behavior. Do not launch an unspecified production run
> or access QB4. Wait for a concrete implementation task after orientation.

## Repository and publication state

- Active checkout: `/home/dmarce1/workspace/cosmictiger`.
- Repository: [dmarce1/cosmictiger](https://github.com/dmarce1/cosmictiger).
- Branch during handoff preparation: `main`.
- Pre-handoff HEAD: `d279f9c4324e0805f1e087b61819255848d93ada`, the October 4
  scaffold commit, titled `Initialize C++20 HPX MPI CUDA and Silo project`.
- Before this handoff, 117 of the 119 files under `cosmictiger/` and `src/`
  existed locally but were untracked. There were also untracked tests, tools,
  configuration, and documentation. The files were present; a clone of the
  scaffold commit would not contain them.
- The user explicitly requested adding, committing, and pushing this project
  during the handoff. Use the commit containing this document and subsequent
  commits as the source baseline. Verify `git status`, `git log`, and the
  remote branch before assuming publication has completed.
- Build products, IDE caches, simulation output, and local profiling reports
  are excluded from the source commit. They are not deleted from this machine.

The CAMB gitlink points to `56a95f78fdd72709c5af6b668141ec0c617777c3`;
its nested `forutils` submodule is `6741ac4f49657a52f41d98d5500513b669162d46`.
Use the recorded gitlinks rather than updating dependencies to their newest
versions as part of routine setup.

For another machine, obtain the published source and initialize submodules:

```sh
git clone --recurse-submodules https://github.com/dmarce1/cosmictiger.git
cd cosmictiger
git status --short --branch
git submodule status --recursive
```

For an existing clone, review local changes before updating it. A plain Git
archive does not include submodule contents. Restarting an existing simulation
also requires the chosen manifest and all referenced checkpoint files; these
are separate from the source repository. Do not infer that a local file named
`simulation.restart` is the user's desired continuation point.

## Architecture and source map

The application starts in [src/main.cpp](../src/main.cpp). It parses application
options, starts HPX, initializes or restores particles on participating
localities, resolves the actual starting epoch and background, publishes
immutable global constants, and dispatches evolution or initial-condition
output. It drains launched tasks and propagates errors. Evolution writes a
manifest only after its final checkpoint files have been produced.

| Area | Principal files | Responsibility |
| --- | --- | --- |
| Entry point | `src/main.cpp`, `cosmictiger/startup.hpp` | Startup, locality actions, initialization or restart orchestration, output manifests |
| Configuration | `src/command_line_options.cpp`, `cosmictiger/global_constants.hpp`, `src/global_constants.cpp` | CLI/INI parsing, derived constants, immutable HPX publication and local copies |
| Background | `cosmictiger/friedmann.hpp`, `src/friedmann.cpp`, `cosmictiger/simulation_time.hpp`, `src/simulation_time.cpp`, `src/evolution/evolution_end.cpp` | Expansion/contraction, physical and conformal times, endpoint and drift integrals |
| Units | `cosmictiger/units.hpp` | Boost.Units scales and semantically distinct position, momentum, velocity, and time conversions |
| Einstein-Boltzmann interface | `cosmictiger/einstein_boltzmann.hpp`, `src/einstein_boltzmann.cpp`, `cosmictiger/camb.hpp`, `src/camb.cpp`, `cmake/camb.cmake` | Solver-independent spectrum interface and external CAMB process adapter |
| Initial conditions | `cosmictiger/particle/particle_initialization.hpp`, `src/particle/particle_initialization.cpp` | Gaussian realization, epoch search, Zel'dovich displacement and initial momentum conversion |
| Particle representation | `cosmictiger/fixed.hpp`, `cosmictiger/spatial_types.hpp`, `cosmictiger/particle/particle_storage.hpp` | Normalized positions, particle arrays, rungs/flags, optional gravity fields |
| Cartesian mesh | `cosmictiger/cartesian_grid.hpp`, `src/cartesian_grid.cpp`, `cosmictiger/box.hpp` | Interior ownership, periodic cell geometry, ghost ranges and cell multipoles |
| Redistribution and ghosts | `cosmictiger/particle/particle_load_balance.hpp`, `src/particle/particle_load_balance.cpp`, `cosmictiger/particle/particle_ghost_exchange.hpp`, `src/particle/particle_ghost_exchange.cpp` | Owned-particle redistribution, ghost packing/exchange and lifetime |
| FFT | `cosmictiger/fft/`, `src/fft/fft_hpx_exchange.cpp` | Radix/vector-radix transforms, packed layouts, communication plans, pencil/slab transforms, FFTW and HPX transport |
| STF and Cartesian operators | `cosmictiger/gravity/stf/`, `cosmictiger/gravity/cartesian/fmm.hpp`, `src/gravity/cartesian/fmm.cpp` | Generated compact tensor operators and runtime wrappers |
| Full symmetric tensors | `cosmictiger/gravity/symmetric/` | Trace-retaining operators needed for complete Plummer gravity |
| PMMM | `cosmictiger/gravity/pmmm/`, `src/gravity/pmmm/pmmm_runtime_convolution.cpp` | Periodic Green coefficients, near-region exclusion, coefficient contraction and runtime rank dispatch |
| Softening and references | `cosmictiger/gravity/ferrers.hpp`, `cosmictiger/gravity/plummer.hpp`, `cosmictiger/gravity/direct/`, `src/gravity/direct/` | Ferrers/Plummer kernels, direct periodic reference, sampled error comparisons |
| Cell-level FMM | `cosmictiger/gravity/fmm/`, `src/gravity/fmm/` | Batched CPU/CUDA M2L, P2L and M2P cell interactions ported from the old tree code; the CPU entries evaluate the tree near field; `cosmictiger/cuda/cuda_warp_partition.cuh` warp list partitioning ([fmm_interactions.md](fmm_interactions.md)) |
| Per-cell kd-trees | `cosmictiger/gravity/tree/`, `src/gravity/tree/` | Geometric x/y/z kd-tree rooted at each mesh cell, in-place position or rung-threshold ordering, center-of-mass, geometric, or old-code minimum-radius node centers ([kd_tree.md](kd_tree.md)) |
| Tree near field | `cosmictiger/gravity/tree/tree_near_field.hpp`, `src/gravity/tree/tree_near_field.cpp` | Optional (`gravity.nearField=tree`) symmetric local FMM walk over the kd-trees replacing near-region direct P2P on the CPU kick ([tree_near_field.md](tree_near_field.md)) |
| Kick | `cosmictiger/gravity/kick/`, `src/gravity/kick/` | Complete near/far field, timestep/rung choice, staggered half-kicks and force sums |
| Drift | `cosmictiger/drift/drift.hpp`, `src/drift/` | CPU/SIMD and CUDA particle position updates using cosmological drift integrals |
| Evolution and restart | `cosmictiger/evolution/`, `src/evolution/` | Integer rung clock, drift/redistribution/gravity sequence, diagnostics, snapshot interpolation and checkpoint metadata |
| CUDA interface | `cosmictiger/cuda/`, `src/cuda/` | HPX futures for runtime operations, allocation, copies, launches and completion; standalone memory pool; pool-backed vectors ported from the old tree code's `device_vector`: host-owned `ManagedVector` with kernel views (`cuda_managed_vector.hpp`) and block-cooperative `DeviceVector` (`cuda_device_vector.hpp`) |
| Silo | `cosmictiger/particle/particle_silo.hpp`, `src/particle/particle_silo.cpp` | Owned particle output, native restart arrays, physical/comoving views and visualization masters |
| Lightcones | `cosmictiger/lightcone/`, `src/lightcone/`, `cosmictiger/healpix/`, `src/healpix/`, `tools/render_lightcone.py` | Crossing detection, sparse HEALPix maps, CSV output and PNG rendering |
| Instrumentation | `cosmictiger/profiler.hpp`, `src/profiler.cpp`, `cosmictiger/flop_counter.hpp` | Statistical CPU sampler, metadata and isolated operation-count utility |
| Validation and concurrency | `cosmictiger/debug.hpp`, `cosmictiger/parallel.hpp`, `cosmictiger/lock_guards.hpp` | Numerical contracts, local FPE guards, bounded HPX tasks and permitted lock guards |

The production sequence is initialization or checkpoint restoration, immutable
constant publication, then repeated time advancement/drift, redistribution and
ghost exchange, cell multipoles, near/far gravity preparation, field evaluation
and kick, and diagnostics/output. The progress table names these phases
`D%`, `X%`, `M%`, `G%`, `K%`, and `O%`.

## Numerical contracts that must survive edits

1. The initial FFT and particle lattice have side `2^initialConditions.meshExponent`
   and exactly `2^(NDIM*meshExponent)` particles. Begin at cell centers, displace,
   and then bin by displaced positions. The evolution mesh is independently
   selected by `gravity.meshExponent`; repartition without changing physical
   particle state or count.
2. `NDIM` is the required dimension macro, currently 3. Particle storage remains
   normalized and kernel-oriented; physical output conversion must not alter
   checkpoint representations.
3. Conformal time is `tau = H0 integral(dt/a)`. Stored canonical momentum is
   `p = a d(x/L)/d tau`. It is not physical peculiar velocity. Initial-condition
   derivative data are converted once at handoff, as described in
   [gravity_kick.md](gravity_kick.md).
4. Rung zero spans the original nominal run duration. Integer ticks determine
   synchronization. Refine immediately; coarsen only when every crossed coarser
   boundary is synchronized. Regular kicks close the old half-step, choose the
   new rung, and open the new half-step as separate stored-momentum updates.
   Initial/final/diagnostic phases have distinct behavior.
5. Width 1 means a 27-cell near region; width 2 means 125 cells. The ghost region,
   direct interactions, and far-field exclusion must use the same selection.
   The executable default is now 1. Lower-level PMMM APIs still use the existing
   `kPmmmNeighborWidth=2` where callers omit their library-level argument; the
   application passes its runtime selection explicitly. Do not confuse these
   API defaults with the command-line default.
6. Runtime FMM order `p` is 2 through 9, default 6. Source rank is `p-1` and local
   rank is `p`; fixed storage capacities are 8 and 9. Preserve inactive-coefficient
   clearing and the saved runtime order on restart.
7. Ferrers is the default softening model, with kernel order 2 and cell fraction
   0.05. Plummer uses full symmetric tensors and its own periodic Green treatment;
   changing only direct P2P is insufficient. Never add the Marcello angular
   momentum correction.
8. Complete each owned sink's field before choosing a timestep, kicking, or
   accumulating its global force. Owned particle ranges determine uniqueness;
   ghost flags alone cannot identify all periodic copies.
9. Checkpoint schema 10 records the current evolution metadata, including the
   near-field solver and opening angle, which a restart may change. Older supported
   schemas are read deliberately; pre-version-7 checkpoints retain their
   historical shared mesh and width 2. Preserve staggered momenta, clock phase,
   original rung-zero duration, energy reference, and diagnostic history.
   Rung-zero extension uses an exact power-of-two remap.
10. Energies and output velocities require a synchronized epoch. Temporary
    synchronized snapshot momenta must not overwrite staggered restart momenta.
    Include the signed Layzer-Irvine work in the conserved energy diagnostic.
11. Output velocity is peculiar velocity only. Keep little-h conversions explicit
    when converting input Mpc/h to ordinary Mpc, and write CGS derived arrays in
    double precision. Never add the Hubble-flow velocity to these views.
12. CPU and CUDA must follow the same phase and state contracts. CUDA P2M uses
    cooperative warp work for crowded cells; CUDA kick combines L2P and near P2P
    in batches. FFT and M2L remain on CPU. Keep buffers alive through actual GPU
    completion futures.

Read [particle_initialization.md](particle_initialization.md),
[friedmann.md](friedmann.md), [particle_drift.md](particle_drift.md),
[gravity_kick.md](gravity_kick.md), [plummer_gravity.md](plummer_gravity.md),
[cartesian_fmm.md](cartesian_fmm.md), and [lightcone.md](lightcone.md) for equations
and implementation detail. Some documents contain dated development notes;
follow the current call path when a historical paragraph disagrees.

## Configuration and operation

Application options use `hpx::program_options`, camelCase components, and dotted
groups. Explicit CLI values override the application INI, which overrides
built-in defaults. `--configurationFile` is for application options. Pass HPX
settings separately, such as `--hpx:threads=2`; do not insert an `[hpx]` section
into the application INI.

| Setting | Executable default | Supplied default.ini |
| --- | --- | --- |
| `simulation.evolve` | true | true |
| `simulation.stopTime` | 0 years from expanding a=1 | 0 |
| `initialConditions.meshExponent` | 5 | 5 |
| `gravity.meshExponent` | follows initialization mesh when omitted | explicitly 5 |
| `gravity.ghostWidth` | 1 after this handoff's requested change | 1 |
| `gravity.fmmOrder` | 6 | 6 |
| `gravity.nearField` | direct | direct |
| `gravity.openingAngle` | 0.55 | 0.55 |
| `execution.cuda.drift`, `.kick`, `.multipole`, `.direct` | off | on |
| `gravity.direct.sinkFraction` | 0, disabled | 0 |
| `initialConditions.seed`, `gravity.direct.seed` | separate streams, each 42 | 42 |
| `output.snapshotCount` | 1 | 1 |
| `output.acceleration`, `output.potential`, `lightcone.enabled` | false | false |

`default.ini` is therefore a supplied run configuration, not an exact dump of
all built-in defaults. `test.ini` was byte-identical to it during inspection.
A CPU-only binary rejects any CUDA execution switch set to on. Even if direct
sampling is disabled, explicitly set `execution.cuda.direct=off` when using
this INI with a CPU-only build.

Invoking the executable without `--simulation.evolve=false` starts initialization
and evolution. For lightweight startup checks, disable evolution explicitly.
For initial-condition-only output, enable `initialConditions.generate`, supply
its output prefix, and disable evolution. Minimum gravity mesh exponent is 2
for width 1, or 3 for width 2. Initial-condition locality decomposition is cubic;
a two-process startup test is not evidence that a two-locality particle run is
a supported production topology.

The physical stop time may be past or future relative to expanding a=1 but must
follow the current state. `simulation.stopTime=end` is for a recollapsing model,
stopping at the positive `minimumScaleFactor` cutoff. Example physical models
are under `configs/evolution/`.

## Build environment

Use CMake 3.25 or newer, a GCC/Clang-compatible C++20 compiler, Ninja or Make,
MPI, HPX with networking and MPI parcelport, Boost headers, pkg-config, serial
FFTW3, Silo, GNU Make and gfortran for CAMB. CUDA builds additionally require
NVCC/CUDA with C++20 support and HPX built with CUDA enabled. Do not link
FFTW3-MPI or link CAMB's implementation into CosmicTiger.

CAMB is staged into each build's `_deps/camb-src`, built with a serial nested
Make invocation, and invoked as a separate executable. GoogleTest is found
locally or fetched at pinned revision `f8d7d77c06936315286eb55f8de22cd23c188571`.

| Preset | Main CPU flags | HPX APEX | Contracts |
| --- | --- | --- | --- |
| release | `-Ofast -march=native -g0 -DNDEBUG` | off | disabled |
| relwithdebinfo | `-Ofast -march=native -g -DNDEBUG` | on | enabled |
| debug | `-O0 -march=native -g3 -UNDEBUG` | on | enabled |
| profile | `-O2 -pg -g -march=native -DNDEBUG` | on | enabled |

CMake adds IEEE-preserving flags in validation builds and selected sensitive
targets. Preserve those overrides. Application flags cannot change the APEX
capability of an installed HPX library. Presets search
`~/local/hpx/Release`, `~/local/hpx/RelWithDebInfo`, `~/local/hpx/Debug`, and
`~/local/hpx/Profile`; override `HPX_DIR` or `CMAKE_PREFIX_PATH` on another host.
Configure separate build directories for each mode/backend.

`./build.sh` offers a terminal menu, initializes the pinned recursive CAMB
submodule, and builds the application. Its default disables tests; enable its
test option or pass `-DBUILD_TESTING=ON` explicitly. Test-job counts do not limit
threads created by individual HPX, MPI, CUDA, or CAMB processes.

For example, with the Release HPX installation present, a separate CPU build is:

```sh
git submodule update --init --recursive -- third_party/camb
cmake -S . -B build_claude_cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$HOME/local/hpx/Release" \
    -DCOSMICTIGER_ENABLE_CUDA=OFF \
    -DBUILD_TESTING=ON
cmake --build build_claude_cpu --parallel 4
ctest --test-dir build_claude_cpu -N
ctest --test-dir build_claude_cpu --output-on-failure --parallel 1
```

These parallel settings are examples for the reported War Thunder OFF state;
recheck the shared resource budget before running them. Full CTest includes
MPI processes and CPU/GPU-specific tests according to configuration. Optional
eight-locality tests require `COSMICTIGER_TEST_EIGHT_LOCALITIES=ON` and a separate
resource check.

To configure the standard CUDA Release build:

```sh
cmake --preset release -DCOSMICTIGER_ENABLE_CUDA=ON \
    -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=OFF
cmake --build --preset release --parallel 4
```

`CMAKE_CUDA_ARCHITECTURES=OFF` leaves architecture selection to the compiler.
Choose an explicit architecture when required by the destination GPU. Do not
assume an old cached architecture describes the current hardware.

## Verification and current evidence

At initial inspection, `release/` was configured as Release/CUDA ON with tests
OFF; `build-pmmm-cpu/` was Release/CUDA OFF with tests ON; `relwithdebinfo/` was
CUDA ON with tests OFF. The CPU test inventory listed 523 CTest entries. These
are existing caches and inventory counts, not evidence that every test passes
on the final source revision.

The handoff changes the application default in `GlobalConstantsData` from 2 to
1, updates the existing option-parsing expectations, and updates the standing
orders and default descriptions. Explicit width 2, library-level defaults, and
the pre-version-7 checkpoint fallback remain unchanged. C++ files changed for
this request are formatted with `clang-format --style=file`.

Validation completed for the default change on October 14, 2026:

| Check | Result |
| --- | --- |
| CPU Release build of `cosmictiger_global_constants_tests`, `cosmictiger_evolution_tests`, and `cosmictiger` | Passed, four build jobs |
| CUDA-enabled Release build of `cosmictiger` | Passed, four build jobs |
| `cosmictiger.evolution` | Passed, 115.85 seconds; includes checkpoint compatibility coverage |
| `cosmictiger.global_constants` | Passed, 3.59 seconds; includes new default and explicit width-2 precedence |
| `cosmictiger.startup` | Passed, 0.68 seconds |
| Two-process `cosmictiger.mpi_startup` | Passed, 1.26 seconds |
| Two-process `cosmictiger.global_constants_remote` | Passed, 4.53 seconds |
| Both rebuilt executables' `--help` | Reported `gravity.ghostWidth arg (=1)` |
| Changed C++ formatting and staged diff whitespace | Passed |

Reproduction commands used for the five CTest entries:

```sh
ctest --test-dir build-pmmm-cpu --output-on-failure --parallel 1 \
    -R '^cosmictiger\.(global_constants|evolution|startup)$'
ctest --test-dir build-pmmm-cpu --output-on-failure --parallel 1 \
    -R '^cosmictiger\.(mpi_startup|global_constants_remote)$'
```

The MPI checks ran on this workstation with access to local sockets, outside
the restricted execution sandbox. No QB4 access or GPU simulation was used.
CUDA application rebuilding is compilation evidence; CUDA evolution runtime
and the complete 523-entry CPU inventory were not rerun. Do not interpret
publication of the accumulated source as fresh validation of every subsystem
or a production benchmark.

Recommended checks for subsequent work:

- `cosmictiger.global_constants` for option defaults, CLI/INI precedence,
  immutable publication, CUDA switches, and restart option validation.
- `cosmictiger.evolution` for evolution and checkpoint behavior, including old
  schemas, saved mesh/width, clock continuation, diagnostics, and lightcones.
- `cosmictiger.startup`, `cosmictiger.mpi_startup`, and
  `cosmictiger.global_constants_remote` for startup/publication changes.
- `cosmictiger_tests` contains FFT, PMMM, direct interactions, Cartesian/STF
  operators, particle storage, redistribution, ghosts, and gravity tests.
  Several CUDA checks are in this executable rather than a single dedicated
  all-CUDA target.
- Separate targets cover units, Friedmann/time, initialization, Silo, HEALPix,
  lightcones, drift, numerical contracts, and the standalone CUDA memory pool.
- For numerical changes, use a validation-enabled build and compare appropriate
  CPU/CUDA and actual MPI cases. In-process logical localities do not establish
  physical multi-process correctness. A CUDA compile or skipped GPU test is not
  GPU runtime validation.

Initial inspection found old `LastTestsFailed.log` files in the local builds,
including initialization, MPI startup/global constants, and a runtime-order
PMMM case. The fresh results above supersede older records for the five named
CTest entries in the tested CPU configuration. Other old failures still need
their timestamps and subsequent partial runs reconciled. Keep precise command,
revision, backend, mode, and pass/fail/skip records for new validation.

## Generated operators and scientific documentation

The STF kernels are generated by `tools/generate_stf_m2l.py`, using helper
modules in `tools/stf/`. Generated headers and `operation_counts.json` live in
`cosmictiger/gravity/stf/generated/` and must be included in source transfers.
Python and clang-format are needed for regeneration, not ordinary compilation.

```sh
python3 tools/generate_stf_m2l.py --check
```

The generator reads the shared dimension definition and emits orders 0 through
9 by default. Change the generator and regenerate rather than patching emitted
arithmetic by hand. Preserve the imported Boost Software License in the STF
directory and other third-party notices alongside the project notice. The
source's coefficients, sign conventions, normalization and operator tests are
the reference for any proposed mathematical rewrite.

Do not confuse coefficient-space FFT M2L experiments with the spatial FFTs
required by PMMM. Historical documentation records removal of the former;
the latter are part of the running gravity path. Any proposed free-space
homogeneity recurrence needs independent justification before use with
periodic/Ewald Green coefficients.

## Known limitations and follow-up context

- Some README and audit passages record earlier stages when PMMM or gravity
  was only a library path. `src/main.cpp` and `src/evolution/evolution.cpp`
  currently connect initialization, PMMM and evolution. Trace active calls.
- `docs/thread_parallelism.md` is a substantial dated audit, not a current
  guarantee that every listed follow-up is still open or already validated.
- `.clangd` currently adds `-std=c++23`, while the build and standing rule use
  C++20. `.clang-format` retains historical settings and does not itself encode
  the complete standard/external/project include grouping. These are existing
  tooling inconsistencies; do not copy them as new coding conventions.
- Existing code has not received a fresh whole-repository conformance audit
  against all later standing rules. Correct affected violations within a
  requested change without turning unrelated work into a bulk rewrite.
- `FlopCounter` is a standalone utility; it is not integrated simulation-wide.
  The standalone CUDA memory pool likewise is not proof that the production
  particle path uses it.
- The statistical profiler samples CPU instruction addresses; it does not
  measure GPU time, waiting time, inclusive timings or invocation counts.
  It is enabled by default through `COSMICTIGER_ENABLE_PROFILING` and may write
  a report even for `--help`.
- Lightcone sparse maps contain raw counts and inverse-radius-squared weights,
  not a finished lensing observable or halo catalogue. Their geometry is the
  periodically replicated simulation volume.
- Full production accuracy, strong/weak scaling, and all combinations of
  CUDA/MPI/restart settings require separate evidence. Preserve all supported
  backends and do not infer application speedups from an isolated kernel.
- Older projects or branch lineages discussed in prior conversations are not
  proof of implemented hydrodynamics, a tree traversal, ROCKSTAR compatibility,
  or a neutrino particle module in this checkout. No such new implementation
  is authorized by this handoff alone.

## Working with Codex and Claude

The user asked whether algorithm development can remain in the Codex conversation
while Claude implements the code. This is feasible as a division of work. It
has not yet activated an automatic Claude connection or delegated a coding task.
No `claude` executable was found on the current shell PATH during preparation;
that does not establish whether another Claude app or installation exists.

Suggested responsibilities if the user adopts this workflow:

| Participant | Responsibility |
| --- | --- |
| User and Codex | Scientific goals, mathematical derivations, algorithm decisions, task scope, acceptance criteria and review |
| Claude | Implementation of the assigned design, focused tests, formatting, documentation and a concrete change report |
| Codex | Review equations and diffs, independently validate important claims, reconcile results and request corrections |
| User | Final project direction and any approval that the task actually requires |

Each implementation assignment should identify the base commit, permitted scope,
equations and invariants, inputs/outputs, numerical tolerances, CPU/CUDA/MPI
coverage, resource budget, expected artifacts, and unresolved decisions. Claude
should return a commit or patch, a concise explanation, exact test commands and
results, and remaining limitations. Do not replace missing numerical evidence
with assurances that the code looks correct.

Both agents can read the same directory. For simultaneous editing, use separate
Git worktrees and separate build/output directories; the worktrees share history
but isolate file edits. A single checkout also shares its Git index, so even
editing different files requires coordination around staging and commits.
The [official OpenAI worktree documentation](https://learn.chatgpt.com/docs/environments/git-worktrees)
describes the isolated-worktree model. Worktrees do not isolate machine resources:
the CPU/GPU budget remains shared. This handoff creates no additional worktree
and launches no Claude process.

## Additional standing user instructions

These policies were supplied directly by the user and must travel with the
project even where the repository's AGENTS.md omits the corresponding global
section.

**Email:** Never send email on the user's behalf, even if asked later. Never
reply, forward, schedule delivery, click Send, or use another account/service/
agent to work around this prohibition. Reading is allowed. Only an explicit
draft request permits creating or editing a Gmail draft; leave it unsent.

**QB4:** Access requires explicit user authorization with a maximum login budget.
Count every new login attempt, including failures, across all agents and tools.
Prefer one owner and a persistent SSH session; commands through that existing
session do not consume new attempts. Reserve each attempt atomically in
`/home/dmarce1/.codex/qb4-access-state.json` before connecting and record its
result. A later explicit user budget supersedes obsolete pause records; do not
add administrator/site-policy verification as another approval requirement.
Stop on a server login/rate warning or exhaustion, and report attempts used
against the authorized ceiling. Do not poll Slurm client commands in loops;
use allocated compute jobs for builds and simulations. No QB4 access is
authorized by this handoff, and zero login attempts were made while preparing it.

**Resources:** Latest explicit War Thunder report is OFF, October 14, 2026.
The newer OFF rule permits the available machine resources and supersedes the
older half-machine/12-core rule. ON means at most six aggregate local threads
across all launched or managed work, including MPI ranks and runtime/library
helpers, and combined CUDA use strictly below half of GPU cores. If the latest
status is unavailable, use the conservative six-thread rule. Never detect the
game automatically or modify unrelated user processes.

**Branches:** All new or renamed Git branches use snake_case, with no hyphen in
any component. A namespace such as `codex/` may be retained, for example
`codex/ghost_width_default`. Do not rename unrelated historical branches or
rewrite published history to enforce this convention.

**C++ includes:** Use three blocks in this order, separated by blank lines:
standard library, external/dependency headers, project headers. Omit empty
blocks and leave a blank line before the code that follows.

**Const:** Put `const` to the right of the qualified type, including `auto`,
template arguments, pointers, and references. Preserve pointee versus pointer
constness. Keep member-function const after the parameter list and retain
`QualifierAlignment: Right` in clang-format.

**Collaboration:** Keep changes scoped; respect requests for analysis only or
a plan before edits. Ground scientific claims in source and reproducible tests.
State exactly what ran, what only compiled, and what remains unverified. Preserve
dirty work and prior numerical defaults unless the task explicitly changes them.

## Complete project standing orders

The following is a verbatim snapshot of `AGENTS.md` after the requested ghost
width update. If this snapshot and a subsequently edited AGENTS.md diverge,
follow the latest explicit user instruction and the current project file.

# Project standing orders

This chat is used for project standing orders. Record new project conventions
in this file so they apply to future work in this repository.

## Maintaining standing orders

Whenever the user adds or clarifies a standing rule, record it here, correct
existing violations within the requested scope, and validate the changes. Once
that work is done, give the user a comprehensive list of all current standing
rules. Include any material validation limits or unfinished corrections.
For a War Thunder status update, record the new status and acknowledge it
briefly; do not repeat the full standing-rule list unless the user asks for it.

## FMM force correction

Do not use or generate the Marcello angular momentum correction in the FMM
kernels or expose it through the gravity APIs.

## Command-line options and global constants

Use `hpx::program_options` for application command-line options. Use `camelCase`
within option names and dots between groups, for example
`gravity.softening.kernelOrder`. Add an option only when an existing execution
path consumes it; do not expose placeholders with no effect.

Keep command-line options and any derived run-wide immutable values in the
globally accessible HPX `GlobalConstants` component. Only the root locality
creates and initializes the authoritative data. Each locality must have a
client with a local copy and a method returning a const reference to that
copy. Publish the data once during initialization and do not change it during
the run. HPX actions transfer values, not references to remote storage.

## Random seeds

Keep independent random seeds and random streams for each purpose. Zel'dovich
initial conditions use `initialConditions.seed`; direct-gravity sink sampling
uses `gravity.direct.seed`. New randomized features must receive their own seed
instead of reusing another feature's seed or generator state. Default every
random seed to `42`.

## Time integration and gravity diagnostics

Use the acceleration-softening timestep criterion with its accuracy parameter
available through command-line and INI options. Advance particles in conformal
time. Rung zero spans the nominal simulation duration; compute that duration
once during initialization and publish it in `GlobalConstants`. Initially set
particle rungs to zero. During the kick, move immediately to any required
shorter timestep. Permit a longer timestep only at synchronization with the
next coarser rung; every crossed rung boundary must satisfy that condition.

At each regular kick, first complete the previous timestep's half-kick using
the old rung, then compute and assign the new rung, then apply the next
timestep's half-kick using that new rung. Perform the half-kicks as separate
momentum updates. The first timestep skips completion of a previous half-kick
because the initial momenta are synchronized; it selects the first rung and
applies only the next half-kick. A final synchronization kick completes only
the previous half-kick, and diagnostic evaluations do not change momenta or
rungs. Apply the same sequence on CPU and CUDA and across restarts.

Specify the stopping epoch as physical years relative to the expanding
present-day a=1 epoch, with zero as the default. Permit future and past offsets
that follow the current simulation state. Evolve the signed background rate
through expansion, turnaround, and contraction. For a recollapsing cosmology,
allow an end-time target, stopping at a documented positive scale-factor cutoff
before the singularity and reporting the predicted end and remaining time.

Restarts preserve the original nominal rung-zero duration, clock phase,
staggered particle momenta, and diagnostic history. An explicit restart
extension may multiply rung zero by 2^n while adding n to every particle rung
and remapping the clock exactly, preserving existing particle timesteps.
Track global force, kinetic energy, gravitational potential energy, signed
Layzer-Irvine expansion work, and their conserved energy sum. Evaluate energies
at one synchronized epoch and count each owned particle once; retain the
original energy reference across restarts.

The kick must support global sums of particle forces and optionally potential.
Accumulate each owned particle's complete force once, after combining all
contributions. Do not retain acceleration or potential in particle storage by
default. Provide separate options to retain each field for SILO output, and
preserve enabled fields with the rest of the particle data.

Report initialization stages and every spectrum and density-scan evaluation
as they begin or complete, flushing terminal output before long work. During
evolution, print and flush one progress line after every completed rung
timestep, including fine rungs, and after any additional snapshot-only sample.
Print progress in aligned columns with labels above the first row and again
after every 25 rows, separated by a blank line. Include redshift `z = 1/a - 1`.
Report the distinct grid, FFT, ghost, multipole, gravity, diagnostics, and
first-snapshot stage, when present, between the evolution banner and first
progress row.
Show physical time in years relative to the expanding present-day epoch,
conformal time since the original handoff in H0^-1, scale factor, physical
da/dt in inverse years, and the highest current particle rung. Mark a progress
line with `*` only after its Silo snapshot has finished writing. Keep energy
and force diagnostics in the CSV.

Include the current step's wall-time percentages for six sequential phases:
`D%` for time advancement and drift, `X%` for redistribution and ghost exchange,
`M%` for cell multipoles, `G%` for overlapping near/far gravity preparation,
`K%` for force evaluation and kick, and `O%` for global diagnostics and output.
Use root-locality elapsed wall time, with overlapping work counted once and
progress-report overhead excluded. Reset the measurements for each
progress row, including initial and snapshot-only evaluations. Use compact labels
and precision to keep the table readable; exceeding 80 columns is allowed when
needed to show the required fields.

## Cosmological output frames

Use Boost.Units for physical unit definitions, run scales, and dimensional
conversion formulas. Keep comoving/proper positions, canonical momentum,
peculiar velocity, and physical/conformal times explicitly distinguished even
when their physical dimensions match. Preserve normalized fixed-point and
floating-point storage in numerical kernels and native checkpoints.

Provide cosmological comoving and physical CGS output views. In both views,
velocity means peculiar velocity: motion relative to the expanding cosmological
frame. Do not add the Hubble-flow term `H(a) * r` or include a Hubble-inclusive
lab velocity in these views. CGS velocity is the physical peculiar velocity in
cm/s. Keep canonical momentum explicitly distinguished from velocity.
Use ordinary Mpc and solar masses in the comoving output view, converting the
existing input and solver conventions with explicit little-h factors. Write
CGS derived arrays in double precision. Synchronize temporary snapshot momenta
to the output epoch without changing staggered restart momenta. Lightcone
conversions use each crossing's epoch.

## Silo particle output

Write only owned particles from the Cartesian grid's interior particle ranges;
never write ghost-cell copies, including periodic copies whose ghost flag is
unset. Save the particle rung value, without packed flags or a ghost field, in
both particle files and visualization masters. Restore particle flags from the
saved rung with ghost and reserved bits cleared. Keep older checkpoints with
packed flags readable, rejecting ghost particles or inconsistent rung values.

## Initial-condition and evolution grids

Keep the initialization FFT mesh matched to the particle lattice: its total
element count equals the particle count. `initialConditions.meshExponent`
controls their common side length, `2^meshExponent`. Start with exactly one
particle at each initialization cell center before Zel'dovich displacement.
After displacement, assign particles to the cells containing their positions.

Allow an independent evolution Cartesian/PMMM FFT mesh through the command-line
and INI option `gravity.meshExponent`; when omitted it matches the initialization
mesh. Repartition the existing particles onto this mesh without changing their
count, masses, positions, momenta, or rungs. Derive particle masses from the
particle count and cell-fraction softening from the evolution cell width.

Set the PMMM ghost-zone and direct-neighbor width at runtime with
`gravity.ghostWidth`, accepting one or two cells and defaulting to one. Apply the
same selected near region to CPU/CUDA direct work and the far-field exclusion.
Publish both mesh sizes and the width in immutable `GlobalConstants` and retain
them in checkpoints; restarts must preserve the saved choices.

## Einstein-Boltzmann solvers

Run Einstein-Boltzmann packages as separate executables through system calls
and file I/O; do not link their implementations into CosmicTiger. Keep the
input and matter-power-spectrum interface independent of the selected solver
so another package or a project solver can replace CAMB. CAMB is the default,
tracked as a Git submodule and downloaded and built by `build.sh`.

## Naming conventions

Apply these conventions throughout project code:

- Filenames: `snake_case` (for example, `particle_tree.cpp` and `particle_tree.hpp`).
- Namespace names: `snake_case`.
- Implementation helper namespaces: use `detail`, without a qualifying prefix
  such as `fixed_detail`.
- Type names: `PascalCase` (including classes, structs, enums, and type aliases).
- Variable names: `camelCase` (including parameters and constants).
- Class data members: `camelCase_`, with a trailing underscore.
- Struct data members: `camelCase`, without a trailing underscore.
- Function names: `camelCase` (including member functions).

Prefix `constexpr` variable and constant names with `k`, followed by the
capitalized camelCase name: `kCompileTimeConstant`. Private class constant data
members retain their trailing underscore: `kCompileTimeConstant_`. Preserve
language- and framework-required names and the explicitly required `NDIM` macro.

Define function- and method-scoped compile-time constants at the start of the
function or method body, in dependency order, and use their names. Prefer
`constexpr` whenever the initializer permits it. Name meaningful fixed values,
coefficients, sizes, masks, and tolerances rather than embedding them in later
expressions. Keep runtime-dependent values at the point where they become
available; do not change evaluation order or object lifetime when moving a
declaration. A constant that depends on a branch-local type or template
instantiation belongs at the start of that callable or valid scope.

These conventions supersede the earlier camelCase filename rule. When renaming
an existing file or identifier, update all includes, uses, and build references.
Preserve names required by external APIs, frameworks, or tools (for example,
`hpx_main` and `CMakeLists.txt`) and existing third-party names.

## File organization

When a filename family beginning with a shared `somename_` prefix would contain
more than one header or more than one source, put that family in corresponding
`somename` folders in the header and source trees. The threshold is two headers
or two sources; one header plus one source alone does not require folders.
Once the threshold is met, group both headers and sources, even if one side
contains only a single file. Keep filenames unchanged and use folder names
without a trailing underscore, as in `cosmictiger/fft` and `src/fft`.

Apply the same prefix grouping within the test and tool source trees, counting
them separately from implementation sources. Keep related umbrella files with
their family. Header-only and source-only families need folders only for the
files that exist. Existing matching family folders already satisfy the rule;
do not add another same-named level. Update includes, imports, generator output
paths, build references, and documentation together when moving files.

## FFT file organization

Keep all FFT implementation headers in `cosmictiger/fft` and all FFT
implementation sources in `src/fft`. Update includes, build references, and
documentation when moving FFT files.

## Dimension counts

Always use a named compile-time constant for dimension counts, even when an
algorithm only supports a fixed number of dimensions. Define spatial dimensions
with `#define NDIM 3` in `cosmictiger/dimensions.hpp`, and use `NDIM` in array
extents, loop bounds, dimension-dependent arithmetic, and related declarations
rather than literal dimension counts. For example, write
`std::array<Coordinate, NDIM>`. Use the macro rather than a `constexpr`
dimension variable. Code generators must read the same definition so regenerated
code retains this convention. `NDIM` is the explicitly requested exception to
camelCase constant naming.

## Class and struct declarations

Use `class` for types that encapsulate behavior. Keep class data members private,
and name them `camelCase_` with one trailing underscore, including static data
members. Methods may be public or private as needed and retain camelCase names
without a member-data suffix. Expose state through methods when access is needed;
use constexpr accessors for compile-time metadata.

Use `struct` only for simple aggregate types with public non-static data members
and no methods. Struct member names must be camelCase without trailing
underscores. Types with constructors, methods, static members, aliases, or nested
types use `class`. Preserve framework-required declarations and names. Apply
these rules throughout new and existing project code, including test helpers
and generated declarations; update all uses when renaming members.

## Template parameters

Use `typename` rather than `class` to declare explicit template type parameters.
Reserve `class` for class type declarations/definitions and scoped enums.
Place all type parameters, including constrained type parameters, before
non-type template parameters. For example, write
`template <typename Real, typename Local, typename Source, int sourceOrder>`.
Preserve type constraints. Update declarations, definitions, aliases,
specializations, explicit arguments, and generator templates together when
reordering parameters; do not invent a new value default to retain an old type
default.

## Code formatting

Follow the repository-root `.clang-format` for all project code we write or
modify. Use clang-format with `--style=file` on changed code before finishing.
Preserve the exact required copyright and license notice when formatting source
files. Keep `.clang-format` and `.clangd` under version control.

Place `const` after the type: `Coordinate const`, `Coordinate const&`, and
`Coordinate const*`. Apply this to declarations, template arguments, and casts,
including `auto const`. Preserve which object or pointee is const when moving
the qualifier; for example, a const pointer remains `Coordinate* const`.
Correct existing declarations when needed. `.clang-format` must retain this
qualifier placement.

Use unqualified `size_t` with an appropriate declaration header, such as
`<stddef.h>`.

Always enclose every `for`, `if`, `else`, `do`, and `while` body in braces,
including empty, single-statement, range-based, and nested bodies. Do this even
when the language permits omitting the braces. Keep automatic brace insertion
enabled in `.clang-format` so formatting retains and applies this convention.

Use postfix increments and decrements in `for` loops: `index++` and `index--`,
rather than `++index` and `--index`. Preserve behavior when an expression uses
the incremented or decremented value; rewrite the expression or loop as needed
instead of changing prefix to postfix blindly.

## Language and FFT dependencies

Use C++20 consistently for host and CUDA code. Use HPX with networking and its
MPI parcelport enabled; validate local and multi-process startup when changing
startup or communication behavior. Use serial FFTW3 for local transforms and
never link FFTW3-MPI.

## CUDA interfaces

Build HPX with `HPX_WITH_CUDA=ON` when project CUDA support is enabled. Route
host CUDA runtime calls and kernel submissions through the project's HPX
interfaces in `cosmictiger/cuda/cuda_runtime.hpp`, including tests. These
interfaces must return futures: queries and allocations return value futures,
and copies, releases, and kernel submissions return completion futures.
GPU completion futures become ready only after the submitted stream work
finishes, and failures propagate through the future. Keep raw CUDA runtime
calls inside the interface implementation; kernel launch callbacks enqueue
kernels on the supplied stream. Keep buffers and caller-owned streams alive
until their completion futures are ready. Perform
HPX scheduling and future operations outside application locks.

Provide independent `execution.cuda.drift`, `execution.cuda.kick`, and
`execution.cuda.multipole` command-line and INI options, each accepting `on` or
`off` and defaulting to `off`. The
selected backend must be used by the evolution path, including initial gravity
output when requested. Reject `on` in a build without CUDA rather than silently
falling back to CPU. A restart may choose any backend without changing its
saved physical state or clock.
With `execution.cuda.multipole=on`, construct the P2M cell multipoles on CUDA.
Use cooperative 32-lane GPU work for crowded cells while keeping a fast path
for sparsely occupied cells; do not serialize every crowded cell in one thread.
Keep FFT and M2L on the CPU. With `execution.cuda.kick=on`, compute L2P
and all direct near-cell P2P interactions on CUDA. Pass each mesh cell's
expansion, owned sink range, and neighbor source ranges to GPU work in batches;
do not launch a kernel separately for each cell-neighbor pair. In the kernel,
initialize each owned sink's field from its cell expansion, add the direct
interactions, then apply one kick and accumulate diagnostics. Do not stage
per-particle accelerations or potentials on the CPU for the handoff.
Provide a batched CUDA M2P evaluator for STF source multipoles at particle
targets, including both potential and acceleration, using the same generated
operator as the CPU evaluator. Keep source and output buffers alive until its
HPX completion future is ready.

## Powers and factorials

Evaluate powers and factorials with compile-time arguments in constant
expressions using `constexpr`. For runtime arguments, memoize computed values
and reuse them rather than recomputing them. Use compile-time tables when the
argument range is known at compile time, including tables indexed at runtime.
For powers of runtime particle coordinates or other changing values, keep the
memoized powers local to the evaluation and reuse them throughout that
evaluation. Apply the same rule to CPU, CUDA, and generated kernel code.

## Floating-point precision

Do not use `long double` without first justifying it to the user and obtaining
their explicit approval. Investigate alternatives before asking, such as
`double`, exact integer or scaled arithmetic, and algorithms that avoid needing
extra precision. Explain why suitable alternatives do not meet the actual
requirements and identify the proposed uses before requesting approval.
Existing uses of `long double` do not establish approval for future work.

## Prefer HPX facilities

Whenever HPX provides a version of a `std::` or `boost::` facility, use the
HPX version in project code, including tests. This applies to types, functions,
algorithms, synchronization primitives, and other utilities. Check the available
HPX APIs before introducing a standard-library or Boost facility, and replace
existing uses with the HPX version when modifying the affected code.

## Wrapped callable annotations

Use `hpx::annotated_function` for every project callable placed behind a task,
continuation, executor, algorithm or collective, parallel or traversal helper,
registered startup/shutdown callback, custom resource deleter, or type-erased
callable wrapper. This includes observers, operational callbacks, stored
function pointers, and host CUDA launch callbacks, in production code and tests.
Annotate at the binding site with a descriptive string-literal label whose
lifetime covers the callable. Preserve captures, reference ownership, moves,
results, exceptions, and empty-callable behavior.

The installed HPX annotation wrapper has a non-const call operator when thread
descriptions are enabled; do not require annotated objects to be const when
invoking them directly. Keep annotation invocation outside application locks,
as it can call HPX. Device kernel arithmetic remains device code, and framework
action registrations retain their action types and registered names.

## Numerical validation

Use `COSMICTIGER_VALIDATION_CHECKS` in every build mode except Release. This
macro covers particle bounds checks as well as numerical and range contracts;
it replaces the former separate particle bounds switch. Keep configuration
branches inside `cosmictiger/debug.hpp` wherever possible, so callers simply use
`expectPositive`, `expectNonNegative`, `expectNonZero`, `expectRange`,
`expectIndex`, and related expectation primitives. These checks become no-ops
in Release. Check bounds, ranges, signs, and invariants as values are consumed;
do not add separate validation passes over cells, particles, or numerical data.

Do not add finite-value classification checks such as `expectFinite`,
`isfinite`, `isnan`, or `isinf` for numerical validation. Use `FpeGuard` in local
host arithmetic scopes to trap invalid operations, division by zero, and
overflow. Never carry a guard across HPX scheduling or suspension: the floating
point environment belongs to an OS thread. Preserve IEEE trapping semantics
in validation-enabled builds. Already-existing quiet NaNs can propagate without
raising an exception; the guard diagnoses arithmetic exceptions, not all stored
nonfinite values. CUDA device code has no hardware floating point exception
flags or masks; the device-compatible guard reports this limitation through
`supportsHardwareTraps()`, while device expectation failures diagnose and trap.

Keep operational error handling (I/O and parsing failures, allocation/capacity
failures, CUDA status errors, communication failures, state transitions, and
algorithmic convergence or step rejection) active when required for correct
behavior in Release.

## Threading and synchronization

Do not use standard-library mutexes or threads (`std::mutex`, its variants,
`std::thread`, or `std::jthread`). Use the corresponding HPX synchronization
primitives and HPX tasks or threads instead, including in tests.

Include `cosmictiger/lock_guards.hpp` for lock guards rather than including
`<mutex>` or `<shared_mutex>` directly in project code. Use its `LockGuard`,
`UniqueLock`, and `SharedLock` aliases, which accept HPX mutex types only.

Never hold an application lock across an HPX call, directly or indirectly.
Restrict locked scopes to local state changes; perform HPX scheduling, future
operations, promise construction/completion, and destruction of HPX objects
outside those scopes. Retain ownership or extract container nodes when needed
so erasing an entry under a lock cannot invoke an HPX object's destructor.
Correct existing violations when encountered.

## CMake build modes

Provide separate Release, RelWithDebInfo, Debug, and Profile builds. Define
`NDEBUG` for every build mode except Debug.

| Mode | C++ compiler flags | HPX APEX integration |
| --- | --- | --- |
| Release | `-Ofast -march=native -g0`, define `NDEBUG` | Disabled |
| RelWithDebInfo | `-Ofast -march=native -g`, define `NDEBUG` | Enabled |
| Debug | `-O0 -march=native -g3`, leave `NDEBUG` undefined | Enabled |
| Profile | `-O2 -pg -g -march=native`, define `NDEBUG` | Enabled |

Use an HPX installation built with the required APEX setting for each mode;
changing an application flag cannot change an installed HPX library's APEX
support. Forward the CPU flags to NVCC's host compiler for CUDA sources.
Profile builds must apply `-pg` during both compilation and linking for gprof.

## Interactive builder

Allow parallel build and test jobs in `build.sh`. Set the job menu limits from
the CPUs available to the process without treating OpenMP thread settings as a
CPU limit. Pass the selected build and test job counts to CMake and CTest,
respectively, and keep their defaults within the available CPU count. Follow
the current aggregate local compute resource budget when running builds and
tests. Present the settings in a full-screen, boxed terminal menu when
`whiptail` is available, with a plain-text fallback otherwise.

## Local compute resource limits

Use the user's explicit War Thunder status reports in this standing-orders chat.
The current reported status is **OFF** (2026-10-14). Update this status when the
user reports a change. Do not use process checks or other automatic detection
to infer whether War Thunder is running. If the latest reported status is
unavailable, use the conservative 6-thread aggregate limit until it is known.

When the reported status is OFF, local builds, tests, benchmarks, and other
compute jobs may use the entire machine, including all available CPU cores and
the GPU. This supersedes the earlier half-machine/12-core local limit. Never
exceed actual system limits; the available resources are shared across all
concurrent tasks and agents. The separate QB4 access pause remains in effect.

While the reported status is ON, limit TOTAL local thread usage for everything
we launch or manage to 6 threads combined across all concurrent work, tasks,
chats, and agents. This is one aggregate thread budget, not a per-job limit or
merely a six-core CPU-usage limit. Count runtime, library, and background helper
threads as well as worker threads. Apply this limit immediately when the user
reports that War Thunder is ON, including in menus or when minimized.

Across all conversations and agents, CUDA work is permitted while the reported
status is ON only when combined CUDA use remains strictly below half of the
available GPU cores. Bound concurrent launches to respect this shared limit.

Account for build processes, concurrent tests, HPX workers, all threads in
every MPI rank, nested parallelism, GPU runtime helpers, and other local work
together. While the reported status is ON, determine the threads already used
by our existing workloads before starting work and allocate only the remaining
shared budget. Setting `-j6` or six HPX workers does not by itself satisfy the
total limit. If total thread usage cannot be established, do not launch
additional parallel work. During sustained work, respond to new user status
reports; when the status changes to ON, reduce or pause our workloads to meet
the shared six-thread budget. Stop or pause a job that cannot safely reduce its
thread count while running.
Do not change the game's resources or unrelated user processes. These checks
apply while agents are working; this file does not install a background monitor
or an operating-system CPU quota.

## Source file copyright and license notice

Place the following exact notice at the head of every project source file we
create or modify, including C++ headers, C++ sources, CUDA sources, and tests.
Preserve existing third-party copyright and license notices in vendored code.
For source languages without C-style block comments, retain the exact notice
text using that language's comment syntax (after any required shebang).

```c
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
```

<!-- BEGIN USER BRANCH NAMING POLICY -->
## Permanent branch naming rule

User instruction, 2026-10-14: All Git branch names must use snake_case.
Never use a hyphen (`-`) in a branch name, in any repository, project, chat,
agent, automation, or delegated task. Use underscores between words.
If a required namespace prefix is used (for example `codex/`), every component
must still be free of hyphens: `codex/wdeos_sound_speed` is valid.
This rule applies to all branches we create or rename and supersedes prior
hyphenated naming conventions. Existing historical names are not permission
to create new hyphenated branches. Do not rewrite published commit history
or rename unrelated existing branches merely to enforce this rule.
<!-- END USER BRANCH NAMING POLICY -->
