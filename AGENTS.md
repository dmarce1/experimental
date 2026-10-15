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
