# CosmicTiger

C++20 project using HPX, MPI, CUDA, Silo, and GoogleTest.

Project layout:

- `src/`: C++ source files.
- `cosmictiger/`: public C++ headers, included as `<cosmictiger/name.hpp>`.
- `tests/`: GoogleTest C++ source files.

See the [thread parallelism audit](docs/thread_parallelism.md) for current HPX
scheduling, synchronization boundaries, and opportunities for additional parallel work.
Pencil/slab FFTs accept the transport directly to overlap up to four owned-buffer
`FftHpxExchange::exchangeAsync` transfers and scatter ready blocks.

## Build and run

Run `./build.sh` in a terminal for an interactive builder. It uses a boxed
terminal menu when `whiptail` is available and a plain-text menu otherwise.
Use the displayed defaults (Release, CUDA enabled, four parallel build jobs),
or select an option to change it. Select "Configure and build" to start.
Before configuring it downloads the pinned CAMB submodule and its nested
forutils dependency. By default it configures without tests and builds the
`cosmictiger` target, its C++ adapter, and the separate native CAMB executable.
The test-suite option also builds the test targets and runs CTest. It leaves the
optional eight-locality MPI tests disabled. The builder shows CMake output as it
runs, saves a log in the selected preset build directory, and reports the stage
that failed with a nonzero exit status. Set parallel jobs according to the
current shared local resource budget; CTest jobs do not limit the worker count
inside an individual MPI test.

The CUDA architecture menu uses `CUDAARCHS` when set, or NVCC's compiler default
otherwise. The compiler default is passed to CMake as
`CMAKE_CUDA_ARCHITECTURES=OFF`, which omits architecture flags and also works when
reconfiguring an existing build directory. Select an explicit architecture list
in the menu to override it.

Requires CMake 3.25+, Git, GNU Make, gfortran, a C++20 compiler, MPI, HPX built
with networking and its MPI parcelport enabled, pkg-config, Silo, and the CUDA toolkit.
The `release`, `relwithdebinfo`, `debug`, and `profile` CMake presets create separate build
directories with these settings:

| Preset | C++ flags | HPX APEX |
| --- | --- | --- |
| `release` | `-Ofast -march=native -g0 -DNDEBUG` | OFF |
| `relwithdebinfo` | `-Ofast -march=native -g -DNDEBUG` | ON |
| `debug` | `-O0 -march=native -g3 -UNDEBUG` | ON |
| `profile` | `-O2 -pg -g -march=native -DNDEBUG` | ON |

The presets look for HPX in `~/local/hpx/Release`, `~/local/hpx/RelWithDebInfo`,
`~/local/hpx/Debug`, and `~/local/hpx/Profile`, respectively. Override `CMAKE_PREFIX_PATH` or `HPX_DIR`
at configure time if your installations are elsewhere. HPX itself must have
been built with `HPX_WITH_APEX=OFF` for Release and `HPX_WITH_APEX=ON` for the
other modes; CMake rejects a mismatched package. Use a single-configuration
generator such as Ninja or Unix Makefiles.
CUDA sources use NVCC optimization flags and forward the listed CPU flags to
the host compiler.
GoogleTest is found locally or downloaded at a pinned revision during configure.

Profile uses `-pg` during both compilation and linking, as required by the
[GNU gprof documentation](https://sourceware.org/binutils/docs/gprof/Compiling.html).
`-O2` supplies optimization and `-g` retains debug information. `NDEBUG` is
defined in every mode except Debug. To collect a profile, configure and build
with `cmake --preset profile` and `cmake --build --preset profile`, run
`./profile/cosmictiger --hpx:threads=4`, then use
`gprof ./profile/cosmictiger gmon.out`. Libraries compiled without `-pg` have
incomplete call graph information; APEX provides HPX task profiling.

Statistical CPU sampling is enabled by default in every build mode through
`COSMICTIGER_ENABLE_PROFILING`. The executable starts the sampler before HPX
creates its OS workers. After each successfully published particle Silo file
(initial conditions or evolution checkpoint), it prints the accumulated report
and updates `cosmictiger.profile.<pid>.txt` in the working directory. Sampling
continues without resetting the counters. Each MPI process produces its own
report; visualization master files do not trigger a duplicate report. On normal
process exit, including handled startup errors and `--help`, it prints and saves
the final report. Sampling currently requires Linux x86-64 or AArch64;
a startup failure is reported explicitly and stops the application.
Configure with `-DCOSMICTIGER_ENABLE_PROFILING=OFF` to define `PROFILE_DISABLE`,
make `PROFILE()` a no-op, and skip automatic sampler startup.

The sampler interrupts each attached OS worker every 10 milliseconds of its
CPU time and records the interrupted instruction address. It adds no sampler
thread and needs no pause/continue markers around HPX calls or worker migration.
The report groups addresses by their actual ELF function symbols and gives
sample counts, estimated exclusive CPU seconds, and CPU percentages. Inlined
work belongs to the enclosing compiled function. Parallel CPU totals can exceed
elapsed wall time; invocation counts, inclusive time, waiting time, and GPU
execution time are not measured. Missing symbols remain unresolved addresses,
and the report lists dropped samples, timer overruns, and setup failures.

Include `<cosmictiger/profiler.hpp>` and place `PROFILE()` in runtime host
functions to register optional signature metadata once. Registration does not
time a scope or override the interrupted address's canonical symbol. The
C++20 registration macro skips constant evaluation in `constexpr` and
`consteval` functions; CUDA device compilation makes it a no-op. Embedders can call
`profile::profileStart(intervalNanoseconds, filename)` before `hpx::init`,
`profile::profileStop()`, `profile::profileReport()`, and
`profile::writeProfileReport(filename)` directly. Starting after the runtime
already exists covers the caller and future OS workers; it does not enumerate
workers that were created earlier. A stopped sampler cannot be restarted in
the same process.

`cosmictiger/flop_counter.hpp` provides the header-only `cosmictiger::FlopCounter<Count, enabled>`
utility for a local CPU worker or CUDA thread. `FlopCounter<>` starts at zero and
stores an exact 64-bit unsigned operation count; assignment, `+=`, `-=`, `reset()`,
`read()`, and conversion to the count type are available on host and device.
`FlopCounter<std::uint64_t, false>` discards updates, has no count storage, and
always reads zero. Validation builds check nonnegative inputs, narrowing,
overflow, and underflow before changing the count. Valid operands and
representable results are caller requirements in Release. The utility is not yet
used by the simulation; it provides neither shared accumulation nor timing.

```sh
git submodule update --init --recursive -- third_party/camb
cmake --preset release
cmake --build --preset release
ctest --preset release
./release/cosmictiger --hpx:threads=4
mpiexec -n 2 ./release/cosmictiger --hpx:threads=2 --hpx:bind=none \
    --hpx:ini=hpx.parcel.mpi.enable=1
```

The executable now generates the linear-density handoff realization and
evolves it by default. `--simulation.evolve=false` retains startup-only or
explicit initial-condition-only operation. Use `mpiexec` for multiple MPI
localities and choose worker counts that respect the aggregate resource limits.

`--configurationFile` reads CosmicTiger application options only. Pass HPX
runtime settings separately, for example `--hpx:threads=4` and
`--hpx:ini=hpx.threadpools.io_pool_size=1`; do not put `[hpx]` or `[hpx.ini]`
sections in the application INI. Use the explicit `--hpx:threads=N` spelling
because the installed HPX build does not apply the `-tN` shorthand.

### Initial-condition command line

The implemented application action generates a density-threshold-selected
Zel'dovich particle realization with CAMB and writes one Silo file per locality:

```sh
./release/cosmictiger --initialConditions.generate --simulation.evolve=false \
    --initialConditions.outputPrefix=initial_conditions \
    --initialConditions.meshExponent=5 \
    --initialConditions.boxLength=100 --initialConditions.seed=42 \
    --hpx:threads=1 --hpx:bind=none
```

Run `./build.sh` first and select Release. It initializes the pinned CAMB
submodule and builds CAMB's separate executable along with CosmicTiger.
The command above uses the fiducial defaults: flat ΛCDM with
`H0=67.4 km/s/Mpc`, `Omega_b h^2=0.0224`, `Omega_c h^2=0.120`,
`Omega_nu h^2=0.00064`, `A_s=2.1e-9`, `n_s=0.965`, and scalar pivot
`0.05 Mpc^-1`. These are rounded
[Planck 2018-like parameters](https://doi.org/10.1051/0004-6361/201833910).
There is one massive neutrino species, two massless species, and total
`N_eff=3.046`. The particles represent combined CDM+baryons. The default
realization has 32³ particles in a 100 Mpc/h box, with the same 32³ Cartesian
cell grid. Each cell starts with one particle at its center before displacement.
The start epoch is selected when the resolved maximum linear density contrast
first reaches 0.1; the executable prints its scale factor, redshift, equal
particle mass, and number of spectrum evaluations.

The files are named `initial_conditions.locality0.silo`,
`initial_conditions.locality1.silo`, and so on. The parent directory of the
prefix must exist; an existing file of the same name is atomically replaced.
Each file contains the particle point mesh, velocities, and rung fields, plus
an `initial_conditions` directory with the cosmology, scale factor, box size,
particle mass, seed, mesh exponent, selected matter component, and density
threshold. Positions are in the periodic unit box; velocities are
`a d(x/L)/d tau`, where `tau = H0 integral(dt/a)` is conformal time in
Hubble-time units. The initializer API returns `d(x/L)/d ln(a)`; executable
startup converts it once before output. The file records these conventions
and the mass/length units.
`readParticleSilo()` restores particles, and
`readInitialConditionsSiloMetadata()` reads and validates the physical metadata.
Particle output contains only owned particles from interior grid cells, with
their rung values; ghost-cell copies and packed flags are omitted. Restarts
restore rungs with ghost and reserved bits cleared. Older checkpoints containing
packed flags remain readable.
`--help` lists the application options by group, with descriptions and defaults,
and exits before starting HPX or a simulation. Use `--hpx:help` for HPX runtime
options. Option names use dots between groups and camelCase within each group. The
`cosmology.*` options set the existing CAMB inputs in
`CosmologyParameters`, with `hubble` in km/s/Mpc and density parameters
named `omega...H2` in physical-density units. The `initialConditions.*`
options select the particle-lattice and initialization FFT exponent, comoving box length in Mpc/h,
random seed, matter component, maximum linear density deviation, scale-factor
search interval, and search/spectrum accuracy controls. The starting
redshift and equal particle mass are derived by the initializer. The
initial-condition action requires a power-of-eight locality count no greater
than the FFT mesh side length, including one locality; a normal two-locality
startup remains supported. CAMB supplies the **linear matter power spectrum
P(k)** to the solver-neutral `EinsteinBoltzmannSolver` interface through files
and a separate process. Its Fortran code is not linked into CosmicTiger.

`initialConditions.meshExponent=m` sets `2^(NDIM*m)` particles and an
initialization FFT with `2^m` elements per axis. `gravity.meshExponent=g`
independently sets the evolution Cartesian grid and gravity FFT to `2^g`
cells per axis. If omitted, the gravity exponent equals the initialization
exponent. The initializer still creates one particle per initialization cell
before displacement; particles are redistributed onto the selected gravity
grid before evolution or requested initial gravity output.

`gravity.ghostWidth` selects one or two ghost-cell layers and defaults to
`1`. The same value selects the direct-neighbor region and the matching
exclusion in the periodic Green function: width one covers `3^3 = 27` cells,
and width two covers `5^3 = 125` cells. Both choices are available in one build;
changing the command-line or INI value requires no recompilation. Gravity
requires a mesh side greater than twice this width, so width two needs
`gravity.meshExponent >= 3`, while width one permits exponent two. Both meshes
must support the selected locality decomposition.

For example, this uses 64³ particles and initialization FFT elements, a 32³
evolution mesh, and one ghost-cell layer:

```sh
./release/cosmictiger --initialConditions.meshExponent=6 \
    --gravity.meshExponent=5 --gravity.ghostWidth=1 \
    --output.checkpointPrefix=run --hpx:threads=1 --hpx:bind=none
```

Random seeds are independent for each purpose and default to `42`.
`initialConditions.seed` controls the Zel'dovich Gaussian realization;
`gravity.direct.seed` controls only the direct-gravity sink sample. Changing
the comparison seed does not alter the initial conditions, and changing the
initial-condition seed does not change the sampling generator's state.

The root locality creates one immutable HPX `GlobalConstants` component from
the parsed options and derived mesh/count values. A client on every locality
fetches a local copy once and returns a `const` reference for later reads.
The evolution path consumes the gravity, timestep, output, and restart controls.

`gravity.softening.model` selects `ferrers` (default) or `plummer`.
`gravity.softening.cellFraction` sets the Ferrers support radius or Plummer
epsilon in evolution-cell widths. Plummer selects full Cartesian multipoles
and softened periodic Ewald filters; its traces and noncompact tails are
retained throughout the force calculation. See the
[Plummer gravity guide](docs/plummer_gravity.md) for the operator API,
normalization, periodic potential convention, and restart behavior.

Select the CUDA execution stages independently with
`--execution.cuda.drift=on`, `--execution.cuda.kick=on`, and
`--execution.cuda.multipole=on`, and `--execution.cuda.direct=on` (or set
`drift = on`, `kick = on`, `multipole = on`, and `direct = on` under
`[execution.cuda]` in an INI file). All default to `off`.
The multipole switch moves cell P2M construction to CUDA. P2M uses one thread
for sparse cells and a 32-lane warp for crowded cells. The CUDA kick evaluates
each cell's L2P field,
adds all near-cell P2P interactions, then updates particle momentum, rung,
optional output fields, and diagnostics on the GPU. FFT and M2L stay on the
CPU; each mesh cell's expansion, sink range, and neighboring source ranges are
passed to a batched CUDA launch. A CUDA-enabled build and an
available GPU are required when a switch is on. On restart, the switches
supplied for the new invocation select its execution backends.
The separate `stf::launchM2pCuda` and `symmetric::launchM2pCuda` APIs evaluate
particle fields from STF and full Cartesian source multipoles in batches.
The current PMMM evolution path has no M2P stage.

### Direct periodic gravity comparison

Set `--gravity.direct.sinkFraction=f` to compare PMMM with a direct periodic
reference solve at the initial or restart epoch. The fraction must lie in
`[0,1]`; `0` disables the comparison, and `1` selects all particles as sinks.
A positive fraction selects `ceil(f*N)` distinct random sinks from the `N`
owned particles. Every owned particle participates as a source, including
sources on other localities; ghost copies are excluded. The direct reference
includes the Ewald real-space and reciprocal-space contributions, the uniform
background, and the same Ferrers or Plummer model as PMMM, including
Plummer's changes to the periodic terms. The
`execution.cuda.direct` switch selects its CPU or CUDA implementation
independently of the PMMM kick backend. This comparison also works with
initial-condition-only output (`initialConditions.generate` together with
`simulation.evolve=false`) and does not require retaining acceleration or
potential fields in particle storage after the comparison. The diagnostic temporarily
stores PMMM fields and gathers all owned source positions on each locality;
only the selected sinks are solved, on their owning localities. A fixed seed
reproduces the sample for a fixed locality/cell/particle ordering.

```sh
./release/cosmictiger --gravity.direct.sinkFraction=0.01 \
    --gravity.direct.seed=42 --initialConditions.seed=42 \
    --execution.cuda.direct=on --hpx:threads=1 --hpx:bind=none
```

The root locality prints acceleration and potential errors using the same
sample for both solutions. For acceleration, let `e_i` be the magnitude of
`a_PMMM(i)-a_direct(i)` and `r_i` the magnitude of `a_direct(i)`. The reported
normalized errors are `L1 = sum(e_i)/sum(r_i)`,
`L2 = sqrt(sum(e_i^2)/sum(r_i^2))`, and
`Linf = max(e_i)/max(r_i)`. Potential uses the corresponding scalar absolute
values in the shared periodic-potential convention. These aggregate ratios
avoid dividing by individual nearly zero particle accelerations. Raw error
norms are also reported; a zero reference norm gives zero relative error when
the corresponding error norm is zero and infinity otherwise. The reference
work grows with the number of selected sinks times the total source count.
The absolute L1 and L2 columns are mean and RMS errors respectively; reference
norms are printed alongside them.

The callable interfaces are `directSolveCpu`, `launchDirectSolveCuda`,
`sampleDirectSinks`, and `compareDirectFields` in `cosmictiger/gravity/direct`.
The independent double-precision Ewald reference uses a unit periodic box,
`alpha=2`, and real/reciprocal cutoffs of four by default; the low-level API
accepts alternative Ewald parameters for convergence checks. This follows the
periodic summation convention in [Nitadori, Appendix B](https://arxiv.org/pdf/1409.5981),
with the gravitational sign and the existing PMMM potential zero point.

### Evolution, physical stopping times, and restart

`simulation.stopTime` is a signed number of **years from the present**, defined
by the expanding background's `a=1` epoch. Its default `0` stops at the present;
`13.8e9` stops 13.8 billion years later. Negative values stop in the past, but
must still follow the selected initial epoch or the checkpoint being resumed.
This replaces `simulation.finalScaleFactor`. The background advances both its
scale factor and signed derivative through a turnaround, while proper and
conformal time continue forward.

```sh
./release/cosmictiger --simulation.stopTime=0 \
    --initialConditions.meshExponent=5 --output.checkpointPrefix=run \
    --hpx:threads=1 --hpx:bind=none

./release/cosmictiger --restart.manifest=run.restart \
    --simulation.stopTime=13.8e9 --output.checkpointPrefix=continued \
    --hpx:threads=1 --hpx:bind=none
```

For a recollapsing background, `--simulation.stopTime=end` computes the future
Big Crunch time. It stops on the contracting branch at
`simulation.minimumScaleFactor` (default `1e-6`), reports the predicted end and
remaining proper time, and saves a finite state. The singularity at `a=0` is
never integrated. A background without a future finite simple turnaround is
rejected for this stopping mode. An unreachable numeric time or unresolved
singular boundary is an operational error, rather than a successful completion.

The PMMM evolution driver performs conformal block-step kicks, drifts every
owned particle, redistributes cells/ownership, refreshes ghosts, and combines
near and periodic far gravity. Rung zero initially spans the nominal conformal
duration. Checkpoints preserve that interval, the original clock origin, the
current epoch/tick and partial drift, signed background state, and energy
history. Restarting with a new endpoint leaves the timestep lattice unchanged.
`--restart.rungZeroExtension=n` multiplies the saved rung-zero interval by `2^n`
and adds `n` to every particle's rung, preserving each existing physical step
and its phase. Longer steps subsequently become available only at legal rung
synchronizations. Out-of-range or inexact remappings are rejected.

The first gravity evaluation prints and flushes detailed initialization progress:
direct neighbor gravity, Ewald filter construction and cached filter FFTs,
multipole FFTs, Fourier contractions and inverse transforms, periodic background
corrections, particle fields and kicks, rung sorting, and global diagnostic sums.
Long CPU Ewald and particle calculations report completed local work about every
two seconds; FFT batches report their completed counts. CPU direct work overlaps
the periodic solve, so its messages can appear between periodic-gravity messages.
With CUDA kicks enabled, range preparation overlaps the periodic solve, followed
by the combined GPU field evaluation and kick. Only the root locality prints;
counts describe its local work, and the diagnostic reduction waits for every
locality. Later timesteps retain the usual concise evolution progress rows.

`output.snapshotCount` selects the number of Silo snapshots (default one, at
the final time). Counts above one include the initial and final states, with
equal conformal-time spacing between them. Output between rung events records
a partial drift without inserting a kick. Each snapshot is a full checkpoint;
files are named `<prefix>.<generation>.snapshot0000.locality0.silo`, and so on.
A single-locality run with `output.snapshotCount=100` writes exactly 100 Silo
files, plus a diagnostics CSV. Native Silo mesh time records conformal time
since the original handoff, in `H0^-1` units; readable evolution datasets also
record the current scale factor, signed derivative, and years from present.
Open `<prefix>.<generation>.visit` in VisIt to view the generation as a time
series. CosmicTiger updates this list after every complete snapshot. For a
multi-locality run, it first writes a Silo master file that combines all
locality particle meshes at that time, then adds the master to the series.
Use VisIt's new-states refresh while a run is active to load snapshots
published after the database was opened. Initial-condition output has its own
`<initialConditions.outputPrefix>.visit` and, when needed, a Silo master.
The `<prefix>.restart` manifest is atomically published only after every
locality file succeeds; a failed new generation leaves the old manifest intact.
Restart requires the same locality decomposition and particle representation.
The saved cosmology, initialization lattice, gravity mesh, ghost width,
softening, timestep accuracy, and optional particle-field choices remain
authoritative. Explicit restart `gravity.meshExponent` or `gravity.ghostWidth`
values must match the checkpoint. Older checkpoints retain their original
shared mesh and use the previous default width of two. Endpoint, output prefix, snapshot count, and requested dyadic
extension are the restart controls. The full evolution checkpoint is needed:
an initial-condition Silo file by itself has no integration clock.

Particle momenta in evolution checkpoints retain their rung-dependent
staggering, including when stopping between scheduled events. Temporary
synchronized momenta are used for diagnostics and physical output; no extra kick is inserted by a
stop or restart. Gravity fields remain optional in particle storage, while
whole-system force and energy sums are always computed. The CSV contains
physical years from present, `a`, signed `da/dt` in inverse years, kinetic
energy `K`, peculiar gravitational energy `U`, signed expansion work `W`,
`K+U+W`, drift from its original value, force components, and the finest rung.
In code units with `q=x/L`, `tau=H0 integral(dt/a)`, and `p=a dq/dtau`,
`K=m sum(|p_sync|^2)/(2a^2)`, `U=m sum(phi0)/(2a)`, and the force sum is
`m sum(g0)/a^2`. Ghosts are excluded. Energies use mass unit `H0^2 L^3/G`
and velocity unit `H0 L`.

Every cosmological particle file also contains `comoving/particles` and
`cgs/particles` visualization meshes. Both views include mass, rung and peculiar
velocity; enabled acceleration and potential fields are converted as well.
The comoving mesh uses ordinary Mpc and solar masses, with peculiar velocity in
km/s. Its acceleration and potential retain the cosmological scalings
`a^2 g_pec` in `(km/s)^2/Mpc` and `a phi_pec` in `(km/s)^2`. The CGS mesh uses
proper cm, grams, peculiar cm/s, peculiar gravitational cm/s² and erg/g.
Neither view adds Hubble flow. Axis and variable units are stored in Silo, and
`output_units` records the scale factors and conventions. The converted arrays
are double precision; original native arrays and restart clocks are unchanged.
Evolution snapshots capture synchronized momentum during the existing CPU or
CUDA kick, before either integration half-kick, and permute it with the particle
data. Direct checkpoint API calls without synchronized output buffers omit the
derived velocity rather than mislabel staggered momentum. All other derived
fields remain available. Multilocality masters expose both converted views.

The diagnostics CSV retains its original columns and appends physical seconds,
energies in `Msun (km/s)^2` and ergs, and force components in
`Msun (km/s)^2/Mpc` and dynes. These are physical peculiar diagnostics; their
existing scale-factor dependence is not applied a second time. The conversion
layer in `<cosmictiger/units.hpp>` uses Boost.Units dimensional arithmetic and
distinct cosmological quantity types. `globalConstants().unitScales()` exposes
the immutable local conversion cache derived from the root's run constants.
Existing Mpc/h inputs, solver interfaces and legacy checkpoint metadata keep
their conventions; the new output views remove little-h factors explicitly.

The Layzer–Irvine diagnostic integrates
`dW/dtau=(da/dtau)/a*(2K+U)` using trapezoidal time quadrature, so
`K+U+W` should be constant up to integration, force, quadrature, and roundoff
error. The work is signed and handles contraction. All energy reference/history
values survive restart. Extra stopping samples can change the quadrature error;
convergence with timestep refinement is the appropriate conservation check.
The driver selects CPU or CUDA gravity/drift and uses HPX collectives; its
cosmology is integrated once per evolution invocation (including restart) from
its saved tick to the requested physical endpoint. Accepted Dormand–Prince
steps retain quartic dense output for scale factor, signed expansion rate,
conformal time, and drift integral. Rungs, snapshots, and lightcone crossings
interpolate this shared trajectory without repeating cosmological quadrature.
Short drift intervals use local polynomial differences to avoid cancellation.
The table is rebuilt on restart; the saved clock and staggered momenta are unchanged.

### Ready-to-run evolution configurations

The three files in `configs/evolution` use a 32³ particle lattice and mesh in a
100 Mpc/h box, seed 42, density handoff threshold 0.1, and 100 snapshots including
both endpoints at equal conformal-time spacing:

| INI | Stop |
| --- | --- |
| [`lcdm_present.ini`](configs/evolution/lcdm_present.ini) | Standard flat Lambda CDM at the present (`a=1`). |
| [`lcdm_ten_ages.ini`](configs/evolution/lcdm_ten_ages.ini) | The same cosmology at cosmic age `10*t0`; `t0=13.792880934981705` Gyr, so the offset is `124.13592841483534` Gyr. |
| [`closed_no_dark_energy.ini`](configs/evolution/closed_no_dark_energy.ini) | Closed `Omega_cold=2`, zero dark energy, through turnaround to contracting `a=1e-6`. The predicted crunch is about `82.859247245` Gyr from present, approximately 0.75 years after the cutoff. |

The age calculation includes the configured radiation and thermal massive
neutrino background. The future INI pins those parameters; if changing them,
recompute `9*t0` for its numeric stopping offset.

From the repository root, run one simulation at a time. These commands use the
validated CPU build and one locality, so each run produces exactly 100 Silo
files. The runtime settings keep HPX at five OS threads; CAMB adds one serial
process during initialization.

```bash
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
runtime=(--hpx:threads=1 --hpx:bind=none
  --hpx:ini=hpx.parcel.enable=0
  --hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0
  --hpx:ini=hpx.threadpools.io_pool_size=1
  --hpx:ini=hpx.threadpools.timer_pool_size=1
  --hpx:ini=hpx.threadpools.parcel_pool_size=1)
./build-pmmm-cpu/cosmictiger --configurationFile=configs/evolution/lcdm_present.ini "${runtime[@]}"
./build-pmmm-cpu/cosmictiger --configurationFile=configs/evolution/lcdm_ten_ages.ini "${runtime[@]}"
./build-pmmm-cpu/cosmictiger --configurationFile=configs/evolution/closed_no_dark_energy.ini "${runtime[@]}"
```

Results go under `output/`, with a separate prefix and final `.restart`
manifest for each cosmology. Runtime scales with clustering and the finest
particle rung; these are simulation configurations, not quick smoke tests.

Initialization prints each CAMB spectrum evaluation, the density scan and
refinement result, and the particle preparation stages. The first evolution
gravity evaluation also reports its grid, FFT, ghost, multipole, force, and
snapshot stages. Evolution prints and flushes a progress row after every
completed rung step, including fine rungs. Additional rows identify initial
and between-rung snapshot samples. A blank line and column header appear before
the first row and again after every 25 rows. Compact labels and reduced display
precision keep each ordinary row to about 108 columns. For example, with
illustrative phase percentages:

```text
*   step k  r      t[yr]        tau          a          z         ad rm    D%    X%    M%    G%    K%    O%
*      0 I  - -1.373e+10  0.000e+00  2.508e-02  3.887e+01  2.455e-10  6   0.0   1.0   5.0  80.0   4.0  10.0
       1 R  6 -1.371e+10  4.178e-02  2.895e-02  3.354e+01  2.283e-10  6   1.0   2.0   6.0  83.0   5.0   3.0
       2 R  6 -1.369e+10  8.356e-02  3.310e-02  2.921e+01  2.135e-10  6   1.0   2.0   6.0  82.0   6.0   3.0
*      2 S  - -1.368e+10  1.070e-01  3.554e-02  2.714e+01  2.059e-10  6   1.0   2.0   5.0  75.0   5.0  12.0
```

`*` means a Silo snapshot finished writing at that state. `step` counts completed
rung events in the current invocation. `k` identifies an initial evaluation
(`I`), completed rung step (`R`), or snapshot-only sample (`S`). `r` is the rung
just advanced; `rm` is the highest particle rung after evaluation.
`t[yr]` is physical years relative to the expanding present-day epoch,
`tau` is conformal time since the original handoff in `H0^-1`, `z = 1/a - 1`
is redshift, and `ad` is physical `da/dt` in inverse years. The diagnostics CSV
retains the full-precision energy and force history.

The final six columns report percentages of the current row's elapsed wall time
on the root locality. Measurements reset for every rung step or snapshot sample;
the initial row covers its initial evaluation and output, excluding setup before
that evaluation. The phases are sequential and their displayed shares sum to
approximately 100%, subject to rounding:

| Column | Phase |
| --- | --- |
| `D%` | Time advancement, optional lightcone sample, and particle drift |
| `X%` | Particle redistribution and ghost exchange |
| `M%` | Cell multipole construction (P2M) |
| `G%` | Near-field preparation and periodic far-field FFT/M2L, counted together while they overlap |
| `K%` | Complete force evaluation, momentum/rung updates, local sums, and rung sorting |
| `O%` | Global reductions, energy history, diagnostic CSV, and snapshot/master/VisIt output |

CPU near-field P2P runs in `G%`; CUDA batches P2P with L2P and the kick in `K%`.
The measurements include waiting for work and communication to finish.
Formatting, printing, and the synchronization that finishes progress reporting
are excluded. These are per-step wall-time shares, so they do not sum CPU or GPU
work across workers or localities.

CUDA is enabled by default and requires HPX built with `HPX_WITH_CUDA=ON`.
Both C++ and CUDA sources use C++20.
Add CUDA sources to the appropriate CMake
target and set `CMAKE_CUDA_ARCHITECTURES` for the GPUs you intend to support.
For a CPU-only build, pass `-DCOSMICTIGER_ENABLE_CUDA=OFF`.

Host CUDA operations use `<cosmictiger/cuda/cuda_runtime.hpp>` and return HPX
futures. Device queries return values, allocation returns a pointer, and copies,
free operations, and kernel submissions return completion futures. Runtime
failures propagate through the futures; out-of-memory allocation throws
`std::bad_alloc`. Blocking allocation and query work runs on HPX's I/O pool.
Kernel and copy completion uses HPX's CUDA stream integration. Chain dependent
submissions through futures, and keep buffers and external streams alive until
completion. Call these interfaces while HPX is running and outside application
locks. Device IDs default to zero and can be supplied explicitly.

`gravity::launchDirectInteractionsCuda` returns `hpx::future<void>` that reports
argument errors, launch errors, and GPU completion. CPU `directInteractionsCpu`
uses bounded sink tasks for backends declaring `supportsConcurrentSinks() == true`;
other backends retain increasing sink order. `DirectFieldBackend` writes
range-relative accelerations and optional potentials, preserving inactive and
ghost outputs. Sources and each sink's reduction order are unchanged. See
[thread parallelism](docs/thread_parallelism.md#3-place-gravity-tasks-above-the-numerical-kernels)
for ownership, completion, and failure contracts. The standalone allocator
view is in `<cosmictiger/cuda/cuda_memory_pool_view.hpp>`; its HPX host owner is
in `<cosmictiger/cuda/cuda_memory_pool.hpp>`. `CudaMemoryPool::create` returns a
future owning the pool, and `release` returns a future reporting CUDA cleanup.
The owner's constructor remains a synchronous convenience. Finish all users
before resetting or releasing the pool.
To omit the test suite, pass `-DBUILD_TESTING=OFF`.

## Cosmological background

`<cosmictiger/friedmann.hpp>` provides `FriedmannCosmology` for adaptive
double-precision scale-factor evolution with radiation, matter, a cosmological
constant, and curvature. Choose physical or conformal time; results include
both derivatives and support expansion, turnaround, and contraction. See
[the Friedmann solver guide](docs/friedmann.md) for units, examples, and accuracy
controls. Link the `cosmictiger_cosmology` CMake target.

## Einstein-Boltzmann solvers and initial-condition spectra

`<cosmictiger/einstein_boltzmann.hpp>` defines a solver-independent contract:
`CosmologyParameters`, `MatterPowerRequest`, `MatterPowerSpectrum`, and the
abstract `EinsteinBoltzmannSolver`. Alternate packages or our own implementation
can implement `compute()` without changing initial-condition callers. Link the
C++ target `cosmictiger_boltzmann` to use these APIs.

`<cosmictiger/camb.hpp>` implements that contract with `CambSolver`. The official
[CAMB repository](https://github.com/cmbant/CAMB) is pinned as a Git submodule in
`third_party/camb`; its required forutils submodule is initialized recursively.
`build.sh` performs this download before CMake configuration. For direct CMake
builds, initialize the submodules with the command above first. The build uses
upstream GNU Make and gfortran in a private copy at
`<build-dir>/_deps/camb-src`, leaving the upstream checkout clean and isolating
application build modes. CAMB retains its upstream optimized Fortran settings
in every application mode. Source/compiler changes rebuild the staged copy.

CAMB remains a **separate executable**. CosmicTiger writes a complete INI file,
launches the executable with HPX's process utilities (`fork`/`execve`, with no
shell), and reads the returned text file. It never links CAMB's Fortran library.
The default executable is `<build-dir>/_deps/camb-src/fortran/camb`; pass another
executable path to `CambSolver` when relocating a build or using another CAMB
installation. Call `compute()` from an HPX task: blocking file I/O and process
waiting run on HPX's existing I/O pool. Each solver process is restricted to one
OpenMP thread, which must be counted in the shared resource budget. Run the
solver once on the console locality and distribute the resulting table as
needed; this API does not automatically start one solver per MPI rank.

```cpp
#include <cosmictiger/camb.hpp>

cosmictiger::CosmologyParameters cosmology;
cosmology.hubble = 67.4;                // H0 in km/s/Mpc
cosmology.omegaBaryonH2 = 0.0224;       // Omega_b * h^2
cosmology.omegaCdmH2 = 0.120;           // Omega_c * h^2
cosmology.scalarAmplitude = 2.1e-9;     // primordial curvature A_s
cosmology.spectralIndex = 0.965;

cosmictiger::MatterPowerRequest request;
request.redshift = 49.0;
request.maximumWaveNumber = 10.0;      // h/Mpc
request.component = cosmictiger::MatterComponent::cdmAndBaryons;

cosmictiger::CambSolver camb;
cosmictiger::EinsteinBoltzmannSolver const& solver = camb;
auto const spectrum = solver.compute(cosmology, request);
double const power = spectrum.power(0.1); // P(k,z) in (Mpc/h)^3
```

The output is the **linear matter power spectrum** `P(k,z)`, for adiabatic
initial perturbations and synchronous-gauge density contrasts. Wave numbers are in `h/Mpc`, power is in `(Mpc/h)^3`, and
`h = H0/100`. The primordial pivot uses `Mpc^-1`. For CAMB's native INI,
`transfer_kmax` uses `h/Mpc` (unlike its Python API). The request selects total
matter (CDM+baryons+massive neutrinos), CDM+baryons, or CDM; these have distinct
mean-density normalizations. The neutrino parameters describe one degenerate
massive group with physical density `omegaNeutrinoH2`, physical species counts
`massiveNeutrinos` and `masslessNeutrinos`, and total `effectiveNeutrinos`
(`N_eff`). All species have a common temperature, so each contributes
`N_eff / (massiveNeutrinos + masslessNeutrinos)`. This is a solver-independent
prescription; the adapter supplies explicit CAMB degeneracies. With zero
massive-neutrino density, set the massive count to zero (and normally the
massless count to three). Dark energy uses the PPF model with
`w(a) = darkEnergyW + darkEnergyWa * (1-a)`.

`samples()` exposes the returned table. `power(k)` interpolates linearly in
`log(k)` and `log(P)` and rejects extrapolation; check that the table covers
all nonzero mesh modes needed by your initial conditions. CAMB's native text
output has about six significant decimal digits. This interface returns
density power only; species transfer functions remain future additions to the
common solver contract. Particle initialization obtains Zel'dovich velocities
from the time derivative of these density amplitudes.

Each run uses a unique temporary directory. Successful runs remove temporary
files. Invalid parameters throw before execution; process failures, missing or
malformed output, and insufficient k coverage throw with the retained run
directory containing `parameters.ini`, `camb.log`, and any output files.

## Particle initialization

`<cosmictiger/particle/particle_initialization.hpp>` supplies `initializeParticles()`.
It takes the CAMB cosmology parameters and a user-selected maximum absolute
linear density contrast, advances the same seeded Gaussian realization using
CAMB spectra, and returns Zel'dovich particle lists at the first bracketed
threshold crossing. The result contains the handoff scale factor/redshift,
equal particle mass, owned particle storage, and populated Cartesian cell and
rung ranges. Mode and field assembly, particle filling, and local density maxima
use bounded HPX tasks before each collective phase. Link the
`cosmictiger_initialization` target. See the
[initialization guide](docs/particle_initialization.md) for the collective API,
units, density criterion, and numerical limits. The startup executable runs
this initializer before evolution by default. To write only initial conditions,
set `simulation.evolve=false` and `initialConditions.generate=true`.

## Distributed FFT

`<cosmictiger/fft/fft.hpp>` implements batched, cubic radix-2 real FFTs and circular
convolutions. The mesh side length is `M=2^m`; the locality count is
`P=2^(3p)`, with `8 <= P <= M`. Each locality owns a cubic block of
`(M/P^(1/3))^3` values. Local blocks store `x` fastest and locality IDs order
their block coordinates as `x`, then `y`, then `z`. Each value is a
`std::array<double, batchSize>`; lanes are independent FFTs and convolutions.

Local FFTW plans are memoized per process by one-dimensional transform length
and transform kind. All FFT strategies and batch sizes reuse compatible plans,
including after individual FFT objects are destroyed. Execution uses private
scratch arrays; cached plans are released at process teardown.

`<cosmictiger/fft/fft_filters.hpp>` provides `FftFilterBank<batchSize>` for filters
created once and reused in later convolutions. Construct one bank per locality
with the same mesh size, locality count, and communication weights. Call
`createFilter(name, localKernel, exchange)` on every locality for each real
kernel; the bank transforms and stores the local spectrum under `name`.
Subsequent `convolve(localInput, name, exchange)` calls use that spectrum and
the bank's retained inverse transform. Use the same filter names and creation
order on every locality, and make each call collectively in the same order.

`setCommunicationWeights(byteWeight, messageWeight)` may change the planner's
weights before the first filter is created. The first filter fixes the selected
plan and spectral layout for the bank; later weight changes are rejected.
After creating a filter, `plan()` reports the selected pencil/slab, packed
vector radix, or full spectrum vector radix strategy, and
`filterValueCount(name)` reports this locality's stored complex values for that
filter. The planner counts payload bytes, nonempty directed sends, and
sequential peer exchange rounds. A batch increases payload bytes but shares
messages and rounds. Filter creation uses a forward FFT; each later
convolution uses one forward FFT, pointwise multiplication, and one inverse.
For the packed vector radix strategy, stored filters use the full complex
vector-radix layout required by its prepositioned multiplier path, even though
the input forward transform uses a packed half-size spectrum. At the pencil
boundary, some localities own no spectral filter values.

`<cosmictiger/fft/fft.hpp>` also exposes `FftDistributed<batchSize>` directly. Its
`FftCommunicationWorkload::twoForwardInputs` mode performs two forward FFTs and
one inverse when both real inputs are supplied for each convolution.

`FftHpxExchange` is the HPX transport adapter. Give every locality the same
ordered HPX locality ID list and a fresh FFT operation ID for each collective
call, and pass an exchange callback to the filter bank:

```cpp
cosmictiger::FftFilterBank<4> filters(M, ids.size(), localIndex, byteWeight, messageWeight);
cosmictiger::FftHpxExchange filterTransport(ids, filterOperationId, localIndex);
auto filterExchange = [&](size_t peer, auto send, auto receive) {
    filterTransport.exchange(peer, send, receive);
};
filters.createFilter("gravity", localKernel, filterExchange);

cosmictiger::FftHpxExchange convolutionTransport(ids, convolutionOperationId, localIndex);
auto convolutionExchange = [&](size_t peer, auto send, auto receive) {
    convolutionTransport.exchange(peer, send, receive);
};
auto result = filters.convolve(localBlock, "gravity", convolutionExchange);
```

Run corresponding calls in the same order on all localities from HPX tasks.
The current executable does not yet create physical filter kernels at startup.
The adapter uses HPX `serialize_buffer` reference payloads for interlocality
transfers and owns same-process payloads where HPX may bypass serialization.
HPX tasks parallelize independent packing and FFTW3 line transforms within a
locality, using at most four chunks per worker and a small-work serial cutoff.
Each transform chunk reuses its private scratch across line/lane pairs;
simultaneous calls have separate scratch. See the [parallelism audit](docs/thread_parallelism.md)
for the scheduling estimate and local timing results.
Each FFTW call is single threaded; the build links `fftw3`, without
FFTW's threads or MPI library.

## Fixed-point real numbers

`<cosmictiger/fixed.hpp>` provides `cosmictiger::Fixed<Integer, dx>`. The integer
can be any standard integer type except `bool`. The positive, finite range `dx`
is a C++20 compile-time parameter and defaults to 1. Integer and floating-point
range literals are accepted; different template arguments define different types.

```cpp
using TenUnitPosition = cosmictiger::Fixed<unsigned int, 10>; // [0, 10)
using Offset = cosmictiger::Fixed<int, 2.0>;           // [-1, 1)
TenUnitPosition position(9.5);
position += TenUnitPosition(1.0);                    // 0.5
Offset offset(0.75);
offset += Offset(0.5);                               // -0.75
```

For an integer with W value bits (including the sign bit), the represented value
is `raw() * dx / 2^W`, with spacing `dx / 2^W`. The upper endpoint wraps to the
lower endpoint. `min()`, `max()`, and `epsilon()` return fixed-point values, and
`fromRaw()` constructs from the integer representation.

Addition, subtraction, negation, multiplication, and division wrap with period
`dx`. Products and quotients truncate toward zero before wrapping and use exact
integer arithmetic, including for 64-bit storage. Floating-point input/output
uses `long double` precision; output can round to an endpoint or lose low bits
when the floating-point type has insufficient precision. Use `toReal<float>()`,
`toReal<double>()`, or `toReal<long double>()` for explicit conversion. The
`toFloat()`, `toDouble()`, and `toLongDouble()` methods are also available.
Comparisons order the represented values within the range. Division by zero is diagnosed by `expectNonZero` in validation builds.
Floating input must be finite; local host conversions use `FpeGuard` to trap
arithmetic exceptions when validation is enabled.

## Numerical validation

`cosmictiger/debug.hpp` provides `expectPositive`, `expectNegative`,
`expectNonNegative`, `expectNonPositive`, `expectNonZero`, inclusive
`expectRange(lower, value, upper)`, exclusive `expectIndex(index, size)`,
`expectEqual`, and `expect`. Values are checked where they are consumed, without
separate numerical validation passes. `COSMICTIGER_VALIDATION_CHECKS` is enabled
in Debug, RelWithDebInfo, and Profile, independently of `NDEBUG`. Release
omits the checks, including evaluation of bounds and boolean contracts.

Host numerical scopes use `FpeGuard` for invalid operations, division by zero,
and overflow. The guard restores the calling thread's exception masks and
pending flags, and must never span HPX scheduling or suspension. Exceptions
raise SIGFPE; they are not recoverable C++ exceptions. A quiet NaN already in
storage does not necessarily raise an arithmetic exception. There are no
finite-value validation scans.

The primitives and guard compile for CUDA devices. Failed device expectations
print a diagnostic and trap. CUDA does not expose floating-point exception
flags or masks: device `FpeGuard` is inert and `supportsHardwareTraps()` returns
false. Operational failures such as I/O, allocation, communication, and CUDA
status errors retain their ordinary error handling in Release.

## Per-locality particle storage

`<cosmictiger/particle/particle_storage.hpp>` provides `ParticleStorage(initialSize)` and
`localParticleStorage()`, a process-local owner for each HPX locality. Initialize
the locality owner with `resize(initialSize)`. Storage uses seven base SoA
allocations: three `Coordinate` position fields (value = raw / 2^32,
range [0,1)), three `float` velocity fields, and one one-byte `Rung` value.
Six bits encode rungs 0..63 (`Rung::maximumRung`), one bit marks a ghost copy
owned by another locality, and one bit is reserved for future use. New particles
have rung zero and `isGhost() == false`. Periodic copies whose owner is this
locality leave the flag false; grid ghost ranges still identify those copies.
CUDA builds allocate each field through the future-returning managed-memory
interface; CPU builds use `malloc`. The storage's synchronous API waits for its
allocation and cleanup futures, and locality storage releases its fields before
HPX shutdown.

Sizes, capacities, and particle indices are unsigned 64-bit values and support
counts beyond 2^32. Byte counts are checked against the allocator's `size_t`
limit before conversion. Unaddressable requests and capacity arithmetic overflow
throw `std::length_error`; allocation exhaustion throws `std::bad_alloc`.

```cpp
auto& particles = cosmictiger::localParticleStorage();
particles.resize(initialSize);
particles.position(0, 0) = cosmictiger::Coordinate::fromRaw(0x80000000u);
particles.velocity(1, 0) = 1.5f;
particles.flags(0).setRung(3);
particles.flags(0).setGhost(false);
particles.append({{}, {}, {}}, {0.0f, 0.0f, 0.0f}, 0);
particles.swapParticles(0, 1);
particles.remove(0);
```

`append` accepts an optional fourth `bool isGhost` argument, defaulting to false.
It rejects rung values above 63 before modifying storage. `Rung` construction
and `setRung()` also throw `std::invalid_argument` for values above 63;
`rung()` reads the value, while `isGhost()` and `setGhost()` access the ghost flag.
Changing either field preserves the other and the reserved bit. `flags(index)`
returns a `Rung` reference, and `rungData()` returns a contiguous `Rung` pointer.

Nonempty capacity is the smallest power of two covering the initial size.
Growth doubles capacity as needed, copying and freeing one field at a time.
`reserve(n)` grows capacity without changing size. `resize(n)` zero-initializes
new live particles; shrinking, `clear()`, and `remove(index)` retain capacity.
`swapParticles(first, second)` exchanges all position, velocity, and flag fields.
Removal replaces the removed particle with the last particle, without preserving
order. Empty storage starts with zero capacity. Storage owns its allocations and
is movable but cannot be copied.

`position(axis, index)`, `velocity(axis, index)`, and `flags(index)` check live
indices and axes when `COSMICTIGER_VALIDATION_CHECKS` is defined.
CMake enables this in Debug, RelWithDebInfo, and Profile; it is disabled in
Release. Invalid access then throws `std::runtime_error` with its source location. With checks
disabled, valid axes and live indices are caller preconditions.

`positionData(axis)`, `velocityData(axis)`, and `rungData()` expose contiguous
fields for kernels and communication; indexing these pointers bypasses checks.
Mutation requires exclusive access and completion of GPU work using the fields.
Reacquire pointers after growth, including failed growth: allocation failure
preserves the live particles and reported capacity, but fields already grown
keep their larger allocations for subsequent retries.

## Cartesian particle grid

`<cosmictiger/cartesian_grid.hpp>` provides `CartesianGrid(m, p, localIndex)`.
Construct one on each HPX locality using its index in the same ordered locality
list as the FFT. The periodic unit cube [0,1)^3 has `2^(3m)` cells, regularly
spaced by `2^-m`, and `2^(3p)` cubic locality blocks. Require `0 <= p <= m`;
zero supports a single cell or locality. Cells and locality blocks use x-fastest
ordering, then y, then z.

Each locality owns `n^3` interior cells with `n=2^(m-p)` and stores `(n+2w)^3`
entries for ghost width `w`. Local coordinates are `[0,n)` for interiors and
`[-w,n+w)` for the stored grid, including all ghost faces, edges, and corners.
The standalone `CartesianGrid` constructor defaults to width one; evolution
passes the runtime `gravity.ghostWidth` setting, whose default is one.
`layout().cellIndex()` and `cellCoordinates()` convert between these coordinates
and the flat storage index. `globalCoordinates()` wraps into the periodic cube;
`cellOwner()` returns the owning locality index and its interior coordinates.
`cellCenter()` returns the wrapped center in the unit cube. Ownership is computed,
so neither global grid entries nor a locality table are allocated.

```cpp
cosmictiger::CartesianGrid grid(m, p, localIndex, ghostWidth);
auto& particles = cosmictiger::localParticleStorage();
grid.setInteriorRange({0, 0, 0}, {begin, end});
grid.setGhostRanges({-1, 0, 0}, {localBegin, localEnd}, {ownerBegin, ownerEnd});
grid.sortParticlesByRung(particles);
const auto& ranges = grid.cell({-1, 0, 0});
const auto owner = grid.layout().cellOwner({-1, 0, 0});
```

Each entry stores two half-open `ParticleRange` values, each with 64-bit
`begin` and `end` indices matching `ParticleStorage::Size`. It also has
`rungParticles`, an array of 64 half-open local particle ranges, one for each
rung 0..63. `sortParticlesByRung(particles)` sorts particles within every cell
from lowest to highest rung and builds these ranges; `sortCellParticlesByRung`
does the same for one cell. Bulk sorting uses bounded HPX tasks over disjoint cell ranges and joins all
tasks before returning. Small grids, one-worker runtimes, and non-HPX callers
use the synchronous path. The sort moves each particle's position, velocity,
and flags together. Empty rungs have empty ranges at the appropriate boundary.
`rungRangesValid` is true after sorting or for an empty cell, and false after a
range setter assigns a nonempty cell. Changing a particle's rung requires
sorting its cell again before using `rungParticles`; direct particle mutation
does not update `rungRangesValid` automatically.

Each entry also stores one `CellBox` (`Box<double>`) with `2 * NDIM`
coordinates for its physical bounds.
It also stores a zero-initialized `CellMultipole multipole` and
`CellExpansion expansion`, both inline arrays of floats. The compiled maximum
tensor ranks are 8 and 9, respectively (81 moment and 101 expansion coefficients,
including the periodic-background trace; the cell size depends on the `Box`
representation). Each cell also retains the double-precision normalized second
moment trace omitted by its STF multipole. Set
`gravity.fmmOrder=p` on the command line, or `fmmOrder=p` in the INI
`[gravity]` section, to select local rank `p` and multipole rank `p-1` at
runtime. The default is 6, preserving source/local ranks 5/6, and the default
build accepts 2–9. Both Ferrers and Plummer gravity use the selected order,
including initial gravity output and CPU/CUDA kicks. Checkpoints save the
order; a restart retains it and rejects an explicit change. Cell arrays keep
their compiled capacities, with unused local coefficients cleared. Every
build includes the full 2–9 runtime range; the former CMake order selections
have been retired. Existing build directories receive the expanded cell
layout when reconfigured and must rebuild all users of that layout.
Each cell additionally stores `fullMultipole` and `fullExpansion` for the
Plummer path. Their capacities contain 165 and 220 floats, adding 1540
bytes per stored cell, including ghosts and cells in Ferrers runs.
`cell(...).physicalBox` stores double-precision bounds for a uniformly spaced
cell. Ghost bounds use the adjacent periodic image: a ghost left of zero has
negative coordinates, and a ghost right of one has coordinates above one.
`physicalBox.containsPeriodic(point)` checks the nearest images of a point with
floating-point or fixed-point coordinates, including `Position`. For `m=0`,
the interior box is [0,1)^3 and its ghost boxes are adjacent images.
`layout().physicalBox()` computes the same bounds without allocating cells.
The reusable `Box<Real>` accepts floating-point and `Fixed` coordinate types.
Its two corners define `[min,max)` intervals on every axis. `expanded()` and
`contracted()` return resized boxes, and `split(axis, position)` returns two
boxes with the right half owning the cut. `intersects()` detects interior
overlap, `touches()` detects boundary contact, and `isDisjoint()` detects a
strict gap. Box and grid tests exercise double precision and 32-bit fixed point.
`localParticles` refers to this locality's particle storage; `ownerParticles`
refers to the storage on the owning locality. Interior updates set both to the
same range; ghost ranges are independent, including when periodic wrapping
returns to the same locality. All ranges start empty at `[0,0)`. Access is const;
use the setters to preserve the interior invariant. `expansion(coordinates)`
provides mutable access to the expansion alone. Range setters clear both gravity
arrays. Mutation requires exclusive access and completed GPU work. The grid
constructor allocates no particles and performs no sorting or exchange.
Ranges must have `begin <= end`; validating their ends against live particle
counts remains the caller's responsibility once particles are populated.

```cpp
grid.computeMultipoles(particles, particleMass); // all stored cells, including ghosts
grid.computeMultipole({0, 0, 0}, particles, particleMass); // one cell
grid.computeMultipoles(); // localParticleStorage(), unit particle mass
```

These functions replace each selected cell's moments and second moment trace using its `localParticles`
range and leave its expansion unchanged. Particle storage has no mass array, so
`particleMass` is a common finite mass, defaulting to `1.0f`. Call after sorting
particles into cell ranges, and after ghost exchange when ghost moments are
needed. Empty ranges produce zero moments. Local range endpoints are checked as each cell is processed in validation
builds; a later failure may leave earlier cells updated. Bulk computation uses bounded HPX tasks across cells while retaining each
cell's particle accumulation order. Every launched task finishes before return,
including after a cell fails; other cells may have completed in any order.
Host arithmetic uses a cell-local `FpeGuard`. Ghost owner ranges may index remote storage and are not read. Particle motion requires
recomputation even if ranges stay unchanged.

The origin is the geometric `layout().cellCenter()`, including for periodic
ghost cells. Moments are raw STF tensors divided by `layout().spacing()^rank`,
with all three dipoles retained; expansions store potential derivatives times
that same scale to their rank, plus an appended normalized Laplacian. Coordinates are differenced and moments summed
in double, then stored as floats. At the default orders these arrays can be
passed directly to `gravity::stf::addM2L<5, 6, false, true>` using the cell spacing for both
scales. Computing moments does not compute a gravitational local expansion.

`<cosmictiger/gravity/pmmm/pmmm_convolution.hpp>` provides
`gravity::PmmmPeriodicConvolution<>`. Construct one per grid locality. Its
`computeExpansions(grid, exchange)` function uses the periodic Green filters and
the cell moments to replace all interior expansions through a distributed FFT
convolution. Its default STF path includes the uniform periodic background, using the stored
second moment trace, and leaves ghost expansions untouched. Green spectra are
cached after the first call. Green setup distributes independent cells across
bounded HPX tasks while preserving each cell's sum order. FFTs batch four
coefficients by default, including the mass/dipole/trace background fields;
packing, products, and publication also use independent parallel loops.
`PmmmPeriodicConvolution<sourceOrder, expansionOrder, batchSize>` selects another
positive batch size; size one retains scalar transforms. Final batch padding
is zero. At the default orders, setup uses 37 FFT calls and a cached solve uses
25, compared with 145 and 91 in the scalar path. Exchanges stay sequential
across batches. `gravity::computePmmmExpansions(grid, exchange)` provides a
one-shot call with the default batch size. The FFT exchange callback uses z-fastest locality
IDs, while the Cartesian grid uses x-fastest IDs; call
`fftLocalityIndex()` to construct the matching FFT transport. All localities
must enter the function concurrently and in the same sequence. The Plummer
path uses the full arrays and includes the background in its Green
derivatives, without an additional STF trace correction. The near-cell
primary interaction excluded by either filter must be evaluated with the
selected softening model by the direct gravity path. The FFT call counts
above describe the default STF path.

Coordinates and locality indices are always checked. Invalid access throws
`std::out_of_range`, invalid topology or ranges throws `std::invalid_argument`,
and unrepresentable counts or allocations throw `std::length_error`.
`CartesianGridLayout` exposes the geometry alone without allocating cell ranges.

## Global particle load balancing

`<cosmictiger/particle/particle_load_balance.hpp>` provides an HPX collective that moves
each owned particle to the locality containing its Cartesian cell, then
partitions each local array into contiguous interior-cell ranges. Link
`cosmictiger_particles` when using this API outside the supplied executables.
Call it from an HPX task on **every** participating locality with matching grid
topology, the same ordered locality list, and a globally unique operation ID:

```cpp
cosmictiger::loadBalanceParticles(particles, grid, localityIds, operationId);
// Or use localParticleStorage() and hpx::find_all_localities():
cosmictiger::loadBalanceParticles(grid, nextOperationId);
```

An operation ID identifies one invocation and must be shared by all participants;
use a fresh ID for every invocation, including retries. The call requires
exclusive access to storage and grid and completed GPU work. Successful return
means every participant has committed its replacement storage and cell ranges.

`layout().particleOwner(position)` computes exact half-open cell membership
from the raw fixed-point coordinates. Redistribution preserves all position,
velocity, and rung fields. It exchanges counts first, sends particle batches
directly between their source and destination, and uses a counting
partition locally. Bounded HPX tasks count and pack outgoing particles, then
count and scatter received particles using private histograms and disjoint
output spans. Chunk-ordered offsets retain the source order before rung sorting.
One-worker runtimes and small particle passes use the synchronous path. Both interior ranges are updated, including empty cells,
in x-fastest order. It then sorts particles in each cell by rung and fills
`rungParticles`; order within a rung is unspecified. The fixed grid
decomposition is unchanged, so uneven physical populations can still give
uneven locality particle counts.

Live particles with `isGhost() == false` outside the union of nonempty ghost
`localParticles` ranges are treated as owned, including before the first interior
ranges are populated. Flagged remote ghosts are excluded even without grid ranges.
Existing ghost copies are excluded and discarded. The operation clears all
ghost ranges because their local and owner indices become stale; refresh ghost
copies separately afterward. It never sorts or exchanges ghost particles.
Overlapping ghost ranges are excluded only once, and out-of-storage ghost
ranges fail validation. Mark ghost copies accurately before calling. Particles
transferred to a new owner retain their numeric rung and have `isGhost == false`.

Every participant prepares its replacement before committing. Validation and
allocation failures before commit preserve every input array and grid range;
A task-launch or allocation failure in the final rung sort is coordinated after
commit; replacement storage remains committed and rung metadata may be incomplete.
Process or network failures require runtime-level recovery. Transfer buffers
use the native particle representation and require matching representation on
all localities. The optional eight-process integration test is enabled with
`-DCOSMICTIGER_TEST_EIGHT_LOCALITIES=ON`; run it only when the shared local compute
budget permits eight workers. Logical-locality tests also run with one HPX worker.

## Ghost particle exchange

`<cosmictiger/particle/particle_ghost_exchange.hpp>` provides exchange and clearing
functions through `cosmictiger_particles`:

```cpp
cosmictiger::exchangeGhostParticles(particles, grid, localityIds, operationId);
cosmictiger::clearGhostParticles(particles, grid);
// Overloads use localParticleStorage() and, for exchange, all HPX localities:
cosmictiger::exchangeGhostParticles(grid, nextOperationId);
cosmictiger::clearGhostParticles(grid);
```

Exchange is collective: call from an HPX task on every participant using the
same ordered locality list, grid topology, and fresh shared operation ID.
Interior ranges must partition the owned particle prefix in x-fastest order,
as produced by load balancing, and positions must still belong to their cells.
If particles have moved between cells, clear ghosts and load balance first.
Both functions require exclusive storage/grid access and completed GPU work.

The exchange copies every particle from each ghost cell's owning interior cell,
including periodic faces, edges, and corners. Each ghost cell receives a
separate contiguous range appended after the owned particles, in stored
x-fastest order. Its `ownerParticles` range refers to the source interior
range. Positions, velocities, numeric rungs, and interior ranges are preserved;
owned particles may be reordered within their cells to group rungs. New ghost
copies are sorted by rung, and their `rungParticles` ranges are populated.
Copies owned by another locality set the ghost flag; same-locality periodic
copies keep it false. Empty cells receive valid empty ranges.
Particle payloads travel directly between owners and recipients; small cell
request/range metadata uses HPX collectives.

Packing and unpacking use bounded HPX tasks over contiguous particle chunks,
including splitting a heavily populated cell across tasks. Prefix offsets
preserve each peer's request order and the stored cell order of the ghost
suffix. Buffers are sized before tasks write disjoint spans; every task finishes
before the next exchange phase or error agreement. After unpacking completes,
cell ranges are published and particles are grouped by rung. Small exchanges
and one-worker execution remain inline.

Repeated exchange replaces the existing ghost suffix. Clearing truncates it
and resets both ranges on all ghost cells, including same-locality periodic
copies. Clearing is local, can be repeated, and preserves capacity, owned
particles, and interior ranges; it also works after owned positions move.
Interior ranges must describe a complete owned prefix. Old ghost metadata is
discarded during refresh and clear because it is no longer consumed.

Exchange stages transfers before committing and checks data as it is consumed.
Allocation failures before commit preserve live particle values, size, and grid
ranges. Validation failures during ghost population may leave partial changes;
storage growth may change capacity or field pointers.
Reacquire raw field pointers after exchange, including failed growth. Native
transfer buffers require matching particle ABI across participants. Successful
return follows a global completion barrier. Logical-locality tests cover the
exchange with one and two HPX workers. A two-process MPI test exercises eight logical
grid localities and actual remote payloads within four HPX workers; the optional
eight-process test uses `COSMICTIGER_TEST_EIGHT_LOCALITIES` as load balancing does.

## Cartesian FMM

The Cartesian STF operators imported from OctotigerII are available through
`<cosmictiger/gravity/stf/operators.hpp>` (fixed-size, header-only, orders 0–9)
and `<cosmictiger/gravity/cartesian/fmm.hpp>` (runtime dispatch and cached lattice
geometry). They include point moments, M2M/M2L/L2L, P2L/M2P, constant local
Laplacians. Runtime `addM2LTargets` evaluates ordered interaction lists in bounded
tasks over disjoint target expansions. Link `cosmictiger_fmm`
for the runtime API or `cosmictiger_stf` for the fixed-size API.
See [Cartesian FMM documentation](docs/cartesian_fmm.md) for normalization,
source provenance, regeneration, and numerical validation.

`<cosmictiger/gravity/symmetric/operators.hpp>` supplies a separate
fixed-size full Cartesian family with template orders, all traces retained,
and Plummer-aware P2L/M2L/M2P/P2P. Its P2M/M2M/L2L operators preserve the
complete moments or local polynomial. See
[full Cartesian multipoles and Plummer gravity](docs/plummer_gravity.md).

## Gravity kicks

The acceleration-softening limiter, conformal rung clock, synchronized rung
changes, global force/potential sums, and optional SILO fields are described in
[the gravity kick guide](docs/gravity_kick.md). `timestep.eta` is configurable
through the command line or `--configurationFile=run.ini`. Rung zero is the
nominal conformal simulation duration, computed once at initialization.
`output.acceleration` and `output.potential` independently enable persistent
fields; both default to false.

When initial gravity output is requested, particles first move onto the selected
`gravity.meshExponent` grid with `gravity.ghostWidth` layers. Their count and
original initialization metadata remain unchanged. Near-field preparation overlaps PMMM
through `hpx::async` and a future. Both branches finish before one combined
diagnostic kick and global force/potential reduction. Temporary near lanes
preserve the existing per-particle arithmetic order; see the [kick guide](docs/gravity_kick.md)
for ownership, scratch storage, and failure behavior.

The local particle drift driver supports CPU or CUDA execution; see
[particle drift](docs/particle_drift.md).

Scalar and SIMD HEALPix indexing and optional particle lightcone output are
described in [the lightcone guide](docs/lightcone.md). Enable sampling with
`--lightcone.enabled=true`; output contains particle crossings and sparse
NESTED maps, with no halo-finder dependency.

### Drift performance checks

The CPU drift uses four double-precision AVX2 lanes per pack when available,
with scalar tails and a scalar fallback on other targets. Production drift
uses the owned particle prefix, avoiding per-particle ghost checks. CUDA uses
256-thread blocks (eight full warps) over contiguous owned particles; only the
final particle tail can have inactive lanes. Both paths preserve the raw
fixed-point low bits and signed periodic shifts. Ghost particle removal only
shrinks logical storage; boundary-cell metadata is reset separately, without
scanning interior cells or repeating the reset during redistribution.

Two optional microbenchmarks are included in the test executables. In a CPU
build, run `cosmictiger_friedmann_tests --gtest_also_run_disabled_tests
--gtest_filter=SimulationTime.DISABLED_BackgroundLookupBenchmark` for table
construction, the former repeated integration, and interpolated interval timing.
Run `cosmictiger_drift_tests --gtest_also_run_disabled_tests
--gtest_filter=DriftCpu.DISABLED_PackedDriftBenchmark --hpx:threads=1
--hpx:bind=none --hpx:localities=1 --hpx:console
--hpx:ini=hpx.parcel.enable=0 --hpx:ini=hpx.parcel.mpi.enable=0
--hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.threadpools.io_pool_size=1
--hpx:ini=hpx.threadpools.timer_pool_size=1` for scalar-versus-packed drift.
Run these serially within the shared resource budget. They measure the named
components, not application-wide speedup or CUDA performance.
