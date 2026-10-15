# CAMB to Zel'dovich particles

To generate a Silo file from the executable, build with `./build.sh` and run:

```sh
./release/cosmictiger --initialConditions.generate --simulation.evolve=false \
    --initialConditions.outputPrefix=initial_conditions \
    --initialConditions.meshExponent=5 \
    --initialConditions.boxLength=100 --initialConditions.seed=1 \
    --hpx:threads=1 --hpx:bind=none
```

This creates `initial_conditions.locality0.silo` with 32³ particles and the
fiducial flat ΛCDM defaults in `CosmologyParameters`. The `cosmology.*` and
`initialConditions.*` option groups appear in `--help`. The starting
redshift is derived from the density threshold and printed at completion.
The Silo metadata records the epoch, cosmology, physical box size and particle
mass, seed, selected matter component, and density criterion. Read it with
`readInitialConditionsSiloMetadata()` from
`<cosmictiger/particle/particle_silo.hpp>`; `readParticleSilo()` restores the
particle and grid data. Ordinary checkpoints without this metadata remain
compatible with the particle reader.

`initialConditions.meshExponent=m` determines the particle count
`N=2^(NDIM*m)` and the initialization FFT side `2^m`. Set
`gravity.meshExponent=g` independently to use an evolution mesh with `2^g`
cells per axis; omitting it retains the initialization resolution. For example,
`--initialConditions.meshExponent=6 --gravity.meshExponent=5` creates 64³
particles with a 64³ initialization FFT and evolves them on a 32³ gravity mesh.
Particles are reassigned to this gravity mesh before evolution and before
requested initial acceleration, potential, or direct-comparison output.
Their count, positions, and initial realization are preserved.

`gravity.ghostWidth` accepts `1` or `2` and defaults to `1`. It controls the
gravity grid's ghost layers, direct-neighbor region, and matching periodic
Green-function exclusion. These are runtime command-line or INI choices;
changing them requires no rebuild. The gravity mesh must have at least four
cells per axis for width one, or eight for width two. Evolution checkpoints
retain both mesh sizes and the chosen width, and restarts preserve them.
The Silo `initial_conditions/mesh_exponent` always describes the original
particle lattice; `grid_global_side` describes the grid actually stored.

Call `initializeParticles()` from an HPX task on **every participating locality**.
Only site zero runs the Einstein-Boltzmann solver; it broadcasts each spectrum.
All sites use the same cosmology, initialization options, and operation ID.
The result contains this site's owned particle list and Cartesian grid, plus
identical global handoff metadata on every site. No gravity calculation,
ghost exchange, or N-body time integration is performed.

```cpp
#include <cosmictiger/particle/particle_initialization.hpp>

cosmictiger::CosmologyParameters cosmology;
cosmology.hubble = 67.4;
cosmology.omegaBaryonH2 = 0.0224;
cosmology.omegaCdmH2 = 0.120;
cosmology.scalarAmplitude = 2.1e-9;

cosmictiger::ParticleInitializationOptions options;
options.meshExponent = 6;            // 64^NDIM particles and initialization FFT elements
options.boxLength = 100.0;           // Comoving Mpc/h
options.seed = 1234;
options.maximumDensityDeviation = 0.1;
options.earliestScaleFactor = 0.001;
options.latestScaleFactor = 1.0;

// Use a new globally unique operation ID for each collective initialization.
auto initial = cosmictiger::initializeParticles(cosmology, options, 42);
double const startingScaleFactor = initial.report.scaleFactor;
auto& particles = initial.particles;
auto& grid = initial.grid;
```

The explicit overload also takes the ordered locality list, this site's index,
and an `EinsteinBoltzmannSolver const&`. This permits another EB package or an
analytic test solver. Logical sites may share one HPX process. A valid topology
has `P=2^(NDIM*p)` sites, FFT side `M=2^meshExponent >= 4`, `P<=M`, and
`p<=meshExponent`. The particle lattice, initialization FFT mesh and temporary
Cartesian cell grid have the same resolution: there is exactly one particle
at each cell center before Zel'dovich displacement. Particles are then assigned
to the cells containing their displaced positions, so final cell occupancies
may differ. The executable subsequently redistributes those particles onto
the independent gravity grid when gravity is requested. Both the locality list and Cartesian blocks
use x-fastest ordering; the initializer converts to the FFT's z-fastest locality
ordering and digit-reversed frequency layout internally.

The threshold is

\[
 \epsilon(a)=\max_{\text{mesh}}|\delta(\mathbf q,a)|,
 \qquad \delta=(\rho-\bar\rho)/\bar\rho.
\]

`maximumDensityDeviation` must be finite and strictly positive. Its mean
density is that of the selected matter component. The default is CDM+baryons,
represented as one cold collisionless population. `cdm` is also available.
`totalMatter` is accepted only without massive neutrino density: a cold lattice
cannot represent neutrino thermal motion. Massive neutrinos still contribute
to CAMB's background and linear evolution when selecting CDM+baryons or CDM.
Separate baryon hydrodynamics and neutrino particle populations are not present.

The earliest scale factor must have `epsilon<=threshold`. The initializer
advances in steps no larger than `maximumLogScaleFactorStep` in `ln(a)`, finds
the first sampled interval with a crossing, and bisects it. It returns the
**above-threshold endpoint**, with a bracket width no larger than
`logScaleFactorTolerance`. `previousScaleFactor` and
`previousDensityDeviation` report the lower endpoint. An early state already
above threshold, an interval with no crossing by `latestScaleFactor`, or an
exhausted `maximumSpectrumEvaluations` budget throws. The allowed interval is
`0<earliestScaleFactor<latestScaleFactor<=1`.

This operational criterion depends on the box, mesh, seed, and retained modes;
it is a resolved linear-density maximum, not a rigorous error estimate of EB
or the nonlinear density of the displaced particles. The search assumes the
usual growing cold-matter regime. A finite scan can miss a transient crossing
and recrossing between trial epochs; decrease the maximum step if examining
nonmonotonic growth. Power-only evolution assumes that each cold density
transfer retains its sign: `sqrt(P)` cannot recover a sign reversal of a signed
transfer function. Early baryon-dominated models with acoustic sign reversals
need a signed transfer interface. No extrapolation beyond the EB spectrum's
k range is used.

For box volume `V=L^NDIM`, the unnormalized FFT density coefficient obeys
`E[|F_k|^2]=N^2 P(k,a)/V`, with `N=M^NDIM`. Each independent mode uses a complex
Gaussian of unit variance; its opposite mode is the exact conjugate. The
counter-based seed depends on the global conjugate pair, so phases are unchanged
across epochs and locality decompositions. The zero mode and all modes touching
a Nyquist plane are removed, preserving zero mean and Hermitian gradients.
Other modes in the Fourier cube are retained. Spectra use CAMB's synchronous
gauge with `k` in `h/Mpc` and `P` in `(Mpc/h)^3`, normalized to the selected
component's own mean density. This is the usual periodic Newtonian Zel'dovich
setup; it does not implement relativistic gauge transformations on horizon
scales. See [CAMB's transfer conventions](https://camb.readthedocs.io/en/latest/transfer_variables.html)
and the [relativistic initial-condition discussion](https://arxiv.org/abs/1505.04756).

The displacement and its derivative are

\[
 \boldsymbol\Psi_{\mathbf k}(a)=
       {i\mathbf k\over k^2}\delta_{\mathbf k}(a),\qquad
 \mathbf x=\mathbf q+\boldsymbol\Psi,
 \qquad \mathbf q_i/L=(\mathbf j_i+1/2)/M.
\]

A half-cell Fourier phase evaluates the fields at the lattice centers.
Positions are wrapped into the fixed-point unit box `[0,1)`. Stored particle
velocities are **`d(x/L)/d ln(a)`**, dimensionless. They use the finite difference
of `sqrt(P(k,a))` at neighboring scale factors, retaining the same mode phases.
This follows the scale-dependent linear growth returned by CAMB instead of
imposing a scale-independent growth factor. `growthDerivativeStep` controls
the separation in `ln(a)`, lies in `[1e-6,0.1]`, and defaults to `1e-3`.
The centered difference is second order. Near `a=1`, a second-order three-point
backward difference keeps the derivative at the handoff epoch without querying
future CAMB redshifts. Direct CAMB Newtonian-gauge velocity transfer functions
are not used.

CAMB's native spectrum files carry about six significant decimal digits, so
smaller finite-difference steps can amplify output rounding. The default
`growthDerivativeStep=1e-3` balances this with truncation error; reducing it
does not automatically improve velocities. The generic interface permits
steps down to `1e-6` for solvers that return more precise spectra.

For a future physical-time solver, convert the stored velocity to proper
peculiar velocity with

\[
 \mathbf u\,[\mathrm{km/s}]=
    a H(a)\,[\mathrm{km/s/Mpc}]\,(L/h)\,[\mathrm{Mpc}]
    {d(\mathbf x/L)\over d\ln a}.
\]

The handoff includes equal particle mass in solar masses/h, computed from the
selected component's present mean density and box volume. Every particle has
rung zero and `isGhost=false`; displaced particles are redistributed to their
Cartesian owners before return. Cell and rung ranges are filled, with ghost
ranges empty. Work is staged in new objects, leaving pre-existing lists untouched.
Preparation and EB errors are shared collectively. As with the FFT transport,
runtime/network failures during an in-progress exchange are not recoverable.

Local mode generation, density-spectrum assembly, displacement/velocity
assembly, and particle filling use coarse HPX tasks over disjoint index ranges.
The minimum chunk size is 1024 entries, with at most four chunks per worker;
small ranges and one-worker runs execute inline. Density maxima are computed
per chunk and combined before the existing global reduction. Each phase joins
all tasks before entering the next collective or releasing its arrays, including
when a task fails. `FpeGuard` stays inside local task arithmetic scopes. The
counter-based seed, per-element arithmetic, spectrum requests, and collective
ordering are unchanged; progress callbacks run on the controlling task.

Link `cosmictiger_initialization` for the initializer. The focused test target is
`cosmictiger_initialization_tests`; the `cosmictiger.initialization` and
`cosmictiger.initialization_parallel` CTest entries exercise one and two workers.
Set runtime and build resources according to the current shared budget,
including HPX service threads and the single-threaded CAMB child process.
