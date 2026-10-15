# Friedmann background evolution

Include `<cosmictiger/friedmann.hpp>` and link the CMake target
`cosmictiger_cosmology`. `FriedmannCosmology` solves for the dimensionless scale
factor and returns its derivatives with respect to both physical time `t` and
conformal time `eta`.

All four density parameters are supplied explicitly at `a=1`:

\[
H(a)^2=H_0^2\left(\frac{\Omega_r}{a^4}+\frac{\Omega_m}{a^3}
+\frac{\Omega_k}{a^2}+\Omega_\Lambda\right),\qquad
dt=a\,d\eta.
\]

Dark energy is a cosmological constant (`w=-1`), as in the legacy `cosmos_`
routines. Radiation and matter must be nonnegative. Curvature may have either
sign; negative `omegaCurvature` describes positive spatial curvature. Signed
dark energy is also supported, including recollapse with negative Lambda.
The Omegas must sum to one within `1e-10`; they are never silently normalized.
`hubbleConstant` must be finite and positive, in inverse units of the times
passed to the solver. For example, supplying H0 in inverse seconds means
supplying time in seconds. The default H0 is one, so default times are in
Hubble-time units.

```cpp
#include <cosmictiger/friedmann.hpp>

cosmictiger::BackgroundCosmologyParameters parameters;
parameters.hubbleConstant = 1.0;
parameters.omegaRadiation = 0.0001;
parameters.omegaMatter = 0.3;
parameters.omegaDarkEnergy = 0.69;
parameters.omegaCurvature = 0.0099;
cosmictiger::FriedmannCosmology cosmology(parameters);

auto initial = cosmology.state(0.01);
auto physical = cosmology.advance(initial, 0.02);
auto conformal = cosmology.advance(initial, 0.02,
    cosmictiger::CosmologicalTime::conformal);

// For a single query, initialize and advance in one call:
auto result = cosmology.evolve(0.01, 0.02);
double a = result.scaleFactor;
double adot = result.physicalDerivative;       // da/dt
double aprime = result.conformalDerivative;   // da/deta = a*adot

// Continue from the entire returned state, retaining its signed velocity:
result = cosmology.advance(result, 0.02);
```

The two `0.02` intervals in the example use different time coordinates and
therefore generally lead to different final scale factors. Every interval is
measured from its supplied initial state. The caller chooses the clock origin;
the API does not assume that either clock is zero at the Big Bang. Negative
intervals evolve backward in time. `state(a, ExpansionBranch::contracting)`
initializes a contracting solution. `evolve` accepts the same branch as its
fourth argument. Returned states can be advanced using either clock.

The solver evolves the signed rate through zero, so a closed universe passes
smoothly through maximum expansion and into contraction. For example:

```cpp
cosmictiger::BackgroundCosmologyParameters closedParameters;
closedParameters.omegaMatter = 2.0;
closedParameters.omegaDarkEnergy = 0.0;
closedParameters.omegaCurvature = -1.0;
cosmictiger::FriedmannCosmology closed(closedParameters);
auto contracting = closed.evolve(1.0, 5.2); // Beyond maximum expansion.
```

## Accuracy and validation

Integration uses the adaptive Dormand–Prince 5(4) Runge–Kutta method in
dimensionless time `s=H0*t` or `s=H0*eta`. Its variables are `a` and
`u=(da/deta)/H0`, with constraint

\[
u^2=Q(a)=\Omega_r+\Omega_m a+\Omega_k a^2+\Omega_\Lambda a^4.
\]

The equations are

\[
\begin{array}{lll}
\text{physical:}& da/ds=u/a,&du/ds=\Omega_m/(2a)+\Omega_k+2\Omega_\Lambda a^2,\\
\text{conformal:}&da/ds=u,&du/ds=\Omega_m/2+\Omega_k a+2\Omega_\Lambda a^3.
\end{array}
\]

These acceleration equations remain regular at a turnaround. Radiation enters
through the initial Friedmann constraint; its absence from `du/ds` is the
result of the conformal velocity transformation. The conformal equations and
time relation also appear in [Boyle and Turok, equations (2), (4a), (5), and
(6)](https://arxiv.org/pdf/2210.01142).

`FriedmannOptions` configures local error control:

- `relativeTolerance`, default `1e-12`, controls the relative scale-factor and
  expansion-rate errors. The scale factor always uses a relative error scale,
  including at small `a`.
- `absoluteTolerance`, default `1e-14`, provides a velocity error floor near
  a turnaround. It multiplies the local component amplitude
  `sqrt(OmegaRadiation + OmegaMatter*a + abs(OmegaCurvature*a*a) +
  abs(OmegaDarkEnergy*a*a*a*a))`; it is not an absolute floor on `a`.
- `maxSteps`, default `100000`, counts both accepted and rejected attempts.

The relative tolerance must be at least `32*epsilon(double)` and less than one;
the absolute tolerance must be nonnegative and less than one. Local tolerances
are not a guarantee of the global error over an arbitrarily long interval.
The solver also monitors the Friedmann constraint and reports lost accuracy.
Both the solver source and its tests are compiled with `-fno-fast-math` in every
build mode to retain finite-value checks and accurate cancellation.

Input states must have finite positive `a`, consistent physical/conformal
derivatives, and satisfy the density constraint within numerical tolerance.
`advance` does not mutate the supplied state. Invalid inputs throw
`std::invalid_argument`; initializing where `H(a)^2<0` throws
`std::domain_error`. Evolution that reaches `a=0`, a finite conformal future
horizon, an unrepresentable numerical value, or a convergence/step limit throws
an exception. It never returns a partial result as a completed interval.

The dedicated tests compare against exact radiation, matter, curvature,
Lambda, matter/radiation, flat matter/Lambda, and recollapsing solutions. They
also check mixed densities, both clocks and branches, reversal, small `a`,
input validation, and singular limits:

```sh
cmake --build release --target cosmictiger_friedmann_tests --parallel 2
ctest --test-dir release -R '^Friedmann\.' --output-on-failure
```

Configure with `BUILD_TESTING=ON` to make the test target available.
