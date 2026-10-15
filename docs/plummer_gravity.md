# Full Cartesian multipoles and Plummer gravity

`gravity.softening.model` selects `ferrers` (the default) or `plummer` for a
new simulation. Plummer runs use full symmetric Cartesian moments, Green
derivatives, and local expansions throughout PMMM. Every trace is retained.
The original STF operator family remains available for Ferrers/Newtonian
gravity.

## Selecting the kernel

```sh
./release/cosmictiger --gravity.softening.model=plummer \
    --gravity.softening.cellFraction=0.05
```

The equivalent INI settings are:

```ini
[gravity.softening]
model = plummer
cellFraction = 0.05
```

`cellFraction` defaults to `0.05` and must be greater than zero and at most
one. The physical interpretation depends on the model:

| Model | Length defined by `cellFraction * evolutionCellWidth` | Kernel behavior |
| --- | --- | --- |
| Ferrers | Compact support radius `q` | Exactly Newtonian outside `q` |
| Plummer | Plummer length `epsilon` | Softened at every finite separation |

These lengths are expressed in the simulation's comoving unit-box coordinates.
There is no implicit conversion between a Ferrers support radius and a
Plummer length. `gravity.softening.kernelOrder` selects Ferrers F0 through F3,
defaulting to F2; it does not change the Plummer kernel.

For a unit source mass, Plummer potential and acceleration are

\[
g_\epsilon(\mathbf r)=-\frac{1}{\sqrt{r^2+\epsilon^2}},\qquad
\mathbf a_\epsilon(\mathbf r)
=-\frac{\mathbf r}{(r^2+\epsilon^2)^{3/2}}.
\]

The same model and length feed near-cell P2P, periodic PMMM, and the direct
periodic reference solve. CPU/CUDA multipole and kick choices continue to use
the independent `execution.cuda.multipole` and `execution.cuda.kick` options;
FFT and M2L remain on the CPU. `execution.cuda.direct` independently selects
the direct reference backend.

The root initializes the model and length in immutable `GlobalConstants`.
Evolution checkpoint archive version eight saves the model. Older supported
archives load as Ferrers. A restart retains its saved model and length;
an explicitly different `gravity.softening.model` is rejected. Execution
backend choices remain independent of that saved physical state.

Implementation: [softening selection](../cosmictiger/gravity/softening.hpp),
[command-line options](../src/command_line_options.cpp),
[evolution checkpoints](../src/evolution/evolution_checkpoint.cpp).

## Representation and operator API

Include
[`cosmictiger/gravity/symmetric/operators.hpp`](../cosmictiger/gravity/symmetric/operators.hpp)
and use namespace `cosmictiger::gravity::symmetric`. `Multipole<Real, order>`
and `Expansion<Real, order>` are fixed-size arrays with

\[
N(p)=\binom{p+3}{3}
\]

coefficients through total rank `p`. The order is a template constant.
Coefficients are rank-major, then increasing `nz`, then increasing `nx`, with
`ny = rank - nz - nx`. `index(nx, ny, nz)`, `coefficientCount(order)`, and
`exponents<order>()` expose this layout. Components such as `zz`, `xzz`, and
`zzzz` are stored explicitly, rather than reconstructed through a trace
constraint.

For source scale `h_s` and target scale `h_t`, the stored quantities are raw
moments and raw derivatives:

\[
M_\beta=\sum_j m_j
\left(\frac{\mathbf x_j-\mathbf c_s}{h_s}\right)^\beta,
\qquad
L_\alpha=h_t^{|\alpha|}\partial^\alpha\Phi(\mathbf c_t).
\]

Neither array absorbs factorials. Operators insert the appropriate
multi-index factorials. In particular, M2L includes every pair
`|beta| <= sourceOrder`, `|alpha| <= localOrder`, requiring Green derivatives
through `sourceOrder + localOrder`:

\[
L_\alpha\mathrel{+}=
\sum_{|\beta|\le p}
\frac{(-1)^{|\beta|}}{\beta!}
\left(\frac{h_s}{h_t}\right)^{|\beta|}
M_\beta\,
h_t^{|\alpha+\beta|}
\partial^{\alpha+\beta}g_\epsilon(\mathbf c_t-\mathbf c_s).
\]

The operator family is host/device compatible:

| Operation | API |
| --- | --- |
| P2M | `addPoint<Real, order>` |
| M2M | `addM2M<Real, order>` |
| P2L | `addP2L<Real, order>` |
| M2L | `addM2L<Real, sourceOrder, localOrder>` |
| L2L | `addL2L<Real, order>` |
| L2P | `evaluateL2P<Real, order>` |
| M2P | `evaluateM2P<Real, order>` |
| P2P | `evaluateP2P<Real>` |
| Full Green derivatives | `greenDerivatives<Real, order>` |

P2M, M2M, and L2L translate source moments or a finite local polynomial;
they do not depend on the softening kernel. P2L, M2L, M2P, and P2P use
the supplied Plummer length, with zero selecting the Newtonian free-space
kernel. `greenDerivatives(separation, epsilon, targetScale)` returns
`targetScale^rank` times each complete derivative. Positive epsilon makes
the free-space kernel regular at coincident coordinates. Physical self
exclusion remains the interaction caller's responsibility.

```cpp
constexpr int kSourceOrder = 5;
constexpr int kLocalOrder = 6;
symmetric::Multipole<double, kSourceOrder> moments{};
symmetric::Expansion<double, kLocalOrder> local{};
symmetric::addPoint<double, kSourceOrder>(moments, mass, sourceOffset, spacing);
symmetric::addM2L<double, kSourceOrder, kLocalOrder>(
    local, moments, targetMinusSource, spacing, spacing, epsilon);
auto const field = symmetric::evaluateL2P<double, kLocalOrder>(
    local, targetOffset, spacing);
```

Translation input and output arrays must be distinct. M2M displacement is
`oldCenter - newCenter`; L2L displacement is `newCenter - oldCenter`.
P2L/M2L/M2P separation points from the source to the target. Returned
acceleration is minus the physical potential gradient.

The batched
[`symmetric::launchM2pCuda`](../cosmictiger/gravity/symmetric/m2p_cuda.hpp)
API evaluates full source tensors at the configured `kMultipoleOrder` and
returns an HPX completion future. Each `M2pCudaSample` supplies its source
pointer, separation, source scale, and softening length. Callers retain the
source/output buffers and supplied CUDA stream until the future is ready.
The PMMM evolution path uses L2P and P2P and has no M2P stage.

## Full periodic Plummer Ewald sum

Let the cubic box have period `L`, volume `V=L^3`, splitting parameter
`alpha=alphaBox/L`, and wave vectors `k_m=2*pi*m/L`. Begin with the Gaussian
integral

\[
g_\epsilon(\mathbf r)
=-\frac{1}{\sqrt\pi}\int_0^\infty
t^{-1/2}\exp[-t(r^2+\epsilon^2)]\,dt.
\]

Split this integral at `alpha^2` and apply Poisson summation to its lower
part. Define

\[
R_{\mathbf n}=\sqrt{|\mathbf r+L\mathbf n|^2+\epsilon^2},\quad
z=\epsilon^2\alpha^2,\quad
b_{\mathbf m}=\frac{|\mathbf k_{\mathbf m}|^2}{4\alpha^2},
\]

\[
J(b,z)=\int_0^\infty
\exp\!\left[-v-\frac{zb}{b+v}\right]dv,\qquad
E_2(z)=\int_1^\infty \frac{e^{-zt}}{t^2}\,dt.
\]

The zero-mean periodic Green function is

\[
G_\epsilon(\mathbf r)=
-\sum_{\mathbf n\in\mathbb Z^3}
\frac{\operatorname{erfc}(\alpha R_{\mathbf n})}{R_{\mathbf n}}
-\frac{4\pi}{V}\sum_{\mathbf m\ne0}
\frac{e^{-b_{\mathbf m}}J(b_{\mathbf m},z)}{|\mathbf k_{\mathbf m}|^2}
e^{i\mathbf k_{\mathbf m}\cdot\mathbf r}
+\frac{\pi}{V\alpha^2}E_2(z).
\]

The final constant cancels the spatial mean of the real-space sum. The
reciprocal sum excludes its zero wave vector and pairs opposite waves to
produce real coefficients. `J(b,0)=E_2(0)=1` recovers the Newtonian Ewald
weights. For positive epsilon both functions change: applying Plummer only
to the primary interaction would omit its periodic tails and give a
different Green function.

Cartesian real-space derivatives follow from derivatives in
`s = r^2 + epsilon^2`:

\[
\frac{d^j}{ds^j}
\left[-\frac{\operatorname{erfc}(\alpha\sqrt{s})}{\sqrt{s}}\right]
=\frac{(-1)^{j+1}}{\sqrt\pi}
\frac{\Gamma(j+\tfrac12,\alpha^2s)}{s^{j+1/2}}.
\]

All Cartesian product-rule terms, including every trace, are retained.
Reciprocal derivatives multiply each mode by `(i*k)^multiIndex`.
The implementation memoizes powers and constructs combinatorial tables at
compile time. Reciprocal radial weights and Cartesian wave coefficients
are cached in the evaluator before its cell loop.

The full free-space trace is

\[
\nabla^2g_\epsilon(\mathbf r)
=\frac{3\epsilon^2}{(r^2+\epsilon^2)^{5/2}}.
\]

Consequently, the periodic trace contains the sum of all Plummer density
tails minus `4*pi/V`. The full Ewald derivatives already include this
background. Do not append the STF path's separate quadratic background
correction to full Cartesian expansions.

[`PmmmPlummerGreenEvaluator<maxRank, neighborWidth>`](../cosmictiger/gravity/pmmm/pmmm_plummer_green.hpp)
supports ranks through 18 and neighbor widths one or two. `evaluate(offset)`
wraps integer mesh offsets and removes the primary interaction in the
selected near cube. `evaluateCells(cellSeparation, subtractPrimary)` accepts
continuous offsets measured in cells and an explicit primary-exclusion
choice. Returned rank-`n` coefficients include `spacing^n`.
The default image and reciprocal cube cutoffs are both four, with
`alphaBox=1.5`; these are finite Ewald truncations, not an exact infinite
sum at arbitrary cutoff choices.

The shared
[`pmmm_plummer_ewald.hpp`](../cosmictiger/gravity/pmmm/pmmm_plummer_ewald.hpp)
helpers evaluate `E_2` by a small-argument series or continued fraction and
`J` by positive Gauss-Legendre quadrature. The omitted integral tail beyond
`v=64` is bounded by `exp(-64)`. The direct CPU/CUDA reference reuses these
weights and precomputes weights for its usual wave-number range.

## Near cells and self interactions

For a near-cell displacement, PMMM uses

\[
G_\epsilon^{\mathrm{far}}(\mathbf r)
=G_\epsilon(\mathbf r)-g_\epsilon(\mathbf r).
\]

Only the primary source is removed. All periodic images and reciprocal
terms remain, including at the origin. Each removed near-cell contribution
is evaluated by direct Plummer P2P. The primary real-space combination is
evaluated as `+erf(alpha*sqrt(s))/sqrt(s)`, using a regular lower-integral
series when needed; this avoids subtracting nearly equal high derivatives
or forming huge inverse epsilon powers at the origin.

An individual particle's own primary contribution is omitted, but its
periodic images and the chosen potential gauge remain. At the origin, the
primary correction is `erf(alpha*epsilon)/epsilon`, whose zero-epsilon
limit is `2*alpha/sqrt(pi)`. Cubic symmetry makes the self acceleration zero.
This convention is shared by PMMM and the direct reference solve.

## PMMM storage and coefficient contraction

[`PmmmPeriodicConvolution`](../cosmictiger/gravity/pmmm/pmmm_convolution.hpp)
selects the full
[`PmmmPlummerConvolution`](../cosmictiger/gravity/pmmm/pmmm_plummer_convolution.hpp)
path for `SofteningModel::plummer`. At each spatial Fourier mode the
coefficient contraction is

\[
\widehat L_\alpha(\mathbf k)
=\sum_{|\beta|\le p}
\frac{(-1)^{|\beta|}}{\beta!}
\widehat M_\beta(\mathbf k)
\widehat G_{\alpha+\beta}(\mathbf k).
\]

This is a correlation over the three Cartesian coefficient indices,
without complex conjugation. The full-tensor path evaluates this sum
directly at each spatial Fourier mode. Spatial FFTs perform the convolution
across mesh cells, while the Cartesian coefficient contraction stays local
to each mode.

The mesh stores both the existing STF arrays and full Cartesian arrays in
each `Cell`. The source/local storage capacities 8/9 contain 165 moments
and 220 local coefficients: 385 additional floats, or 1540 bytes per
stored cell, including ghost cells and cells in Ferrers runs. The default
active source/local ranks 5/6 use 56 moments and 84 local coefficients.
Their rank-11 Green tensor has 364 coefficients; runtime order 9 uses a
rank-17 Green tensor with 1140 coefficients, increasing the cached spectrum
and FFT workspace requirements relative to the STF representation.
`gravity.fmmOrder=p` selects source/local ranks `p-1`/`p` at runtime for
both softening models. Every build accepts orders 2–9 and defaults to
6. Lower orders reduce the Green and source/local FFT coefficient counts;
cell storage retains its compiled capacity. Checkpoints retain the selected
order, and restarts reject explicit changes. The former CMake order
selections have been retired so every executable includes the full runtime
range. Reconfigure and rebuild existing build directories to update all
cell-layout users.

## Numerical references and tests

The incomplete-gamma derivatives use the
[NIST DLMF incomplete-gamma identities](https://dlmf.nist.gov/8.8).
The independent unsplit Fourier reference follows from the
[modified Bessel integral, DLMF 10.32.10](https://dlmf.nist.gov/10.32.E10):

\[
\widetilde g_\epsilon(\mathbf k)
=-\frac{4\pi\epsilon}{|\mathbf k|}K_1(\epsilon|\mathbf k|),
\qquad \mathbf k\ne0.
\]

[`tests/pmmm/plummer_green.cpp`](../tests/pmmm/plummer_green.cpp)
compares full Cartesian Ewald derivatives with that independent Bessel
Fourier sum, varies the Ewald splitting parameter, checks primary removal,
checks the tiny-epsilon origin limit, and compares derivative orders with
continuous finite differences. The
[PMMM convolution test](../tests/pmmm/plummer_convolution.cpp) checks
both near widths, coefficient indexing, cached filter reuse, and preserved
ghost expansions. Runtime validation results belong with the actual build
and backend used; host tests alone do not establish CUDA or distributed
MPI execution.

Validation on 2026-10-06 completed with one HPX locality and one worker:

- The Release application and affected CPU/CUDA test targets built successfully.
- All 14 symmetric-operator and Plummer-Green tests passed in both Release
  and Profile after removing the experimental M2L implementation. Profile
  enabled numerical contracts and floating-point traps, used an APEX-enabled
  HPX installation, and built without CUDA.
- The selected Release integration/regression run passed 48 tests, including
  all four CPU/CUDA P2M and kick combinations, full trace moments in sparse
  and crowded cells, batched full-tensor CUDA M2P, softened periodic direct
  forces, both PMMM near widths, and existing Ferrers paths. GPU execution
  used an RTX 3090. Two validation-only tests and one physical multi-locality
  test were skipped in that run.
- All 15 checkpoint tests passed, including Plummer round trips and readers
  for checkpoint versions 2 through 7. Command-line parsing and local
  GlobalConstants publication/serialization also passed.
- Formatting and whitespace checks passed. Physical multi-process execution
  was not tested.

The integration tests caught and corrected a CPU/CUDA discrepancy in the
separate second-moment trace field during full-moment construction. Both
backends now replace that field together with the full moments.
