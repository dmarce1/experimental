# HEALPix and lightcone output

`cosmictiger/healpix/healpix.hpp` provides the project's scalar and batched SIMD
HEALPix implementation. Link `cosmictiger_healpix`. `Healpix(nside, ordering)`
supports RING and NESTED pixel numbering, direction-to-pixel lookup,
pixel-center directions, and exact integer ordering conversion. Indices are
64-bit. NESTED requires a power-of-two `nside`; RING also accepts other positive
resolutions. The maximum is `2^29`.

`vectorToPixel` accepts a finite, nonzero Cartesian direction of any magnitude.
`vectorsToPixels` accepts spans of directions and output indices. It uses native
double-precision SIMD arithmetic on supported x86 targets, scalar `atan2` and
integer finishing, and a scalar tail/fallback. `nativeLaneCount()` reports the
compiled lane width. This is CPU SIMD; it does not launch GPU kernels. The
polar formulas retain transverse information when normalized z rounds to one.
Compile these routines with IEEE arithmetic and contraction disabled, as the
CMake target does, to retain matching scalar and SIMD boundary decisions.

The equations and integer indexing adapt the HEALPix reference implementation.
The affected files preserve its GPL notices and credit Martin Reinecke,
the Max-Planck-Society, and the original HEALPix authors, including the NASA/JPL
project. The scientific reference is Górski et al. (2005), ApJ 622, 759,
[HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere](https://doi.org/10.1086/427976).
See also the [official HEALPix project](https://healpix.sourceforge.io/).

## Enabling lightcone output

The module is disabled by default. For example, add these options to a normal
evolution command:

```text
--lightcone.enabled=true --lightcone.nside=64
--lightcone.maximumRadius=1 --lightcone.outputPrefix=results/lightcone
```

The same settings work in an INI `[lightcone]` section. `observerX`, `observerY`,
and `observerZ` default to zero and must lie in `[0,1)`. Coordinates and
`maximumRadius` are comoving box lengths. `observationTime` is physical years
relative to the expanding present-day `a=1` epoch, with zero as the default.
The observer epoch must follow the original initial-condition epoch. The run
can stop before or after that observation time; only the past lightcone is
sampled. `nside` must be a power of two because output uses NESTED ordering.

For a drift with fixed canonical momentum p, the code finds crossings of

```text
|q_begin + image - observer + p * (D(tau) - D_begin)|
    = c_box * (tau_observer - tau)
D(tau) = integral(d tau / a)
c_box = 299792.458 / (100 * boxLength)
```

`boxLength` is in Mpc/h and conformal time is in H0^-1. The root solve evaluates
the same cosmological background and tick-anchored drift integral used by the
position update, including partial steps introduced by snapshots. Each interval
owns `(begin,end]`. Only interior ownership ranges contribute; ghost copies are
excluded. All relevant periodic images are enumerated, including when the
radius exceeds one box length. Images intentionally represent distinct sky
objects. The geometric model is the Euclidean, periodically replicated
simulation volume.

Sampling happens before wrapping and redistribution, after previous CUDA work
has completed, and works with either CPU or CUDA drift selection. It reads
particles without changing their trajectories. The root solve requires
subluminal peculiar motion. A crossing exactly at the observer is omitted
because it has no sky direction.

## Files, maps, and restarts

Each invocation prints a unique output series prefix. Each locality and drift
segment writes a crossing CSV and sparse map CSV. A `.complete` marker is
published last; consumers must require it before reading a pair. Metadata records the observer,
time interval, units, NESTED ordering, and resolution. Catalogs include the
crossing time, scale factor, observer-relative position, periodic image,
HEALPix pixel, and peculiar velocity `p/a`. The original columns retain unit-box
positions and H0*L velocities. CSV version 2 appends explicitly labeled output
views: comoving positions and radii in plain Mpc, peculiar velocities in km/s,
proper positions and radii in cm, and peculiar velocities in cm/s. These views
do not include Hubble flow. Physical positions use each object's crossing scale
factor, not the scale factor at either end of the drift. The writer converts
the already computed `p/a` velocity without dividing by `a` again.

The appended `conformalTimeSeconds` and `tickOffsetSeconds` columns express the
existing conformal clock and tick offset in seconds; they are not elapsed
proper time. Unit metadata records the comoving box size, the H0^-1 time scale,
and the velocity conversion. The Mpc columns have no hidden `h` factor even
though the input `boxLength` is specified in Mpc/h. A local particle index is only
a diagnostic index for that segment, not a persistent particle identifier.

The map contains occupied pixels only, with particle counts and sums of
`1/radius^2` in inverse squared box lengths. Version 2 retains those columns and
appends weights in inverse squared comoving Mpc and inverse squared proper cm.
The proper weight is the sum of `1/(a_crossing * radius * L_cm)^2`, with each
object's own crossing epoch applied before summation. These are raw counts and weights,
not a normalized density, lensing convergence, or halo catalog. To form a
full-sky map, sum matching pixel rows across the localities and desired time
segments of a series. Missing pixels have zero weight. No distributed dense
map or halo/ROCKSTAR processing is performed.

Evolution diagnostics likewise preserve their original columns and append
explicit astronomical and CGS units. Kinetic energy, gravitational potential
energy, signed expansion work, the conserved energy sum, and absolute energy
drift all use the same constant energy conversion: solar masses times squared
km/s, or erg. Forces use solar masses times squared km/s per Mpc, or dyne.
These quantities retain the existing synchronized peculiar-energy and force
definitions; the conversion does not add expansion velocity or another power
of the scale factor. `physicalTimeSeconds` retains the expanding present-day
epoch as zero, and `daDtPerSecond` is the signed physical scale-factor derivative.

Checkpoint version 5 preserves the lightcone configuration and conformal
observer epoch; versions 2–4 load with sampling disabled. Restart retains saved
settings unless explicitly overridden through CLI/INI. Rung-zero extension
does not move the observer epoch. A restart writes a new output series; when
combining it with an earlier series, keep only earlier segments ending at or
before the selected checkpoint, then append the restart segments. Replaying an
older checkpoint must not be combined with the superseded later segments.

Cost grows with the number of owned particles and periodic images. The module
stores one segment's crossings and sparse occupied pixels, so high image counts
or very long drift intervals can still require substantial memory. Roots are
solved with double precision and cosmological integration accuracy; snapshot
splitting can also introduce the existing fixed-point position quantization.

## PNG sky images

After the run completes, render its completed catalogs with:

```bash
python3 tools/render_lightcone.py output/my-run/lightcone.*.crossings.csv \
  --output-directory output/my-run/png --title 'Flat ΛCDM lightcone'
```

The renderer requires NumPy and Matplotlib and produces `lightcone_counts.png`
and `lightcone_weights.png`. It combines completed segments from one invocation,
rejects mixed run prefixes, and skips segments without a `.complete` marker.
PNGs use equal-solid-angle longitude/sin(latitude) bins in a Mollweide projection,
with logarithmic color scales and gray empty bins. They visualize the crossing
catalog; the original NESTED HEALPix CSV maps remain available for analysis.
Keep each simulation's output in its own directory so shell wildcards select
one series. The default angular image grid is 128 by 64; use
`--longitude-bins=256` for a 256 by 128 grid.
