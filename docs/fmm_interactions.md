# Cell-level FMM interactions

`cosmictiger/gravity/fmm/fmm_interactions.hpp` provides batched cell-level
FMM interactions on CPU and CUDA. They are ports of the CPU and CUDA kernels
of the old tree code (`cosmictiger_old`, `src/gravity.cpp` and
`src/gravity.cu` at `eb17708`). They are intended as an alternative to direct
P2P for selected near-region cell pairs. **They are not yet connected to the
kick, the evolution path, or any option.**

| Old kernel | Operation | CPU entry | CUDA entry |
| --- | --- | --- | --- |
| `cpu_gravity_cc` / `cuda_gravity_cc_direct` | Source cell multipole to target cell expansion (M2L) | `addM2lCpu` | `launchM2lCuda` |
| `cpu_gravity_cp` / `cuda_gravity_cp_direct` | Source cell particles to target cell expansion (P2L) | `addP2lCpu` | `launchP2lCuda` |
| `cpu_gravity_pc` / `cuda_gravity_pc_direct` | Source cell multipole to target cell particles (M2P) | `addM2pCpu` | `launchM2pCuda` |

The old `pp` kernels are direct P2P, which `gravity/direct` already provides
with Ferrers softening. The old `cc_ewald` kernel is replaced in this code by
the PMMM far field, so neither was ported.

## Conventions

The kernels call the generated STF operators (`M2lKernel`, `P2lKernel`,
`M2pKernel`) directly; the generator now emits P2L and M2L as
`__host__ __device__` so that the CPU and CUDA backends evaluate the same
polynomial. Runtime FMM order `p` (2 through 9) uses source rank `p-1` and
local rank `p`. Multipoles and expansions use the `CellMultipole` and
`CellExpansion` layouts, with the cell spacing as both scales. Only the active
ranks are read or accumulated, and the expansion's appended Laplacian is left
unchanged because the exterior Newtonian contributions are harmonic.

Each separation is the target cell center minus the source cell center, for
the periodic image chosen by the caller. It is never wrapped, so it remains
correct when the near region spans more than half of a small mesh. Particle
offsets are taken from their own cell center with the fixed-point minimum
image. The kernels evaluate the unsoftened exterior Newtonian kernel; the
caller is responsible for choosing pairs that satisfy an acceptance criterion
and for keeping softened pairs on the direct path.

The CPU backend evaluates in double precision and adds each target's complete
sum once. The CUDA backend evaluates in single precision: one warp per target
for M2L and P2L, with a fixed-order butterfly reduction, and one block per
target cell for M2P. All outputs are accumulated. Targets in one request must
not alias.

## Defects found in the old kernels

These defects were found during the port; none is carried into the new code.

1. CPU `pp` aborted unconditionally (`"Called CPU pp"`). Its unreachable body
   added an uninitialized `rinv1` to the potential when `do_phi` was false,
   because `gsoft` sets the potential only when requested. A local
   `const int& count = maxi;` also shadowed the particle count.
2. CUDA `pp` had the same uninitialized-potential problem in its sink-per-lane
   branch, where `F.phi += pot` was unconditional. It also lacked a warp
   synchronization between finishing reads of the shared source tile and the
   next `memcpy_async` into that tile, a write-after-read race under
   independent thread scheduling.
3. CUDA `cp` re-initialized the shared `cuda::barrier` at the top of every
   batch without first synchronizing the warp, so lane 0 could initialize it
   while other lanes were still returning from the previous phase.
4. CPU `cp` and `pp` padded source lanes with position 0 and mass 0. A padded
   lane coinciding with the sink gives `0 * inf`, an invalid operation.
5. CPU `cp` charged `count * greens_function(...)` flops for every SIMD batch,
   using the total source count instead of the batch count.
6. `cp` ignored the particle mass (unit masses were assumed) and `do_phi`.
7. CPU `cc` and `pc` kept scratch in `static thread_local` vectors, which is
   only safe while no HPX suspension occurs between filling and use.

## Tests

`cosmictiger.fmm_interactions` (`tests/fmm/`) checks every order against the
generated reference operators, ninth-order fields against a direct Newtonian
sum (and fails at order 3), minimum-image offsets across the periodic
boundary, accumulation and preservation of inactive coefficients, parallel
targets, request validation, and, in CUDA builds, device results against the
CPU at every order. Device coefficients of rank 8 and 9 have a 10x looser
tolerance because single-precision cancellation costs about one digit there.

## Warp partition of interaction lists

`cosmictiger/cuda/cuda_warp_partition.cuh` provides `partitionWarp` and
`appendWarp`, device primitives for splitting one entry per lane among several
output lists. They replace the old walk's `compute_indices_array`, which ran a
shuffle prefix scan for each of the cc, leaf, cp, pc, and next lists. Because
each entry selects at most one list, one `__ballot_sync` per list gives every
count, and `__popc(ballot & lanemask_lt)` gives each lane's stable offset.
`appendWarp` writes nothing when any list would overflow, so the caller can
grow the lists and retry the same entries. Like the FMM kernels, these are not
yet used by any walk.

Microbenchmark on an RTX 3090 (sm_86, CUDA 12.8): five lists, 64 Mi entries,
one-sixth of them assigned to no list.

| Version | Time | Instructions | Registers |
| --- | --- | --- | --- |
| Shuffle prefix scan (old) | 1.53 ms | 781 M | 39 |
| Ballot and population count | 0.95 ms | 239 M | 26 |

Both versions issue identical global stores. Keep the per-list loops fully
unrolled. Without `#pragma unroll` the compiler kept the per-list sizes in
local memory, which increased DRAM writes about fivefold and made the ballot
version 2.4 times slower than the scan.

`cosmictiger.cuda_warp_partition` checks one, five, and nine lists against a
host stable partition, including partial final batches and overflow.
