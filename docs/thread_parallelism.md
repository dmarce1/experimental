# Thread parallelism: current use and opportunities

Source audit: 2026-10-05. This describes the current working tree, including
uncommitted and untracked code; HEAD `d279f9c` alone does not contain that code.
The review covered application sources, public headers, tests, build settings,
and the CAMB integration boundary. Vendored CAMB internals were not audited.
Source links identify the implementation and line at the time of the audit.

CosmicTiger already uses HPX for concurrent locality actions, parallel FFT
loops, asynchronous message delivery, and collective synchronization. Particle
redistribution, bulk cell work, and initialization loops use coarse HPX tasks.
Gravity target and CPU sink batches use coarse HPX tasks as described in
section 3.
Ghost packing and unpacking now also use bounded particle chunks.
Near-field preparation now overlaps the PMMM expansion solve before one combined
kick. The remaining priorities are independent two-input FFT forwards and
Silo block conversion. Repeated FFTW plan setup was removed
in the 2026-10-05 follow-up described in section 2. Particle counting/packing,
received-cell partitioning, rung sorting, and cell multipoles were subsequently
parallelized as described below. Pencil/slab redistributions now overlap bounded
peer transfers and receive scatter through an owned-buffer async interface.
FFT local work now uses bounded worker-aware chunks and one reusable scratch
array (or real/complex pair) per transform chunk.

The initial audit identified source-derived opportunities rather than measured
bottlenecks. The implementation follow-ups add focused correctness tests and
a limited local FFT timing comparison below. No QB4 access was used.

## Execution model and application reachability

An HPX task is lightweight scheduled work, not a new OS thread for each task.
Several runnable tasks can share the configured worker pool. Inside an HPX
task, waiting for an unfinished HPX future suspends that task so other work can
run; it still prevents that task's dependent statements from proceeding.
Ordinary blocking library calls do not automatically gain this behavior.

`hpx::async(hpx::launch::async, ...)` is useful both for local work and for
actions on another locality. Its future carries completion, a result, or an
exception. Launching all independent tasks and then collecting their futures
permits concurrency. Repeatedly launching one task and immediately calling
`.get()` leaves no overlap between those iterations. A future alone does not
create work, and extra workers do not parallelize an ordinary C++ loop.
See the [HPX async API](https://docs.hpx.dev/latest/html/libs/core/async_base/api/async.html)
and [single-node guide](https://docs.hpx.dev/latest/html/manual/writing_single_node_hpx_applications.html).

The executable publishes constants, optionally generates initial conditions,
and writes Silo output. The current [initial-gravity path](../src/main.cpp)
also exchanges ghosts, forms multipoles, computes PMMM expansions, and evaluates
combined far/near forces when acceleration or potential output is enabled.
These operations now have application callers; a full evolving N-body timestep
loop is still separate. The [CUDA pool owner](../cosmictiger/cuda/cuda_memory_pool.hpp)
remains a library interface.

## Existing scheduling and synchronization

### Startup and locality tasks

| Path | When and how it runs | Completion boundary |
| --- | --- | --- |
| [Global constants publication](../src/global_constants.cpp#L89) | Root creates the component, then launches an installation action on every locality. Each client retrieves its immutable local copy through an action. | Component creation precedes fan-out; `wait_all` and `get` complete installation before use. The immediate fetch in each client is a real initialization dependency. |
| [Initial-condition dispatch](../src/main.cpp#L85) | Root launches one `hpx::async<Action>` per locality, including itself. Each action owns its initialization result and writes its locality's file. | All actions are launched before the collection loop. Every future is drained, retaining the first exception to rethrow afterward. |
| [CAMB execution](../src/camb.cpp#L310) | `run_as_os_thread` offloads process launch, file processing, and the blocking child wait. The caller gets the future's result before proceeding. | This frees the HPX worker during the wait; it does not overlap multiple spectrum requests in the current initializer. |

The installed Release HPX headers report 1.11.0. Its
`hpx/runtime_local/run_as_os_thread.hpp` uses an `io_pool_executor`, so this is
service-pool work, not a dedicated new OS thread guaranteed for every request.
CAMB runs on logical initialization site zero; spectra are broadcast to the
other sites. Its child is explicitly restricted to one thread by both the
[INI parameters](../src/camb.cpp#L205) and
[OpenMP/OpenBLAS environment](../src/camb.cpp#L223).

### FFT computation

The common [FFT helpers](../cosmictiger/fft/fft_parallel.hpp) use
`ParallelChunkPlan` and `parallelForChunks`: at most four contiguous tasks per
HPX worker, with a minimum estimated workload per chunk. Small ranges,
one-worker runtimes, and calls outside HPX run directly. Every launched task is
drained before return or rethrow. Numerical guards stay inside callbacks and
never span scheduling or waits.

| Implementation | Work already parallel within a locality | Limits and serial boundaries |
| --- | --- | --- |
| [Full-complex vector radix](../cosmictiger/fft/fft_vector_radix.hpp) | Conversions, butterfly arithmetic, twiddles, normalization, and independent FFT line/lane pairs. | Worker-aware chunks reuse one complex scratch array per chunk. Communication completes before butterfly arithmetic. |
| [Packed vector radix](../cosmictiger/fft/fft_packed_vector_radix.hpp) | Packing, products, butterflies, twiddles, and FFT line/lane pairs. | Scratch is allocated per chunk and reused across its transforms. |
| [Pencil/slab](../cosmictiger/fft/fft_pencil_slab.hpp) | Independent real Z pencil/lane pairs; spectral products and endpoint handling; separate XY row and column phases. | Row and column phases join between axes. Each chunk reuses complex scratch or a real/complex pair; plans are memoized across calls. |
| [Pencil boundary, P=M](../cosmictiger/fft/fft_pencil_boundary.hpp) | Z pencil/lane pairs and XY line/lane pairs use the same worker-aware chunks. | Odd localities send their half-plane and return an empty spectrum. Even localities perform XY transforms and spectral products. All ranks do not participate in every phase. |
| [Redistribution](../cosmictiger/fft/fft_redistribution.hpp) | Count and pack passes use worker-aware chunks with a 4096-entry minimum and separate counts/offsets. Async transports allow four outstanding peer transfers; ready receive continuations scatter into disjoint destinations. | Prefix accumulation and submissions remain ordered. Synchronous-only callbacks retain serial peer traversal/scatter; each async window joins before the next starts. |
| [Generic spectral multiplication](../cosmictiger/fft/fft.hpp#L191) | Independent frequencies use the same helper. | [Two-input convolution](../cosmictiger/fft/fft.hpp#L160) runs left forward, right forward, multiply, and inverse in order. |

The project links [serial FFTW3](../CMakeLists.txt#L77). HPX provides threading
around separate FFT executions; neither FFTW worker threads nor FFTW-MPI are
used. First-time plan creation uses the shared
[HPX planner mutex](../cosmictiger/fft/fft_fftw.hpp#L37). Execution happens
outside that lock, using private arrays and the new-array execute interface.
This matches [FFTW's thread-safety contract](https://www.fftw.org/fftw3_doc/Thread-safety.html):
execution may share a plan across independent arrays, while planning and
destruction must be serialized. A plan must outlive every execution using it.
Cached plans are destroyed at process teardown after runtime work has finished,
without calling HPX synchronization after runtime shutdown.

### FFT and particle communication

For a non-self exchange,
[FftHpxExchange::completeExchange](../src/fft/fft_hpx_exchange.cpp) registers
an incoming channel future and launches the outgoing action asynchronously
for nonempty directions, then drains both futures and copies received data.
Self-exchange uses a local `memmove`. Delivery and waiting tasks can progress
with a single HPX worker. The span-based `exchange()` completes before return;
`exchangeAsync(peer, ownedOutgoing, receiveCount)` returns a future owning the
input/output buffers and channel handle through completion.

The [transport contract](../cosmictiger/fft/fft_hpx_exchange.hpp) requires
sequential submissions on each instance and matching operation/call identities
across participants. Concurrent FFT operations require distinct operation IDs.
Tags are reserved before async task scheduling, so execution/completion order
cannot change message identities. Outstanding futures may be collected in any
order and may outlive the endpoint. The transport uses HPX channel communicators
without an application mailbox or application locks. Missing peer traffic has
no cancellation protocol; local draining protects buffers but does not provide
collective recovery after communication or submission failure. The
[filter bank](../cosmictiger/fft/fft_filters.hpp) still requires sequential
calls and consistent filter-creation order.

[Particle load balancing](../src/particle/particle_load_balance.cpp#L307)
and [ghost exchange](../src/particle/particle_ghost_exchange.cpp#L266) already
start all nonempty non-self destination sends before joining their futures. Subsequent
receives call `get()` in site order. Metadata collectives, error agreement, and
completion barriers divide preparation, transfer, validation, and commit.
The [transfer agreement](../src/particle/particle_load_balance.cpp#L337)
prevents a receiver waiting forever for a payload that another participant
failed to send; it is not merely an unnecessary performance barrier.

The installed HPX [channel implementation](/home/dmarce1/local/hpx/Release/include/hpx/collectives/detail/channel_communicator.hpp:131)
dispatches `get()` synchronously to the receiving site's own channel. Its
destination `set()` action stores the payload before its completion future
becomes ready. Since both particle paths join all sends and complete transfer
error agreement before collecting receives, these payloads are already present.
Queuing their receive futures or retaining shared futures would not overlap
network delivery with preparation under this protocol. Receive collection
therefore remains serial; useful transfer overlap requires a separate redesign
of failure agreement and payload lifetime.

Both transports retain remote reference buffers until completion. Same-process
logical participants use owning payload copies because local delivery can
bypass serialization. These are lifetime requirements for any asynchronous
redesign, not optional copies to remove casually.

### Particle work, gravity, and I/O

| Area | Current local computation and concurrency |
| --- | --- |
| [Initial conditions](../src/particle/particle_initialization.cpp#L125) | The adaptive epoch search is ordered. Mode generation, density assembly and local maxima, final fields, and particle filling use bounded chunks around internally parallel FFT calls. Each phase joins before collective futures broadcast spectra or combine errors/maxima. |
| [Particle redistribution](../src/particle/particle_load_balance.cpp) / [ghost exchange](../src/particle/particle_ghost_exchange.cpp) | Redistribution counts/packs outgoing particles and counts/scatters received particles in bounded HPX chunks with private histograms and disjoint output spans. Ghost payload packing and suffix population use bounded particle chunks across peers and cells, retaining request/suffix order. Small request/range prefix preparation remains serial; sends run between coordinated phases. |
| [Cell multipoles](../src/cartesian_grid.cpp#L75) | Independent cell chunks run concurrently. Each cell checks its range as consumed and retains its ordered double-precision accumulation, with a local FPE guard. |
| [Rung sorting](../src/cartesian_grid.cpp#L90) | Independent cell chunks run concurrently; each cell retains its existing counting/permutation algorithm and checks as consumed. Cell particle ranges must be disjoint. |
| [PMMM convolution](../cosmictiger/gravity/pmmm/pmmm_convolution.hpp) | Green setup uses bounded independent cell tasks. Four coefficients share each FFT/exchange by default. Source/filter packing, spectral contractions, inverse unpacking, and expansion publication use disjoint parallel loops. Cached Green spectra avoid repeated setup; exchanges remain sequential and ordered across localities. |
| [FMM/STF operators and cache](../src/gravity/cartesian/fmm.cpp) | Scalar operators execute on caller-owned arrays; `addM2LTargets` places bounded tasks over independent target rows, retaining each row's source order. The cache uses shared reads and exclusive insertion under an HPX shared mutex; construction is outside the lock. |
| [CPU direct interactions](../cosmictiger/gravity/direct/direct_interactions.hpp) | Backends declaring concurrent sink support receive bounded sink tasks; legacy backends keep increasing sink order. Each sink retains lane-batched source accumulation and reduction order. `DirectFieldBackend` stores accelerations and optional potentials; the kick backend updates canonical momentum and synchronized rungs, with atomic scalar diagnostic sums and optional per-sink output fields. |
| [CUDA direct interactions](../cosmictiger/gravity/direct/direct_interactions_cuda.cuh#L30) | 128 GPU threads per block, one sink at a time per thread, cooperative source tiles, and block barriers. The [future-returning launch](../src/gravity/direct/direct_interactions.cu) reports validation, launch, and execution failures and becomes ready after GPU completion. |
| [CUDA memory pool](../cosmictiger/cuda/cuda_memory_pool.hpp) | Atomic allocation ownership permits concurrent CPU/GPU users with suitable backing memory. The standalone view schedules no tasks. The HPX host owner allocates and frees through futures and is not used by the simulation. Reset and destruction require all users to finish. |
| [Silo output](../src/particle/particle_silo.cpp#L337) | Block conversion and writes are synchronous. Locality actions write distinct files concurrently across processes; the [current caller](../src/main.cpp#L50) performs output on its HPX task. |
| [Friedmann integration](../src/friedmann.cpp#L95) | Serial adaptive integration of a small state. Runge-Kutta stages depend on earlier stages; tasking individual stages would not expose useful independence. Independent cosmology solves could be separate tasks if a caller needs them. |

## Implemented particle and cell tasks, 2026-10-05

[ParallelChunkPlan and parallelForChunks](../cosmictiger/parallel.hpp) divide
contiguous ranges into at most four chunks per HPX worker. Particle passes use
a minimum grain of 4096 particles; cell passes use 64 cells. Fewer than two
useful chunks, a one-worker runtime, and calls outside HPX use the synchronous
path. Tasks run on the existing HPX pool. Every launched task is joined,
including after another callback or launch fails, before captured storage can
be destroyed or a subsequent collective begins.

Outgoing redistribution uses private destination-rank counts and prefix offsets
for disjoint packing. Chunks skip the union of ghost intervals and flagged
ghosts without allocating a particle-index list. Received payloads are traversed
as a source-ordered stream; sparse private cell histograms and chunk-ordered
prefixes assign disjoint scatter intervals without a workers-times-cell-count
table. Stable chunk order preserves the previous partition order before each
cell's existing rung permutation. Collective identities and transport remain
unchanged. Errors before commit preserve input; a task-launch/allocation failure
in the final rung sort is reported through the completion agreement after
replacement has committed and may leave incomplete rung metadata.

Bulk multipoles and rung sorting distribute independent cells across coarse
tasks. Multipoles retain each cell's particle accumulation order. Each cell's
range is checked when consumed in validation builds; arithmetic has a local
`FpeGuard`, with no guard spanning task launch or a future wait. A cell failure
can leave other cells updated, and all launched chunks are drained before
rethrowing. Ghost owner ranges are not read. Sorting requires disjoint local
particle ranges. The public exclusive-access contract still applies.

Initialization still sorts rungs immediately after load balancing, which already
sorts them. Eliminating that duplicate pass is a separate small follow-up.

### Implemented ghost packing and unpacking, 2026-10-05

Requested-cell counts establish per-peer prefix offsets and one logical copy
stream across all peers. Nonempty spans carry source ranges and cumulative
ends; response entries for empty requests remain in their original order.
Peer buffers are sized before tasks run. Particle chunks intersect those
spans and write disjoint offsets, preserving peer request order and splitting
large cells across workers without allocating a particle-index array.

Received ghost targets already have assigned suffix ranges in stored cell
order. Copy chunks intersect those ranges, skip empty cells, and write disjoint
particle indices after storage growth. Each cell's ranges are published once
after all chunks finish, then the existing rung sort runs. Positions, velocities,
rungs, optional acceleration/potential fields, owner ranges, and same-locality
periodic flags retain their contracts.

Both copy phases target at least 4096 particles per chunk and at most four
chunks per worker across the entire locality, with inline small/one-worker
execution. All launched tasks are drained before phase error agreement, even
after a copy or launch fails. Collective/channel phase ordering and payload
lifetimes are unchanged. Metadata preparation stays serial; this implementation
parallelizes the particle copies rather than introducing extra validation passes.

Ghost regression coverage includes a 10,003-particle singleton cell copied to
26 periodic targets and eight logical ranks with one large cell, one small
cell, and six empty ranks. Exact sequence and bitwise field snapshots check
positions, velocities, rungs, both optional gravity fields, owner/suffix ranges,
two refreshes, and repeated clear. Source cells are sorted before capturing
the order reference because the existing rung permutation is not stable.
A large invalid-owner case confirms packing failures reach every participant,
preserve live storage, and permit retry with a fresh operation ID after repair.

The particle/storage/grid/multipole/chunk selection ran 72 tests with both one
and two HPX workers. Release passed 58 and skipped 14 (12 validation-contract
cases and two remote MPI cases); validation-enabled runs passed 70 and skipped
the two MPI cases. The validation harness uses Release HPX with
`COSMICTIGER_VALIDATION_CHECKS=1`, `-fno-fast-math`, `-ftrapping-math`, and
`-mno-daz-ftz`; it is not a native Debug/APEX build. Builds used one compiler
job and runtime checks reported at most six total OS threads. Formatting and
diff whitespace checks passed. Multi-process MPI and CUDA were not rerun for
this follow-up; the local task changes preserve the existing communication
phases. No ghost phase speedup was measured.

## Implemented PMMM setup and coefficient batching, 2026-10-05

[Periodic Green block construction](../cosmictiger/gravity/pmmm/pmmm_periodic_green.hpp)
uses bounded HPX chunks with a grain of four cells and at most four chunks per
worker. Each cell retains the original real-space and reciprocal-space sum
order. Each scalar cell owns its `FpeGuard`; no guard spans scheduling or a
future wait. All chunks finish before any distributed FFT begins. A one-worker
runtime or non-HPX caller follows the synchronous path.

[PmmmPeriodicConvolution](../cosmictiger/gravity/pmmm/pmmm_convolution.hpp)
now defaults to four coefficient lanes per FFT. Its optional third template
parameter selects another positive batch size, with size one providing the
scalar path. Source fields, Green coefficients, local expansions, and the five
mass/dipole/second-moment background fields use deterministic batches. The
constant-one Green field occupies the next available Green lane. Unused final
lanes remain zero. Spectra stay packed by batch, avoiding separate copies of
each coefficient; contractions access the appropriate batch and lane and retain
each local coefficient's original term order.

Pack/unpack, frequency products, background correction, and publication use
independent local loops. Each instance and exchange callback still require
sequential calls; this change batches messages without launching concurrent
transforms through shared communication state. The one-shot API uses the new
default. Green spectra are committed only after every filter batch completes.

At the default source/expansion orders 5/6, filter setup drops from 145 FFT
calls to 37, and cached solves drop from 91 to 25 (10 forward and 15 inverse).
A first solve therefore uses 62 calls instead of 236. These are call counts,
not measured wall-time speedups: per-call scratch and messages now carry four
lanes, and the last batch may include padding. The total scalar arithmetic
remains comparable. PMMM remains a library/test path pending a simulation
gravity caller.

### Implemented: asynchronous FFT transfer overlap

[FftHpxExchange::exchangeAsync](../cosmictiger/fft/fft_hpx_exchange.hpp) accepts
an owned send vector and expected receive count, returning
`hpx::future<std::vector<Value>>`. Submissions reserve tags synchronously;
ordinary HPX tasks own buffers and a channel handle until send and receive
complete. Errors are carried by the future after launched directions are drained.
The existing span API and callable synchronous adapter remain available.

[Redistribution](../cosmictiger/fft/fft_redistribution.hpp) detects this explicit
capability and submits at most four peers per window in increasing mask order.
Ready `.then(hpx::launch::async, ...)` continuations scatter into disjoint output
indices while other transfers remain pending. The local block is scattered
after the first window is submitted. Every launched continuation is joined
before the next window or return, including error paths. A retained shared
receive future allows draining if continuation attachment fails. Prepared sends
remain the existing total local block; outstanding receive storage is bounded
by four peer payloads, rather than all peers.

Pass the transport object directly; a lambda exposing only the synchronous
call hides this capability. The boundary FFT preserves the capability through
its logical-to-physical peer adapter. Legacy callbacks retain their original
order. The planner's `communicationPhases` still counts serialized logical peer
rounds; it is not a count of serial waits with this transport.

This adds communication/scatter overlap to pencil/slab library paths. Current
initialization and PMMM use vector radix, whose dependent butterfly exchanges
still complete before arithmetic; they gain no new overlap from this path.
No concurrent forward transforms, solver processes, or file writers were added.
No wall-time speedup is claimed. Particle receive collection was checked and
left unchanged for the protocol reasons above.

## Audit priorities and implementation follow-ups

Priority reflects active call paths, independence, and implementation risk.
Measured phase times could change this ordering.

The leading remaining tasks are:

1. **Launch independent left/right FFT forwards concurrently:** give each input
   separate supported FFT/transport state and globally consistent operation
   IDs, then join both spectra before multiplying and inverse transforming.
   This remains useful for the two-input convolution interface, whose direct
   callers are currently tests rather than the PMMM gravity path.
2. **Parallelize Silo block conversion:** convert positions and pack rung/flag
   arrays in disjoint chunks before their existing file writes. Velocity,
   acceleration, and potential already use direct particle-array pointers.
   Keep calls using one Silo file handle ordered; particle storage must remain
   fixed through conversion and output.

### 1. Chunk local initialization work with async tasks and futures (implemented)

Implemented 2026-10-05 in
[particle initialization](../src/particle/particle_initialization.cpp): mode
creation, density-spectrum assembly, final displacement/velocity fields, and
particle filling now use `ParallelChunkPlan` and `parallelForChunks`. The
minimum chunk size is 1024 entries, with at most four chunks per HPX worker.
This is a scheduling heuristic; small ranges and one-worker runtimes stay
inline. Output storage is allocated before launch and each callback writes
only its own contiguous range. No extra OS threads are created.

The density scan computes one maximum per chunk, joins all tasks, and reduces
those maxima before the existing global maximum collective. Seeded modes still
depend only on their global conjugate-pair index. Per-element arithmetic and
phase boundaries are retained. Every callback scopes its own `FpeGuard` around
local arithmetic, and every launched future is drained before return or
rethrow. Errors, including density-reduction allocation or scheduling errors,
reach the existing collective agreement before the next phase. Progress
observers and the external spectrum solver remain on the controlling path.

### 2. FFT task granularity, scratch reuse, and plan memoization (implemented)

Implemented 2026-10-05: all four FFT strategies now use a common bounded
scheduling policy. Simple loops estimate work from batch entries; FFT lines use
`length * max(1, floor(log2(length)))`, saturated at the scheduling target to
avoid overflow. Chunks target at least 4096 such units, with at most four chunks
per worker. A range stays inline unless it has enough work for two chunks.
The estimate is a heuristic, not a measured flop count or a universal optimum.

`parallelForFftWithScratch` constructs scratch once per chunk and reuses it
across that chunk's line/lane pairs. Complex transforms use one vector; real
transforms use a real vector and a half-spectrum vector. Every execution fills
all of its input entries, including the real-transform endpoints. Scratch is
private to one invocation and freed after the joined phase, so concurrent calls
need no shared workspace lock. Axis ordering and communication boundaries are
unchanged. Redistribution count/pack passes also use bounded worker-aware chunks
with a 4096-entry minimum and stable prefix offsets.

Implemented 2026-10-05: [memoizedFftwPlan](../cosmictiger/fft/fft_fftw.hpp#L133)
retains plans across calls and FFT object lifetimes. The process-wide cache
keys by transform length and kind: in-place complex forward/backward, or
out-of-place real-to-complex/complex-to-real. Precision, unit stride, and
`FFTW_ESTIMATE | FFTW_UNALIGNED` are fixed. Compatible plans are shared across
all FFT strategies, batch sizes, and logical localities in that process.

Cache hits allocate no planning scratch. Concurrent misses recheck under the
planner mutex so only one plan is created per key. Callers borrow handles;
there is no eviction or clear operation, and retained plan memory grows with
the distinct keys requested. Only the cache destroys plans, at process
teardown after execution has quiesced. Execution continues on independent
private arrays through FFTW's new-array functions. Normal FFT object
destruction no longer destroys cached plans.

This removes the fixed eight-chunk ceiling and per-line execution allocations.
Scratch is reused within each phase; it is not retained in a shared global or
thread-local buffer between calls. Receive scatter overlaps peer transfers as
described above.

### 3. Place gravity tasks above the numerical kernels

Implemented 2026-10-05 using the existing bounded `ParallelChunkPlan` and
`parallelForChunks` helpers. Tasks partition targets or sinks; they retain each
destination's complete source order and introduce no shared numerical reduction.
There are at most four chunks per HPX worker, and small batches, one-worker
runtimes, and callers outside HPX execute synchronously.

[`addM2LTargets`](../cosmictiger/gravity/cartesian/fmm.hpp) accepts one immutable
interaction span per preallocated local expansion. Sources use a distinct
owning field, and each task writes disjoint local rows. Entries carry source
index, physical separation, source scale, and the centered-source flag;
targets supply their normalization scale. Existing sums, optional traces, and
empty rows are preserved. The minimum grain is 16 targets. Index, shape, and
geometry checks happen when interactions are consumed; no validation or
interaction-count prepass was introduced.

[`directInteractionsCpu`](../cosmictiger/gravity/direct/direct_interactions.hpp)
requires an explicit `static constexpr bool supportsConcurrentSinks()` returning
true before splitting sinks. Calls share the same backend instance: outputs for
different sinks must be disjoint, source positions and flags must be immutable,
and callbacks must not read another sink's changing outputs. Backends without
the capability, or returning false, keep increasing callback order. This retains
the existing shared-vector recording backend safely. Source traversal, self
exclusion, SIMD lane accumulation, and lane reduction order are unchanged.

The source-aware sink grain targets roughly 4096 source/sink pairs, clamped to
8–64 sinks, using only source-range length. Both grain choices are unbenchmarked
heuristics. `DirectFieldBackend` opts in and overwrites one range-relative
acceleration and optional potential per active owned sink, leaving ghost and
inactive outputs untouched. Caller-owned output arrays must have capacity for
the declared range, remain live through completion, and be disjoint from each
other and particle inputs. The kick backend also opts in; it updates each sink independently and uses
atomic additions for the small diagnostic accumulator. See the
[conformal kick guide](gravity_kick.md) for the integration rule.

All launched chunks are drained before returning or rethrowing an exception;
completed writes and partial sums are not rolled back. Arithmetic guards stay
inside scalar work and end before backend callbacks, scheduling, or future
waits, so a supported backend may suspend. These library paths do not add a
gravity timestep to the initialization executable. FMM cache access retains its
existing synchronization.

### 4. Further asynchronous interfaces

Implemented near/PMMM overlap: the [initial-gravity caller](../src/main.cpp)
launches read-only near preparation through
[`kickPmmmParticlesOverlapped`](../cosmictiger/gravity/kick/pmmm_kick.hpp), runs
the existing expansion convolution, and joins the near future before the
single combined kick and global force/potential/error reduction. Raw direct
lanes retain the far-first, increasing-lane arithmetic order; near preparation
does not publish fields, select rungs, or accumulate diagnostics. Particle data
and ranges stay fixed while PMMM reads moments/trace and writes expansions.
The [kick guide](gravity_kick.md) describes temporary storage and error behavior.
All launched near work is joined after a far failure; a near failure still
allows the rank to enter the same FFT sequence. These local task joins do not
repair an already-stranded distributed FFT operation.

Near preparation and staged application use a flat stream of candidate sinks,
with per-cell range/prefix metadata and at most four chunks per worker. The
initial sink grain is eight; a crowded cell can span chunks, and empty cells
do not schedule individual tasks. Sorted cells use their cached active suffix;
unsorted cells retain flag checks over the full owned range. These grain and
storage choices are unbenchmarked. The existing fused kick remains available
without near-lane scratch when the far expansion is already ready.

| Candidate | Possible future-based design | Required change or limitation |
| --- | --- | --- |
| Two-input FFT convolution | Compute `FFT(left)` and `FFT(right)` concurrently, then multiply their spectra and inverse-transform the product. Multiplication waits for both forward results. | Use separate supported FFT/transport state and globally consistent operation IDs. Compare batch lanes first; each existing transform may already occupy the worker pool. This applies to two-input convolution; a precomputed multiplier needs only one forward transform. |
| Particle receive collection | Retain the current collection loop under the existing transfer protocol. | Completed sends have already populated the local HPX channels before collection begins. Queuing get futures adds no network overlap here; changing this requires redesigning transfer-error agreement and cleanup. |
| Final EB spectra | Once the handoff epoch is known, overlap the center/before/after solver requests on root, then broadcast deterministically. | [spectrum()](../src/particle/particle_initialization.cpp#L393) mutates evaluation counts and collective identities. Preassign IDs/budget and require reentrant solvers or separate instances. The solver interface and mutable test doubles do not guarantee reentrancy. Bound CAMB children and service threads; enough service-pool capacity or nonblocking child completion is needed. |
| Silo output | Offload blocking I/O to an appropriate HPX service executor and pipeline bounded conversion buffers. | Keep one writer per file; verify installed Silo/HDF5 thread safety before concurrent library calls within a process. Own an immutable snapshot until writing completes. Current initialization has little subsequent computation to overlap. |
| Future CPU/GPU gravity pipeline | Schedule CPU work and communication against a future representing actual CUDA completion. | Release and Debug HPX now have CUDA enabled. The [CUDA interfaces](../cosmictiger/cuda/cuda_runtime.hpp) provide futures for actual stream completion and retain kernel launch captures until completion. Chain dependent submissions and preserve external buffer ownership; blocking runtime queries and allocations use the HPX I/O pool. |

The initialization [test configuration](../CMakeLists.txt#L214) uses an I/O
pool of size one. With the current blocking child wait, additional async solver
tasks would queue their CAMB jobs serially on that pool. More HPX worker tasks
alone do not create concurrent CAMB execution.

The adaptive initial-condition scan/bisection remains dependent on previous
results. At the FFT `P=M` boundary, restoring balanced spectral work across odd
and even localities needs an algorithm/layout change; additional futures alone
cannot redistribute that work.

## How to use futures safely in these changes

For independent local chunks, a vector of `hpx::future<Result>` is sufficient:
launch the bounded group, perform any independent caller work, then collect
all results. For longer dependency chains, `when_all(...).then(...)` or
`hpx::dataflow` can make readiness explicit. Use `shared_future` only when
multiple consumers need the same result, with immutable result ownership.
Redistribution now uses shared receive futures and async `.then()` scatter
continuations. `when_all` and `dataflow` remain opportunities. The [CUDA completion bridge](../src/cuda/cuda_runtime.cpp)
uses asynchronous `.then()` continuations, and its tests compose allocation and
copy futures; an FFT transport test also uses `.then()`. See [when_all](https://docs.hpx.dev/latest/html/libs/core/async_combinators/api/when_all.html)
and [dataflow](https://docs.hpx.dev/latest/html/libs/core/async_base/api/dataflow.html).

`when_all` signals readiness of its contained futures; consume those futures
to observe their errors. Installed HPX 1.11.0's `wait_all` can itself rethrow
stored errors after waiting; it must not be described as exception-blind.
Use a deliberate drain-and-report policy, including a failure while launching
only part of a task group. The
[main action collection loop](../src/main.cpp#L92) demonstrates draining
already-launched tasks after an execution error. See
[HPX wait_all](https://docs.hpx.dev/latest/html/libs/core/async_combinators/api/wait_all.html).

The important project-specific conditions are:

- Keep all HPX scheduling, waits, promise operations, and HPX-object destruction
  outside application locks. A synchronous continuation may execute during
  promise completion. Preserve the mailbox's extract-before-destroy pattern.
- Preallocate before workers start. Do not resize, reserve, append, or replace
  particle storage while tasks hold its pointers. Even failed growth can change
  field pointers. GPU work must also finish before storage mutation.
- The [convenience particle storage](../cosmictiger/particle/particle_storage.hpp#L433)
  is process-wide, not task-local. Concurrent logical sites need separate
  explicit storage/grid objects.
- Keep distributed operation IDs and phase order consistent on every
  participant. A local task failure must be shared before peers enter an
  exchange that cannot finish without that participant.
- Bound task counts, outstanding messages, temporary arrays, and child
  processes. Reuse the HPX pool; avoid another independent CPU thread pool.
- Separate host submission, network completion, and GPU completion. Each
  future must have a clear completion/lifetime contract.

## Existing coverage and measurements still needed

Existing tests provide useful correctness anchors. The initial audit inspected
them without execution; implementation validation is recorded separately below.

| Coverage in source/build definitions | What it establishes when run |
| --- | --- |
| [Memoization tests](../tests/fft/fft.cpp#L141) | Reused plans survive FFT object destruction; concurrent cold lookups share compatible handles and execute on independent arrays; odd/even/unit real transforms and invalid keys are covered. |
| [Logical FFT locality tasks](../tests/fft/fft.cpp#L275), [channel transport tests](../tests/fft/fft_hpx_exchange.cpp), and [single-worker FFT entry](../CMakeLists.txt#L268) | Cooperative progress, transport ordering/lifetimes, continuation reentry, and numerical FFT behavior with one worker. |
| [Two-process FFT entry](../CMakeLists.txt#L304) | Remote reference-payload transport; this test is not an entire distributed FFT convolution. |
| [Initialization tests](../tests/particle/particle_initialization.cpp#L131) | Coarse initialization chunks, seeded reproducibility, eight logical sites, and solver/worker error propagation; separate CTest entries request one and two workers. |
| [Particle single-worker and remote entries](../CMakeLists.txt#L314) | Migration and periodic ghost exchange, including logical sites sharing processes; eight-process checks require an explicit build option. |
| [Cache test](../tests/cartesian/cartesian_fmm.cpp#L154), [host pool tasks](../tests/cuda/cuda_memory_pool_host.cpp#L166), [GPU pool stress](../tests/cuda/cuda_memory_pool_cuda.cu#L228) | Concurrent cache reuse, allocator ownership, and host/device allocation overlap. |
| [PMMM logical-locality test](../tests/pmmm/pmmm_convolution.cpp#L268) | Distributed-algorithm behavior emulated by HPX tasks and promise/future transport in one process. |

Before selecting an optimization, time representative phases separately:
CAMB wait, local mode/field preparation, FFT planning/execution/packing/exchange,
particle packing/sorting, and Silo output. For PMMM distinguish first-call
Green setup from cached solves. Sweep one worker and a few larger authorized
worker counts with fixed problem sizes, reporting wall time, task granularity,
peak temporary memory, and total OS threads. Use APEX-capable build modes for
task evidence; do not infer bottlenecks just from occurrences of `.get()`.

After an implementation, retain one-worker progress tests, add relevant
multiworker/decomposition comparisons, exercise empty/skewed partitions and
failure cleanup, and validate actual multi-process communication when changing
transport. Preserve deterministic per-cell arithmetic where possible; explicitly
assess numerical changes if reduction order changes. GPU overlap needs device
completion and lifetime checks, not just a successful launch return code.

Build parallelism and test-process parallelism are separate from HPX workers.
[Presets](../CMakePresets.json) request four build jobs and two CTest scheduling
slots. [General tests](../tests/test_main.cpp#L39) default to two HPX workers;
[MPI test options](../CMakeLists.txt#L284) request two workers per rank. CTest
`PROCESSORS` values are scheduling weights, not enforced total-thread limits.
HPX service threads, MPI/CUDA helpers, and CAMB children also consume resources.

Follow [AGENTS.md](../AGENTS.md) and the user-provided QB4 policy for any later
measurements. The latest explicit game status is OFF (2026-10-05), permitting available
local resources shared across concurrent work. If reported ON, the six-thread
aggregate limit includes service/helper threads, with the separate GPU limit; do not infer game status from processes. QB4 remains paused until
explicitly resumed. No benchmark commands here authorize remote access.
## Implementation verification, 2026-10-05

FFTW plan memoization and the repository-wide validation migration were built
and tested in isolated local build directories. Release ran 301 CPU tests:
260 passed and 41 validation-only or locality-dependent tests were skipped.
The validation-enabled CPU/CUDA suite ran 315 tests: 312 passed and three
multi-process-only tests were skipped. The CUDA expectation/FpeGuard kernel
ran successfully. One-worker FFT tests passed 34 of 35 cases; the skipped
remote-payload case passed in a separate two-process MPI invocation. That
invocation also passed ghost exchange; two-process constants publication
passed separately. All 18 Friedmann tests and 17 Boltzmann/CAMB tests passed;
initialization passed 12 tests with its opt-in smoke test skipped.

The nine standalone primitive/guard tests verify enabled expectations,
Release bound elision, SIGFPE trapping, nested-mask/pending-flag restoration,
and subnormal arithmetic. Eight pass with validation enabled (the disabled
case is skipped); two pass in Release (the enabled-only cases are skipped).
Validation is enabled by CMake in every mode except Release. Native Debug
configuration could not run because installed HPX lacks APEX; enabled behavior
was verified against available Release HPX using isolated compiler/linker
launchers defining COSMICTIGER_VALIDATION_CHECKS and enforcing IEEE semantics.
Local MPI runs used Open MPI with local socket access permitted. The earlier
opal_ifinit socket errno=1 message was a sandbox restriction. MVAPICH is not installed locally; evaluating it
requires a separate HPX build and compatible or serial Silo/HDF5, because the
current HPX and system Silo/HDF5 libraries depend on Open MPI. No vendor switch
was performed. The standing-orders file now records War Thunder ON, so further
compute work must fit the shared six-thread budget. No QB4 access was attempted. Formatting, exact source notices, and whitespace were checked.


### Particle and cell follow-up verification

The CPU-only Release build passed 43 of 56 focused tests with both one and two
HPX workers. Eleven validation-only cases and two physical-locality cases were
skipped in each run. The new large redistribution, bitwise cell comparisons,
chunk-boundary overflow, and exception-draining tests all passed. Eight logical
participants exercised the collective paths within one runtime. Formatting,
source notices, and whitespace checks passed.

A temporary test main counted its own OS threads without a monitoring process:
one-worker runs observed four then five threads; two-worker runs observed five
then six. Networking and jemalloc background threads were disabled, and builds
used one job. A temporary fatal-runtime-error hook exited before HPX could
create an additional shutdown helper. The production test main is unchanged.
Physical MPI runs and CUDA execution were not repeated for this follow-up under
the active six-thread limit. Validation-enabled builds were not rerun for these
new task changes; the earlier validation-migration results above predate them.
No performance speedup has been measured.

### PMMM follow-up verification

CPU-only Release and validation-enabled builds passed the focused PMMM suite
with both one and four HPX workers. Release passed 19 of 21 cases, skipping the
validation-only input rejection and physical-locality case. Validation enabled
passed 20 of 21, skipping only the physical-locality case. Bitwise Green cell
comparisons, scalar-versus-four-lane default-order expansions, partial batches,
and cold/cached exchange counts all passed. The distributed order-2 test
observed 150 versus 45 exchange callbacks on cold calls and 72 versus 24 on
cached calls, per logical endpoint.

A separate two-process local Open MPI run passed the PMMM coefficient-batch test
and the existing FFT remote-reference-payload test. Eight logical PMMM
endpoints shared the two processes, with one HPX worker per process and bounded
service pools. The test checked dipoles, the second-moment background correction,
unchanged ghost expansions, cold/cached agreement, and 108/57 scalar versus
36/21 batched exchange callbacks. No QB4 access was attempted.

Enabled validation used an isolated Release-HPX build with
`COSMICTIGER_VALIDATION_CHECKS` and strict floating-point compiler/linker flags;
installed HPX still lacks the APEX required for native Debug configuration.
CUDA execution and wall-time speedups were not evaluated for these host changes.
Formatting, exact source notices, and whitespace were checked.


### Asynchronous FFT overlap validation follow-up, 2026-10-05

CPU-only Release and a strict-FP validation-enabled build against the installed
Release HPX passed the following checks. The latter defines
`COSMICTIGER_VALIDATION_CHECKS` and applies `-fno-fast-math -ftrapping-math
-mno-daz-ftz` during compilation and linking; it is not a native Debug/APEX build.

| Run | Passed / selected | Skips |
| --- | --- | --- |
| Release FFT, one worker | 37 / 40 | Two validation-only cases; physical MPI payload case. |
| Release FFT + PMMM regression, two workers | 56 / 61 | Three validation-only cases; two physical MPI cases. |
| Validation-enabled FFT, one worker | 39 / 40 | Physical MPI payload case. |
| Validation-enabled FFT + PMMM regression, two workers | 59 / 61 | Two physical MPI cases. |
| Release initialization regression, one worker | 11 / 12 | Validation-only option case. |
| Validation-enabled initialization regression, one worker | 12 / 12 | None. |

Real CAMB initialization was excluded from these regression selections. The
new transport cases check delayed peer construction, endpoint/input-buffer
destruction, twelve outstanding tags, reverse result collection, asymmetric
and empty directions, self exchange, payload-size errors, and simultaneous
operation IDs. A gated redistribution test requires four submissions before
fulfilling the first window and completes them in reverse order. Actual HPX
channels exercise direct/staged pencil convolution plus eight- and 64-logical
endpoint boundary layouts; the latter covers nontrivial peer remapping.

A temporary test main checked runtime thread counts at startup and completion
and exits immediately on a fatal HPX runtime error to avoid its extra shutdown
helper. One-worker runs reported four then five OS threads; two-worker runs
reported five then six. Networking and jemalloc background threads were disabled;
builds used one compiler job. All runtime runs were sequential, with no other
managed compute job active in the process audit.

The remote payload test now includes eight outstanding async owned-buffer
transfers, but its new path was not run across processes. The current reported
War Thunder ON six-thread aggregate limit cannot accommodate two MPI ranks
and their service/helper threads. Multi-process validation remains outstanding.
These local results establish ordering, lifetimes, cooperative progress, and
numerical equivalence; they do not establish network overlap speedups or remote
failure recovery. No GPU test or QB4 access was used for this host-only change.

## Gravity target and sink follow-up validation (2026-10-05)

The CPU Release library and tests built successfully. A separate CPU tree used
the installed Release HPX with `COSMICTIGER_VALIDATION_CHECKS=1`,
`-fno-fast-math`, `-ftrapping-math`, and `-mno-daz-ftz`; it also built successfully.
This exercises validation and arithmetic traps, but is not a native Debug/APEX
build. Each run selected `DirectInteractionsCpu*`, `CartesianFmm*`,
`ParallelChunks*`, `StfM2l*`, and `Ferrers*`:

| Configuration | HPX workers | Passed | Skipped |
| --- | --- | --- | --- |
| Release | 1 | 37 | 10 |
| Release | 2 | 39 | 8 |
| Validation enabled | 1 | 44 | 3 |
| Validation enabled | 2 | 47 | 0 |

Release skips expectation-failure cases; one-worker runs additionally skip
tests requiring multiple chunks. The ten new target/sink tests cover bitwise
scalar agreement, cancellation-sensitive M2L ordering, centered sources,
normalization scales, traces, empty lists, output alias/range contracts,
partial sink chunks, self exclusion, coincident particles, rung/ghost filtering,
legacy backend ordering, optional potentials, concurrent suspending callbacks,
and drainage after failures. Existing STF oracle and Ferrers checks also passed.
Both affected CUDA translation units (the direct library and direct tests)
compiled with validation enabled. No GPU execution or multi-process MPI run
was performed; this change adds local scheduling and no transport behavior.

The temporary bounded test main disabled networking and allocator background
threads. It sampled four/five OS threads in one-worker runs and five/six in
two-worker runs, with the validation two-worker run sampling six at both ends.
One validation build was accidentally started before the Release two-worker
run had exited. It was paused when noticed and resumed after that run finished;
the aggregate count during that brief overlap was not measured, so compliance
with the six-thread aggregate limit cannot be claimed for that interval.
The remaining builds and runtime checks ran sequentially after process audits.
No QB4 access was used. Chunk sizes have not been benchmarked, and the current
initialization executable still has no consuming gravity timestep.

## FFT granularity and scratch follow-up (2026-10-05)

The new `FftParallel` tests exercise inline small/empty ranges, bounded scratch
allocation counts, reuse across uneven tail chunks, and scratch destruction
after another chunk fails. Concurrent full and packed transforms test private
scratch on the same FFT object, three independent lanes, all axes, lengths from
2 through 32, and bitwise agreement with separate calls. Existing pencil/slab,
boundary, distributed-layout, filter, and PMMM tests exercise the same helpers.

A local timing experiment compared the saved pre-change FFT headers with the
new headers, using the same Release compiler flags, serial FFTW plans, input
arrays, and bounded HPX test main. Each case uses three lanes, warmed plans,
and seven samples; values below are median microseconds per forward/inverse
round trip. Repetition counts ranged from 2 to 128 depending on mesh size.
Each before/after executable ran separately. The latest resource report was
War Thunder ON, so these are limited local samples rather than an isolated
machine benchmark or a test of larger worker pools.

| Mesh side | Full, 1 worker: before → after | Packed, 1 worker | Full, 2 workers | Packed, 2 workers |
| --- | ---: | ---: | ---: | ---: |
| 4 | 20.394 → 5.160 | 17.015 → 3.273 | 26.355 → 9.145 | 22.123 → 5.427 |
| 8 | 54.362 → 35.824 | 33.468 → 19.707 | 53.079 → 49.469 | 39.535 → 21.064 |
| 16 | 332.012 → 324.646 | 166.401 → 151.856 | 289.869 → 303.921 | 165.197 → 160.975 |
| 32 | 3305.481 → 3539.380 | 1585.717 → 1704.305 | 2535.772 → 2540.333 | 1368.391 → 1385.670 |
| 64 | 53962.440 → 58372.146 | 22859.546 → 20751.514 | 47420.877 → 46594.211 | 15310.565 → 15341.427 |

Small transforms benefit from avoiding unnecessary scheduling and allocation.
Larger two-worker results are roughly unchanged in this sample; one-worker
large results are mixed, including slower full transforms. The 4096-unit cutoff
is therefore an initial heuristic, not an established optimum or a general
speedup claim. Pencil-path allocation reduction is structural: each phase now
allocates scratch per chunk instead of per line; pencil wall times were not
benchmarked in this experiment.

Final CPU regression results, selecting `Fft*:Pmmm*` in the current shared tree
(including the subsequently added PMMM kick tests):

| Build | HPX workers | Passed | Skipped |
| --- | ---: | ---: | ---: |
| Release | 1 | 65 | 5 |
| Release | 2 | 65 | 5 |
| Validation enabled | 1 | 68 | 2 |
| Validation enabled | 2 | 68 | 2 |

Release skips three validation-contract cases and two remote MPI cases.
Validation skips only those two MPI cases. Initialization regression tests,
excluding the real CAMB smoke test, passed 11/12 in Release (one contract case
skipped) and 12/12 with validation enabled. The validation tree uses Release
HPX plus `COSMICTIGER_VALIDATION_CHECKS=1`, `-fno-fast-math`, `-ftrapping-math`,
and `-mno-daz-ftz`; these are not native Debug/APEX results. Formatting and
whitespace checks passed. No CUDA execution, multi-process MPI test, or QB4
access was used for this local scheduling change.

Runtime samples reported four/five OS threads with one worker and five/six
with two. A validation build was started in the same tool batch that revealed
another active build; it was stopped. Later launch checks refused to overlap
with active builds. The user's later OFF report was verified in the designated
status chat, but automatic approval review still required direct confirmation
in this chat before exceeding six threads. It blocked concurrent linking and
the proposed eight-worker run. Final checks therefore used exclusive one- and
two-worker runs. Scaling beyond two workers, including the test assertion that
larger pools use more than eight scratch chunks, remains unrun.

## Near-field and PMMM overlap verification (2026-10-05)

The application now uses the collective overlap wrapper after ghost exchange
and moment construction. Near scratch is local and temporary; FFT sequence
IDs, locality mapping, and the final diagnostic reduction are unchanged.

The new tests compare particle fields and rung/range metadata bitwise against
the fused kick across 256 combinations of Ferrers order, kick phase, retained
fields, potential sums, and sorted/unsorted active sets. A crowded cell with
65 active and five inactive sinks exercises splitting within a cell and
skipping empty spans. Existing eight-logical-locality ghost inputs also match
the fused fields exactly. Global floating sums use a tolerance because their
existing atomic accumulation order is not fixed.

Promise-gated tasks show near preparation can finish while far work remains
suspended, with no premature momentum, rung, output-field, or sum updates.
Both branch failure directions allow retry without an early kick. Tests cover
range failures, move/one-use contracts, and consuming a failed staged kick so
cached lanes cannot kick partially updated particles twice. Cold and cached
real periodic convolution also match the serial ordering. Gates release and
drain launched tasks on exceptional exits.

An initial two-worker test run failed in GoogleTest's OS-thread-local trace
cleanup after an HPX suspension. Moving trace scopes after all scheduling and
future waits resolved it; production arithmetic did not change for that fix.

The application and test target built in both configurations. Selecting
`Pmmm*:Kick*:Fft*:ParticleGhostExchangeTest.*:ParallelChunks.*` gave:

| Build | HPX workers | Passed | Skipped |
| --- | ---: | ---: | ---: |
| Release | 1 | 114 | 14 |
| Release | 2 | 114 | 14 |
| Validation enabled | 1 | 124 | 4 |
| Validation enabled | 2 | 124 | 4 |

Release skips ten validation-only cases and four cases requiring more than one
physical locality. Validation skips only those four distributed cases. The
validation harness uses Release HPX with `COSMICTIGER_VALIDATION_CHECKS=1`,
`-fno-fast-math`, `-ftrapping-math`, and `-mno-daz-ftz`, not a native Debug/APEX
installation. Serial builds and runtime checks ran exclusively; runtime
samples reported at most six OS threads. Formatting and whitespace checks
passed. Multi-process MPI, CUDA, full executable initialization/output, and
larger worker pools were not rerun in this follow-up. No phase speedup was
measured and no QB4 access was used.


### Initialization chunking validation (2026-10-05)

The Release and validation-enabled CPU initialization targets rebuilt and were
run with one and two HPX workers, using the bounded runtime harness. Both
application targets also built successfully. Each Release
run passed 11 of 14 tests; two validation-only cases and the opt-in real-CAMB
smoke test were skipped. Each validation-enabled run passed 13 of 14 tests,
with only real CAMB skipped. The validation build uses the installed Release HPX
with application validation checks and strict floating-point flags; this is not
a native Debug/APEX configuration.

The expanded tests exercise 4096-cell seeded repeatability and analytic growth,
and compare 32768 particles across one versus eight logical localities. A
truncated spectrum makes selected sites fail inside density-assembly chunks;
all eight sites receive the agreed error, including sites whose chunks succeed.
A separate exact snapshot check compared the original serial initializer with
both one- and two-worker chunked runs: all 852040 bytes of sorted particle
positions, velocity bits, rung/ghost values, and handoff report fields matched.

CTest now registers both one- and two-worker initialization entries. No phase
timing or speedup claim is made. Physical multi-process MPI, CUDA, and the
opt-in real-CAMB smoke test were not repeated for this local-loop change.
