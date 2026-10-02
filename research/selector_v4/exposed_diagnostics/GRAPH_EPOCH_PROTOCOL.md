# Fixed-capture Resource Graph metadata diagnostic

This is a separate development experiment, using the committed normal
FlashInfer75544a17 source. It does not change that source, the running
0a5c735/c34d5d9 qualification sources, the consumed ten-case canary, or promotion.

The public Graph runner still prohibits `rebind_same_geometry`. This experiment
does not bypass that method and call the result a certified runner. It explicitly
captures the uncertified Resource operation once, then owns its captured
metadata and workspaces through a separate research-only transaction.

## Frozen population and operations

Use the previously exposed query lengths `[3,39,107,175,243,311,411]` and
cached-prefix lengths `[22528,16512,11840,8256,2624,672,80]`. Page size is1,
Q heads32, KV heads8, head dimension128, unsplit FA2, causal=false.
Run FP16/BF16, NHD/HND, packed/tuple K/V, Graph1/Graph16:16 cells per card.
Both GPU4090 and GPU5090 allocations have an initial one-hour limit; no retry.

Each cell must use the same captured graph object and owned buffer addresses
across three announced metadata epochs. Each epoch changes physical page
indices and actual Q/K/V payloads, calls the real original Native planner,
validates the same ordered geometry and15 planner values, copies scheduling
state and metadata into the existing captured buffers, and replays the graph.

Each epoch establishes current Native reference O/LSE before Resource replay;
Resource and subsequent Native-after-Resource must both match that reference.
Output and page
digests must actually change. Four traced calls per epoch must show4 or64
attention launches, all with actual65536 shared-memory bytes. Retain each raw
CUPTI/Kineto trace; zero profiler occupancy estimates are not residency evidence.

Update-time replay raises before either Native or Resource execution. An
unannounced physical metadata write, including an inference tensor lacking a
version counter, retires the graph: the diagnostic reads GPU metadata values.
A changed ordered query partition with the same total Q storage must also
retire and use current Native. A poisoned graph object's replay method ensures
these rejections cannot accidentally execute the stale graph.

Private workspaces are fully copied on transition. GPU readback, synchronization,
Native planning, a pre-Resource Native control, payload updates, copying and four
calls are included in the recorded transaction wall time. This contains a
validation control and must not be interpreted as deployment-step timing.
This deliberately conservative implementation is
not a low-overhead serving path. Three diagnostic observations provide no
performance confidence interval. Pure replay timing is not substituted for it.

## Required evidence and interpretation

Each job requires Slurm COMPLETED0:0, shell exit0, exactly24 pytest cases
(8 CPU boundaries plus16 CUDA cells), zero skipped/failed/error cases, all16
result files with three epochs each, actual Resource launch traces, immutable
helper/package/SDK pre/post checks, and an independent full-instruction SASS
comparison of the actual Native and Resource libraries compiled in that job.
Pytest8.4.2 and its existing dependencies are copied into a diagnostic-private
directory from the already hash-bound normal wheel validation test support.
Their own complete file ledger is checked before and after each allocation;
the shared Python environment and validation source are never installed into.

Every submission has a durable pre-submission intent and confirmed receipt;
unknown outcomes block duplicate submission. A finite controller archives only
after both jobs stop and preserves any failure or partial test population.
No re-sampling or continuation replaces failed records.

Even a functional PASS means only fixed-geometry physical-metadata updates on
this minimal driver. It cannot authorise managed v2 Graph selection, real
SGLang Graph lifecycle updates, fresh release, full HTTP, historical2/432 closure,
or default/serving promotion. Every result retains those authorities as false.
