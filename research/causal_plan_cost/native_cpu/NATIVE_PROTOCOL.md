# Native implementation check (CPU only)

This is a post-evaluation engineering intervention, not a new trained policy.
The source targets the already-frozen causal representation and sklearn models.
No refitting, threshold tuning, new hardware data, or GPU launch is involved.

Required invariants: compare every feature, log-cost prediction and chosen policy
on all registered training, test and diagnostic metadata, both head shapes and
both SM counts. A single differing decision aborts the benchmark. Preserve
scikit-learn tree float32 input conversion; do not compile with fast math.

After equivalence, measure the same native selection 21 blocks of five calls
per held-out geometry. Include Python argument construction, FFI, all six policy
feature computations, all model evaluations and result conversion. Also report
a pre-existing-buffer path separately. Allocation and compilation of the model
are offline. Record host/platform and hashes. Apply both per-case median and
per-case p95 selection costs to the unchanged GPU-cycle observations only as
additive estimates. Neither estimate is a measured integrated plan/run cycle or
vLLM serving speedup. No cross-hardware scaling claim is permitted from host CPU
overhead. A final live-system check remains necessary.

This optimization does not establish scientific novelty: compiled inference and
low-overhead plan selection have extensive prior art. It tests whether Python
implementation overhead, rather than frozen decision quality, is the immediate
reason the candidate failed the net-overhead screen.
