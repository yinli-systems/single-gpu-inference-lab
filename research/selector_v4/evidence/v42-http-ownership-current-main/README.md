# Latest pinned SGLang HTTP ownership PASS

Upstream baae019, candidate3825c5a4, full36-layer BF16 Qwen3-4B-Instruct-2507 on ParaCloud RTX4090. Job1644258 COMPLETED0:0,12m56s. Official FlashInfer0.7.0.post1 wheel SHA256c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716 is isolated from the shared environment. All49 registered CPU tests passed with full production imports.

Each arm executes11 requests:10 expected complete plus one intentional n=2 streaming disconnect. Original8/10; both reused-RID replacements are aborted by stale cleanup. Candidate10/10, zero stale-owner aborts, zero remaining owned states after cleanup. Batch, n=2 child ownership, streaming disconnect and recovery are covered. Both arms completed upstream prefill CUDA Graph initialization before HTTP requests.

43 raw files (source ledgers, binding, observer/probe/launcher, SSE responses, cleanup events, model/server/allocation logs) are SHA256-bound, archived and verified again locally. Resource cap is disabled. This is an independent request-lifetime diagnostic and makes no attention-performance or historical2/432 token-parity claim.
