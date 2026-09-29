# Delayed cleanup ownership: review packet

## What changed

The proposed change binds delayed HTTP stream cleanup to the original request objects
and ReqState identities. Reusing a textual RID no longer authorizes an older cleanup
task to cancel a replacement. The public method signature, two-second grace period,
and normal active-request abort remain unchanged. This is a reliability change, not
a CUDA speedup or a fix for complete-length token-value divergence.

A real Qwen HTTP reproduction ran original and modified owned-server processes on
one RTX4090. In the original, the second request in each of two reused-RID epochs
was aborted at66 and65 tokens instead of128. Logged cleanup owner identities differed
from the registered target request identities. Four unique-RID controls completed.
The modified server completed all8/8 fixed requests with no stale cleanup aborts.
The original completed6/8. All expected failed outputs and their SSE frames are retained.

## Prior work and source binding

SGLang PR38850 (scripted-runtime test-harness reliability) already discusses delayed
cleanup and request-ID reuse. It uses a non-streaming workaround in that harness.
This packet does not claim discovery priority or authorship of that PR. The reviewed
upstream method at211b1d9784d15845566081cc25c8103ce43cc4f2 matches the installed original
method. Full installed-file SHA256:952fdfcf54b13af6851993d4b7599d4b1efeb6aa630a7554584a1a047eda94f2.

References:
- https://github.com/sgl-project/sglang/pull/38850
- https://github.com/sgl-project/sglang/blob/211b1d9784d15845566081cc25c8103ce43cc4f2/python/sglang/srt/managers/tokenizer_manager.py

## Tests and limitations

Twelve CPU tests execute the exact original method and the proposed replacement,
covering replacement before/during the delay, the same object with a new state,
active, finished, removed and batched requests. Tests reproduce the original stale
abort before asserting the fix. GPU execution checks two sequential reused-RID epochs
and four unique-ID controls in an owned single-worker HTTP server.

This is not full production validation. Remaining review topics: batched n>1 sampling,
multi-worker routing, cancellation during pre-dispatch, disconnect handling, callback
failure propagation, and upstream registered test integration. Holding ReqState references
for the grace period can retain result buffers; memory-retention cost should be reviewed
or replaced with generation/weak-reference tokens without reintroducing ABA problems.
Other cleanup paths are outside this narrow patch. No shared installation was modified,
no official upstream PR was submitted, and no inference default was enabled.

The source-exact patch is included in the campaign archive under
http-v1/abort-source-patch-v1/tokenizer-abort-ownership.patch. `abort_ownership.py`
can regenerate it from the reviewed installed source, refusing a different method.
