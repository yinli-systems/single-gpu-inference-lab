# Bounded layer/operator and physical-KV evidence

Separate correctness diagnostic, not performance or a replacement for historical failures.
Pinned 36-layer Qwen, standard math, fixed seed42, overlap disabled, original pristine/cap.
Clone inputs/outputs of each decoder layer norm, QKV projection, RadixAttention,
output projection and MLP projections into graph-owned shadow tensors. The original tensors
and returned values are never mutated. Decode graph size is read from the actual runner,
not inferred from request count. Prefill captures cannot overwrite decode banks.

At the two exact historical common-prefix hashes before token indices74 and114, retain
up to8 snapshots per server: all36 layer shadows and each request's actual paged KV rows,
plus physical slot indices and logits. No model weights are copied. Store raw tensors only
on the research filesystem; public receipts can publish their hashes. The observer alters
launches/memory/timing and may remove or create scheduling-sensitive observations.

A differing hidden/KV tensor locates the earliest OBSERVED divergence in this instrumented
run, not automatically the first cause in the old uninstrumented run. Equal histories do
not imply equal KV/hidden inputs. No root-cause claim without checking actual operator inputs.
One GPU,30-minute bound,two servers. Abort on unsupported layouts, missing shadows or errors.
No changed performance configuration is promoted and no new performance repeats are submitted.
