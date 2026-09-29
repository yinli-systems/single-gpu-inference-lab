# Matched sampler-boundary diagnostic

Separate from the frozen standard/deterministic HTTP replay. Both pristine and cap use
seed 42 and disable overlap scheduling. Only test-owned server processes install a
sampler observer. It does not modify logits, input tensors, weights or token choices.
CPU snapshots alter timing and may remove the original failure: absence of divergence
is not evidence of root-cause resolution. No measurements here are performance data.

For each sampled step record actual sequence lengths, input IDs, request-pool slots,
reconstructed complete history hashes where possible, top-eight pre-sampling logits,
full-row logit hashes and sampled token IDs. Save at most eight full-logit snapshots
near sequence lengths 201..203 and 241..243 per process. Unknown history is explicitly
marked and cannot support a same-input comparison. A sampler-boundary disagreement
is not by itself the first attention/operator disagreement or proof of a race.

One single-GPU job, pristine then cap, maximum 25 minutes; no retry loop. Source bound
before submission. Original failures remain HOLD. All raw snapshots retained, never
mixed into HTTP throughput or confidence intervals.
