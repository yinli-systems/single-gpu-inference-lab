# Parity remediation: bounded diagnostic protocol

Frozen before new observations. Previous timing matrices, failures, and HOLD remain unchanged.
Primary task: reproduce the two cap-only historical divergences without declaring baseline
non-determinism, harmless near-ties, or a race absent without direct evidence.

Two one-GPU jobs: standard execution and the installed upstream deterministic mode.
Each job runs pristine, cap, pristine, cap in new server processes on its assigned GPU.
These are correctness diagnostics, not performance repeats. The deterministic mode changes
execution policy in BOTH arms and is not credited as a novel optimization.

Each server uses the same model/source as historical serving and replays all three old
workloads in their original order for three blocks (including block zero). Then it runs
both exposed requests individually twice and evaluates their common teacher-forced prefixes
with one-token top-logprob readout. These prefix probes change prefill/decode context; they
are not a trace of the original failing autoregressive operator.

Require hashes for historical receipts, unchanged workload construction, version and native
header identity. Separate within-arm repeatability, cross-arm token equality, and equality
against historical trajectories. Diagnostic timings are never merged into performance data.

No per-layer causal attribution without actual same-input layer evidence. Any nonfinite output,
request failure, context mismatch or unexplained token mismatch prevents release. Lack of
reproduction does not resolve the old failure. Preserve startup failures and all new outputs.

Secondary CPU audit: re-evaluate ALL confirmatory cells, denominator=432 cap requests for
historical HTTP; 1296 counts all three arms, not 1296 independently tested cap requests.
Retain all blocks. Report units from JSON fields, not untyped log lines. Generalization is
limited to the frozen shape distribution and conditional controls, never universal.

Keep native source and runtime environments immutable. Do not change live weights,
shared packages, unrelated jobs, clocks, permissions or credentials. Bounded limits:
2 GPUs, <=40 minutes each, at most four server launches per job, no automatic resubmission.
Default OFF; no upstream production PR until parity and no-regression gates pass.
