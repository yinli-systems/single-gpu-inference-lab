# Official-path applicability audit — 2026-09-28

Sources were read directly, not inferred from titles or version labels.

## What is established by the official sources

- https://flashinfer.ai/2026/09/22/autotuner-v2.html explicitly separates eager recurring host/device cost from captured CUDA Graph replay. The deployment context belongs in the saved winner's identity. Its reported tactic-selection results are not end-to-end serving speedups.
- https://flashinfer.ai/2026/09/22/flashinfer-v07.html introduces the opt-in v0.7 redesign and persistent result reuse. This is not evidence that an arbitrary pre-existing wrapper automatically participates in tuning.
- https://docs.flashinfer.ai/api/pod.html documents auxiliary-structure reuse across layers. POD is a different paged/fused surface from this experiment's ragged wrapper.
- The exact pinned ragged wrapper ALSO already documents layer reuse: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/prefill.py (example and note around lines3200–3282). Thus the applicability is not merely an analogy from POD. Reuse itself is unequivocally prior art.

The official Qwen3-4B and Qwen3-8B configuration files both list36 hidden layers,32 attention heads,8 KV heads and head_dim128:
https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
Both configurations specify BF16 weights. Our FP16 attention replay uses their common dimensions, not their weights or precision, and is not a two-model benchmark.

## What was actually inspected in the installed environment

Pinned installation:
`/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/lib/python3.12/site-packages/flashinfer`

Installed version0.6.18. Full prefill.py SHA256:
`743953a8d301d89c6943dc48a8f6b393c0d636b71b165ae1842c2c65c7defaab`.

AST inspection found `BatchPrefillWithRaggedKVCacheWrapper` at lines3193–4415. Its implementation calls the compiled module's `plan` at3927 and `ragged_run` at4377. Explicit backend='fa2' bypasses automatic backend choice. The entire installed prefill.py has zero occurrences of `AutoTuner`, `choose_one`, `autotune_v2`, and `MeasurementPolicy`. Runtime import confirmed that both `flashinfer` and `flashinfer.autotuner` do not export `autotune_v2`; the latter also does not export `MeasurementPolicy`. The autotuner package itself exists, but existence is not registration of the FA2 ragged operation.

Therefore **Autotuner v2 is not a directly callable strong control for this pinned0.6.18 FA2 path**. We do not label an ineffective tuning context or our TRAIN-selected fixed split as the official autotuner. We keep stock FA2 auto and a TRAIN-selected memory-bounded fixed policy, both with equal plan reuse and actual eager wall timing. A future0.7 upgrade, registered custom runner, different backend, or graph-mode test must be identified as a separate integration/version experiment. Any empirical per-shape tuner would need independent selection/timing samples, cold tuning cost and compatible cache identity; none has been claimed here.

## Scope of the current cache

This experiment tests an in-process, capacity-two decision LRU and one currently active materialized plan per arm. It does not implement FlashInfer v2's persistent store, cross-process reuse, atomic file publication or rank convergence. All arms start each episode cold, then execute the same four-segment state sequence. The unchanged sequence at36 layers runs144 attention calls under one plan; alternating/eviction sequences replan once per36-layer segment. The primary reported quantity is the directly timed episode, not CPU cost divided by36.

Execution modes unsupported by this harness are rejected before a plan is reused. No eager-to-graph transfer, model-level speedup, or new invention of caching is asserted.
