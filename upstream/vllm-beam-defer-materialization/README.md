# vLLM offline beam search — rank candidates before building their histories

`0001-rank-before-materializing.patch`: each step produces 2·B² candidates (B beams × 2B
logprobs) and used to copy the parent's full token list — which starts with the whole prompt —
and logprob list for every one of them before keeping B. Candidates are now scored from
(parent length + 1, last token, cumulative logprob) with the same stable reverse sort, and only
survivors are built. EOS completions are still built immediately, in order.

| Evidence | File |
| --- | --- |
| Per-step beam-logic CPU time on the real `LLM.beam_search` with a fake engine, main vs change (B 4–64, prompt 128–4096) | `results/cpu-step-bench-{main,change}.json`, `harness/beam_harness.py` (`bench`) |
| 640 full runs, byte-identical outputs (and the 0.29.0 port identical too) | `results/differential-dumps.sha256`, `harness/beam_harness.py` (`diff`) |
| End to end on L20 and GPU bit-equality | [`../vllm-beam-e2e-l20/`](../vllm-beam-e2e-l20/) |

CPU beam logic per step (ms, one pinned core): B=8/p4096 1.59 → 0.17, B=16/p4096 14.2 → 0.47,
B=32/p1024 15.4 → 1.25, B=32/p4096 52.8 → 1.54, B=64/p1024 95.2 → 6.05. On main the cost grows
with prompt length; with the change it barely does.

The new unit tests fail on an off-by-one candidate length (4/6) and on a changed tie order (5/6).
