# Pairing swap: same marginals, different cost

**Question.** The partition result (1×2048 vs 8×256) changes the number of prefills, so a reviewer
can ask whether count or fragmentation explains it. What is the cleanest possible counterexample
to "the aggregate step state determines step cost"?

**Construction.** Two partial prefills at KV depths (k_a, k_b) get chunks (q_a, q_b) in state A
and the swapped (q_b, q_a) in state B. Between the two states the following are identical:
- prefill count, Σq and Σk;
- the multiset of chunks and the multiset of depths, and therefore their max, min and variance;
- the decode state (one background decoder) and the padded token count.

**Only the pairing of chunk to depth changes.** The attention work differs by exactly
ΔW = (q_a − q_b)(k_a − k_b); the q(q+1)/2 term is identical.

The states are built through vLLM's own scheduler, and every executed step is checked against
the intended (chunks, depths):
- the two prefixes are put in the prefix cache first;
- both requests (prefix + fresh suffix) are submitted together with `AsyncLLM` while one
  background request decodes, with `max_num_batched_tokens = q_a + q_b + 1`, so the next step
  executes exactly (q_a @ k_a, q_b @ k_b) plus the decode row;
- 20 trials per state, A and B alternating, fresh suffix tokens every trial.

[`scripts/measure_pairing_swap.py`](../../../scripts/measure_pairing_swap.py). Pre-registered in
addendum 8 A of
[`docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md`](../../../docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md).

## Result

[`summary.md`](summary.md). Qwen3-4B. Predicted Δ uses the published partition-fit slope (L20 6.5,
A100 3.3 ms per million units of attention work).

| setup | Δ measured, 5 configs (ms) | measured / predicted | control (ΔW = 0) | PS1 (±25%) | PS2 (\|Δ\| ≤ 1 ms) |
| --- | --- | --- | ---: | --- | --- |
| L20, vLLM 0.29 | +18.6 … +54.9 | 0.86 – 1.01 | −0.09 ms | **held** (5/5) | **held** |
| L20, vLLM 0.30 | +18.5 … +55.0 | 0.86 – 1.01 | −0.05 ms | **held** (5/5) | **held** |
| A100, vLLM 0.29 | +7.0 … +27.6 | 0.65 – 0.89 | +0.01 ms | **failed** (2/5 within ±25%) | **held** |

- **The counterexample holds on every setup.** Swapping only the pairing changes the step by
  7–28 ms on the A100 and 19–55 ms on the L20, always in the predicted direction. The control
  (equal chunks, so the swap changes nothing) moves by ≤ 0.09 ms.
- **Identifiability bound.** A and B map to the same aggregate vector, so any deterministic
  predictor on aggregate features has mean absolute error ≥ |Δ|/2 on the pair: 9–27 ms on the
  L20, 3.5–14 ms on the A100, whatever its capacity. This is the missing information, measured
  directly.
- **Magnitude on the A100 (pre-registered PS1 failed).** The measured effect is 65–89% of what
  the partition-fit slope predicts; on the L20 it is 86–101%. The pairing (cross) term therefore
  costs less per unit on the A100 than the overall attention-work slope fitted on partitions
  implies. The q(q+1)/2 part of the proxy, identical in both states here, carries relatively more
  of the fitted slope on that GPU. The sign and the existence of the effect are not in question;
  the single-slope proxy is an approximation whose cross-term coefficient is GPU-dependent.
- **vLLM 0.30 reproduces 0.29 to within 0.7 ms** on every config.

## Getting the measurement right (three failed attempts, kept)

Three setups produced no usable data before the one reported. Their directories were kept on the
hosts and are not used.
1. **In-process engine** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) skips the engine busy loop that
   hosts the iteration tracer.
2. **The `LLM` class** turns stats logging off, and iteration details only run with it on.
3. **Sequential submission** from `LLM.generate` let the idle engine schedule the first request
   before the second arrived, splitting the pair across steps.

The final harness keeps one request decoding so the engine is mid-step when the pair arrives.
Analysis classifies each executed step by its own (chunks, depths), so a trial that still split
simply does not appear: 94–100% of trials were captured.

## Reproduce

```bash
for i in 0 1 2 3 4 5; do python scripts/measure_pairing_swap.py --model Qwen/Qwen3-4B --config-index $i --trace-dir OUT; done
python scripts/measure_pairing_swap.py --analyze --trace-dir OUT --slope 6.5 --output OUT/pairswap.json
```

Traces in `raw/*/` are gzipped; gunzip them before `--analyze`.
