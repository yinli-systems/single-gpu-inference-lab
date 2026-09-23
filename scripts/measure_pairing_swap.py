#!/usr/bin/env python3
"""Pairing-swap counterexample: two prefill steps with identical marginals, different q-KV pairing.

Two partial prefills at KV depths (k_a, k_b) receive chunks (q_a, q_b) in state A and the swapped
(q_b, q_a) in state B. Prefill count, total tokens, total KV, the multisets of chunks and depths,
their max/min/variance, decode state and padded token count are all identical; only the pairing
changes. The attention-work difference is (q_a - q_b)(k_a - k_b); the q(q+1)/2 term is identical.

Construction (exact, through vLLM's own scheduler): the prefixes of length k_a and k_b are put in
the prefix cache first; then both requests (prefix + fresh suffix of length q) are submitted
together, while one background request is decoding, with max_num_batched_tokens = q_a + q_b + 1,
so the next engine step executes exactly (q_a @ k_a, q_b @ k_b) plus the one decode row. The shallow request is always first in the batch. Every executed step
is checked against the intended (chunks, depths) in the engine trace; mismatches are dropped and
counted. States alternate A, B, B, A, ... and every trial uses fresh suffix tokens.

  python scripts/measure_pairing_swap.py --model M --config-index I --trace-dir DIR     (one per config)
  python scripts/measure_pairing_swap.py --analyze --trace-dir DIR --slope S --output OUT.json
One configuration per process (the tracer reads its output path at import; one engine per process
gives a fresh cache); analysis runs after all processes have exited and flushed their traces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (label, k_a, k_b, q_a, q_b): state A pairs the small chunk with the shallow depth
CONFIGS = [
    ("k4k/16k q256/768", 4096, 16384, 256, 768),
    ("k4k/16k q128/896", 4096, 16384, 128, 896),
    ("k4k/16k q384/640", 4096, 16384, 384, 640),
    ("k8k/24k q256/768", 8192, 24576, 256, 768),
    ("k0/16k q256/768", 0, 16384, 256, 768),
    ("k4k/16k q512/512 (control, dW=0)", 4096, 16384, 512, 512),
]


def run(args):
    """AsyncLLM with one background decoder so the engine is mid-step when A and B are submitted
    together; the next step is then exactly [decode x1] + (q1 @ k_a) + (q2 @ k_b) with budget
    q_a + q_b + 1 (the decode row is identical in both states)."""
    import asyncio
    rnd = random.Random(args.config_index)
    toks = lambda n: [rnd.randrange(1000, args.vocab) for _ in range(n)]
    label, ka, kb, qa, qb = CONFIGS[args.config_index]
    stem = f"pairswap-{args.config_index}"
    os.environ["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{stem}.jsonl")
    os.environ["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{stem}.steps.jsonl")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def go():
        eng = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
            enable_prefix_caching=True, max_num_batched_tokens=qa + qb + 1, max_num_seqs=8,
            enable_logging_iteration_details=True, disable_log_stats=False, seed=0))
        n = [0]

        async def gen(ids, max_tokens):
            n[0] += 1
            async for _ in eng.generate({"prompt_token_ids": ids}, SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True), f"r{n[0]}"):
                pass

        pa, pb = toks(ka), toks(kb)
        for p in (pa, pb):
            if p:
                await gen(p, 1)                                  # put the prefixes in the cache
        stop = asyncio.Event()

        async def blocker():                                     # keeps exactly one decode running
            while not stop.is_set():
                await gen(toks(16), 4000)

        bt = asyncio.create_task(blocker())
        await asyncio.sleep(1.0)
        plan = []
        for i in range(args.repeats):
            plan += ["A", "B"] if i % 2 == 0 else ["B", "A"]
        for _ in range(4):                                       # warm-up of this batch shape, not recorded
            await asyncio.gather(gen(pa + toks(qa), 1), gen(pb + toks(qb), 1))
        intended = []
        for state in plan:
            q1, q2 = (qa, qb) if state == "A" else (qb, qa)
            await asyncio.sleep(0.05)
            await asyncio.gather(gen(pa + toks(q1), 1), gen(pb + toks(q2), 1))
            intended.append([state, [q1, q2], [ka, kb]])
        stop.set(); bt.cancel()
        eng.shutdown()
        return intended

    intended = asyncio.run(go())
    (args.trace_dir / f"{stem}.plan.json").write_text(json.dumps(
        {"config_index": args.config_index, "label": label, "k": [ka, kb], "q": [qa, qb], "model": args.model, "intended": intended}) + "\n")


def analyze(args):
    from step_trace_join import load_joined
    results = {"configs": []}
    for plan_path in sorted(args.trace_dir.glob("pairswap-*.plan.json")):
        pl = json.load(open(plan_path))
        (ka, kb), (qa, qb), intended = pl["k"], pl["q"], pl["intended"]
        rows, info = load_joined(plan_path.with_name(plan_path.name.replace(".plan.json", ".jsonl")))
        # Each executed step identifies its own state by its (chunks, depths); order-independent, so a
        # trial whose two prefills landed in different steps simply does not appear. The first 4
        # A-shaped steps are the warm-up.
        pre = [r for r in rows if r["ctx_reqs"] == 2 and r["gen_reqs"] == 1 and list(r.get("ctx_depths", [])) == [ka, kb]
               and r["cuda_ms"] == r["cuda_ms"]]
        got = {"A": [], "B": []}; warm = 4
        for r in pre:
            ch = list(r["ctx_chunks"])
            if ch == [qa, qb] and warm:
                warm -= 1; continue
            if ch == [qa, qb]:
                got["A"].append(r["cuda_ms"])
            elif ch == [qb, qa]:
                got["B"].append(r["cuda_ms"])
        if qa == qb:                       # control: A and B are the same shape; split by order
            allr = got["A"]; got = {"A": allr[0::2], "B": allr[1::2]}
        mismatched = len(intended) - len(got["A"]) - len(got["B"])
        wa = qa * ka + qb * kb; wb = qb * ka + qa * kb
        cfg = {"label": pl["label"], "model": pl["model"], "k": [ka, kb], "q_state_A": [qa, qb], "q_state_B": [qb, qa],
               "cross_work_A_M": wa / 1e6, "cross_work_B_M": wb / 1e6, "dW_A_minus_B_M": (wa - wb) / 1e6,
               "cuda_ms_A": got["A"], "cuda_ms_B": got["B"], "trials_not_in_one_step": mismatched,
               "steps_found": len(pre), "join_match_frac": info["match_frac"]}
        if got["A"] and got["B"]:
            cfg["median_A_ms"] = statistics.median(got["A"]); cfg["median_B_ms"] = statistics.median(got["B"])
            cfg["delta_A_minus_B_ms"] = cfg["median_A_ms"] - cfg["median_B_ms"]
            if args.slope:
                cfg["predicted_delta_ms"] = args.slope * cfg["dW_A_minus_B_M"]
        results["configs"].append(cfg)
        pred = f" predicted {cfg['predicted_delta_ms']:+6.2f}" if "predicted_delta_ms" in cfg else ""
        print(f"{pl['label']:34s} dW {cfg['dW_A_minus_B_M']:+6.2f} M | A {cfg.get('median_A_ms', float('nan')):7.2f}  "
              f"B {cfg.get('median_B_ms', float('nan')):7.2f}  delta {cfg.get('delta_A_minus_B_ms', float('nan')):+6.2f} ms{pred} "
              f"(n {len(got['A'])}/{len(got['B'])}, mismatched {mismatched})", flush=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--repeats", type=int, default=20, help="trials per state per config")
    ap.add_argument("--config-index", type=int, help=f"0..{len(CONFIGS) - 1} (run mode)")
    ap.add_argument("--vocab", type=int, default=151_000)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true", help="parse every pairswap-*.plan.json in --trace-dir")
    ap.add_argument("--slope", type=float, help="ms per million attention work, for the predicted delta")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
