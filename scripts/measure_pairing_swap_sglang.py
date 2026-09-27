#!/usr/bin/env python3
"""Pairing swap on SGLang: same construction and configurations as measure_pairing_swap.py (vLLM).

Two prefixes of lengths (k_a, k_b) are put in the radix cache; then both requests (prefix + fresh
suffix of q tokens) are submitted together while one background request decodes, with
chunked_prefill_size = q_a + q_b, so the next extend batch is exactly (q1 @ k_a, q2 @ k_b). SGLang
does not mix decode rows into extend batches by default, so, unlike vLLM, the step has no decode
row. States alternate A, B, B, A, ... with fresh suffix tokens every trial. Each executed extend
forward is identified by its own (prefix, extend) lengths from the step tracer in
integrations/sglang_step_tracer (put that directory on PYTHONPATH); trials that split across
batches do not appear.

  PYTHONPATH=integrations/sglang_step_tracer python scripts/measure_pairing_swap_sglang.py --model M --config-index I --trace-dir DIR
  python scripts/measure_pairing_swap_sglang.py --analyze --trace-dir DIR --output OUT.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_pairing_swap import CONFIGS  # noqa: E402


def run(args):
    import asyncio
    rnd = random.Random(args.config_index)
    toks = lambda n: [rnd.randrange(1000, args.vocab) for _ in range(n)]
    label, ka, kb, qa, qb = CONFIGS[args.config_index]
    stem = args.trace_dir / f"sglswap-{args.config_index}"
    assert os.environ.get("SGL_EXP_STEP_TRACE") is None
    os.environ["SGL_EXP_STEP_TRACE"] = str(stem) + ".steps.jsonl"
    import sglang as sgl

    eng = sgl.Engine(model_path=args.model, context_length=args.max_model_len, mem_fraction_static=0.8,
                     chunked_prefill_size=qa + qb, max_running_requests=8, random_seed=0, log_level="warning")

    async def go():
        async def gen(ids, max_tokens):
            async for _ in await eng.async_generate(input_ids=ids, stream=True,
                                                    sampling_params={"max_new_tokens": max_tokens, "temperature": 0, "ignore_eos": True}):
                pass

        pa, pb = toks(ka), toks(kb)
        for p in (pa, pb):
            if p:
                await gen(p, 1)
        stop = asyncio.Event()

        async def blocker():
            while not stop.is_set():
                await gen(toks(16), 4000)

        async def pair(ids1, ids2):
            # one batched request: both reach the scheduler in the same message, so they cannot be split
            # across extend batches by arrival timing (separate submissions were, in the harness smoke)
            await eng.async_generate(input_ids=[ids1, ids2],
                                     sampling_params=[{"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}] * 2)

        bt = asyncio.create_task(blocker())
        await asyncio.sleep(1.0)
        plan = []
        for i in range(args.repeats):
            plan += ["A", "B"] if i % 2 == 0 else ["B", "A"]
        for _ in range(4):
            await pair(pa + toks(qa), pb + toks(qb))
        intended = []
        for state in plan:
            q1, q2 = (qa, qb) if state == "A" else (qb, qa)
            await asyncio.sleep(0.05)
            await pair(pa + toks(q1), pb + toks(q2))
            intended.append([state, [q1, q2], [ka, kb]])
        stop.set(); bt.cancel()
        return intended

    intended = asyncio.get_event_loop().run_until_complete(go())
    eng.shutdown()
    Path(str(stem) + ".plan.json").write_text(json.dumps(
        {"config_index": args.config_index, "label": label, "k": [ka, kb], "q": [qa, qb], "model": args.model,
         "engine": "sglang " + sgl.__version__, "intended": intended}) + "\n")


def analyze(args):
    results = {"configs": []}
    for plan_path in sorted(args.trace_dir.glob("sglswap-*.plan.json")):
        pl = json.load(open(plan_path))
        (ka, kb), (qa, qb), intended = pl["k"], pl["q"], pl["intended"]
        rows = []
        for f in glob.glob(str(plan_path).replace(".plan.json", ".steps.jsonl") + ".*"):
            rows += [json.loads(l) for l in open(f)]
        got = {"A": [], "B": []}; warm = 4
        pre = []
        for r in sorted(rows, key=lambda r: r["id"]):
            if not r["mode"].startswith("EXTEND") or len(r["extend"]) != 2:
                continue
            pair = dict(zip(r["prefix"], r["extend"]))
            if sorted(r["prefix"]) != sorted([ka, kb]):
                continue
            pre.append(r)
            q_at_a, q_at_b = pair.get(ka), pair.get(kb)
            if (q_at_a, q_at_b) == (qa, qb):
                if warm:
                    warm -= 1; continue
                got["A"].append(r["cuda_ms"])
            elif (q_at_a, q_at_b) == (qb, qa):
                got["B"].append(r["cuda_ms"])
        if qa == qb:
            allr = got["A"]; got = {"A": allr[0::2], "B": allr[1::2]}
        wa = qa * ka + qb * kb; wb = qb * ka + qa * kb
        cfg = {"label": pl["label"], "engine": pl.get("engine"), "k": [ka, kb], "q_state_A": [qa, qb], "q_state_B": [qb, qa],
               "dW_A_minus_B_M": (wa - wb) / 1e6, "cuda_ms_A": got["A"], "cuda_ms_B": got["B"],
               "trials_not_in_one_step": len(intended) - len(got["A"]) - len(got["B"]), "steps_found": len(pre)}
        if got["A"] and got["B"]:
            cfg["median_A_ms"] = statistics.median(got["A"]); cfg["median_B_ms"] = statistics.median(got["B"])
            cfg["delta_A_minus_B_ms"] = cfg["median_A_ms"] - cfg["median_B_ms"]
        results["configs"].append(cfg)
        print(f"{pl['label']:34s} dW {cfg['dW_A_minus_B_M']:+6.2f} M | A {cfg.get('median_A_ms', float('nan')):7.2f}  "
              f"B {cfg.get('median_B_ms', float('nan')):7.2f}  delta {cfg.get('delta_A_minus_B_ms', float('nan')):+6.2f} ms "
              f"(n {len(got['A'])}/{len(got['B'])}, not in one step {cfg['trials_not_in_one_step']})", flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--config-index", type=int)
    ap.add_argument("--vocab", type=int, default=151_000)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
