#!/usr/bin/env python3
"""Step time of one prefill chunk at a fixed cached depth, swept over chunk size (vLLM, tracer v2).

A prefix of length K is put in the prefix cache; then single requests (prefix + fresh suffix of q
tokens) are submitted while one background request decodes, with max_num_batched_tokens large
enough that the whole suffix runs in one step: the step is exactly (q @ K) plus the decode row.
q runs over --chunks in a shuffled order per round, --repeats rounds, fresh suffix tokens every
trial. Each executed step is identified by its own (chunk, depth) in the trace.

  python scripts/measure_chunk_staircase.py --model M --depth K --trace-dir DIR     (one depth per process)
  python scripts/measure_chunk_staircase.py --analyze --trace-dir DIR --output OUT.json
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

CHUNKS = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]


def run(args):
    import asyncio
    rnd = random.Random(args.depth)
    toks = lambda n: [rnd.randrange(1000, args.vocab) for _ in range(n)]
    stem = f"stair-k{args.depth}"
    os.environ["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{stem}.jsonl")
    os.environ["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{stem}.steps.jsonl")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def go():
        eng = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
            enable_prefix_caching=True, max_num_batched_tokens=max(CHUNKS) + 64, max_num_seqs=8,
            enable_logging_iteration_details=True, disable_log_stats=False, seed=0))
        n = [0]

        async def gen(ids, max_tokens):
            n[0] += 1
            async for _ in eng.generate({"prompt_token_ids": ids}, SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True), f"r{n[0]}"):
                pass

        prefix = toks(args.depth)
        if prefix:
            await gen(prefix, 1)
        stop = asyncio.Event()

        async def blocker():
            while not stop.is_set():
                await gen(toks(16), 4000)

        bt = asyncio.create_task(blocker())
        await asyncio.sleep(1.0)
        for q in CHUNKS:                      # warm-up of every shape, not recorded
            await gen(prefix + toks(q), 1)
        order = []
        for r in range(args.repeats):
            qs = CHUNKS[:]; rnd.shuffle(qs); order += qs
        for q in order:
            await asyncio.sleep(0.03)
            await gen(prefix + toks(q), 1)
        stop.set(); bt.cancel()
        eng.shutdown()
        return order

    order = asyncio.run(go())
    (args.trace_dir / f"{stem}.plan.json").write_text(json.dumps({"depth": args.depth, "model": args.model, "order": order,
                                                                   "warmup_per_chunk": 1}) + "\n")


def analyze(args):
    from step_trace_join import load_joined
    out = {"depths": []}
    for plan_path in sorted(args.trace_dir.glob("stair-k*.plan.json")):
        pl = json.load(open(plan_path))
        K = pl["depth"]
        rows, info = load_joined(plan_path.with_name(plan_path.name.replace(".plan.json", ".jsonl")))
        by_q = {q: [] for q in CHUNKS}
        seen = {q: 0 for q in CHUNKS}
        for r in rows:
            if r["ctx_reqs"] != 1 or r["gen_reqs"] != 1 or list(r.get("ctx_depths", [])) != [K] or r["cuda_ms"] != r["cuda_ms"]:
                continue
            q = r["ctx_chunks"][0]
            if q in by_q:
                seen[q] += 1
                if seen[q] > pl["warmup_per_chunk"]:
                    by_q[q].append(r["cuda_ms"])
        med = {q: statistics.median(v) for q, v in by_q.items() if v}
        qs = sorted(med)
        inc = {f"{a}->{b}": med[b] - med[a] for a, b in zip(qs, qs[1:])}
        out["depths"].append({"depth": K, "median_ms": med, "n": {q: len(v) for q, v in by_q.items()}, "increments_ms": inc,
                              "join_match_frac": info["match_frac"]})
        print(f"depth {K}: " + "  ".join(f"{q}:{med[q]:.2f}" for q in qs))
        print("   increments: " + "  ".join(f"{k} {v:+.2f}" for k, v in inc.items()))
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--depth", type=int)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=151_000)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
