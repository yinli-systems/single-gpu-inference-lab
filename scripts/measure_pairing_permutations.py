#!/usr/bin/env python3
"""Permutation campaign: n prefills with fixed chunk and depth multisets, re-paired, measured live (vLLM).

For one configuration (n cached prefixes at depths k_1..k_n, n chunk sizes q_1..q_n), every *state* is a
pairing pi: request j (prefix of depth k_j) gets a fresh suffix of q_pi(j) tokens. All states have the
same prefill count, token sum, KV sum, chunk multiset, depth multiset and decode state; only the pairing,
and so the coupling C = sum_j q_pi(j) k_j, differs. States: same order (C_max), opposite order (C_min),
random pairings, two different pairings with equal C, and an order null (the same-order pairing
submitted in reverse order: an identical state with a different batch row order).

Construction as in measure_pairing_swap.py: prefixes are put in the prefix cache first (in an order set
by --layout-seed, which changes their physical block placement), then all n requests of a state are
enqueued while scheduling is paused (keeping the prefix cache) and one background request is decoding,
then scheduling resumes with max_num_batched_tokens = sum q + 1, so the next step executes exactly the
n chunks at their depths plus one decode row. Every state goes through the same pause. Design: randomized
complete blocks -- each block runs every state once in a fresh random order, fresh suffix tokens every
trial. Each executed step is identified by its own (chunks, depths) in the engine trace.

  python scripts/measure_pairing_permutations.py --model M --config NAME --layout-seed S --trace-dir DIR
  python scripts/measure_pairing_permutations.py --analyze --trace-dir DIR --output OUT.json
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
from geometry_theory import coupling, coupling_range  # noqa: E402

# name -> (depths k_j of the n prefixes, chunk multiset q, equal-coupling pair or None). The equal-C pair
# is two different pairings with the same C near the middle of [C_min, C_max], chosen offline by
# exhaustive search (n <= 8) or 200k random pairings (n = 16) as the pair differing most per request.
CONFIGS = {
    "n2": ([4096, 16384], [256, 768], None),
    "n4": ([0, 4096, 12288, 24576], [128, 256, 512, 1024],
           ([128, 1024, 512, 256], [512, 256, 1024, 128])),
    "n8": ([0, 2048, 4096, 6144, 8192, 12288, 16384, 24576], [64, 128, 192, 256, 320, 384, 448, 512],
           ([64, 128, 320, 384, 512, 448, 256, 192], [512, 384, 64, 128, 256, 192, 320, 448])),
    "n16": ([1024 * i for i in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28)], [32 * i for i in range(1, 17)],
            ([32, 64, 512, 256, 448, 160, 416, 288, 480, 384, 128, 352, 224, 320, 192, 96],
             [512, 480, 192, 448, 96, 320, 128, 160, 288, 32, 352, 64, 416, 224, 384, 256])),
}
N_RANDOM = 6


def states(depths, chunks, eqc, seed=0):
    """-> list of (label, pairing), pairing[j] = chunk given to prefix j (depths ascending).
    same / opposite: C_max / C_min; rand*: random pairings; eqC-a / eqC-b: two different pairings with equal
    C; order-null: the 'same' pairing submitted in reverse order (identical state, different batch order)."""
    rnd = random.Random(seed)
    qs = sorted(chunks)
    out = [("same", qs[:]), ("opposite", qs[::-1])]
    seen = {tuple(qs), tuple(qs[::-1])}
    if eqc:
        out += [("eqC-a", list(eqc[0])), ("eqC-b", list(eqc[1]))]
        seen |= {tuple(eqc[0]), tuple(eqc[1])}
    tries = 0
    while sum(l.startswith("rand") for l, _ in out) < N_RANDOM and tries < 10000:
        p = qs[:]; rnd.shuffle(p); tries += 1
        if tuple(p) not in seen:
            seen.add(tuple(p)); out.append((f"rand{sum(l.startswith('rand') for l, _ in out) + 1}", p))
    out.append(("order-null", qs[:]))
    return out


def run(args):
    import asyncio
    depths, chunks, eqc = CONFIGS[args.config]
    depths = sorted(depths)
    st = states(depths, chunks, eqc)
    rnd = random.Random(1000 * args.layout_seed + len(args.config))
    toks = lambda n: [rnd.randrange(1000, args.vocab) for _ in range(n)]
    stem = f"perm-{args.config}-L{args.layout_seed}"
    os.environ["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{stem}.jsonl")
    os.environ["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{stem}.steps.jsonl")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def go():
        eng = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
            enable_prefix_caching=True, max_num_batched_tokens=sum(chunks) + 1, max_num_seqs=len(chunks) + 4,
            enable_logging_iteration_details=True, disable_log_stats=False, seed=0))
        n = [0]

        async def gen(ids, max_tokens):
            n[0] += 1
            async for _ in eng.generate({"prompt_token_ids": ids}, SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True), f"r{n[0]}"):
                pass

        prefixes = [toks(k) for k in depths]
        order = list(range(len(depths)))
        random.Random(args.layout_seed).shuffle(order)           # prefix creation order -> physical block layout
        for j in order:
            if prefixes[j]:
                await gen(prefixes[j], 1)
        stop = asyncio.Event()

        async def blocker():
            while not stop.is_set():
                await gen(toks(16), 4000)

        async def submit(pairing, reverse_equal=False):
            reqs = [prefixes[j] + toks(pairing[j]) for j in range(len(depths))]
            if reverse_equal:
                reqs = reqs[::-1]                                  # same prefix->chunk map, other submission order
            # Requests reach the engine one by one; with many of them the engine starts scheduling before
            # the last arrives (n = 16 split in the harness smoke). Freeze scheduling (keeping the prefix
            # cache), enqueue all n, then resume: the first step after resume holds all n chunks.
            await eng.pause_generation(mode="keep", clear_cache=False)
            tasks = [asyncio.create_task(gen(r, 1)) for r in reqs]
            await asyncio.sleep(args.enqueue_s)
            await eng.resume_generation()
            await asyncio.gather(*tasks)

        bt = asyncio.create_task(blocker())
        await asyncio.sleep(1.0)
        for _ in range(args.warmup):
            for _label, p in st:
                await submit(p)
        intended = []
        for b in range(args.blocks):
            blk = st[:]; rnd.shuffle(blk)
            for label, p in blk:
                await asyncio.sleep(0.05)
                await submit(p, reverse_equal=(label == "order-null"))
                intended.append({"block": b, "state": label, "pairing": p})
        stop.set(); bt.cancel()
        eng.shutdown()
        return intended

    intended = asyncio.run(go())
    (args.trace_dir / f"{stem}.plan.json").write_text(json.dumps(
        {"config": args.config, "layout_seed": args.layout_seed, "depths": depths, "chunks": chunks, "model": args.model,
         "states": st, "blocks": args.blocks, "warmup_rounds": args.warmup, "enqueue_s": args.enqueue_s, "intended": intended}) + "\n")


def analyze(args):
    from step_trace_join import load_joined
    out = {}
    for plan_path in sorted(args.trace_dir.glob("perm-*.plan.json")):
        pl = json.load(open(plan_path))
        depths, st = pl["depths"], pl["states"]
        rows, info = load_joined(plan_path.with_name(plan_path.name.replace(".plan.json", ".jsonl")))
        by_map = {}
        for label, p in st:
            by_map.setdefault(tuple(p), []).append(label)
        n_warm = pl["warmup_rounds"] * len(st)
        seq = []
        for r in rows:
            if r["ctx_reqs"] != len(depths) or r["gen_reqs"] != 1 or r["cuda_ms"] != r["cuda_ms"]:
                continue
            if sorted(r.get("ctx_depths", [])) != depths:
                continue
            m = dict(zip(r["ctx_depths"], r["ctx_chunks"]))
            seq.append((tuple(m[k] for k in depths), r["cuda_ms"]))
        seq = seq[n_warm:]                                       # warm-up rounds come first and are dropped
        # Trials run strictly one after another and a captured trial yields exactly one step with all n
        # chunks, so candidate steps follow the plan order. A trial whose requests split across steps
        # yields no candidate; strict matching (never skipping ahead) keeps it from taking a later trial's step.
        plan = pl["intended"]
        times = {label: [] for label, _ in st}
        block_of = {label: [] for label, _ in st}
        missing = 0
        i = 0
        for t in plan:
            if i < len(seq) and seq[i][0] == tuple(t["pairing"]):
                times[t["state"]].append(seq[i][1]); block_of[t["state"]].append(t["block"]); i += 1
            else:
                missing += 1
        res_extra = {"trials": len(plan), "missing": missing, "unassigned_candidate_steps": len(seq) - i}
        res = {"config": pl["config"], "layout_seed": pl["layout_seed"], "depths": depths, "chunks": pl["chunks"],
               "join_match_frac": info["match_frac"], "states": {}, **res_extra}
        lo, hi = coupling_range(pl["chunks"], depths)
        res["C_min_M"], res["C_max_M"] = lo / 1e6, hi / 1e6
        for label, p in st:
            v = times[label]
            res["states"][label] = {"pairing": p, "C_M": coupling(p, depths) / 1e6, "n": len(v),
                                    "median_ms": statistics.median(v) if v else None, "cuda_ms": v, "blocks": block_of[label]}
        out[plan_path.stem.replace(".plan", "")] = res
        print(f"{plan_path.stem:28s} join {info['match_frac']:.3f} | " + "  ".join(
            f"{lab}:{res['states'][lab]['median_ms']:.2f}(n{res['states'][lab]['n']})" if res['states'][lab]['median_ms'] else f"{lab}:—"
            for lab, _ in st))
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--layout-seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--enqueue-s", type=float, default=0.15, help="time the engine stays paused while a state's requests are enqueued")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--vocab", type=int, default=151_000)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
