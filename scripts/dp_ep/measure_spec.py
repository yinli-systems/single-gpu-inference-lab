#!/usr/bin/env python3
"""Task 27: speculative-decoding acceptance skew across synchronized DP/EP ranks.

Server-wide n-gram speculation (no draft model): `--speculative-config
'{"method":"ngram","num_speculative_tokens":K,"prompt_lookup_max":4,"prompt_lookup_min":2}'`.
Rank r = port P_r (multi-port external LB). Each rank runs B temperature-0 decode streams whose
prompt *kind* sets the acceptance regime:
  rep   a repeated 8-token pattern (128 tokens): the greedy continuation repeats it, n-gram
        lookup proposes K tokens every step and they are accepted -> verification width 1+K
  rnd   random token ids (128 tokens): no n-gram match in the history -> width 1, or K
        rejected drafts when a spurious match exists
Cells: rep|rep, rnd|rnd (homogeneous), rep|rnd, rnd|rep (skewed, swapped for asymmetry).
Questions: does the low-acceptance rank pay the high-acceptance rank's verification width
(padded execution tokens, step period), i.e. is its own generation rate lower in rep|rnd than in
rnd|rnd; and does the high-acceptance rank keep its rate in rep|rnd vs rep|rep?
Metrics per rank: generation rate (tokens/s per stream inside the window; ITL is bimodal under
speculation so the rate is the user-visible signal), ITL p50/p95 (reported), step period /
CUDA time / num_tokens / padded_tokens from VLLM_EXP_STEP_TRACE, and the spec-decode acceptance
counters from each rank's /metrics (delta over the window).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_contagion import itl_summary, rand_tokens, stream_completion  # noqa: E402

SPEC_METRICS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:generation_tokens_total", "vllm:num_requests_running")


def rep_tokens(rng, n, period=8, vocab=151000):
    pat = [rng.randrange(1000, vocab) for _ in range(period)]
    return [pat[i % period] for i in range(n)]


async def metrics(client, base):
    out = {}
    try:
        r = await client.get(f"{base}/metrics")
        for line in r.text.splitlines():
            for m in SPEC_METRICS:
                if line.startswith(m):
                    mm = re.match(r"^\S+(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
                    if mm:
                        out[m] = out.get(m, 0.0) + float(mm.group(2))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
    return out


async def run_cell(args, rng, kind0, kind1, B, ports, model):
    bases = [f"http://localhost:{p}" for p in ports]
    async with httpx.AsyncClient(timeout=900) as client:
        dec_store = {base: [] for base in bases}
        dec_tasks = []
        for base, kind in zip(bases, (kind0, kind1)):
            for _ in range(B):
                prompt = rep_tokens(rng, 128) if kind == "rep" else rand_tokens(rng, 128)
                dec_tasks.append(asyncio.create_task(stream_completion(client, base, model, prompt, args.decode_tokens, {}, dec_store[base])))
        await asyncio.sleep(args.settle_s)
        m0 = {base: await metrics(client, base) for base in bases}
        t_window0 = time.monotonic()
        await asyncio.sleep(args.window_s)
        t_window1 = time.monotonic()
        m1 = {base: await metrics(client, base) for base in bases}
        await asyncio.sleep(args.tail_s)
        for t in dec_tasks:
            t.cancel()
        await asyncio.gather(*dec_tasks, return_exceptions=True)
    spec = {}
    for base in bases:
        d = {k: m1[base].get(k, 0.0) - m0[base].get(k, 0.0) for k in SPEC_METRICS if k != "vllm:num_requests_running"}
        d["accept_rate"] = d["vllm:spec_decode_num_accepted_tokens_total"] / d["vllm:spec_decode_num_draft_tokens_total"] if d.get("vllm:spec_decode_num_draft_tokens_total") else None
        # tokens/s of this rank from the engine counter (a streamed chunk carries every token accepted in a step, so chunk counts undercount)
        d["gen_tok_s"] = d.get("vllm:generation_tokens_total", 0.0) / (t_window1 - t_window0)
        spec[base] = d
    return {"kind0": kind0, "kind1": kind1, "B": B, "t_window0": t_window0, "t_window1": t_window1, "spec": spec,
            "decode": {base: [{k: v for k, v in s.items() if k != "t_send"} for s in v] for base, v in dec_store.items()}}


def rate_summary(res):
    out = {}
    t0, t1 = res["t_window0"], res["t_window1"]
    for base, streams in res["decode"].items():
        per = [sum(1 for t in s["token_times"] if t0 <= t <= t1) / (t1 - t0) for s in streams]
        per.sort()
        out[base] = {"streams": len(per), "tok_s_per_stream_p50": per[len(per) // 2] if per else 0.0, "tok_s_total": sum(per)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs=2, required=True)
    ap.add_argument("--batch-sizes", default="8,32")
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--tail-s", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rec = {"schema_version": 1, "result_type": "dp_ep_spec_skew", "args": {k: str(v) for k, v in vars(args).items()}, "cells": []}

    async def run():
        for rep in range(args.repeats):
            plan = [(k0, k1, B) for B in [int(x) for x in args.batch_sizes.split(",")] for k0, k1 in (("rep", "rep"), ("rnd", "rnd"), ("rep", "rnd"), ("rnd", "rep"))]
            random.Random(rep).shuffle(plan)
            for k0, k1, B in plan:
                res = await run_cell(args, rng, k0, k1, B, args.ports, args.model)
                res["repeat"] = rep
                res["itl_window"] = itl_summary(res)
                res["rate"] = rate_summary(res)
                rec["cells"].append(res)
                print(f"[r{rep}] {k0}|{k1} B={B}: " + "; ".join(f"{b.split(':')[-1]} gen {res['spec'][b]['gen_tok_s']:.0f} tok/s ({v['tok_s_per_stream_p50']:.1f} chunks/s/stream) ITL p50 {res['itl_window'][b].get('p50', float('nan')):.1f} accept {res['spec'][b].get('accept_rate') if res['spec'][b].get('accept_rate') is None else round(res['spec'][b]['accept_rate'], 2)} drafts {res['spec'][b].get('vllm:spec_decode_num_drafts_total', 0):.0f} err {res['itl_window'][b]['errors']}" for b, v in res["rate"].items()), flush=True)
                args.output.write_text(json.dumps(rec) + "\n")
                await asyncio.sleep(2.0)
    asyncio.run(run())


if __name__ == "__main__":
    main()
