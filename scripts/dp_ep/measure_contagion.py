#!/usr/bin/env python3
"""Task 21: cross-rank feature contagion under DP/EP synchronized steps.

Question: does an expensive request-local feature on DP rank 0 (its own GPU, its own engine
core and API server) raise the inter-token latency of unrelated plain requests on DP rank 1,
because the ranks rendezvous every step (DP coordination all-reduce + per-layer EP collectives)?

Deterministic placement via the multi-port external LB: rank r = port P_r. Each cell runs B
decode streams per rank for a fixed window; the "feature" rank's streams carry the feature,
the other rank's streams are plain (temperature 0, no logprobs). Cells:
  plain|plain            baseline
  F|plain, plain|F       feature on rank 0 / rank 1 (swap detects rank asymmetry)
Features F:
  logprobs   top-5 logprobs per generated token (GPU top-k + CPU pythonisation/serialisation)
  struct     structured output, an unbounded EBNF grammar (xgrammar bitmask build + apply
             every step, grammar advance per token; ignore_eos keeps the FSM live)
  sampling   temperature 1, top-p/top-k/min-p, repetition/presence/frequency penalties
  plogp      prompt logprobs: a train of 2048-token prompts (max_tokens 8, prompt_logprobs=1)
             issued back-to-back on the feature rank during the window; control cell `pfx`
             issues the same prompt train without prompt_logprobs, so the prefill cost itself
             is separated from the feature cost
The B plain decoders on the other rank are the "unrelated client B" whose ITL is the metric.
Per-rank engine/step traces (VLLM_EXP_STEP_TRACE) give step period and per-step GPU time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

GRAMMAR = r"""
root ::= line*
line ::= word " " number "\n"
word ::= [a-z]+
number ::= [0-9]+
"""

FEATURES = {
    "plain": {},
    "logprobs": {"logprobs": 5},
    "logprobs20": {"logprobs": 20},
    "struct": {"structured_outputs": {"grammar": GRAMMAR}},
    "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "min_p": 0.05, "repetition_penalty": 1.2, "presence_penalty": 0.5, "frequency_penalty": 0.5, "seed": 7},
    # decomposition of `sampling` (round 2): which part of the sampler path is exported to the peer rank
    "temp": {"temperature": 1.0, "seed": 7},
    "topk": {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "min_p": 0.05, "seed": 7},
    "pen": {"temperature": 1.0, "repetition_penalty": 1.2, "presence_penalty": 0.5, "frequency_penalty": 0.5, "seed": 7},
}


def rand_tokens(rng, n, vocab=151000):
    return [rng.randrange(1000, vocab) for _ in range(n)]


async def stream_completion(client, base, model, prompt, max_tokens, extra, store=None):
    t0 = time.monotonic(); first = None; ts = []; n = 0
    live = {"t_send": t0, "token_times": ts}
    if store is not None:
        store.append(live)
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    try:
        async with client.stream("POST", f"{base}/v1/completions", json=body) as r:
            if r.status_code != 200:
                live["error"] = (await r.aread())[:300].decode(errors="replace"); return live
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    d = json.loads(line[6:])
                    if d.get("choices") and d["choices"][0].get("text"):
                        now = time.monotonic(); ts.append(now)
                        if first is None:
                            first = now
                    if d.get("usage"):
                        n = d["usage"]["completion_tokens"]
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        live["error"] = repr(e)[:300]
    live.update({"ttft_ms": ((first or time.monotonic()) - t0) * 1e3, "e2e_ms": (time.monotonic() - t0) * 1e3, "tokens": n})
    return live


async def blocking_completion(client, base, model, prompt, max_tokens, extra, store):
    """Non-streaming request (vLLM rejects prompt_logprobs with stream=True); ttft := e2e."""
    t0 = time.monotonic(); live = {"t_send": t0, "token_times": []}
    store.append(live)
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    body.update(extra)
    try:
        r = await client.post(f"{base}/v1/completions", json=body)
        if r.status_code != 200:
            live["error"] = r.text[:300]
        else:
            live["tokens"] = r.json().get("usage", {}).get("completion_tokens", 0)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        live["error"] = repr(e)[:300]
    live.update({"ttft_ms": (time.monotonic() - t0) * 1e3, "e2e_ms": (time.monotonic() - t0) * 1e3})
    return live


async def prompt_train(client, base, model, rng, L, extra, until, store):
    """Back-to-back short-generation requests with long prompts until `until` (monotonic)."""
    while time.monotonic() < until:
        await blocking_completion(client, base, model, rand_tokens(rng, L), 8, extra, store)


async def run_cell(args, rng, feat0, feat1, B, ports, model):
    bases = [f"http://localhost:{p}" for p in ports]
    async with httpx.AsyncClient(timeout=900) as client:
        dec_store = {base: [] for base in bases}
        train_store = {base: [] for base in bases}
        dec_tasks, train_tasks = [], []
        for base, feat in zip(bases, (feat0, feat1)):
            extra = FEATURES.get(feat, {})
            for _ in range(B):
                dec_tasks.append(asyncio.create_task(stream_completion(client, base, model, rand_tokens(rng, 128), args.decode_tokens, extra, dec_store[base])))
        await asyncio.sleep(args.settle_s)
        t_window0 = time.monotonic()
        until = t_window0 + args.window_s
        for base, feat in zip(bases, (feat0, feat1)):
            if feat in ("plogp", "pfx"):
                extra = {"prompt_logprobs": 1} if feat == "plogp" else {}
                train_tasks.append(asyncio.create_task(prompt_train(client, base, model, rng, args.train_prompt_len, extra, until, train_store[base])))
        await asyncio.sleep(args.window_s)
        t_window1 = time.monotonic()
        await asyncio.gather(*train_tasks, return_exceptions=True)
        await asyncio.sleep(args.tail_s)
        for t in dec_tasks:
            t.cancel()
        await asyncio.gather(*dec_tasks, return_exceptions=True)
    return {"feat0": feat0, "feat1": feat1, "B": B, "t_window0": t_window0, "t_window1": t_window1,
            "decode": {base: [{k: v for k, v in s.items() if k != "t_send"} for s in v] for base, v in dec_store.items()},
            "train": {base: [{k: v for k, v in s.items() if k != "token_times"} for s in v] for base, v in train_store.items()}}


def itl_summary(res):
    summ = {}
    for base, streams in res["decode"].items():
        gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if res["t_window0"] <= a <= res["t_window1"]]
        errs = sum(1 for s in streams if s.get("error"))
        if gaps:
            gaps.sort(); summ[base] = {"n": len(gaps), "p50": gaps[len(gaps) // 2], "p95": gaps[int(len(gaps) * .95)], "max": gaps[-1], "errors": errs}
        else:
            summ[base] = {"n": 0, "errors": errs}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs=2, required=True)
    ap.add_argument("--features", default="logprobs,struct,sampling,plogp,pfx")
    ap.add_argument("--batch-sizes", default="8,32")
    ap.add_argument("--no-swap", action="store_true", help="only F|plain cells (feature on rank 0)")
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--train-prompt-len", type=int, default=2048)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--tail-s", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rec = {"schema_version": 1, "result_type": "dp_ep_feature_contagion", "args": {k: str(v) for k, v in vars(args).items()}, "cells": []}

    async def run():
        for rep in range(args.repeats):
            plan = []
            for B in [int(x) for x in args.batch_sizes.split(",")]:
                plan.append(("plain", "plain", B))
                for f in args.features.split(","):
                    plan.append((f, "plain", B))
                    if not args.no_swap:
                        plan.append(("plain", f, B))
            random.Random(rep).shuffle(plan)
            for f0, f1, B in plan:
                res = await run_cell(args, rng, f0, f1, B, args.ports, args.model)
                res["repeat"] = rep
                res["itl_window"] = itl_summary(res)
                rec["cells"].append(res)
                tr = "; ".join(f"train {b.split(':')[-1]}: {len(v)} reqs ttft p50 {sorted(s.get('ttft_ms', 0) for s in v)[len(v) // 2]:.0f} ms" for b, v in res["train"].items() if v)
                print(f"[r{rep}] {f0}|{f1} B={B}: " + "; ".join(f"{k.split(':')[-1]} ITL p50 {v.get('p50', float('nan')):.1f} p95 {v.get('p95', float('nan')):.1f} max {v.get('max', float('nan')):.0f} err {v['errors']}" for k, v in res["itl_window"].items()) + (" | " + tr if tr else ""), flush=True)
                args.output.write_text(json.dumps(rec) + "\n")
                await asyncio.sleep(2.0)
    asyncio.run(run())


if __name__ == "__main__":
    main()
