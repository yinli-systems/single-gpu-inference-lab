#!/usr/bin/env python3
"""DP+EP synchronized-wave geometry (campaign M1).

Drives a running vLLM DP=2 / EP=2 deployment with controlled per-rank work.
Two client modes:
  --ports P0 P1   multi-port external LB: requests are placed on a rank
                  deterministically (rank r = port P_r)
  --ports P       internal load balancer: the engine places requests; the same
                  request stream is sent and the balancer's choice is observed

Cells (M1a, mechanism; each cell = one measurement window):
  dd   decode | decode      B decode streams on each rank
  pp   prefill | prefill    one long prompt (L tokens, chunked by the server's
                            budget q) on each rank, plus B decoders on each
  pd   prefill | decode     one long prompt on rank 0 while rank 1 runs B decoders
                            (rank 0 also keeps B decoders so decode rows are equal)
The decode streams are 128-token prompts with long generations whose per-token
arrival times are recorded (TPOT / ITL per rank); the prefill's TTFT and e2e are
recorded. Per-rank engine/step/EP traces are written by the server-side tracers
(VLLM_EXP_ITER_TRACE / VLLM_EXP_STEP_TRACE / VLLM_EXP_EP_TRACE, rank-suffixed).

M1b (serving A/B) uses the same client with a mixed arrival stream
(--mixed-stream) under internal LB vs phase-aligned placement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


def rand_tokens(rng, n, vocab=151000):
    return [rng.randrange(1000, vocab) for _ in range(n)]


async def stream_completion(client, base, model, prompt, max_tokens, store=None):
    """store: a list; a dict with a live 'token_times' list is registered at start so a cancelled
    stream still leaves its arrival times behind."""
    t0 = time.monotonic(); first = None; ts = []; n = 0
    live = {"t_send": t0, "token_times": ts}
    if store is not None:
        store.append(live)
    async with client.stream("POST", f"{base}/v1/completions", json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                                                                      "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])
                if d.get("choices") and d["choices"][0].get("text"):
                    now = time.monotonic(); ts.append(now)
                    if first is None:
                        first = now
                if d.get("usage"):
                    n = d["usage"]["completion_tokens"]
    out = {"t_send": t0, "ttft_ms": ((first or time.monotonic()) - t0) * 1e3, "e2e_ms": (time.monotonic() - t0) * 1e3, "tokens": n, "token_times": ts}
    live.update({k: v for k, v in out.items() if k != "token_times"})
    return out


async def run_cell(args, rng, cell, B, L, ports, model):
    """Returns per-rank decode token arrival times and prefill timings for the window."""
    bases = [f"http://localhost:{p}" for p in ports]
    r0, r1 = bases[0], bases[-1]
    async with httpx.AsyncClient(timeout=900) as client:
        dec_store = {r: [] for r in bases}
        # decode streams: B per rank (multi-port) or 2B to the single LB port
        dec_tasks = []
        targets = bases if len(bases) == 2 else [bases[0]] * 2
        for base in targets:
            for _ in range(B):
                dec_tasks.append(asyncio.create_task(stream_completion(client, base, model, rand_tokens(rng, 128), args.decode_tokens, dec_store[base])))
        await asyncio.sleep(args.settle_s)
        t_window0 = time.monotonic()
        pre = []
        if cell == "pp":
            pre = await asyncio.gather(*(stream_completion(client, base, model, rand_tokens(rng, L), args.prefill_gen) for base in targets[:2]))
        elif cell == "pd":
            pre = [await stream_completion(client, r0 if len(bases) == 2 else bases[0], model, rand_tokens(rng, L), args.prefill_gen)]
        else:
            await asyncio.sleep(args.decode_window_s)
        t_window1 = time.monotonic()
        # let decoders run a little past the prefill, then stop them
        await asyncio.sleep(args.tail_s)
        for t in dec_tasks:
            t.cancel()
        await asyncio.gather(*dec_tasks, return_exceptions=True)
    return {"cell": cell, "B": B, "L": L, "t_window0": t_window0, "t_window1": t_window1, "prefill": pre,
            "decode": {base: [{"token_times": s["token_times"]} for s in v] for base, v in dec_store.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs="+", required=True)
    ap.add_argument("--cells", default="dd,pp,pd")
    ap.add_argument("--batch-sizes", default="8,16,32")
    ap.add_argument("--prefix-lens", default="4096,8192")
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--prefill-gen", type=int, default=16)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--decode-window-s", type=float, default=6.0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rec = {"schema_version": 1, "result_type": "dp_ep_waves", "args": {k: str(v) for k, v in vars(args).items()}, "mode": "multi-port" if len(args.ports) == 2 else "internal-lb", "cells": []}

    async def run():
        for rep in range(args.repeats):
            plan = [(c, B, L) for c in args.cells.split(",") for B in [int(x) for x in args.batch_sizes.split(",")] for L in ([int(x) for x in args.prefix_lens.split(",")] if c != "dd" else [0])]
            random.Random(rep).shuffle(plan)
            for cell, B, L in plan:
                res = await run_cell(args, rng, cell, B, L, args.ports, args.model)
                res["repeat"] = rep
                # quick per-rank ITL summary
                summ = {}
                for base, streams in res["decode"].items():
                    gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if res["t_window0"] <= a <= res["t_window1"]]
                    if gaps:
                        gaps.sort(); summ[base] = {"n": len(gaps), "p50": gaps[len(gaps) // 2], "p95": gaps[int(len(gaps) * .95)], "max": gaps[-1]}
                res["itl_window"] = summ
                rec["cells"].append(res)
                pf = f" prefill ttft {res['prefill'][0]['ttft_ms']:.0f} ms" if res["prefill"] else ""
                print(f"[r{rep}] {cell} B={B} L={L}: " + "; ".join(f"{k.split(':')[-1]} ITL p50 {v['p50']:.1f} p95 {v['p95']:.1f} max {v['max']:.0f}" for k, v in summ.items()) + pf, flush=True)
                args.output.write_text(json.dumps(rec) + "\n")
                await asyncio.sleep(2.0)
    asyncio.run(run())


if __name__ == "__main__":
    main()
