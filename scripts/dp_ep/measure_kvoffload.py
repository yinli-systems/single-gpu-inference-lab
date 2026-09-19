#!/usr/bin/env python3
"""Task 26 real-mover check: vLLM's own CPU offloading connector as the KV mover on rank 0.

The hog experiments (measure_pcie.py) inject synthetic copy-engine traffic. This harness drives
real KV movement through vLLM 0.29.0's native OffloadingConnector (`--kv-offloading-size <GiB>`,
prefix caching on) on rank 0 only, by workload, while rank 1 runs B plain decode streams:
  plain   rank 0 runs B plain decode streams too (baseline, no prefill, no KV movement)
  store   rank 0 runs a back-to-back train of *unique* L-token prompts (8 output tokens): every
          prefill chunk produces new full blocks that the connector stores GPU->CPU (D2H, copy
          engine, ~L x 192 KiB per request; rate bounded by the prefill rate)
  load    rank 0 first primes P distinct L-token prompts (P x L tokens > the GPU KV cache, so the
          LRU evicts them from GPU while the CPU tier keeps them), then re-requests them round
          robin: each request is a CPU->GPU load of L x 192 KiB instead of a recompute
The offload arm is server-level: run the job twice (OFFLOAD_GIB=16 and 0). store(ON) - store(OFF)
isolates the D2H store cost on rank 1 at identical rank-0 prefill work; load(ON) vs load(OFF) is
the product comparison (H2D load vs recompute). Per cell: rank-1 ITL p50/p95 in the window, rank-0
train TTFT/e2e and request count, plus the step traces; /metrics before/after each cell give the
connector's transfer counters when exposed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_contagion import blocking_completion, itl_summary, rand_tokens, stream_completion  # noqa: E402


async def train(client, base, model, prompts, until, store, cycle):
    """Back-to-back requests: unique prompts (cycle=False) or round robin over `prompts`."""
    i = 0
    while time.monotonic() < until:
        p = prompts[i % len(prompts)] if cycle else prompts.pop() if prompts else None
        if p is None:
            break
        await blocking_completion(client, base, model, p, 8, {}, store)
        i += 1


async def run_cell(args, rng, kind, B, ports, model):
    bases = [f"http://localhost:{p}" for p in ports]
    async with httpx.AsyncClient(timeout=900) as client:
        dec_store = {base: [] for base in bases}
        train_store = []
        dec_tasks = []
        for _ in range(B):
            dec_tasks.append(asyncio.create_task(stream_completion(client, bases[1], model, rand_tokens(rng, 128), args.decode_tokens, {}, dec_store[bases[1]])))
        if kind == "plain":
            for _ in range(B):
                dec_tasks.append(asyncio.create_task(stream_completion(client, bases[0], model, rand_tokens(rng, 128), args.decode_tokens, {}, dec_store[bases[0]])))
        prompts = []
        if kind == "load":  # prime P prompts (stored to CPU as they are computed; P*L > GPU KV so they leave the GPU)
            prompts = [rand_tokens(rng, args.prompt_len) for _ in range(args.prime_count)]
            for p in prompts:
                await blocking_completion(client, bases[0], model, p, 8, {}, [])
        elif kind == "store":
            prompts = [rand_tokens(rng, args.prompt_len) for _ in range(400)]
        await asyncio.sleep(args.settle_s)
        t_window0 = time.monotonic()
        until = t_window0 + args.window_s
        train_task = None
        if kind in ("store", "load"):
            train_task = asyncio.create_task(train(client, bases[0], model, prompts, until, train_store, cycle=(kind == "load")))
        await asyncio.sleep(args.window_s)
        t_window1 = time.monotonic()
        if train_task is not None:
            await asyncio.gather(train_task, return_exceptions=True)
        await asyncio.sleep(args.tail_s)
        for t in dec_tasks:
            t.cancel()
        await asyncio.gather(*dec_tasks, return_exceptions=True)
    tr = [t for t in train_store if not t.get("error")]
    ttft = sorted(t["ttft_ms"] for t in tr)
    return {"kind": kind, "B": B, "t_window0": t_window0, "t_window1": t_window1,
            "train": {"n": len(train_store), "errors": len(train_store) - len(tr), "ttft_p50_ms": ttft[len(ttft) // 2] if ttft else None,
                      "kv_GiB_moved_est": len(tr) * args.prompt_len * args.kv_bytes_per_token / 2**30},
            "decode": {base: [{k: v for k, v in s.items() if k != "t_send"} for s in v] for base, v in dec_store.items()}}


async def metrics(port):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"http://localhost:{port}/metrics")
            return [l for l in r.text.splitlines() if "offload" in l.lower() or "kv_transfer" in l.lower() or "prefix_cache" in l.lower()]
    except Exception as e:  # noqa: BLE001
        return [repr(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs=2, required=True)
    ap.add_argument("--kinds", default="plain,store,load")
    ap.add_argument("--batch-sizes", default="8,32")
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--prime-count", type=int, default=10, help="load: distinct prompts primed before the window (10 x 4k = 40k tokens > 27k GPU KV)")
    ap.add_argument("--kv-bytes-per-token", type=int, default=192 * 1024, help="Qwen1.5-MoE-A2.7B bf16: 24 layers x 2 x 16 heads x 128 x 2 B")
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--tail-s", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rec = {"schema_version": 1, "result_type": "dp_ep_kv_offload_real_mover", "args": {k: str(v) for k, v in vars(args).items()}, "cells": []}

    async def run():
        for rep in range(args.repeats):
            plan = [(k, B) for B in [int(x) for x in args.batch_sizes.split(",")] for k in args.kinds.split(",")]
            random.Random(rep).shuffle(plan)
            for kind, B in plan:
                m0 = await metrics(args.ports[0])
                res = await run_cell(args, rng, kind, B, args.ports, args.model)
                res["repeat"] = rep
                res["itl_window"] = itl_summary(res)
                res["metrics_rank0_before"], res["metrics_rank0_after"] = m0, await metrics(args.ports[0])
                rec["cells"].append(res)
                tr = res["train"]
                print(f"[r{rep}] {kind} B={B}: " + "; ".join(f"{k.split(':')[-1]} ITL p50 {v.get('p50', float('nan')):.1f} p95 {v.get('p95', float('nan')):.1f} err {v['errors']}" for k, v in res["itl_window"].items())
                      + f" | train n {tr['n']} err {tr['errors']} ttft p50 {tr['ttft_p50_ms'] or 0:.0f} ms ~{tr['kv_GiB_moved_est']:.1f} GiB KV", flush=True)
                args.output.write_text(json.dumps(rec) + "\n")
                await asyncio.sleep(2.0)
    asyncio.run(run())


if __name__ == "__main__":
    main()
