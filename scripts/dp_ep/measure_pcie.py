#!/usr/bin/env python3
"""Task 26: PCIe arbitration — bulk KV movement on rank 0's GPU vs EP communication of rank 1.

Question: on PCIe-only GPUs (2x4090, no NVLink, no P2P: NCCL moves EP tensors through host
shared memory), does a CPU<->GPU bulk transfer on rank 0's GPU degrade the decode latency of
rank 1 (which moves no KV) because every step rendezvous at the EP collectives and the transfer
shares rank 0's PCIe link / root complex with those collectives?

Both ranks run B plain decode streams (multi-port placement: rank r = port P_r). Each cell
starts `pcie_hog.py` on GPU 0 (a separate process in the same allocation) with one spec:
  off              no hog (baseline)
  idle             hog process alive on GPU 0 (context + pinned buffers) but never copying
  h2d:1.0          continuous host->GPU0 bursts (1.5 GiB = 8k tokens of KV per burst)
  d2h:1.0          continuous GPU0->host
  both:1.0         alternating h2d / d2h bursts
  h2d:0.5          50 % duty (burst, then an equal pause)
  h2d:0.25         25 % duty
  h2d:cap4         bandwidth-capped to ~4 GB/s (chunked copies with sleeps)
  d2d:1.0          control: GPU0-internal copies (copy engine busy, no PCIe traffic)
  <dir>:<duty>:<GiB> optional third field = bytes per burst in GiB (default 1.5)
  d2h-sm:1.0       `-sm` suffix: the same bytes moved by a Triton kernel (SM-issued PCIe writes/reads)
                   instead of the copy engine (cudaMemcpyAsync) — separates requester from bytes
The hog is started at the beginning of the cell (allocates, warms up, then waits for a start
file); the start file is created exactly at t_window0 so bursts and the window coincide.
Metrics per rank: ITL of the decode streams inside the window (p50/p95), plus the per-rank step
traces (VLLM_EXP_STEP_TRACE) for step period and GPU time, and the hog's per-burst GB/s log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_contagion import itl_summary, rand_tokens, stream_completion  # noqa: E402


def hog_cmd(spec, args, log, start_file):
    spec, _, numa = spec.partition("@")  # "d2h:1.0@6" = hog process + pinned buffer on NUMA node 6
    parts = spec.split(":")
    d = parts[0]
    engine = "ce"
    if d.endswith("-sm"):
        d, engine = d[:-3], "sm"
    duty = parts[1] if len(parts) > 1 else "1.0"
    gib = float(parts[2]) if len(parts) > 2 else args.burst_gib
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "pcie_hog.py"), "--device", args.hog_device, "--bytes", str(int(gib * 2**30)),
           "--chunk-mb", str(args.chunk_mb), "--gpu-buf-mb", str(args.gpu_buf_mb), "--duration", str(args.window_s), "--log", str(log), "--start-file", str(start_file)]
    cmd += ["--dir", d, "--engine", engine]  # "idle" keeps the context + pinned buffers alive on GPU 0 for the window without copying
    if numa:
        cmd += ["--numa-node", numa]
    if duty.startswith("cap"):
        cmd += ["--rate-gbs", duty[3:], "--chunk-mb", "16"]
    else:
        cmd += ["--duty", duty]
    return cmd


async def run_cell(args, rng, spec, B, ports, model, cell_idx):
    bases = [f"http://localhost:{p}" for p in ports]
    log = args.hog_dir / f"hog-{cell_idx:03d}-{spec.replace(':', '_')}-B{B}.jsonl"
    start_file = args.hog_dir / f"start-{cell_idx:03d}"
    hog = None
    if spec != "off":
        hog = subprocess.Popen(hog_cmd(spec, args, log, start_file), stdout=open(str(log) + ".out", "w"), stderr=subprocess.STDOUT)
    async with httpx.AsyncClient(timeout=900) as client:
        dec_store = {base: [] for base in bases}
        dec_tasks = []
        for base in bases:
            for _ in range(B):
                dec_tasks.append(asyncio.create_task(stream_completion(client, base, model, rand_tokens(rng, 128), args.decode_tokens, {}, dec_store[base])))
        await asyncio.sleep(args.settle_s)
        t_window0 = time.monotonic()
        start_file.touch()
        await asyncio.sleep(args.window_s)
        t_window1 = time.monotonic()
        await asyncio.sleep(args.tail_s)
        for t in dec_tasks:
            t.cancel()
        await asyncio.gather(*dec_tasks, return_exceptions=True)
    hog_rc = None
    if hog is not None:
        try:
            hog_rc = hog.wait(timeout=30)
        except subprocess.TimeoutExpired:
            hog.kill(); hog_rc = "killed"
    bursts = []
    if log.exists():
        bursts = [json.loads(l) for l in log.read_text().splitlines() if l.strip() and "t_start" in l]
    return {"spec": spec, "B": B, "t_window0": t_window0, "t_window1": t_window1, "hog_rc": hog_rc, "hog_log": str(log),
            "hog": {"bursts": len(bursts), "GB_s_p50": sorted(b["GB_s"] for b in bursts)[len(bursts) // 2] if bursts else None,
                    "active_s": sum(b["t_end"] - b["t_start"] for b in bursts), "bytes": sum(b["bytes"] for b in bursts)},
            "decode": {base: [{k: v for k, v in s.items() if k != "t_send"} for s in v] for base, v in dec_store.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs=2, required=True)
    ap.add_argument("--specs", default="off,idle,h2d:1.0,d2h:1.0,both:1.0,h2d:0.5,h2d:0.25,h2d:cap4,d2d:1.0")
    ap.add_argument("--batch-sizes", default="8,32")
    ap.add_argument("--burst-gib", type=float, default=1.5)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--gpu-buf-mb", type=int, default=512)
    ap.add_argument("--hog-device", default="cuda:0")
    ap.add_argument("--hog-dir", type=Path, required=True)
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--tail-s", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=63)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.hog_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rec = {"schema_version": 1, "result_type": "dp_ep_pcie_arbitration", "args": {k: str(v) for k, v in vars(args).items()}, "cells": []}

    async def run():
        idx = 0
        for rep in range(args.repeats):
            plan = [(s, B) for B in [int(x) for x in args.batch_sizes.split(",")] for s in args.specs.split(",")]
            random.Random(rep).shuffle(plan)
            for spec, B in plan:
                res = await run_cell(args, rng, spec, B, args.ports, args.model, idx); idx += 1
                res["repeat"] = rep
                res["itl_window"] = itl_summary(res)
                rec["cells"].append(res)
                h = res["hog"]
                print(f"[r{rep}] {spec} B={B}: " + "; ".join(f"{k.split(':')[-1]} ITL p50 {v.get('p50', float('nan')):.1f} p95 {v.get('p95', float('nan')):.1f} max {v.get('max', float('nan')):.0f} err {v['errors']}" for k, v in res["itl_window"].items())
                      + f" | hog {h['bursts']} bursts {h['bytes'] / 2**30:.1f} GiB active {h['active_s']:.1f} s p50 {h['GB_s_p50'] or 0:.1f} GB/s rc {res['hog_rc']}", flush=True)
                args.output.write_text(json.dumps(rec) + "\n")
                await asyncio.sleep(2.0)
    asyncio.run(run())


if __name__ == "__main__":
    main()
