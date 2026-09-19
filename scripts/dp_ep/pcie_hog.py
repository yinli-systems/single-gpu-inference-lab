#!/usr/bin/env python3
"""Task 26: PCIe traffic injector standing in for bulk KV movement on ONE DP rank's GPU.

Runs as a separate process on `--device` (the rank-0 GPU of a DP2/EP2 server in the same Slurm
allocation) and moves `--bytes` per burst between pinned host memory and the GPU, in the given
direction, with a duty cycle: burst, then sleep so that bursts occupy `--duty` of wall time
(`--duty 1.0` = continuous). Bytes per burst should be KV-equivalent: Qwen1.5-MoE-A2.7B keeps
24 layers x 2 (K,V) x 16 kv-heads x 128 x bf16 = 192 KiB per token -> 4k/8k/16k tokens =
0.75 / 1.5 / 3.0 GiB. Prints achieved GB/s per burst so the injected bandwidth is a measured
number, and writes one JSON line per burst to --log (t_start, t_end monotonic, bytes, dir, GB/s)
so bursts can be aligned with the server's step trace (same node clock).

This is an upper-bound proxy: the real vLLM CPU-offload connector copies on a side stream inside
the worker process; a separate process shares the PCIe link and root complex the same way but
not the worker's copy engine scheduling.
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bytes", type=float, default=1.5 * 2**30, help="bytes per burst (default 1.5 GiB = 8k tokens of KV)")
    ap.add_argument("--dir", choices=["h2d", "d2h", "both"], default="h2d")
    ap.add_argument("--duty", type=float, default=1.0, help="fraction of wall time spent copying (1.0 = continuous)")
    ap.add_argument("--chunk-mb", type=int, default=64, help="copy granularity (MiB), like a connector's per-block copies")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to run")
    ap.add_argument("--start-delay", type=float, default=0.0)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    n = int(args.bytes)
    chunk = args.chunk_mb * 2**20
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(n, dtype=torch.uint8, device=dev)
    stream = torch.cuda.Stream(device=dev)
    time.sleep(args.start_delay)
    t_end_all = time.monotonic() + args.duration
    with open(args.log, "a") as f:
        while time.monotonic() < t_end_all:
            d = args.dir if args.dir != "both" else ("h2d" if int(time.monotonic() * 10) % 2 == 0 else "d2h")
            t0 = time.monotonic()
            with torch.cuda.stream(stream):
                for off in range(0, n, chunk):
                    end = min(off + chunk, n)
                    if d == "h2d":
                        gpu[off:end].copy_(host[off:end], non_blocking=True)
                    else:
                        host[off:end].copy_(gpu[off:end], non_blocking=True)
            stream.synchronize()
            t1 = time.monotonic()
            gbs = n / (t1 - t0) / 1e9
            f.write(json.dumps({"t_start": t0, "t_end": t1, "bytes": n, "dir": d, "GB_s": gbs}) + "\n"); f.flush()
            print(f"{d} {n / 2**30:.2f} GiB in {(t1 - t0) * 1e3:.0f} ms = {gbs:.1f} GB/s", flush=True)
            if args.duty < 1.0:
                time.sleep((t1 - t0) * (1 - args.duty) / args.duty)


if __name__ == "__main__":
    main()
