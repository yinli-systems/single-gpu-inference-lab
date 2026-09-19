#!/usr/bin/env python3
"""Task 26: PCIe traffic injector standing in for bulk KV movement on ONE DP rank's GPU.

Runs as a separate process on `--device` (the rank-0 GPU of a DP2/EP2 server in the same Slurm
allocation) and moves `--bytes` per burst between pinned host memory and the GPU, in the given
direction, with a duty cycle: burst, then sleep so that bursts occupy `--duty` of wall time
(`--duty 1.0` = continuous). Bytes per burst should be KV-equivalent: Qwen1.5-MoE-A2.7B keeps
24 layers x 2 (K,V) x 16 kv-heads x 128 x bf16 = 192 KiB per token -> 4k/8k/16k tokens =
0.75 / 1.5 / 3.0 GiB. Prints achieved GB/s per burst so the injected bandwidth is a measured
number, and writes one JSON line per burst to --log (t_start, t_end = time.monotonic() of this
node, bytes, dir, GB/s) so bursts can be aligned with the server's step trace (same clock).

Directions: h2d / d2h / both (alternating bursts) move data over the GPU's PCIe link;
`d2d` copies GPU->GPU inside the same device (copy engine busy, *no* PCIe traffic) and is the
control that separates "a second context is active on GPU 0" from PCIe arbitration.

GPU memory is bounded by --gpu-buf-mb (the server holds ~0.85 of the card): bursts cycle the
host buffer through a smaller device window, which keeps the PCIe bytes per burst exact.
`--rate-gbs` caps the average transfer rate by sleeping after every chunk (bandwidth-capped arm).
`--start-file`: after allocation + one warm-up copy the hog waits (10 ms polls) until that file
exists, so the harness can align burst start with its measurement window; `--duration` counts
from that moment.

This is an upper-bound proxy: the real vLLM CPU-offload connector copies on a side stream inside
the worker process; a separate process shares the PCIe link and root complex the same way but
not the worker's copy engine scheduling.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bytes", type=float, default=1.5 * 2**30, help="bytes per burst (default 1.5 GiB = 8k tokens of KV)")
    ap.add_argument("--dir", choices=["h2d", "d2h", "both", "d2d", "idle"], default="h2d", help="idle = keep the context/buffers alive without copying")
    ap.add_argument("--duty", type=float, default=1.0, help="fraction of wall time spent copying (1.0 = continuous)")
    ap.add_argument("--chunk-mb", type=int, default=64, help="copy granularity (MiB), like a connector's per-block copies")
    ap.add_argument("--gpu-buf-mb", type=int, default=512, help="device window size (MiB); bursts cycle through it")
    ap.add_argument("--rate-gbs", type=float, default=0.0, help="cap the average rate (GB/s) by sleeping after each chunk; 0 = no cap")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to run (from the start signal)")
    ap.add_argument("--start-delay", type=float, default=0.0)
    ap.add_argument("--start-file", default=None, help="wait until this file exists before the first burst")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    n = int(args.bytes)
    chunk = args.chunk_mb * 2**20
    gn = min(n, args.gpu_buf_mb * 2**20)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(gn, dtype=torch.uint8, device=dev)
    gpu2 = torch.empty(gn, dtype=torch.uint8, device=dev) if args.dir == "d2d" else None
    stream = torch.cuda.Stream(device=dev)
    # warm-up: one chunk each way so context/pinned mappings are established before the window
    with torch.cuda.stream(stream):
        gpu[:chunk].copy_(host[:chunk], non_blocking=True)
        host[:chunk].copy_(gpu[:chunk], non_blocking=True)
    stream.synchronize()
    print(f"hog ready on {args.device}: {n / 2**30:.2f} GiB pinned, {gn / 2**20:.0f} MiB device window, dir {args.dir} duty {args.duty} rate {args.rate_gbs}", flush=True)
    if args.start_file:
        while not os.path.exists(args.start_file):
            time.sleep(0.01)
    time.sleep(args.start_delay)
    t_begin = time.monotonic()
    t_end_all = t_begin + args.duration
    k = 0
    with open(args.log, "a") as f:
        f.write(json.dumps({"event": "start", "t": t_begin, "args": vars(args)}) + "\n"); f.flush()
        while time.monotonic() < t_end_all:
            if args.dir == "idle":
                time.sleep(0.05); continue
            d = args.dir if args.dir != "both" else ("h2d" if k % 2 == 0 else "d2h")
            k += 1
            t0 = time.monotonic()
            with torch.cuda.stream(stream):
                for off in range(0, n, chunk):
                    end = min(off + chunk, n)
                    g0 = off % gn; g1 = g0 + (end - off)
                    if g1 > gn:
                        g0, g1 = 0, end - off
                    if d == "h2d":
                        gpu[g0:g1].copy_(host[off:end], non_blocking=True)
                    elif d == "d2h":
                        host[off:end].copy_(gpu[g0:g1], non_blocking=True)
                    else:  # d2d control: same device, no PCIe traffic
                        gpu2[g0:g1].copy_(gpu[g0:g1], non_blocking=True)
                    if args.rate_gbs > 0:
                        stream.synchronize()
                        el = time.monotonic() - t0
                        want = (end) / (args.rate_gbs * 1e9)
                        if want > el:
                            time.sleep(want - el)
            stream.synchronize()
            t1 = time.monotonic()
            gbs = n / (t1 - t0) / 1e9
            f.write(json.dumps({"t_start": t0, "t_end": t1, "bytes": n, "dir": d, "GB_s": gbs}) + "\n"); f.flush()
            print(f"{d} {n / 2**30:.2f} GiB in {(t1 - t0) * 1e3:.0f} ms = {gbs:.1f} GB/s", flush=True)
            if args.duty < 1.0:
                time.sleep((t1 - t0) * (1 - args.duty) / args.duty)
        f.write(json.dumps({"event": "end", "t": time.monotonic(), "bursts": k}) + "\n"); f.flush()


if __name__ == "__main__":
    main()
