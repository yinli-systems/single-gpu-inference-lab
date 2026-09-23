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

`--engine ce` (default) moves bytes with cudaMemcpyAsync = the copy engine, which is what vLLM
0.29.0's OffloadingConnector uses for GPU->CPU (`ops.swap_blocks_batch`). `--engine sm` moves the
same bytes with a Triton kernel that dereferences the pinned host pointer directly (the shape of
vLLM's `swap_blocks_triton` CPU->GPU fast path, and of NCCL's SHM-transport sender), so the two
engines separate "PCIe bytes" from "which DMA requester issues them".
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def make_sm_copy():
    import triton
    import triton.language as tl

    @triton.jit
    def _copy_kernel(src_addr, dst_addr, off_src_words, off_dst_words, n_words, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        src = tl.load(src_addr).to(tl.pointer_type(tl.int64)) + off_src_words
        dst = tl.load(dst_addr).to(tl.pointer_type(tl.int64)) + off_dst_words
        start = pid * BLOCK
        while start < n_words:
            idx = start + tl.arange(0, BLOCK)
            mask = idx < n_words
            tl.store(dst + idx, tl.load(src + idx, mask=mask, other=0), mask=mask)
            start += nprog * BLOCK

    return _copy_kernel


def set_mempolicy_bind(node):
    """set_mempolicy(2): MPOL_BIND to `node` (None = MPOL_DEFAULT). Returns 0 or -errno; needs no CPU affinity."""
    import ctypes
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if node is None:
            rc = libc.syscall(238, 0, None, 0)  # SYS_set_mempolicy, MPOL_DEFAULT
        else:
            mask = (ctypes.c_ulong * 2)(0, 0)
            mask[node // 64] = 1 << (node % 64)
            rc = libc.syscall(238, 2, mask, 128)  # MPOL_BIND, maxnode = 128 bits
        return int(rc) if rc == 0 else -ctypes.get_errno()
    except Exception as e:  # noqa: BLE001
        return repr(e)[:80]


def numa_placement(addr, nbytes, samples=256):
    """NUMA node of `samples` pages spread over [addr, addr+nbytes), via move_pages(2) with a null target (query mode).

    Works for driver-owned pinned mappings that /proc/self/numa_maps does not account; returns {"N<node>": count}."""
    import ctypes
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        page = os.sysconf("SC_PAGE_SIZE")
        npages = max(nbytes // page, 1)
        idx = sorted({(i * npages) // samples for i in range(samples)})
        pages = (ctypes.c_void_p * len(idx))(*[addr + i * page for i in idx])
        status = (ctypes.c_int * len(idx))()
        rc = libc.syscall(279, 0, len(idx), pages, None, status, 0)  # SYS_move_pages on x86_64
        if rc != 0:
            return {"error": f"move_pages rc {rc} errno {ctypes.get_errno()}"}
        out = {}
        for st in status:
            out[f"N{st}" if st >= 0 else f"err{st}"] = out.get(f"N{st}" if st >= 0 else f"err{st}", 0) + 1
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:100]}


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
    ap.add_argument("--engine", choices=["ce", "sm"], default="ce", help="ce = cudaMemcpyAsync (copy engine); sm = Triton kernel over the pinned host pointer")
    ap.add_argument("--sm-count", type=int, default=12, help="Triton programs for --engine sm (vLLM's swap_blocks_triton uses 12)")
    ap.add_argument("--numa-node", type=int, default=None, help="pin this process to the CPUs of NUMA node N before allocating the pinned buffer (first-touch places it there); the placement is read back with move_pages(2)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    n = int(args.bytes)
    chunk = args.chunk_mb * 2**20
    gn = min(n, args.gpu_buf_mb * 2**20)
    numa = {"requested": args.numa_node}
    if args.numa_node is not None:
        # memory policy first (works even when Slurm's cpuset excludes that node's CPUs), CPU affinity as best effort
        numa["mempolicy_rc"] = set_mempolicy_bind(args.numa_node)
        cpus = set()
        for part in open(f"/sys/devices/system/node/node{args.numa_node}/cpulist").read().strip().split(","):
            a, _, b = part.partition("-"); cpus.update(range(int(a), int(b or a) + 1))
        cpus &= os.sched_getaffinity(0)
        if cpus:
            os.sched_setaffinity(0, cpus)
        else:
            numa["affinity_note"] = "no CPU of that node in the allowed cpuset; memory policy only"
    numa["affinity"] = sorted(os.sched_getaffinity(0))
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    host.fill_(0)  # touch every page from this CPU set so the placement is decided here, not by the first copy
    numa["pages_by_node"] = numa_placement(host.data_ptr(), n)
    if args.numa_node is not None:
        set_mempolicy_bind(None)  # back to the default policy for everything allocated after the pinned buffer
    gpu = torch.empty(gn, dtype=torch.uint8, device=dev)
    gpu2 = torch.empty(gn, dtype=torch.uint8, device=dev) if args.dir == "d2d" else None
    stream = torch.cuda.Stream(device=dev)
    sm_copy = None
    if args.engine == "sm":
        kernel = make_sm_copy()
        addr_host = torch.tensor([host.data_ptr()], dtype=torch.int64, device=dev)
        addr_gpu = torch.tensor([gpu.data_ptr()], dtype=torch.int64, device=dev)
        addr_gpu2 = torch.tensor([gpu2.data_ptr()], dtype=torch.int64, device=dev) if gpu2 is not None else None

        def sm_copy(d, off, end, g0):  # same byte ranges as the copy-engine branches below; offsets in 8-byte words
            nw = (end - off) // 8
            if d == "h2d":
                kernel[(args.sm_count,)](addr_host, addr_gpu, off // 8, g0 // 8, nw, BLOCK=1024)
            elif d == "d2h":
                kernel[(args.sm_count,)](addr_gpu, addr_host, g0 // 8, off // 8, nw, BLOCK=1024)
            else:
                kernel[(args.sm_count,)](addr_gpu, addr_gpu2, g0 // 8, g0 // 8, nw, BLOCK=1024)
    # warm-up: one chunk each way so context/pinned mappings (and the Triton JIT) are established before the window
    with torch.cuda.stream(stream):
        gpu[:chunk].copy_(host[:chunk], non_blocking=True)
        host[:chunk].copy_(gpu[:chunk], non_blocking=True)
        if sm_copy is not None:
            sm_copy("h2d", 0, chunk, 0); sm_copy("d2h", 0, chunk, 0)
    stream.synchronize()
    if sm_copy is not None:  # the SM path must move the same bytes: check one chunk round-trip
        host[:chunk].fill_(7); host[chunk:2 * chunk].zero_()
        with torch.cuda.stream(stream):
            sm_copy("h2d", 0, chunk, 0); sm_copy("d2h", chunk, 2 * chunk, 0)
        stream.synchronize()
        assert bool((host[chunk:2 * chunk] == 7).all()), "sm copy round-trip mismatch"
    print(f"hog ready on {args.device}: {n / 2**30:.2f} GiB pinned, {gn / 2**20:.0f} MiB device window, dir {args.dir} engine {args.engine} duty {args.duty} rate {args.rate_gbs}", flush=True)
    if args.start_file:
        while not os.path.exists(args.start_file):
            time.sleep(0.01)
    time.sleep(args.start_delay)
    t_begin = time.monotonic()
    t_end_all = t_begin + args.duration
    k = 0
    with open(args.log, "a") as f:
        f.write(json.dumps({"event": "start", "t": t_begin, "args": vars(args), "numa": numa}) + "\n"); f.flush()
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
                    if sm_copy is not None:
                        sm_copy(d, off, end, g0)
                    elif d == "h2d":
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
            f.write(json.dumps({"t_start": t0, "t_end": t1, "bytes": n, "dir": d, "engine": args.engine, "GB_s": gbs}) + "\n"); f.flush()
            print(f"{d}/{args.engine} {n / 2**30:.2f} GiB in {(t1 - t0) * 1e3:.0f} ms = {gbs:.1f} GB/s", flush=True)
            if args.duty < 1.0:
                time.sleep((t1 - t0) * (1 - args.duty) / args.duty)
        f.write(json.dumps({"event": "end", "t": time.monotonic(), "bursts": k}) + "\n"); f.flush()


if __name__ == "__main__":
    main()
