#!/usr/bin/env python3
"""Per-step sequence inside store cells (rank 0): chunk steps with host period vs cuda_ms, and
what precedes each chunk. Also per-cell: distribution of (period - cuda_ms) for chunk steps, the
run structure (prompt boundaries = num_tokens < 512 chunk), and store metrics windows."""
import glob, json, os, sys
import numpy as np

d = sys.argv[1].rstrip("/")
rank = sys.argv[2] if len(sys.argv) > 2 else ".dp0"
nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rec = json.load(open(d + "/kvoffload.json"))
steps = [json.loads(l) for l in open(d + "/trace/step.jsonl" + rank) if l.strip()]
for i in range(1, len(steps)):
    steps[i]["period"] = (steps[i]["t"] - steps[i - 1]["t"]) * 1e3
    steps[i]["prev_tokens"] = steps[i - 1]["num_tokens"]
    steps[i]["prev_cuda"] = steps[i - 1]["cuda_ms"]
steps[0]["period"] = float("nan"); steps[0]["prev_tokens"] = -1; steps[0]["prev_cuda"] = float("nan")
for c in rec["cells"]:
    if c["kind"] != "store":
        continue
    t0, t1 = c["t_window0"], c["t_window1"]
    s0 = [s for s in steps if t0 <= s["t"] <= t1]
    ch = [s for s in s0 if s["num_tokens"] >= 256]
    cm = np.array([s["cuda_ms"] for s in ch]); pr = np.array([s["period"] for s in ch])
    # chunk steps whose previous step was also a chunk (store of prev chunk pending) vs first chunks
    after_chunk = np.array([s["prev_tokens"] >= 256 for s in ch])
    print(f"\n=== store B={c['B']} rep={c['repeat']} rank{rank}: {len(ch)} chunk steps; cuda p50 {np.median(cm):.1f}; period p50 {np.median(pr):.1f}; "
          f"period-cuda p50 {np.median(pr-cm):.1f}; after-chunk cuda p50 {np.median(cm[after_chunk]) if after_chunk.any() else float('nan'):.1f} (n={after_chunk.sum()}) "
          f"first-chunk cuda p50 {np.median(cm[~after_chunk]) if (~after_chunk).any() else float('nan'):.1f} (n={(~after_chunk).sum()}); "
          f"cuda hist: <60 {np.mean(cm<60):.2f} 60-75 {np.mean((cm>=60)&(cm<75)):.2f} 75-95 {np.mean((cm>=75)&(cm<95)):.2f} >95 {np.mean(cm>=95):.2f}")
    # print a stretch of consecutive steps from the middle of the window
    mid = len(s0) // 2
    for s in s0[mid:mid + nshow]:
        print(f"  step {s['step']:7d} t+{(s['t']-t0):6.2f}s tok {s['num_tokens']:4d} pad {s['padded_tokens']:4d} reqs {s['num_reqs']:3d} cg {s['cg_mode']:<9s} cuda {s['cuda_ms']:7.1f} period {s['period']:7.1f} prev_tok {s['prev_tokens']:4d}")
