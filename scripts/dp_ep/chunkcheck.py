#!/usr/bin/env python3
"""Task 26 real mover: the prefill-chunk step's own CUDA time per store cell, connector on vs off.

usage: chunkcheck.py <results-dir> [<results-dir> ...]    (c26kv-* DP2 dirs or c26kv1-* DP=1 control dirs)

Reads kvoffload.json (cell windows) and every trace/step.jsonl* file (rank-suffixed .dp0/.dp1 at DP2, whatever
the tracer wrote at DP=1). Per store cell and rank: chunk steps (num_tokens >= 256) CUDA p50/p95/max, decode-only
steps CUDA p50, chunk tokens / padded / cg_mode, the chunk-step CUDA p50 by window third (uniform vs bursty) and the
fraction of chunk steps above 70 ms (the 85-ms mode seen at DP2 with the connector on, job 1602608).
"""

import glob
import json
import os
import sys

import numpy as np

CHUNK_MIN = 256

for d in sys.argv[1:]:
    d = d.rstrip("/")
    print("==", os.path.basename(d))
    rec = json.load(open(d + "/kvoffload.json"))
    traces = sorted(glob.glob(d + "/trace/step.jsonl*"))
    st = {os.path.basename(p).replace("step.jsonl", "") or ".dp0": [json.loads(l) for l in open(p) if l.strip()] for p in traces}
    print("   step traces:", {k: len(v) for k, v in st.items()}, "| cells:", len(rec["cells"]), "| server:", rec["args"].get("ports"))
    for c in rec["cells"]:
        if c["kind"] != "store":
            continue
        t0, t1 = c["t_window0"], c["t_window1"]
        for rk, steps in st.items():
            s0 = [s for s in steps if t0 <= s["t"] <= t1]
            ch = [s for s in s0 if s["num_tokens"] >= CHUNK_MIN]
            dec = [s for s in s0 if s["num_tokens"] < CHUNK_MIN]
            if not ch:
                print(f"store B={c['B']} rep={c['repeat']} rank{rk}: no chunk steps in window ({len(s0)} steps)")
                continue
            cm = np.array([s["cuda_ms"] for s in ch])
            ts = np.array([s["t"] for s in ch])
            edges = np.quantile(ts, [0, 1 / 3, 2 / 3, 1])
            thirds = [np.median(cm[(ts >= edges[i]) & (ts <= edges[i + 1])]) for i in range(3)]
            print(f"store B={c['B']} rep={c['repeat']} rank{rk}: chunk steps {len(ch)} cuda p50 {np.median(cm):.1f} p95 {np.quantile(cm, .95):.1f} max {cm.max():.0f}; "
                  f"decode-only steps {len(dec)} cuda p50 {np.median([s['cuda_ms'] for s in dec]) if dec else float('nan'):.1f}; "
                  f"chunk tokens p50 {np.median([s['num_tokens'] for s in ch]):.0f} padded {np.median([s['padded_tokens'] for s in ch]):.0f} "
                  f"cg {sorted(set(str(s['cg_mode']) for s in ch))} reqs p50 {np.median([s['num_reqs'] for s in ch]):.0f}; "
                  f"by third {[f'{p:.1f}' for p in thirds]}; frac > 70 ms {(cm > 70).mean():.2f}; "
                  f"train n {c['train']['n']} ttft p50 {c['train']['ttft_p50_ms'] or 0:.0f} ms")
