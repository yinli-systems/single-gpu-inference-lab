#!/usr/bin/env python3
"""Whole-run time series of eager (cg NONE) step CUDA time, per rank: consecutive-eager-step runs
(prompt chunk trains) collapsed to (t_rel, n_steps, tokens p50, cuda p50), with mode flips marked,
plus the cell timeline. Usage: modeseries.py <results-dir> [thresh_ms=65]"""
import glob, json, os, sys
import numpy as np

d = sys.argv[1].rstrip("/")
th = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0
rec = json.load(open(d + "/kvoffload.json"))
cells = rec["cells"]
t_ref = min(c["t_window0"] for c in cells)
print("== cells (t_rel of window start, kind, B, rep, train n, ttft p50)")
for c in cells:
    print(f"  {c['t_window0']-t_ref:7.1f}s .. {c['t_window1']-t_ref:7.1f}s  {c['kind']:5s} B={c['B']:2d} r{c['repeat']} train {c['train']['n']:3d} ttft {c['train']['ttft_p50_ms'] or 0:5.0f} ms")
for p in sorted(glob.glob(d + "/trace/step.jsonl*")):
    rk = os.path.basename(p).replace("step.jsonl", "") or ".dp0"
    steps = [json.loads(l) for l in open(p) if l.strip()]
    eager = [s for s in steps if str(s["cg_mode"]).endswith("NONE") and s["num_tokens"] >= 256]
    print(f"\n== rank{rk}: {len(steps)} steps, {len(eager)} eager >=256-token steps; runs of consecutive eager steps (gap < 0.5 s):")
    runs = []
    for s in eager:
        if runs and s["t"] - runs[-1][-1]["t"] < 0.5:
            runs[-1].append(s)
        else:
            runs.append([s])
    prev_mode = None
    for r in runs:
        cm = np.array([s["cuda_ms"] for s in r])
        mode = "SLOW" if np.median(cm) > th else "fast"
        flip = "  <-- FLIP" if prev_mode is not None and mode != prev_mode else ""
        mixed = " (mixed: %.2f slow)" % (cm > th).mean() if 0.15 < (cm > th).mean() < 0.85 else ""
        t_rel = r[0]["t"] - t_ref
        cell = next((f"{c['kind']} B={c['B']} r{c['repeat']}" for c in cells if c["t_window0"] - 60 <= r[0]["t"] <= c["t_window1"] + 5), "between")
        print(f"  t {t_rel:8.1f}s n {len(r):3d} tok p50 {np.median([s['num_tokens'] for s in r]):4.0f} cuda p50 {np.median(cm):6.1f} min {cm.min():6.1f} max {cm.max():6.1f} {mode}{mixed}{flip}  [{cell}]")
        prev_mode = mode
