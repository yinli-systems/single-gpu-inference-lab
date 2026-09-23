#!/usr/bin/env python3
"""Scan every results dir with step traces: eager 512-token steps (cg NONE, num_tokens == 512)
per rank -> count, cuda p50, fraction > 65 ms (the 85-ms 'slow mode' of job 1602608), plus the
number of mode flips in time order. Usage: modescan.py results/*/"""
import glob, json, os, sys
import numpy as np

for d in sorted(sys.argv[1:], key=lambda p: os.path.getmtime(p)):
    d = d.rstrip("/")
    traces = sorted(glob.glob(d + "/trace/step.jsonl*"))
    if not traces:
        continue
    out = []
    for p in traces:
        rk = os.path.basename(p).replace("step.jsonl", "") or "(single)"
        try:
            steps = [json.loads(l) for l in open(p) if l.strip()]
        except Exception as e:
            out.append(f"{rk}: unreadable ({e})"); continue
        eg = [s for s in steps if str(s["cg_mode"]).endswith("NONE") and s["num_tokens"] == 512 and s["cuda_ms"] < 1000]
        if len(eg) < 5:
            out.append(f"{rk}: {len(eg)} eager-512 steps"); continue
        cm = np.array([s["cuda_ms"] for s in eg])
        slow = cm > 65
        flips = int(np.sum(slow[1:] != slow[:-1]))
        out.append(f"{rk}: n {len(eg):4d} cuda p50 {np.median(cm):5.1f} p10 {np.quantile(cm,.1):5.1f} p90 {np.quantile(cm,.9):5.1f} slow-frac {slow.mean():.2f} flips {flips}")
    print(os.path.basename(d)[:64].ljust(64), " | ".join(out))
