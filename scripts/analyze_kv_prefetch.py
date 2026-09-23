#!/usr/bin/env python3
"""Oracle KV-prefetch upper bound: TTFT_resume(lead) per prefix, the knee, the
CPU->GPU load time, and the contention cost on concurrent decoders.

Input: measure_kv_prefetch.py JSON(s). For each (prefix P, arm, lead): resume
TTFT median / min–max over repeats. Knee = smallest lead whose TTFT is within
5% of the retained arm. Load time = reactive TTFT − retained TTFT. With
background decoders: inter-token latency of the decoders inside the
[prefetch start, resume end] window versus outside it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    for rf in args.results:
        rec = json.load(open(rf))
        cells = defaultdict(list)
        for t in rec["trials"]:
            cells[(t["P"], t["arm"], t["lead_ms"])].append(t)
        Ps = sorted({k[0] for k in cells})
        leads = sorted({k[2] for k in cells if k[1] == "C"})
        print(f"\n## {rf.stem}: resume TTFT (ms), median over repeats (min–max)")
        print("| prefix | A retained | B reactive (load on critical path) | " + " | ".join(f"C lead {l}" for l in leads) + " | knee lead | load ms | oracle gain |")
        print("| ---: | ---: | ---: | " + " | ".join("---:" for _ in leads) + " | ---: | ---: | ---: |")
        out = {}
        for P in Ps:
            def stat(arm, lead=0):
                v = [t["resume"]["ttft_ms"] for t in cells.get((P, arm, lead), [])]
                return (float(np.median(v)), min(v), max(v)) if v else (float("nan"),) * 3
            a, b = stat("A"), stat("B")
            cs = {l: stat("C", l) for l in leads}
            knee = next((l for l in leads if cs[l][0] <= a[0] * 1.05), None)
            load = b[0] - a[0]
            best_c = min(cs.values(), key=lambda x: x[0])[0]
            gain = (b[0] - best_c) / b[0]
            out[P] = {"retained": a, "reactive": b, "oracle": {l: cs[l] for l in leads}, "knee_lead_ms": knee, "load_ms": load, "oracle_gain": gain}
            fmt = lambda s: f"{s[0]:.0f} ({s[1]:.0f}–{s[2]:.0f})"
            print(f"| {P} | {fmt(a)} | {fmt(b)} | " + " | ".join(fmt(cs[l]) for l in leads) + f" | {knee if knee is not None else '—'} | {load:.0f} | {gain*100:.0f}% |")
        # background contention
        bg = rec.get("background_token_times") or []
        if bg:
            gaps_in, gaps_out = [], []
            windows = [(t["t_prefetch_start"], t["t_resume_end"]) for t in rec["trials"] if t["arm"] != "A"]
            fill = [(t["t_trial_start"], t["t_prefetch_start"]) for t in rec["trials"]]
            for ts in bg:
                ts = np.array(ts)
                for a_, b_ in zip(ts, ts[1:]):
                    gap = (b_ - a_) * 1e3
                    if any(w0 <= a_ <= w1 for w0, w1 in windows):
                        gaps_in.append(gap)
                    elif not any(f0 <= a_ <= f1 for f0, f1 in fill):
                        gaps_out.append(gap)
            gi, go = np.array(gaps_in), np.array(gaps_out)
            print(f"\nbackground decoders ({len(bg)} runs): ITL inside prefetch/resume windows p50 {np.median(gi):.1f} / p95 {np.quantile(gi,.95):.1f} / p99 {np.quantile(gi,.99):.1f} / max {gi.max():.0f} ms (n={len(gi)}); "
                  f"outside (steady, excluding filler) p50 {np.median(go):.1f} / p95 {np.quantile(go,.95):.1f} / p99 {np.quantile(go,.99):.1f} ms (n={len(go)})")
            out["background_itl"] = {"in": {"p50": float(np.median(gi)), "p95": float(np.quantile(gi,.95)), "p99": float(np.quantile(gi,.99)), "max": float(gi.max()), "n": len(gi)},
                                     "out": {"p50": float(np.median(go)), "p95": float(np.quantile(go,.95)), "p99": float(np.quantile(go,.99)), "n": len(go)}}
            # per-arm ITL max inside the window (does the load stall decoders?)
            print("| prefix | arm | lead | decoder ITL inside window: p50 / p95 / max |"); print("| ---: | --- | ---: | ---: |")
            for (P, arm, lead), ts_ in sorted(cells.items()):
                if arm == "A":
                    continue
                g = []
                for t in ts_:
                    w0, w1 = t["t_prefetch_start"], t["t_resume_end"]
                    for ts in bg:
                        ts = np.array(ts); sel = ts[(ts >= w0) & (ts <= w1)]
                        g += list(np.diff(sel) * 1e3)
                if g:
                    g = np.array(g); print(f"| {P} | {arm} | {lead} | {np.median(g):.1f} / {np.quantile(g,.95):.1f} / {g.max():.0f} |")
        report[rf.stem] = out
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
