#!/usr/bin/env python3
"""Task 26 analysis: does bulk PCIe traffic on rank 0's GPU raise the decode latency of rank 1?

Input: a results dir with pcie.json (from measure_pcie.py), hog/*.jsonl (per-burst logs),
hog-standalone.jsonl (link bandwidth without server traffic) and trace/step.jsonl.dp{0,1}.

Per cell: both ranks' ITL inside the window (p50/p95), per-rank step period and GPU time from
the step traces (window-selected; same node monotonic clock), and the hog's achieved GB/s and
active fraction. Windows of duty<1 cells are additionally split into hog-active vs hog-pause
steps (burst intervals from the hog log) so the per-step effect during transfers is visible even
when the window average dilutes it.
Pooled by (spec, B) with medians over repeats; ratios vs the `off` cell of the same B.
Gate (pre-registered): rank-1 p95 TPOT +10 % with rank-0 transfer = strong; <5 % kill.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, p)) if len(a) else float("nan")


def fmt(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def in_bursts(t, bursts):
    return any(b["t_start"] <= t <= b["t_end"] for b in bursts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rec = json.load(open(args.dir / "pcie.json"))
    step = {r: load_jsonl(args.dir / "trace" / f"step.jsonl.dp{r}") for r in (0, 1)}
    st = {r: np.array([x["t"] for x in step[r]]) for r in (0, 1)}
    sa = load_jsonl(args.dir / "hog-standalone.jsonl")
    if sa:
        print("standalone hog bandwidth (no server): " + ", ".join(f"{d} p50 {fmt(q([b['GB_s'] for b in sa if b.get('dir') == d], .5))} GB/s" for d in ("h2d", "d2h", "d2d")) + "\n")
    rows = []
    print("| spec | B | rep | ITL r0 p50/p95 | ITL r1 p50/p95 | err | step r0 period / cuda p50 | step r1 period / cuda p50 | r1 cuda p50 in-burst / pause | hog bursts / active frac / GB/s |")
    print("| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |")
    for c in rec["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        itl = {}
        for r, (base, streams) in enumerate(c["decode"].items()):
            gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if t0 <= a <= t1]
            itl[r] = {"p50": q(gaps, .5), "p95": q(gaps, .95), "n": len(gaps), "errors": sum(1 for s in streams if s.get("error"))}
        bursts = [b for b in load_jsonl(Path(c["hog_log"])) if "t_start" in b] if c.get("hog_log") and Path(c["hog_log"]).exists() else []
        sp = {}
        for r in (0, 1):
            sel = [step[r][i] for i in np.where((st[r] >= t0) & (st[r] <= t1))[0]]
            iv = np.diff(np.array([s["t"] for s in sel])) * 1e3 if len(sel) > 1 else np.array([])
            sp[r] = {"n": len(sel), "period_p50": q(iv, .5), "period_p95": q(iv, .95), "cuda_p50": q([s["cuda_ms"] for s in sel], .5), "cuda_p95": q([s["cuda_ms"] for s in sel], .95),
                     "tokens_p50": q([s["num_tokens"] for s in sel], .5)}
            if bursts:
                inb = [s["cuda_ms"] for s in sel if in_bursts(s["t"], bursts)]
                outb = [s["cuda_ms"] for s in sel if not in_bursts(s["t"], bursts)]
                sp[r].update({"cuda_p50_inburst": q(inb, .5), "cuda_p50_pause": q(outb, .5), "n_inburst": len(inb), "n_pause": len(outb)})
        active = sum(b["t_end"] - b["t_start"] for b in bursts if t0 <= b["t_start"] <= t1)
        hog = {"bursts": len(bursts), "active_frac": active / (t1 - t0), "GB_s_p50": q([b["GB_s"] for b in bursts], .5), "rc": c.get("hog_rc")}
        row = {"spec": c["spec"], "B": c["B"], "rep": c["repeat"], "itl": itl, "step": sp, "hog": hog}
        rows.append(row)
        print(f"| {c['spec']} | {c['B']} | {c['repeat']} | {fmt(itl[0]['p50'])}/{fmt(itl[0]['p95'])} | {fmt(itl[1]['p50'])}/{fmt(itl[1]['p95'])} | {itl[0]['errors']}/{itl[1]['errors']} | "
              f"{fmt(sp[0]['period_p50'])} / {fmt(sp[0]['cuda_p50'])} | {fmt(sp[1]['period_p50'])} / {fmt(sp[1]['cuda_p50'])} | "
              f"{fmt(sp[1].get('cuda_p50_inburst', float('nan')))} / {fmt(sp[1].get('cuda_p50_pause', float('nan')))} | {hog['bursts']} / {fmt(hog['active_frac'], 2)} / {fmt(hog['GB_s_p50'])}{'' if hog['rc'] in (None, 0) else ' rc=' + str(hog['rc'])} |")
    g = defaultdict(list)
    for x in rows:
        g[(x["spec"], x["B"])].append(x)
    pooled = {}
    for k, xs in g.items():
        pooled[k] = {f"itl{r}_{p}": float(np.nanmedian([x["itl"][r][p] for x in xs])) for r in (0, 1) for p in ("p50", "p95")}
        for r in (0, 1):
            pooled[k][f"period{r}"] = float(np.nanmedian([x["step"][r]["period_p50"] for x in xs]))
            pooled[k][f"cuda{r}"] = float(np.nanmedian([x["step"][r]["cuda_p50"] for x in xs]))
            pooled[k][f"cuda{r}_inburst"] = float(np.nanmedian([x["step"][r].get("cuda_p50_inburst", np.nan) for x in xs]))
        pooled[k]["hog_GB_s"] = float(np.nanmedian([x["hog"]["GB_s_p50"] for x in xs])); pooled[k]["active_frac"] = float(np.nanmedian([x["hog"]["active_frac"] for x in xs]))
    print("\n### rank-1 (no KV movement) latency vs rank-0 PCIe traffic (pooled medians over repeats; ratio vs `off` at the same B)\n")
    print("| spec | B | hog GB/s / active | r1 ITL p50 off → hog (ratio) | r1 ITL p95 off → hog (ratio) | r1 step period off → hog | r1 cuda p50 off → hog (in-burst) | r0 ITL p50 off → hog (ratio) | r0 cuda p50 off → hog (in-burst) |")
    print("| --- | ---: | --- | --- | --- | --- | --- | --- | --- |")
    summary = []
    for (spec, B), v in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if spec == "off":
            continue
        base = pooled.get(("off", B))
        if base is None:
            continue
        s = {"spec": spec, "B": B, "hog_GB_s": v["hog_GB_s"], "active_frac": v["active_frac"]}
        for r in (0, 1):
            for p in ("p50", "p95"):
                s[f"r{r}_{p}_off"], s[f"r{r}_{p}_hog"] = base[f"itl{r}_{p}"], v[f"itl{r}_{p}"]; s[f"r{r}_{p}_ratio"] = v[f"itl{r}_{p}"] / base[f"itl{r}_{p}"]
            s[f"period{r}_off"], s[f"period{r}_hog"] = base[f"period{r}"], v[f"period{r}"]
            s[f"cuda{r}_off"], s[f"cuda{r}_hog"], s[f"cuda{r}_inburst"] = base[f"cuda{r}"], v[f"cuda{r}"], v[f"cuda{r}_inburst"]
        summary.append(s)
        print(f"| {spec} | {B} | {fmt(v['hog_GB_s'])} / {fmt(v['active_frac'], 2)} | {fmt(s['r1_p50_off'])} → {fmt(s['r1_p50_hog'])} ({fmt(s['r1_p50_ratio'], 2)}) | {fmt(s['r1_p95_off'])} → {fmt(s['r1_p95_hog'])} ({fmt(s['r1_p95_ratio'], 2)}) | "
              f"{fmt(s['period1_off'])} → {fmt(s['period1_hog'])} | {fmt(s['cuda1_off'])} → {fmt(s['cuda1_hog'])} ({fmt(s['cuda1_inburst'])}) | {fmt(s['r0_p50_off'])} → {fmt(s['r0_p50_hog'])} ({fmt(s['r0_p50_ratio'], 2)}) | {fmt(s['cuda0_off'])} → {fmt(s['cuda0_hog'])} ({fmt(s['cuda0_inburst'])}) |")
    if summary:
        worst = max(summary, key=lambda s: s["r1_p95_ratio"] if not np.isnan(s["r1_p95_ratio"]) else 0)
        print(f"\nlargest rank-1 ITL p95 effect: {worst['spec']} B={worst['B']}: p95 ×{worst['r1_p95_ratio']:.2f}, p50 ×{worst['r1_p50_ratio']:.2f}; gate: rank-1 p95 +10 % strong, <5 % kill")
    args.output.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
