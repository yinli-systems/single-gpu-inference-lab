#!/usr/bin/env python3
"""Campaign G1: cross-rank DP padding amplification, graph (arm A) vs eager (arm B) oracle.

Inputs: one results dir per arm (waves.json + trace/step.jsonl.dp{0,1}); the step trace is
valid in both modes (`num_tokens` = true scheduled tokens, `padded_tokens` = tokens actually
executed after CUDA-graph bucket + DP padding, `cg_mode`, `cuda_ms` = this rank's step GPU
time incl. waiting inside collectives, `t` = host monotonic seconds at step start).

Per cell window [t_window0, t_window1] and rank:
  true / exec tokens   median num_tokens / padded_tokens over the window's steps
  A_pad                R * max_r(true_r) / sum_r(true_r), per step, paired by nearest `t`
                       (same node clock; ranks step in lockstep)
  exec_amp             sum_r(padded_r) / sum_r(true_r): work actually executed vs useful
  cg_mode              distribution of the runtime graph mode
  step ms              cuda_ms p50 / p95; step interval p50 (t[i+1]-t[i]) = wave period
  ITL / TTFT           user-visible decode inter-token latency per rank, prefill TTFT
Cross-arm (same cell,B,L; repeats pooled): T_A/T_B for the decode rank's ITL p50, for the
prefill TTFT and for the wave period; oracle C = min(A,B) per cell; gain vs the default (A).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, p)) if len(a) else float("nan")


def analyze_dir(d: Path):
    waves = json.load(open(d / "waves.json"))
    step = {r: load_jsonl(d / "trace" / f"step.jsonl.dp{r}") for r in (0, 1)}
    t = {r: np.array([x["t"] for x in step[r]]) for r in (0, 1)}
    rows = []
    for c in waves["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        sel = {r: [step[r][i] for i in np.where((t[r] >= t0) & (t[r] <= t1))[0]] for r in (0, 1)}
        if not sel[0] or not sel[1]:
            continue
        # pair r0 steps with the nearest r1 step (lockstep waves; reject pairs further apart than half a period)
        t1s = np.array([s["t"] for s in sel[1]])
        period = float(np.median(np.diff(np.array([s["t"] for s in sel[0]])))) if len(sel[0]) > 1 else 0.05
        pairs = []
        for s0 in sel[0]:
            j = int(np.argmin(np.abs(t1s - s0["t"])))
            if abs(t1s[j] - s0["t"]) <= period / 2:
                pairs.append((s0, sel[1][j]))
        n = np.array([[a["num_tokens"], b["num_tokens"]] for a, b in pairs], dtype=float)
        p = np.array([[a["padded_tokens"], b["padded_tokens"]] for a, b in pairs], dtype=float)
        a_pad = 2 * n.max(1) / n.sum(1) if len(n) else np.array([np.nan])
        exec_amp = p.sum(1) / n.sum(1) if len(n) else np.array([np.nan])
        per_rank = {}
        for r in (0, 1):
            cm = np.array([s["cuda_ms"] for s in sel[r]])
            iv = np.diff(np.array([s["t"] for s in sel[r]])) * 1e3
            per_rank[r] = {"steps": len(sel[r]), "true_p50": q([s["num_tokens"] for s in sel[r]], .5), "exec_p50": q([s["padded_tokens"] for s in sel[r]], .5),
                           "exec_max": max(s["padded_tokens"] for s in sel[r]), "cg": dict(Counter(s["cg_mode"] for s in sel[r])),
                           "cuda_p50": q(cm, .5), "cuda_p95": q(cm, .95), "cuda_mean": float(cm.mean()), "period_p50": q(iv, .5), "period_p95": q(iv, .95)}
        itl = {}
        for r, (base, streams) in enumerate(c["decode"].items()):
            gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if t0 <= a <= t1]
            itl[r] = {"p50": q(gaps, .5), "p95": q(gaps, .95), "n": len(gaps)}
        rows.append({"cell": c["cell"], "B": c["B"], "L": c["L"], "rep": c["repeat"], "pairs": len(pairs), "A_pad_p50": q(a_pad, .5), "A_pad_max": float(np.nanmax(a_pad)),
                     "exec_amp_p50": q(exec_amp, .5), "exec_amp_mean": float(np.nanmean(exec_amp)), "rank": per_rank, "itl": itl,
                     "ttft_ms": c["prefill"][0]["ttft_ms"] if c["prefill"] else float("nan")})
    return waves["args"], rows


def fmt(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def print_arm(label, rows):
    print(f"\n### arm {label}\n")
    print("| cell | B | L | rep | pairs | true r0/r1 p50 | exec r0/r1 p50 (max) | A_pad p50/max | exec_amp p50 | cg r0 | cg r1 | cuda r0 p50/p95 | cuda r1 p50/p95 | period r0 p50 | ITL r0 p50/p95 | ITL r1 p50/p95 | TTFT |")
    print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: |")
    for x in rows:
        r0, r1 = x["rank"][0], x["rank"][1]
        cg = lambda r: ",".join(f"{k.replace('CUDAGraphMode.', '')}:{v}" for k, v in sorted(r["cg"].items(), key=lambda kv: -kv[1]))
        print(f"| {x['cell']} | {x['B']} | {x['L']} | {x['rep']} | {x['pairs']} | {fmt(r0['true_p50'], 0)}/{fmt(r1['true_p50'], 0)} | {fmt(r0['exec_p50'], 0)}/{fmt(r1['exec_p50'], 0)} ({r0['exec_max']}/{r1['exec_max']}) | "
              f"{fmt(x['A_pad_p50'], 2)}/{fmt(x['A_pad_max'], 2)} | {fmt(x['exec_amp_p50'], 2)} | {cg(r0)} | {cg(r1)} | {fmt(r0['cuda_p50'])}/{fmt(r0['cuda_p95'])} | {fmt(r1['cuda_p50'])}/{fmt(r1['cuda_p95'])} | {fmt(r0['period_p50'])} | "
              f"{fmt(x['itl'].get(0, {}).get('p50'))}/{fmt(x['itl'].get(0, {}).get('p95'))} | {fmt(x['itl'].get(1, {}).get('p50'))}/{fmt(x['itl'].get(1, {}).get('p95'))} | {fmt(x['ttft_ms'], 0)} |")


def pool(rows):
    """(cell,B,L) -> pooled metrics over repeats (median of per-rep values)."""
    g = defaultdict(list)
    for x in rows:
        g[(x["cell"], x["B"], x["L"])].append(x)
    out = {}
    for k, xs in g.items():
        med = lambda f: float(np.nanmedian([f(x) for x in xs]))
        out[k] = {"n": len(xs), "itl1_p50": med(lambda x: x["itl"][1]["p50"]), "itl1_p95": med(lambda x: x["itl"][1]["p95"]), "itl0_p50": med(lambda x: x["itl"][0]["p50"]),
                  "ttft": med(lambda x: x["ttft_ms"]), "period": med(lambda x: x["rank"][0]["period_p50"]), "cuda0": med(lambda x: x["rank"][0]["cuda_p50"]), "cuda1": med(lambda x: x["rank"][1]["cuda_p50"]),
                  "A_pad": med(lambda x: x["A_pad_p50"]), "exec_amp": med(lambda x: x["exec_amp_mean"]), "exec1": med(lambda x: x["rank"][1]["exec_p50"]), "true1": med(lambda x: x["rank"][1]["true_p50"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", type=Path, help="graph-mode results dir (default runtime: graph + max-DP padding)")
    ap.add_argument("--arm-b", type=Path, help="eager results dir (true per-rank shapes)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    res = {}
    for label, d in (("A", args.arm_a), ("B", args.arm_b)):
        if d is None:
            continue
        a, rows = analyze_dir(d)
        res[label] = {"dir": str(d), "args": a, "rows": rows}
        print_arm(f"{label} ({d.name})", rows)
    if "A" in res and "B" in res:
        pa, pb = pool(res["A"]["rows"]), pool(res["B"]["rows"])
        print("\n### oracle per cell (repeats pooled by median; T_A/T_B < 1 means the default graph arm is faster)\n")
        print("| cell | B | L | A_pad (A) | exec_amp (A) | r1 exec/true (A) | ITL r1 p50 A / B / ratio | ITL r1 p95 A / B | TTFT A / B / ratio | period A / B / ratio | oracle gain ITL r1 p50 | oracle gain TTFT |")
        print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: |")
        gains = []
        for k in sorted(pa):
            if k not in pb:
                continue
            a, b = pa[k], pb[k]
            r_itl = a["itl1_p50"] / b["itl1_p50"]; r_ttft = a["ttft"] / b["ttft"]; r_per = a["period"] / b["period"]
            g_itl = max(0.0, 1 - min(a["itl1_p50"], b["itl1_p50"]) / a["itl1_p50"]); g_ttft = max(0.0, 1 - min(a["ttft"], b["ttft"]) / a["ttft"]) if not np.isnan(a["ttft"]) else float("nan")
            gains.append((k, g_itl, g_ttft, r_itl, r_ttft))
            print(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(a['A_pad'], 2)} | {fmt(a['exec_amp'], 2)} | {fmt(a['exec1'], 0)}/{fmt(a['true1'], 0)} | {fmt(a['itl1_p50'])} / {fmt(b['itl1_p50'])} / {fmt(r_itl, 2)} | {fmt(a['itl1_p95'])} / {fmt(b['itl1_p95'])} | "
                  f"{fmt(a['ttft'], 0)} / {fmt(b['ttft'], 0)} / {fmt(r_ttft, 2)} | {fmt(a['period'])} / {fmt(b['period'])} / {fmt(r_per, 2)} | {fmt(100 * g_itl)}% | {fmt(100 * g_ttft) if not np.isnan(g_ttft) else '—'}% |")
        gi = [g for _, g, _, _, _ in gains]; gt = [g for _, _, g, _, _ in gains if not np.isnan(g)]
        print(f"\noracle gain over default (A): decode-rank ITL p50 mean {100 * np.mean(gi):.1f}% max {100 * np.max(gi):.1f}% (cells where B wins: {sum(g > 0 for g in gi)}/{len(gi)}); "
              f"TTFT mean {100 * np.mean(gt):.1f}% max {100 * np.max(gt):.1f}% (B wins: {sum(g > 0 for g in gt)}/{len(gt)})")
        print("gate: <5% kill / 5-10% engineering-only / >10% alive / >20% strong")
        res["oracle"] = [{"cell": k[0], "B": k[1], "L": k[2], "gain_itl1_p50": g1, "gain_ttft": g2, "ratio_itl1": r1, "ratio_ttft": r2} for k, g1, g2, r1, r2 in gains]
    args.output.write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
