#!/usr/bin/env python3
"""Analyze live trace-replay runs (replay_trace_serving.py) per pre-registration addenda 4 and 5.

  --dir RESULTS [--dir ...]   one or more result directories (<cond>-r<k>.json + trace/<cond>-r<k>.jsonl[.gz])

Per condition (name without -r<k>): mean (min-max) over repeats of primary goodput (TTFT <= 5 s,
TPOT <= 100 ms), all SLO pairs, TTFT and TPOT percentiles. From the engine trace (if present):
P(prefills per step > 1) and P(geometry error > 5 ms) over prefill steps, geometry error =
slope * ((sum q)(sum kv) - sum q_i kv_i) / 1e6 as in simulate_geometry_prevalence.py.
For phased runs (workload shift): per-phase goodput, the per-phase hindsight best over the stock
arms, and regret(arm) = sum_phases (best - arm) / sum_phases best.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY = "ttft<=5s,tpot<=100ms"
STOCK = re.compile(r"^(b\d+|agentx|default|x[\d.]+-(default|agentx|b8192))$")


def geometry(trace: Path, slope: float):
    op = gzip.open if trace.suffix == ".gz" else open
    n = multi = over5 = 0
    with op(trace, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            ch, dp = r.get("ctx_chunks") or [], r.get("ctx_depths") or []
            if not ch:
                continue
            n += 1
            multi += len(ch) > 1
            geo = slope * (sum(ch) * sum(dp) - sum(q * k for q, k in zip(ch, dp))) / 1e6
            over5 += geo > 5
    return {"prefill_steps": n, "p_multi": multi / n if n else None, "p_geo_gt5ms": over5 / n if n else None}


def load_runs(dirs, slope):
    runs = defaultdict(list)
    for d in dirs:
        for jf in sorted(d.glob("*.json")):
            m = re.match(r"(.+)-r(\d+)$", jf.stem)
            if not m:
                continue
            rep = json.load(open(jf))
            if rep.get("result_type") != "trace_replay_serving":
                continue
            run = {"rep": int(m.group(2)), "dir": d.name, "summary": rep["summary"], "phases": rep.get("phases")}
            for cand in (d / "trace" / f"{jf.stem}.jsonl", d / "trace" / f"{jf.stem}.jsonl.gz"):
                if cand.exists():
                    run["geometry"] = geometry(cand, slope)
                    break
            runs[(d.name, m.group(1))].append(run)
    return runs


def mr(xs, f="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f.format(np.mean(xs)) + (f" ({f.format(min(xs))}–{f.format(max(xs))})" if len(xs) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, nargs="+", required=True)
    ap.add_argument("--slope", type=float, default=3.3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runs = load_runs(args.dir, args.slope)
    out = {"primary_slo": PRIMARY, "conditions": {}}
    print(f"| run set | condition | reps | offered req/s | goodput {PRIMARY} | TTFT p50 / p90 / p99 (s) | TPOT p50 / p99 (ms) | P(prefills>1) | P(geo>5 ms) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (d, cond), rs in sorted(runs.items()):
        g = lambda f: [f(r) for r in rs]
        row = {"reps": len(rs), "offered_rps": float(np.mean(g(lambda r: r["summary"]["offered_rps"]))),
               "goodput": {k: float(np.mean(g(lambda r, k=k: r["summary"]["goodput_rps"][k]))) for k in rs[0]["summary"]["goodput_rps"]},
               "goodput_primary_runs": g(lambda r: r["summary"]["goodput_rps"][PRIMARY]),
               "ttft_p50_p90_p99": [float(np.mean(g(lambda r, i=i: r["summary"]["ttft_p50_p90_p99"][i]))) for i in range(3)],
               "tpot_p50_p90_p99": [float(np.mean(g(lambda r, i=i: r["summary"]["tpot_p50_p90_p99"][i]))) for i in range(3)],
               "errors": sum(g(lambda r: r["summary"]["errors"])),
               "geometry": {k: float(np.mean([r["geometry"][k] for r in rs if r.get("geometry") and r["geometry"][k] is not None]))
                            for k in ("p_multi", "p_geo_gt5ms")} if any(r.get("geometry") for r in rs) else None}
        if rs[0]["phases"]:
            row["phases"] = {p["name"]: {"goodput_primary": float(np.mean([r["phases"][i]["summary"]["goodput_rps"][PRIMARY] for r in rs])),
                                         "offered_rps": p["summary"]["offered_rps"],
                                         "ttft_p90": float(np.mean([r["phases"][i]["summary"]["ttft_p50_p90_p99"][1] or 0 for r in rs]))}
                             for i, p in enumerate(rs[0]["phases"])}
        out["conditions"][f"{d}/{cond}"] = row
        geo = row["geometry"] or {}
        print(f"| {d} | {cond} | {len(rs)} | {row['offered_rps']:.2f} | {mr(row['goodput_primary_runs'])} | "
              + " / ".join(f"{x:.2f}" for x in row["ttft_p50_p90_p99"]) + " | "
              + f"{row['tpot_p50_p90_p99'][0] * 1000:.0f} / {row['tpot_p50_p90_p99'][2] * 1000:.0f} | "
              + (f"{geo.get('p_multi', float('nan')) * 100:.1f}% | {geo.get('p_geo_gt5ms', float('nan')) * 100:.1f}% |" if geo else "— | — |"))
    # workload shift: per-phase hindsight best over stock arms, regret
    for d in sorted({d for d, _ in runs}):
        conds = {c: v for k, v in out["conditions"].items() if k.startswith(d + "/") and v.get("phases") for c in [k.split("/", 1)[1]]}
        if not conds:
            continue
        phases = list(next(iter(conds.values()))["phases"])
        stock = [c for c in conds if STOCK.match(c)]
        best = {p: max(conds[c]["phases"][p]["goodput_primary"] for c in stock) for p in phases}
        best_arm = {p: max(stock, key=lambda c: conds[c]["phases"][p]["goodput_primary"]) for p in phases}
        tot = sum(best.values())
        print(f"\n## {d}: per-phase primary goodput (req/s); hindsight best over stock arms {stock}")
        print("| arm | " + " | ".join(phases) + " | regret |")
        print("| --- |" + " ---: |" * (len(phases) + 1))
        reg = {}
        for c in sorted(conds):
            vals = [conds[c]["phases"][p]["goodput_primary"] for p in phases]
            reg[c] = sum(best[p] - v for p, v in zip(phases, vals)) / tot if tot else None
            print(f"| {c} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {reg[c] * 100:.1f}% |")
        print("| *hindsight best* | " + " | ".join(f"{best[p]:.2f} ({best_arm[p]})" for p in phases) + " | 0 |")
        within5 = {c: all(conds[c]["phases"][p]["goodput_primary"] >= 0.95 * best[p] for p in phases) for c in stock}
        out.setdefault("shift", {})[d] = {"phases": phases, "hindsight_best": best, "hindsight_best_arm": best_arm, "regret": reg,
                                          "stock_within_5pct_every_phase": within5}
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
