#!/usr/bin/env python3
"""Registered analysis of the H100 capacity-planning campaign
(docs/preregistration/2026-10-02-capacity-planning.md).

Inputs: the live replay outputs (one replay_trace_serving.py JSON per (trace, config, rate) cell) and
the simulator predictions made before any live run (predictions/sim-{aggregate,meanfield,geometry}.json).

p99 TTFT is recomputed from the per-request TTFTs with np.quantile (linear), the definition the
simulator uses, over requests without errors.

Per (trace, config) and SLO (p99 TTFT <= 1 s, <= 2 s), the *capacity* of a source (live or a clock)
is the largest rate scale of the live grid whose p99 TTFT meets the SLO, provided every smaller grid
rate meets it too; 0 if the smallest grid rate already misses.

  python scripts/analyze_capacity_planning.py --live DIR --pred DIR --output verdicts.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

CLOCKS = ("aggregate", "meanfield", "geometry")
SLOS = (1.0, 2.0)
GRID = {"azure-code": (0.75, 1.0, 1.5, 2.0, 3.0), "mooncake-agent": (0.5, 0.6, 0.7, 0.8)}
CONFIGS = ("b2048", "b8192", "b4096t1024")


def load_json(p: Path):
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
        return json.load(f)


def live_p99(path: Path) -> float:
    reqs = [r for r in load_json(path)["requests"] if not r.get("error") and r.get("ttft_s") is not None]
    return float(np.quantile([r["ttft_s"] for r in reqs], 0.99))


def capacity(p99_by_rate: dict, slo: float, grid) -> float:
    cap = 0.0
    for r in grid:
        if p99_by_rate[r] <= slo:
            cap = r
        else:
            break
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=Path, required=True, help="dir with {trace}-{config}-x{rate}.json[.gz]")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    pred = {}
    for c in CLOCKS:
        for tr, v in load_json(args.pred / f"sim-{c}.json")["results"].items():
            for key, sm in v["runs"].items():
                cfg, rate = key.split("@")
                pred[(c, tr, cfg, float(rate[1:]))] = sm["ttft_s_p50_p99"][1]

    live, missing = {}, []
    for tr, grid in GRID.items():
        for cfg in CONFIGS:
            for r in grid:
                hits = sorted(args.live.glob(f"{tr}-{cfg}-x{r:g}.json*"))
                if hits:
                    live[(tr, cfg, r)] = live_p99(hits[0])
                else:
                    missing.append(f"{tr}-{cfg}-x{r:g}")

    out = {"missing_cells": missing, "cells": {}, "capacity": {}, "verdicts": {}}
    for (tr, cfg, r), lp in sorted(live.items()):
        out["cells"][f"{tr}-{cfg}-x{r:g}"] = {"live_p99_s": lp, **{
            f"{c}_p99_s": pred[(c, tr, cfg, r)] for c in CLOCKS}, **{
            f"{c}_rel_err": (pred[(c, tr, cfg, r)] - lp) / lp for c in CLOCKS}}

    decisions = []
    for tr, grid in GRID.items():
        for cfg in CONFIGS:
            if any((tr, cfg, r) not in live for r in grid):
                continue
            for slo in SLOS:
                caps = {"live": capacity({r: live[(tr, cfg, r)] for r in grid}, slo, grid)}
                for c in CLOCKS:
                    caps[c] = capacity({r: pred[(c, tr, cfg, r)] for r in grid}, slo, grid)
                out["capacity"][f"{tr}-{cfg}-slo{slo:g}s"] = caps
                decisions.append((tr, cfg, slo, caps))

    def med_abs_err(clock, trace):
        e = [abs(v[f"{clock}_rel_err"]) for k, v in out["cells"].items() if k.startswith(trace)]
        return float(np.median(e)) if e else None

    az = [v for k, v in out["cells"].items() if k.startswith("azure-code")]
    v = out["verdicts"]
    v["K1_median_abs_rel_err_azure"] = {c: med_abs_err(c, "azure-code") for c in CLOCKS}
    v["K1_aggregate_overpredicts_azure"] = f"{sum(1 for x in az if x['aggregate_rel_err'] > 0)}/{len(az)}"
    k1a = v["K1_median_abs_rel_err_azure"]
    v["K1"] = (k1a["geometry"] is not None and k1a["geometry"] < k1a["aggregate"]
               and sum(1 for x in az if x["aggregate_rel_err"] > 0) >= 9)
    match = {c: sum(1 for *_, caps in decisions if caps[c] == caps["live"]) for c in CLOCKS}
    v["K2_capacity_matches_live"] = {c: f"{match[c]}/{len(decisions)}" for c in CLOCKS}
    v["K2"] = match["geometry"] >= 8 and match["geometry"] > match["aggregate"]
    ratios = []
    for tr, cfg, slo, caps in decisions:
        if tr == "azure-code" and slo == 1.0:
            if caps["aggregate"] > 0:
                ratios.append(caps["live"] / caps["aggregate"])
            else:  # the aggregate clock says no grid rate meets the SLO
                ratios.append("inf" if caps["live"] > 0 else 1.0)
    v["K3_azure_live_over_aggregate_capacity_slo1s"] = ratios
    v["K3"] = sum(1 for x in ratios if x == "inf" or (x != "inf" and x >= 1.5)) >= 2
    moon = [caps for tr, cfg, slo, caps in decisions if tr == "mooncake-agent"]
    grid = GRID["mooncake-agent"]

    def idx(cap):
        return grid.index(cap) if cap in grid else -1  # 0 = below the grid

    # Control: on the agent trace the knee is sharp, so every clock and live agree within one grid step.
    v["K4"] = bool(moon) and all(max(idx(caps[s]) for s in ("live", *CLOCKS)) - min(idx(caps[s]) for s in ("live", *CLOCKS)) <= 1
                                 for caps in moon)
    for k in ("K1", "K2", "K3", "K4"):
        v[k] = bool(v[k])
    print(json.dumps(v, indent=1))
    for k, caps in out["capacity"].items():
        print(f"{k:34s} " + "  ".join(f"{s}={caps[s]:g}" for s in ("live", *CLOCKS)))
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
