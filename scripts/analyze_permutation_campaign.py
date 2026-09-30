#!/usr/bin/env python3
"""Registered analysis of the permutation campaign (docs/preregistration/2026-09-28-permutation-campaign.md).

Input: the output of `measure_pairing_permutations.py --analyze` (per config and layout seed: every
state's pairing, coupling C and per-trial CUDA times with their block index).

Per (config, layout seed) cell:
- P1  Spearman rho between C and the median step time over the distinct pairings (all states except
      order-null); n >= 4 cells only.
- P2  T(same) - T(opposite): Hodges-Lehmann estimate of the within-block paired difference and a 95%
      bootstrap CI over blocks.
- P3  order-null vs same: 90% bootstrap CI of the paired median difference inside [-eps, +eps] (TOST
      at alpha 0.05), eps = max(0.5 ms, 2% of median T(same)).
- P5  eqC-a vs eqC-b: the same equivalence test (n4, n8, n16).
- P6  measured / predicted range, predicted = slope x (C_max - C_min), inside the registered band. The
      slope is the GPU's c_X (M2n cross-attention coefficient, split-term.json): 3.23 ms/M on the A100,
      band 0.6-1.0 (the defaults); 6.33 ms/M on the L20, band 0.8-1.15 (addendum 1).
- floor (T_max - T_min) / 2 over the state medians: the minimax floor of any marginal-only predictor
  on this collision class.
Across layout seeds of one config:
- P4  for every state, 90% CI of the difference of per-seed medians (bootstrap over trials) inside
      [-eps, +eps], and Spearman rho of the state medians across seeds.

  python scripts/analyze_permutation_campaign.py --input perm.json --output verdicts.json [--boot 10000]
      [--slope 3.23 --p6-band 0.6 1.0]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np



def hodges_lehmann(d):
    d = np.asarray(d)
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def paired(cell, a, b):
    """Within-block differences T(a) - T(b) over blocks where both were captured."""
    A = dict(zip(cell["states"][a]["blocks"], cell["states"][a]["cuda_ms"]))
    B = dict(zip(cell["states"][b]["blocks"], cell["states"][b]["cuda_ms"]))
    return np.array([A[k] - B[k] for k in sorted(set(A) & set(B))])


def boot_ci(d, rnd, boot, level):
    stats = [np.median(rnd.choice(d, size=len(d), replace=True)) for _ in range(boot)]
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--slope", type=float, default=3.23, help="c_X in ms/M for P6")
    ap.add_argument("--p6-band", type=float, nargs=2, default=(0.6, 1.0), metavar=("LO", "HI"))
    args = ap.parse_args()
    data = json.load(open(args.input))
    rnd = np.random.default_rng(0)
    out = {"cells": {}, "layout": {}, "p6_slope_ms_per_M": args.slope, "p6_band": list(args.p6_band)}
    for key, c in sorted(data.items()):
        st = c["states"]
        med = {k: v["median_ms"] for k, v in st.items() if v["median_ms"] is not None}
        eps = max(0.5, 0.02 * med["same"])
        res = {"config": c["config"], "layout_seed": c["layout_seed"], "eps_ms": eps, "trials": c.get("trials"),
               "missing": c.get("missing"), "medians": med}
        distinct = [k for k in med if k != "order-null"]
        if len(distinct) >= 3:
            res["P1_rho"] = spearman([st[k]["C_M"] for k in distinct], [med[k] for k in distinct])
            res["P1"] = res["P1_rho"] >= 0.8
        d = paired(c, "same", "opposite")
        res["P2_hl_ms"] = hodges_lehmann(d)
        res["P2_ci95"] = boot_ci(d, rnd, args.boot, 0.95)
        res["P2"] = res["P2_ci95"][0] > 0
        d = paired(c, "order-null", "same")
        res["P3_ci90"] = boot_ci(d, rnd, args.boot, 0.90)
        res["P3"] = -eps < res["P3_ci90"][0] and res["P3_ci90"][1] < eps
        if "eqC-a" in st:
            d = paired(c, "eqC-a", "eqC-b")
            res["P5_hl_ms"] = hodges_lehmann(d)
            res["P5_ci90"] = boot_ci(d, rnd, args.boot, 0.90)
            res["P5"] = -eps < res["P5_ci90"][0] and res["P5_ci90"][1] < eps
        rng_meas = med["same"] - med["opposite"]
        res["P6_measured_ms"] = rng_meas
        res["P6_predicted_ms"] = args.slope * (c["C_max_M"] - c["C_min_M"])
        res["P6_ratio"] = rng_meas / res["P6_predicted_ms"]
        res["P6"] = args.p6_band[0] <= res["P6_ratio"] <= args.p6_band[1]
        res["floor_ms"] = (max(med[k] for k in distinct) - min(med[k] for k in distinct)) / 2
        out["cells"][key] = res
        print(f"{key:16s} rho {res.get('P1_rho', float('nan')):+.2f} | same-opp HL {res['P2_hl_ms']:+6.2f} ms CI95 "
              f"[{res['P2_ci95'][0]:+.2f}, {res['P2_ci95'][1]:+.2f}] | order-null CI90 [{res['P3_ci90'][0]:+.2f}, {res['P3_ci90'][1]:+.2f}] (eps {eps:.2f})"
              + (f" | eqC CI90 [{res['P5_ci90'][0]:+.2f}, {res['P5_ci90'][1]:+.2f}]" if "P5_ci90" in res else "")
              + f" | range meas/pred {res['P6_ratio']:.2f} | floor {res['floor_ms']:.2f} ms")
    by_cfg = {}
    for key, c in data.items():
        by_cfg.setdefault(c["config"], []).append(c)
    for cfg, cs in sorted(by_cfg.items()):
        if len(cs) < 2:
            continue
        a, b = sorted(cs, key=lambda c: c["layout_seed"])[:2]
        states = [k for k in a["states"] if a["states"][k]["cuda_ms"] and b["states"][k]["cuda_ms"]]
        eps = max(0.5, 0.02 * a["states"]["same"]["median_ms"])
        ok, cis = True, {}
        for k in states:
            xa, xb = np.array(a["states"][k]["cuda_ms"]), np.array(b["states"][k]["cuda_ms"])
            diffs = [np.median(rnd.choice(xa, len(xa))) - np.median(rnd.choice(xb, len(xb))) for _ in range(args.boot // 5)]
            lo, hi = np.quantile(diffs, [0.05, 0.95])
            cis[k] = [float(lo), float(hi)]
            ok &= -eps < lo and hi < eps
        rho = spearman([a["states"][k]["median_ms"] for k in states], [b["states"][k]["median_ms"] for k in states])
        out["layout"][cfg] = {"eps_ms": eps, "ci90_by_state": cis, "P4_equivalent": bool(ok), "P4_rho": rho, "P4": bool(ok) and rho >= 0.9}
        print(f"layout {cfg:5s} all states within +-{eps:.2f} ms: {ok}; rho of state medians across seeds {rho:+.2f}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
