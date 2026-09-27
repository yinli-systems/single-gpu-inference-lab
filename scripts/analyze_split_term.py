#!/usr/bin/env python3
"""Split the attention-work term into cross X = sum q_i k_i and self S = sum q_i (q_i + 1) / 2.

Implements the definitions frozen in docs/preregistration/2026-09-27-h100-hopper-replication.md:
- M2s = M2n with the single attention-work feature replaced by X/1e6 and S/1e6 (same Ridge,
  same filters, same primary/reverse splits as analyze_m2_variants.py);
- c_X, c_S: raw-unit coefficients (ms per million units) of M2s fitted on all filtered
  prefill-containing steps of a dataset (never on pairing-swap steps);
- single slope: mean of the six partition fits ms = a + b * W of analyze_partition_geometry.py;
- pairing-swap predictions c_X * dX and slope * dX against a measured pairswap.json.

X and S are recovered exactly from steps.csv: the tracer writes attn_proxy = sum q(k + q/2) +
sum of decode depths, so X = attn_proxy - sum q^2/2 - gen_kv_sum.

  python scripts/analyze_split_term.py --steps NAME=steps.csv[:pairswap.json] ... --output out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_m2_variants as V  # noqa: E402
import analyze_partition_geometry as P  # noqa: E402
import analyze_step_cost_v2 as A  # noqa: E402


def add_split(r):
    q = r["ctx_chunks"]
    r["attn_self"] = sum(c * (c + 1) / 2 for c in q)
    r["attn_cross"] = r["attn_proxy"] - sum(c * c / 2 for c in q) - r["gen_kv_sum"]
    return r


def m2n(r):
    return [x for i, x in enumerate(A.features(r, 2)) if i != 2]


def m2s(r):
    return m2n(r)[:-1] + [r["attn_cross"] / 1e6, r["attn_self"] / 1e6]


FEATURES = {"M0": lambda r: A.features(r, 0), "M2n": m2n, "M2s": m2s}


def ood(cells):
    out = {}
    for split, (tr, te) in V.splits(cells).items():
        ytr = np.array([r["cuda_ms"] for r in tr]); yte = np.array([r["cuda_ms"] for r in te])
        rep = {"n_train": len(tr), "n_test": len(te)}
        for name, f in FEATURES.items():
            res = yte - A.Ridge().fit(np.array([f(r) for r in tr]), ytr).predict(np.array([f(r) for r in te]))
            rep[name] = {"mae": float(np.mean(np.abs(res))), "signed_mean": float(res.mean())}
        out[split] = rep
    return out


def split_coefficients(cells):
    rows = [r for rs, _ in cells.values() for r in rs]
    X = np.array([m2s(r) for r in rows]); y = np.array([r["cuda_ms"] for r in rows])
    m = A.Ridge().fit(X, y)
    raw = m.w / m.sd
    c_x, c_s = float(raw[-2]), float(raw[-1])
    return {"n_steps": len(rows), "c_X_ms_per_M": c_x, "c_S_ms_per_M": c_s, "c_S_over_c_X": c_s / c_x}


def single_slope(steps_csv: Path):
    rep = P.analyze(P.load(steps_csv))
    slopes = [f["ms_per_M"] for b in rep.values() for f in b["fits"].values()]
    return {"partition_slopes_ms_per_M": slopes, "mean_slope_ms_per_M": float(np.mean(slopes)),
            "spread_vs_mean": [float(s / np.mean(slopes) - 1) for s in slopes]}


def pairing(pairswap: Path, c_x: float, slope: float):
    out = []
    for c in json.load(open(pairswap))["configs"]:
        if "delta_A_minus_B_ms" not in c:
            continue
        dx = c["dW_A_minus_B_M"]; meas = c["delta_A_minus_B_ms"]
        row = {"label": c["label"], "dX_M": dx, "measured_ms": meas, "n": [len(c["cuda_ms_A"]), len(c["cuda_ms_B"])]}
        if dx:
            row.update({"split_pred_ms": c_x * dx, "split_ratio": meas / (c_x * dx),
                        "single_pred_ms": slope * dx, "single_ratio": meas / (slope * dx)})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True, help="NAME=steps.csv or NAME=steps.csv:pairswap.json")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = {}
    for spec in args.steps:
        name, paths = spec.split("=", 1)
        steps_csv, _, ps = paths.partition(":")
        cells = V.load_cells(Path(steps_csv), not args.no_filter)
        for rs, _ in cells.values():
            for r in rs:
                add_split(r)
        rep = {"ood": ood(cells), "split": split_coefficients(cells), "single": single_slope(Path(steps_csv))}
        if ps:
            rep["pairing"] = pairing(Path(ps), rep["split"]["c_X_ms_per_M"], rep["single"]["mean_slope_ms_per_M"])
        report[name] = rep
        s, g = rep["split"], rep["single"]
        print(f"## {name}: c_X {s['c_X_ms_per_M']:.3f}  c_S {s['c_S_ms_per_M']:.3f}  c_S/c_X {s['c_S_over_c_X']:.2f}  "
              f"single slope {g['mean_slope_ms_per_M']:.3f} (spread {min(g['spread_vs_mean']):+.1%}..{max(g['spread_vs_mean']):+.1%})")
        for split, r in rep["ood"].items():
            print(f"   {split}: " + "  ".join(f"{m} {r[m]['mae']:.2f}" for m in FEATURES))
        for p in rep.get("pairing", []):
            extra = (f"split {p['split_pred_ms']:+6.2f} ({p['split_ratio']:.2f})  single {p['single_pred_ms']:+6.2f} ({p['single_ratio']:.2f})"
                     if "split_pred_ms" in p else "control")
            print(f"   {p['label']:34s} measured {p['measured_ms']:+6.2f}  {extra}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
