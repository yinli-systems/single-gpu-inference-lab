#!/usr/bin/env python3
"""Phase 0 of risk-calibrated prefill scheduling: is a point step-time predictor enough?

Reads per-iteration traces (``VLLM_EXP_ITER_TRACE`` JSONL: cost coordinates
plus the engine's iteration timer), derives the decode-facing step time as the
gap between consecutive result-ready instants (async scheduling overlaps
execution with scheduling, so the raw ``ms`` field is not the step cost), fits
a deliberately simple predictor on *calibration* cells, and asks two questions
on *shifted* cells:

  1. How large and how stable are the prediction residuals (p50/p90/p95/p99)?
  2. If a scheduler trusted the point prediction against a deadline, how often
     would the real step miss it (false-safe rate), versus a controller that
     adds the calibration residual quantile (risk-calibrated)?

Predictor: least squares on features that a scheduler knows before the step:
  [1, gen_reqs, gen_kv_sum/1e4, ctx_tokens, ctx_tokens*ctx_kv_max/1e6, ctx_reqs]
Only steps with a prefill chunk are used for the deadline question (decode-only
steps are what the deadline protects; the prefill step is what threatens it).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np


def pct(a, q):
    return float(np.quantile(a, q)) if len(a) else float("nan")


def load_trace(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # result-ready instants; step time = gap to the previous result
    ready = np.array([r["t"] + r["ms"] / 1e3 for r in rows])
    step_ms = np.diff(ready) * 1e3
    feats = []
    for r in rows[1:]:
        feats.append(
            [
                1.0,
                r["gen_reqs"],
                r["gen_kv_sum"] / 1e4,
                r["ctx_tokens"],
                r["ctx_tokens"] * r["ctx_kv_max"] / 1e6,
                r["ctx_reqs"],
            ]
        )
    X = np.array(feats)
    meta = np.array([[r["ctx_tokens"], r["ctx_kv_max"], r["gen_reqs"], r["total_tokens"]] for r in rows[1:]])
    keep = (step_ms > 0) & (step_ms < 5000) & (meta[:, 3] > 0)
    return X[keep], step_ms[keep], meta[keep]


def parse_cell(name: str):
    m = re.match(r"bg(\d+)-L(\d+)-chunk(\d+)", name)
    return {"bg": int(m.group(1)), "L": int(m.group(2)), "chunk": int(m.group(3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--calib", default="bg8-L4096,bg8-L8192", help="cells (bg-L prefixes) used to fit")
    ap.add_argument("--deadlines-ms", default="25,50,100")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cells = {}
    for f in sorted(args.trace_dir.glob("*.jsonl")):
        X, y, meta = load_trace(f)
        cells[f.stem] = (X, y, meta, parse_cell(f.stem))
    calib_prefixes = args.calib.split(",")
    is_calib = {k: any(k.startswith(p + "-") for p in calib_prefixes) for k in cells}

    Xc = np.vstack([cells[k][0] for k in cells if is_calib[k]])
    yc = np.concatenate([cells[k][1] for k in cells if is_calib[k]])
    w, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    resid_c = yc - Xc @ w
    # residual quantile learned on calibration, used by the risk-calibrated rule
    q_alpha = pct(resid_c, 1 - args.alpha)

    deadlines = [float(d) for d in args.deadlines_ms.split(",")]
    report = {"weights": w.tolist(), "calibration_cells": [k for k in cells if is_calib[k]],
              "calibration_residual": {"p50": pct(resid_c, .5), "p90": pct(resid_c, .9), "p95": pct(resid_c, .95), "p99": pct(resid_c, .99), "n": int(len(yc))},
              "alpha": args.alpha, "q_alpha_ms": q_alpha, "cells": {}}
    print(f"fit on {len(yc)} steps; weights {np.round(w, 3).tolist()}; calib residual p50/p95/p99 = "
          f"{pct(resid_c,.5):.2f}/{pct(resid_c,.95):.2f}/{pct(resid_c,.99):.2f} ms; q_{1-args.alpha:.2f} = {q_alpha:.2f} ms")
    print("| cell | calib? | steps | prefill steps | actual p50 | resid p50 | resid p95 | resid p99 | " + " | ".join(f"miss@{int(d)}ms point / risk" for d in deadlines) + " |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in deadlines) + " |")
    for k, (X, y, meta, cell) in cells.items():
        pred = X @ w
        resid = y - pred
        pre = meta[:, 0] > 0  # steps that include prefill work
        row = {**cell, "calibration": is_calib[k], "steps": int(len(y)), "prefill_steps": int(pre.sum()),
               "actual_p50_ms": pct(y, .5), "actual_p99_ms": pct(y, .99),
               "residual": {"p50": pct(resid, .5), "p90": pct(resid, .9), "p95": pct(resid, .95), "p99": pct(resid, .99)},
               "deadline": {}}
        cols = []
        for d in deadlines:
            yp, pp = y[pre], pred[pre]
            safe_point = pp <= d
            safe_risk = (pp + q_alpha) <= d
            miss_point = float(np.mean(yp[safe_point] > d)) if safe_point.any() else float("nan")
            miss_risk = float(np.mean(yp[safe_risk] > d)) if safe_risk.any() else float("nan")
            row["deadline"][str(int(d))] = {"point_safe_frac": float(safe_point.mean()) if len(yp) else float("nan"),
                                          "point_miss_rate": miss_point,
                                          "risk_safe_frac": float(safe_risk.mean()) if len(yp) else float("nan"),
                                          "risk_miss_rate": miss_risk}
            cols.append(f"{miss_point*100:5.1f}% / {miss_risk*100:5.1f}%" if len(yp) else "—")
        report["cells"][k] = row
        print(f"| {k} | {'yes' if is_calib[k] else 'no'} | {len(y)} | {int(pre.sum())} | {pct(y,.5):.1f} | {pct(resid,.5):+.1f} | {pct(resid,.95):+.1f} | {pct(resid,.99):+.1f} | " + " | ".join(cols) + " |")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
