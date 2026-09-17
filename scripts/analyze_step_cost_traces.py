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
    # tokens-only competitor: no KV-read term (columns 0,1,3,5 = 1, gen_reqs, ctx_tokens, ctx_reqs)
    TOK = [0, 1, 3, 5]
    w_tok, *_ = np.linalg.lstsq(Xc[:, TOK], yc, rcond=None)

    # Per-cell: prefill-step cost by KV-depth bucket, and held-out error of both models
    print("\n## Prefill-step cost by KV-read depth (steps containing a prefill chunk)\n")
    print("| cell | KV bucket | steps | chunk tokens (median) | actual step p50 | p95 | tokens-only pred p50 | +KV pred p50 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    buckets = [(0, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 1 << 30)]
    kv_rows = []
    for k, (X, y, meta, cell) in cells.items():
        pre = meta[:, 0] > 0
        for lo, hi in buckets:
            sel = pre & (meta[:, 1] >= lo) & (meta[:, 1] < hi)
            if sel.sum() < 5:
                continue
            p_tok = X[sel][:, TOK] @ w_tok
            p_kv = X[sel] @ w
            kv_rows.append({"cell": k, "kv_lo": lo, "kv_hi": min(hi, 1 << 20), "steps": int(sel.sum()),
                            "chunk_median": float(np.median(meta[sel, 0])), "actual_p50": pct(y[sel], .5), "actual_p95": pct(y[sel], .95),
                            "tokens_only_pred_p50": pct(p_tok, .5), "kv_pred_p50": pct(p_kv, .5)})
            print(f"| {k} | {lo//1024}k–{min(hi,1<<20)//1024}k | {int(sel.sum())} | {np.median(meta[sel,0]):.0f} | {pct(y[sel],.5):.1f} | {pct(y[sel],.95):.1f} | {pct(p_tok,.5):.1f} | {pct(p_kv,.5):.1f} |")
    report_kv = kv_rows
    # residual quantile learned on calibration, used by the risk-calibrated rule
    q_alpha = pct(resid_c, 1 - args.alpha)

    deadlines = [float(d) for d in args.deadlines_ms.split(",")]
    report = {"weights": w.tolist(), "weights_tokens_only": w_tok.tolist(), "kv_depth_table": report_kv,
              "calibration_cells": [k for k in cells if is_calib[k]],
              "calibration_residual": {"p50": pct(resid_c, .5), "p90": pct(resid_c, .9), "p95": pct(resid_c, .95), "p99": pct(resid_c, .99), "n": int(len(yc))},
              "alpha": args.alpha, "q_alpha_ms": q_alpha, "cells": {}}
    print(f"fit on {len(yc)} steps; weights {np.round(w, 3).tolist()}; calib residual p50/p95/p99 = "
          f"{pct(resid_c,.5):.2f}/{pct(resid_c,.95):.2f}/{pct(resid_c,.99):.2f} ms; q_{1-args.alpha:.2f} = {q_alpha:.2f} ms")
    print("\n## Residuals on prefill steps (fit on calibration cells) and deadline miss rates\n")
    print("| cell | calib? | prefill steps | actual p50 | tokens-only resid p50 / p95 | +KV resid p50 / p95 | " + " | ".join(f"miss@{int(d)}ms point / risk" for d in deadlines) + " |")
    print("| --- | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in deadlines) + " |")
    for k, (X, y, meta, cell) in cells.items():
        pred = X @ w
        resid = y - pred
        pre = meta[:, 0] > 0  # steps that include prefill work
        resid_tok = (y - X[:, TOK] @ w_tok)[pre]
        resid_kv = resid[pre]
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
        row["residual_prefill_steps"] = {"tokens_only": {"p50": pct(resid_tok, .5), "p95": pct(resid_tok, .95)}, "with_kv": {"p50": pct(resid_kv, .5), "p95": pct(resid_kv, .95)}}
        print(f"| {k} | {'yes' if is_calib[k] else 'no'} | {int(pre.sum())} | {pct(y[pre],.5):.1f} | {pct(resid_tok,.5):+.1f} / {pct(resid_tok,.95):+.1f} | {pct(resid_kv,.5):+.1f} / {pct(resid_kv,.95):+.1f} | " + " | ".join(cols) + " |")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
