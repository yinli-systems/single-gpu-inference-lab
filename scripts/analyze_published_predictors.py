#!/usr/bin/env python3
"""Two published step-cost predictors against the pairing swap and the geometry splits.

Vidur (microsoft/vidur @ 8383d29, vidur/execution_time_predictor/sklearn_execution_time_predictor.py
lines 852-870): the prefill-attention time of a batch is looked up at
(sum of kv_cache_size, round(sqrt(sum of prefill_chunk_size^2))^2). Both states of a pairing swap
map to the same key, so Vidur predicts A = B; its error on the pair is at least |delta|/2.

LLMVisor (arXiv 2608.08382, Eq. for T_B): T = beta + a1 sum p_i + a2 sum c_i + a3 sum p_i^2 + a4 |B|,
with p_i the tokens processed for request i in the step and c_i its context (KV) tokens. Also
invariant under the swap. Here it is fitted by least squares on the same primary/reverse geometry
splits and filters as analyze_m2_variants.py and compared with M2n.

  python scripts/analyze_published_predictors.py --steps NAME=steps.csv ... --pairswap NAME=pairswap.json ... --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_m2_variants as V  # noqa: E402
import analyze_step_cost_v2 as A  # noqa: E402


def vidur_key(batch):
    return (sum(k for _, k in batch), round(sum(q * q for q, _ in batch) ** 0.5) ** 2)


def llmvisor(r):
    q = r["ctx_chunks"]; g = r["gen_reqs"]
    return [1.0, (sum(q) + g) / 1e3, (r["ctx_kv_sum"] + r["gen_kv_sum"]) / 1e4, (sum(c * c for c in q) + g) / 1e6, len(q) + g]


def m2n(r):
    return [x for i, x in enumerate(A.features(r, 2)) if i != 2]


def ols_mae(tr, te, f):
    Xtr = np.array([f(r) for r in tr]); ytr = np.array([r["cuda_ms"] for r in tr])
    Xte = np.array([f(r) for r in te]); yte = np.array([r["cuda_ms"] for r in te])
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    res = yte - Xte @ w
    return float(np.mean(np.abs(res))), float(res.mean())


def ridge_mae(tr, te, f):
    ytr = np.array([r["cuda_ms"] for r in tr]); yte = np.array([r["cuda_ms"] for r in te])
    res = yte - A.Ridge().fit(np.array([f(r) for r in tr]), ytr).predict(np.array([f(r) for r in te]))
    return float(np.mean(np.abs(res))), float(res.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=[])
    ap.add_argument("--pairswap", nargs="*", default=[])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    out = {"splits": {}, "pairing": {}}
    for spec in args.steps:
        name, path = spec.split("=", 1)
        cells = V.load_cells(Path(path), True)
        out["splits"][name] = {}
        for split, (tr, te) in V.splits(cells).items():
            lv = ols_mae(tr, te, llmvisor); mn = ridge_mae(tr, te, m2n)
            out["splits"][name][split] = {"LLMVisor_mae": lv[0], "LLMVisor_signed": lv[1], "M2n_mae": mn[0], "M2n_signed": mn[1],
                                          "ratio": lv[0] / mn[0]}
            print(f"{name:24s} {split:32s} LLMVisor {lv[0]:7.2f} ({lv[1]:+7.2f})  M2n {mn[0]:5.2f}  ratio {lv[0] / mn[0]:6.1f}x")
    for spec in args.pairswap:
        name, path = spec.split("=", 1)
        rows = []
        for c in json.load(open(path))["configs"]:
            ka, kb = c["k"]; qa, qb = c["q_state_A"]
            A_, B_ = [(qa, ka), (qb, kb)], [(qb, ka), (qa, kb)]
            same = vidur_key(A_) == vidur_key(B_)
            d = c.get("delta_A_minus_B_ms")
            rows.append({"label": c["label"], "vidur_key_A": vidur_key(A_), "vidur_key_B": vidur_key(B_), "vidur_invariant": same,
                         "delta_ms": d, "floor_ms": abs(d) / 2 if d is not None else None})
            print(f"{name:24s} {c['label']:34s} Vidur key A {vidur_key(A_)} B {vidur_key(B_)} same={same}  floor {abs(d) / 2 if d is not None else float('nan'):.2f} ms")
        out["pairing"][name] = rows
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
