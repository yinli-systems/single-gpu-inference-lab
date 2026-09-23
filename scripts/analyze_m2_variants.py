#!/usr/bin/env python3
"""M0 vs M2 vs M2n (M2 without the aggregate prefill-KV sum) on the geometry splits, from steps.csv.

M2n is the variant pre-registered in docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md:
`analyze_step_cost_v2.features(r, 2)` with index 2 (ctx_kv_sum / 1e4) removed; same Ridge, same
split definitions as analyze_step_cost_v2.py. For each split it reports MAE, P95 |error| and the
mean signed error (actual - predicted; negative = over-prediction), the signed error by number of
concurrent prefills, and, as a diagnostic of model form vs extrapolation, the same models fitted on
train + test together and scored on test.

Filters: first engine iteration of each server and steps > 10x their cell's median prefill step
are dropped (the `--exclude-first-iteration --exclude-over-median-x 10` setting); `--no-filter`
reproduces the unfiltered published numbers.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_step_cost_v2 as A  # noqa: E402

INT = ("i", "ctx_reqs", "ctx_tokens", "ctx_kv_sum", "ctx_kv_max", "gen_reqs", "gen_tokens", "gen_kv_sum", "gen_kv_max",
       "total_tokens", "padded_tokens")
FLOAT = ("cuda_ms", "attn_proxy", "ctx_kv_var", "gen_kv_var")


def load_cells(path: Path, filtered: bool):
    cells = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["cuda_ms"] in ("", "nan"):
            continue
        d = dict(r)
        for k in INT:
            d[k] = int(float(r[k])) if r[k] not in ("", "None") else 0
        for k in FLOAT:
            d[k] = float(r[k]) if r[k] not in ("", "None") else 0.0
        d["ctx_chunks"] = [int(x) for x in r["ctx_chunks"].split()] if r["ctx_chunks"] else []
        if d["ctx_tokens"] > 0:
            cells[d["cell"]].append(d)
    out = {}
    for c, rs in cells.items():
        if filtered:
            rs = [r for r in rs if r["i"] != 0]
            med = np.median([r["cuda_ms"] for r in rs])
            rs = [r for r in rs if r["cuda_ms"] <= 10 * med]
        out[c] = (rs, A.cell_meta(c))
    return out


def splits(cells):
    multi_cell = lambda m: m["kind"] in ("part", "part2") and m["n_prefill"] > 1
    one_cell = lambda m: m["kind"] in ("part", "part2") and m["n_prefill"] == 1
    pick = lambda cf, rf: [r for k, (rs, m) in cells.items() if cf(m) for r in rs if rf(r)]
    return {
        "primary_geometry_one_to_multi": (pick(lambda m: not multi_cell(m) and m["kind"] != "graph", lambda r: r["ctx_reqs"] == 1),
                                          pick(multi_cell, lambda r: r["ctx_reqs"] >= 2)),
        "reverse_geometry_multi_to_one": (pick(multi_cell, lambda r: r["ctx_reqs"] >= 2), pick(one_cell, lambda r: r["ctx_reqs"] == 1)),
    }


FEATURES = {
    "M0": lambda r: A.features(r, 0),
    "M2": lambda r: A.features(r, 2),
    "M2n": lambda r: [x for i, x in enumerate(A.features(r, 2)) if i != 2],
}


def evaluate(tr, te):
    ytr = np.array([r["cuda_ms"] for r in tr]); yte = np.array([r["cuda_ms"] for r in te])
    rep = {}
    for name, f in FEATURES.items():
        Xtr = np.array([f(r) for r in tr]); Xte = np.array([f(r) for r in te])
        res = yte - A.Ridge().fit(Xtr, ytr).predict(Xte)
        ins = yte - A.Ridge().fit(np.r_[Xtr, Xte], np.r_[ytr, yte]).predict(Xte)
        by_n = collections.defaultdict(list)
        for r, e in zip(te, res):
            by_n[r["ctx_reqs"]].append(e)
        rep[name] = {"mae": float(np.mean(np.abs(res))), "p95_abs": float(np.quantile(np.abs(res), .95)), "signed_mean": float(res.mean()),
                     "signed_mean_by_prefill_count": {int(k): [float(np.mean(v)), len(v)] for k, v in sorted(by_n.items())},
                     "in_sample_mae": float(np.mean(np.abs(ins))), "in_sample_p95_abs": float(np.quantile(np.abs(ins), .95))}
    return {"n_train": len(tr), "n_test": len(te), "models": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True, help="NAME=path/to/steps.csv")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    out = {}
    for spec in args.steps:
        name, path = spec.split("=", 1)
        cells = load_cells(Path(path), not args.no_filter)
        out[name] = {s: evaluate(tr, te) for s, (tr, te) in splits(cells).items()}
        print(f"## {name}")
        for s, rep in out[name].items():
            print(f"  {s} (train {rep['n_train']}, test {rep['n_test']})")
            for m, v in rep["models"].items():
                byn = ", ".join(f"{k}:{e:+.1f}" for k, (e, n) in v["signed_mean_by_prefill_count"].items() if n >= 10)
                print(f"    {m:4s} MAE {v['mae']:6.2f}  P95 {v['p95_abs']:6.2f}  signed {v['signed_mean']:+6.2f}  "
                      f"[by prefill count {byn}]  in-sample MAE {v['in_sample_mae']:.2f}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
