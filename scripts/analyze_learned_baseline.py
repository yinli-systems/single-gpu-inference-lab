#!/usr/bin/env python3
"""LPRS-style learned baseline vs the physical model (pre-registration addendum 8 B).

MLP (3 layers: 2 x 64 ReLU hidden, scikit-learn, early stopping) on 16 aggregate / marginal
features. It sees everything an aggregate scheduler state holds (and marginal statistics of the
chunks and depths) but not the per-request q-KV pairing. Compared with M0 and M2n (Ridge, as in
analyze_step_cost_v2.py) on the same splits and filters (analyze_m2_variants.load_cells).

  LB1  primary / reverse split MAE, MLP averaged over 5 seeds
  LB2  sample efficiency: N one-prefill training steps (5 draws), test on the multi-prefill cells
  LB3  geometry exposure: a fraction f of the training set replaced by multi-prefill steps from
       the 2x512 / 4x512 cells; test on the 4x256 / 8x256 cells only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_step_cost_v2 as A  # noqa: E402
from analyze_m2_variants import FEATURES, load_cells, splits  # noqa: E402


def mlp_feats(r):
    ch = r.get("ctx_chunks") or [r["ctx_tokens"]]
    g = max(r["gen_reqs"], 1); c = max(r["ctx_reqs"], 1)
    return [r["gen_reqs"], r["gen_kv_sum"] / 1e4, r["gen_kv_max"] / 1e4, r["gen_kv_sum"] / g / 1e4,
            r["ctx_reqs"], r["ctx_tokens"] / 1e3, max(ch) / 1e3, r["ctx_kv_sum"] / 1e4, r["ctx_kv_max"] / 1e4,
            r["ctx_kv_sum"] / c / 1e4, r["ctx_tokens"] * r["ctx_kv_sum"] / 1e7, r["total_tokens"] / 1e3,
            r["padded_tokens"] / 1e3, 1.0 if r["cg_mode"] == "NONE" else 0.0, sum(q * q for q in ch) / 1e6,
            r.get("gen_kv_var", 0.0) / 1e8]


def fit_predict(kind, tr, te, seed=0):
    ytr = np.array([r["cuda_ms"] for r in tr])
    if kind == "MLP":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", early_stopping=True,
                                                         max_iter=3000, random_state=seed, validation_fraction=0.15))
        m.fit(np.array([mlp_feats(r) for r in tr]), ytr)
        return m.predict(np.array([mlp_feats(r) for r in te]))
    f = FEATURES[kind]
    return A.Ridge().fit(np.array([f(r) for r in tr]), ytr).predict(np.array([f(r) for r in te]))


def mae(te, p):
    return float(np.mean(np.abs(np.array([r["cuda_ms"] for r in te]) - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True, help="NAME=path/to/steps.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for spec in args.steps:
        name, path = spec.split("=", 1)
        cells = load_cells(Path(path), True)
        sp = splits(cells)
        rep = {"LB1": {}, "LB2": {}, "LB3": {}}
        for s, (tr, te) in sp.items():
            rep["LB1"][s] = {"M0": mae(te, fit_predict("M0", tr, te)), "M2n": mae(te, fit_predict("M2n", tr, te)),
                             "MLP": float(np.mean([mae(te, fit_predict("MLP", tr, te, k)) for k in range(args.seeds)]))}
        tr, te = sp["primary_geometry_one_to_multi"]
        rng = np.random.default_rng(0)
        for n in (32, 64, 128, 256, 512, 1024, 2048):
            if n > len(tr):
                continue
            res = {"M0": [], "M2n": [], "MLP": []}
            for k in range(args.seeds):
                sub = [tr[i] for i in rng.choice(len(tr), n, replace=False)]
                for kind in res:
                    res[kind].append(mae(te, fit_predict(kind, sub, te, k)))
            rep["LB2"][n] = {kind: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for kind, v in res.items()}
        expo = [r for c, (rs, m) in cells.items() if c.startswith(("part-2x512", "part2-4x512")) for r in rs if r["ctx_reqs"] >= 2]
        test3 = [r for c, (rs, m) in cells.items() if c.startswith(("part-4x256", "part2-8x256")) for r in rs if r["ctx_reqs"] >= 2]
        for f in (0.0, 0.01, 0.05, 0.10, 0.25):
            res = {"M0": [], "M2n": [], "MLP": []}
            for k in range(args.seeds):
                n_mix = int(round(f * len(tr)))
                mix = [expo[i] for i in rng.choice(len(expo), min(n_mix, len(expo)), replace=False)] if n_mix else []
                base = [tr[i] for i in rng.choice(len(tr), len(tr) - len(mix), replace=False)]
                for kind in res:
                    res[kind].append(mae(test3, fit_predict(kind, base + mix, test3, k)))
            rep["LB3"][f] = {kind: float(np.mean(v)) for kind, v in res.items()}
        out[name] = rep
        print(f"## {name}")
        for s, v in rep["LB1"].items():
            print(f"  LB1 {s:32s} M0 {v['M0']:7.1f}  MLP {v['MLP']:7.1f}  M2n {v['M2n']:5.2f}")
        print("  LB2 N:   " + "  ".join(f"{n}: M0 {v['M0'][0]:.0f} / MLP {v['MLP'][0]:.1f} / M2n {v['M2n'][0]:.2f}" for n, v in rep["LB2"].items()))
        print("  LB3 f:   " + "  ".join(f"{f:.0%}: M0 {v['M0']:.1f} / MLP {v['MLP']:.1f} / M2n {v['M2n']:.2f}" for f, v in rep["LB3"].items()), flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
