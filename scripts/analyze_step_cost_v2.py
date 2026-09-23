#!/usr/bin/env python3
"""Aggregate (M0) vs geometry (M1) vs execution (M2) step-cost predictors under
explicit distribution shift, on tracer-v2 traces (engine iteration geometry
joined by sequence to runner CUDA timing; see step_trace_join.py).

Target: direct CUDA elapsed time of the model step (ms), prefill-containing
steps only (those are the steps a deadline-aware prefill scheduler has to
price; decode-only steps are what the deadline protects).

Models (ridge on standardized features, lambda=1e-2, deliberately simple):
  M0  strongest *aggregate* coordinate: decode batch, aggregate decode KV,
      aggregate prefill KV-read depth, prefill tokens, prefill tokens x
      aggregate prefill KV.  Plus M0-lookup: conservative P99 table over
      (decode batch, prefill tokens/256, aggregate KV/4k) bins.
  M1  M0 + request/KV geometry: prefill request count, prefill KV max and
      variance, largest query chunk, prefill tokens x prefill KV max, decode
      KV max/variance.
  M1+exec  M1 + eager flag, padded tokens, attention-work proxy (all features;
      on one-prefill training data the proxy and the aggregate interaction are
      collinear, so this fit does not extrapolate across geometry either)
  M2  physics-structured geometry model: aggregates (decode batch, decode KV,
      prefill KV, prefill tokens) + prefill count + execution (eager flag,
      padded tokens) + attention-work proxy sum_i q_i*kv_i + q_i(q_i+1)/2
      *in place of* tokens x aggregate KV. Geometry statistics that duplicate
      an aggregate on one-prefill data (KV max == KV sum) are deliberately
      left out: a fit cannot tell them apart there and extrapolates wrongly.

Splits (train -> test; never random):
  primary geometry-OOD / aggregate-in-support: every one-prefill step from the
          ctx/load/part-1x/skew cells -> multi-prefill steps of the partition
          cells (2x512, 4x256, 4x512, 8x256).  Test rows whose M0 coordinates
          all lie inside the training range are reported separately.
  reverse multi-prefill partition cells -> one-prefill partition cell (the
          direction in which the aggregate coordinate *under*prices)
  ctx     cells with L <= 16k -> L = 32k        (secondary)
  batch   cells with decode batch <= 16 -> 32   (secondary)
Metrics: MAE, median/P95 |err|, underprediction rate, P50/P95/P99 positive
residual (actual - predicted), and the miss rate of a nominal alpha=5%
deadline rule (predict + train residual q95 <= deadline) at 50/100/200 ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from step_trace_join import load_joined

LAMBDA = 1e-2


def load_cell(iter_path: Path):
    rows, info = load_joined(iter_path)
    return [r for r in rows if r["total_tokens"] > 0], info


def m0(r):
    return [r["gen_reqs"], r["gen_kv_sum"] / 1e4, r["ctx_kv_sum"] / 1e4, r["ctx_tokens"] / 1e3,
            r["ctx_tokens"] * r["ctx_kv_sum"] / 1e7]


def features(r, model):
    f = m0(r)
    if model == 1 or model == 3:
        f = f + [r["ctx_reqs"], r["ctx_kv_max"] / 1e4, r.get("ctx_kv_var", 0.0) / 1e8, max(r.get("ctx_chunks", []), default=0) / 1e3,
                 r["ctx_tokens"] * r["ctx_kv_max"] / 1e7, r["gen_kv_max"] / 1e4, r.get("gen_kv_var", 0.0) / 1e8]
    if model == 3:  # M1 + execution features, aggregate interaction kept (collinear with the proxy on one-prefill data)
        f = f + [1.0 if r["cg_mode"] == "NONE" else 0.0, r["padded_tokens"] / 1e3, r.get("attn_proxy", 0.0) / 1e6]
    if model == 2:  # M2: the aggregate interaction term is *replaced* by the per-request attention-work proxy
        f = f[:4] + [r["ctx_reqs"], 1.0 if r["cg_mode"] == "NONE" else 0.0, r["padded_tokens"] / 1e3, r.get("attn_proxy", 0.0) / 1e6]
    return f


MODELS = [(0, "M0"), (1, "M1"), (3, "M1+exec"), (2, "M2")]


class Ridge:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        A = Z.T @ Z + LAMBDA * len(y) * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        self.b = float(y.mean())
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b


class LookupP99:
    """Conservative aggregate lookup: P99 of training cost in the bin of
    (decode batch, prefill tokens // 256, aggregate KV // 4096); an unseen bin
    falls back to the ridge M0 (counted as out of support)."""

    def key(self, r):
        return (r["gen_reqs"], r["ctx_tokens"] // 256, (r["ctx_kv_sum"] + r["gen_kv_sum"]) // 4096)

    def fit(self, rows, y, fallback):
        self.tab = {}
        for r, v in zip(rows, y):
            self.tab.setdefault(self.key(r), []).append(v)
        self.tab = {k: float(np.quantile(v, .99)) for k, v in self.tab.items()}
        self.fb = fallback
        return self

    def predict(self, rows, X):
        fb = self.fb.predict(X)
        hit = np.array([self.key(r) in self.tab for r in rows])
        return np.array([self.tab.get(self.key(r), f) for r, f in zip(rows, fb)]), hit


def cell_meta(name):
    kind = name.split("-")[0]
    bg = re.search(r"bg(\d+)", name); L = re.search(r"L(\d+)", name); part = re.search(r"part2?-(\d)x(\d+)", name)
    chunk = re.search(r"chunk(\d+)$", name)
    return {"kind": kind, "bg": int(bg.group(1)) if bg else 8, "L": int(L.group(1)) if L else 16384,
            "n_prefill": int(part.group(1)) if part else 1, "chunk": int(chunk.group(1)) if chunk else None}


def metrics(y, p):
    e = y - p
    pos = np.clip(e, 0, None)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "med_abs": float(np.median(np.abs(e))),
            "p95_abs": float(np.quantile(np.abs(e), .95)), "underpred_rate": float(np.mean(e > 0)),
            "pos_resid_p50": float(np.quantile(pos, .5)), "pos_resid_p95": float(np.quantile(pos, .95)),
            "pos_resid_p99": float(np.quantile(pos, .99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--csv", type=Path, help="per-step CSV (all joined steps, all cells)")
    ap.add_argument("--exclude-over-median-x", type=float, default=None,
                    help="drop prefill steps slower than this multiple of their cell's median prefill "
                         "step (off by default; used for one-off kernel-compile steps on a fresh machine)")
    ap.add_argument("--exclude-first-iteration", action="store_true",
                    help="drop each server's engine iteration 0 (warm-up; off by default)")
    args = ap.parse_args()
    excluded_steps = []

    cells, joins, decode_cells = {}, {}, {}
    all_rows = []
    for d in args.trace_dir:
        for f in sorted(d.glob("*.jsonl")):
            if f.name.endswith(".steps.jsonl"):
                continue
            rows, info = load_cell(f)
            joins[f.stem] = info
            for r in rows:
                all_rows.append({"cell": f.stem, **r})
            decode_cells[f.stem] = [r for r in rows if r["ctx_tokens"] == 0 and np.isfinite(r["cuda_ms"])]
            pre = [r for r in rows if r["ctx_tokens"] > 0 and np.isfinite(r["cuda_ms"])]
            if args.exclude_first_iteration:
                excluded_steps += [{"cell": f.stem, "i": r["i"], "cuda_ms": r["cuda_ms"], "reason": "first iteration"}
                                   for r in pre if r["i"] == 0]
                pre = [r for r in pre if r["i"] != 0]
            if pre and args.exclude_over_median_x:
                med = float(np.median([r["cuda_ms"] for r in pre]))
                excluded_steps += [{"cell": f.stem, "i": r["i"], "cuda_ms": r["cuda_ms"], "cell_median_ms": med,
                                    "reason": f"> {args.exclude_over_median_x:g}x cell median"}
                                   for r in pre if r["cuda_ms"] > args.exclude_over_median_x * med]
                pre = [r for r in pre if r["cuda_ms"] <= args.exclude_over_median_x * med]
            if pre:
                cells[f.stem] = (pre, cell_meta(f.stem))
    print(f"cells with prefill steps: {len(cells)}; join match fraction min "
          f"{min(j['match_frac'] for j in joins.values()):.4f} (offsets {sorted(set(j['offset'] for j in joins.values()))})")
    if args.csv:
        keys = ["cell", "i", "runner_step", "cuda_ms", "gap_ms", "ms", "ctx_reqs", "ctx_tokens", "ctx_kv_sum", "ctx_kv_max", "ctx_kv_var",
                "gen_reqs", "gen_tokens", "gen_kv_sum", "gen_kv_max", "gen_kv_var", "total_tokens", "padded_tokens", "cg_mode", "attn_proxy"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(keys + ["ctx_chunks"])
            for r in all_rows:
                w.writerow([r.get(k) for k in keys] + [" ".join(map(str, r.get("ctx_chunks", [])))])

    one = lambda r: r["ctx_reqs"] == 1
    multi = lambda r: r["ctx_reqs"] >= 2
    non_multi_cell = lambda m: not (m["kind"] in ("part", "part2") and m["n_prefill"] > 1) and m["kind"] != "graph"
    splits = {
        "primary_geometry_one_to_multi": (non_multi_cell, one, lambda m: m["kind"] in ("part", "part2") and m["n_prefill"] > 1, multi),
        "reverse_geometry_multi_to_one": (lambda m: m["kind"] in ("part", "part2") and m["n_prefill"] > 1, multi, lambda m: m["kind"] in ("part", "part2") and m["n_prefill"] == 1, one),
        "ctx_le16k_to_32k": (lambda m: m["kind"] == "ctx" and m["L"] <= 16384, one, lambda m: m["kind"] == "ctx" and m["L"] == 32768, one),
        "batch_le16_to_32": (lambda m: m["kind"] in ("ctx", "load") and m["bg"] <= 16 and m["L"] == 16384, one, lambda m: m["kind"] == "load" and m["bg"] == 32, one),
    }
    report = {"target": "cuda_ms", "alpha": args.alpha, "lambda": LAMBDA, "splits": {},
              "exclude_over_median_x": args.exclude_over_median_x,
              "exclude_first_iteration": args.exclude_first_iteration, "excluded_steps": excluded_steps}
    for split, (tr_cell, tr_row, te_cell, te_row) in splits.items():
        tr = [r for k, (rows, m) in cells.items() if tr_cell(m) for r in rows if tr_row(r)]
        te = [r for k, (rows, m) in cells.items() if te_cell(m) for r in rows if te_row(r)]
        if not tr or not te:
            print(f"\n## {split}: missing cells (train {len(tr)}, test {len(te)})"); continue
        ytr = np.array([r["cuda_ms"] for r in tr]); yte = np.array([r["cuda_ms"] for r in te])
        A0tr = np.array([m0(r) for r in tr]); A0te = np.array([m0(r) for r in te])
        in_support = np.all((A0te >= A0tr.min(0)) & (A0te <= A0tr.max(0)), axis=1)
        print(f"\n## split {split}: train {len(tr)} prefill steps ({len({k for k,(rows,m) in cells.items() if tr_cell(m)})} cells), test {len(te)} "
              f"({in_support.mean()*100:.0f}% of test rows have every M0 coordinate inside the training range)")
        hdr = "| model | MAE | med abs | P95 abs | underpred | pos resid p50 / p95 / p99 | @50 miss(safe%) | @100 | @200 |"
        print(hdr); print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        rep = {"n_train": len(tr), "n_test": len(te), "test_in_m0_support_frac": float(in_support.mean()), "models": {}}
        fitted = {}
        for mi, name in MODELS:
            Xtr = np.array([features(r, mi) for r in tr]); Xte = np.array([features(r, mi) for r in te])
            model = Ridge().fit(Xtr, ytr); fitted[name] = model
            q = float(np.quantile(ytr - model.predict(Xtr), 1 - args.alpha))
            preds = {name: (model.predict(Xte), q, None)}
            if mi == 0:
                lk = LookupP99().fit(tr, ytr, model)
                p, hit = lk.predict(te, Xte)
                preds["M0-lookupP99"] = (p, 0.0, hit)
            for nm, (pte, qq, hit) in preds.items():
                met = metrics(yte, pte)
                dl, cols = {}, []
                for d in (50, 100, 200):
                    safe = (pte + qq) <= d
                    miss = float(np.mean(yte[safe] > d)) if safe.any() else float("nan")
                    dl[str(d)] = {"safe_frac": float(safe.mean()), "miss_rate": miss}
                    cols.append(f"{miss*100:.0f}%({safe.mean()*100:.0f})" if safe.any() else "—(0)")
                entry = {**met, "train_q_alpha_ms": qq, "deadline": dl, "residuals": (yte - pte).tolist()}
                if hit is not None:
                    entry["lookup_hit_frac"] = float(hit.mean())
                if in_support.any() and not in_support.all():
                    entry["in_support"] = metrics(yte[in_support], pte[in_support])
                    entry["out_of_support"] = metrics(yte[~in_support], pte[~in_support])
                rep["models"][nm] = entry
                extra = f" (lookup hit {hit.mean()*100:.0f}%)" if hit is not None else ""
                print(f"| {nm}{extra} | {met['mae']:.1f} | {met['med_abs']:.1f} | {met['p95_abs']:.1f} | {met['underpred_rate']*100:.0f}% | "
                      f"{met['pos_resid_p50']:.1f} / {met['pos_resid_p95']:.1f} / {met['pos_resid_p99']:.1f} | " + " | ".join(cols) + " |")
        if in_support.any() and not in_support.all():
            print(f"in-support test rows only (n={int(in_support.sum())}): " + "; ".join(
                f"{nm} pos resid p95 {e['in_support']['pos_resid_p95']:.1f} / p99 {e['in_support']['pos_resid_p99']:.1f}" for nm, e in rep["models"].items()))
        report["splits"][split] = rep

    print("\n## Prefill-step cost by cell (steps containing prefill work)")
    print("| cell | n | chunk tokens p50 | cuda p50 | cuda p95 | gap p50 | padded p50 | cg_mode |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    same = {}
    for k, (rows, m) in cells.items():
        c = np.array([r["cuda_ms"] for r in rows]); g = np.array([r["gap_ms"] for r in rows])
        modes = sorted(set(r["cg_mode"] for r in rows))
        same[k] = {"meta": m, "n": len(rows), "chunk_p50": float(np.median([r["ctx_tokens"] for r in rows])), "cuda_p50": float(np.median(c)),
                   "cuda_p95": float(np.quantile(c, .95)), "gap_p50": float(np.nanmedian(g)), "padded_p50": float(np.median([r["padded_tokens"] for r in rows])), "modes": modes}
        print(f"| {k} | {len(rows)} | {same[k]['chunk_p50']:.0f} | {same[k]['cuda_p50']:.1f} | {same[k]['cuda_p95']:.1f} | {same[k]['gap_p50']:.1f} | {same[k]['padded_p50']:.0f} | {','.join(modes)} |")
    report["cells"] = same

    print("\n## Decode-only steps by cell")
    print("| cell | n | reqs p50 | padded tokens p50 | decode KV sum p50 | cuda p50 | cuda p95 | gap p50 | cg_mode |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    decs = {}
    for k, rows in decode_cells.items():
        if not rows:
            continue
        c = np.array([r["cuda_ms"] for r in rows]); g = np.array([r["gap_ms"] for r in rows])
        modes = sorted(set(r["cg_mode"] for r in rows))
        decs[k] = {"n": len(rows), "reqs_p50": float(np.median([r["gen_reqs"] for r in rows])), "padded_p50": float(np.median([r["padded_tokens"] for r in rows])),
                   "gen_kv_sum_p50": float(np.median([r["gen_kv_sum"] for r in rows])),
                   "cuda_p50": float(np.median(c)), "cuda_p95": float(np.quantile(c, .95)), "gap_p50": float(np.nanmedian(g)), "modes": modes}
        d = decs[k]
        print(f"| {k} | {d['n']} | {d['reqs_p50']:.0f} | {d['padded_p50']:.0f} | {d['gen_kv_sum_p50']:.0f} | {d['cuda_p50']:.2f} | {d['cuda_p95']:.2f} | {d['gap_p50']:.2f} | {','.join(modes)} |")
    report["decode_cells"] = decs
    report["joins"] = joins
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
