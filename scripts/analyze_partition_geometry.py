#!/usr/bin/env python3
"""Section 2.4 of l20-prefill-cost-geometry, from any per-step CSV.

Partition of the same prefill budget at an equal aggregate coordinate: decode batch 8, a full
prefill budget per step, binned by the aggregate KV depth of the step (prefill KV read + decode
KV), one row per bin and one column per geometry; then each geometry's step time regressed on the
per-request attention-work proxy alone (sum_i q_i * (kv_i + (q_i + 1) / 2), in millions).

Input: the per-step CSV written by `analyze_step_cost_v2.py --csv` (one row per joined engine
iteration). Several CSVs can be given (e.g. one per GPU) and are reported side by side.

Outliers: a step slower than 10x its cell's median is excluded from the fits (and reported). On a
fresh machine the first step of a new prefill shape can include a one-off kernel compile (one
2.9 s step on the A100, against a ~125 ms median); binned medians are unaffected either way.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

SERIES = {
    1024: [("part-1x1024-chunk1024", "1x1024"), ("part-2x512-chunk1024", "2x512"),
           ("part-4x256-chunk1024", "4x256")],
    2048: [("part2-1x2048-chunk2048", "1x2048"), ("part2-4x512-chunk2048", "4x512"),
           ("part2-8x256-chunk2048", "8x256")],
}
BINS = [(0, 4096), (4096, 8192), (8192, 12288), (12288, 16384), (16384, 20480)]
DECODE_BATCH = 8


def load(path: Path):
    return list(csv.DictReader(open(path)))


def full_budget_steps(rows, cell, budget):
    out = []
    for r in rows:
        if r["cell"] != cell or r["cuda_ms"] in ("", "nan"):
            continue
        if int(r["gen_reqs"]) != DECODE_BATCH:
            continue
        ctx = int(r["ctx_tokens"])
        if not (budget - DECODE_BATCH <= ctx <= budget):
            continue
        out.append({"cuda_ms": float(r["cuda_ms"]),
                    "agg_kv": int(r["ctx_kv_sum"]) + int(r["gen_kv_sum"]),
                    "proxy_m": float(r["attn_proxy"]) / 1e6})
    return out


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


OUTLIER_X_MEDIAN = 10.0


def drop_outliers(st):
    if not st:
        return st, []
    med = statistics.median(s["cuda_ms"] for s in st)
    keep = [s for s in st if s["cuda_ms"] <= OUTLIER_X_MEDIAN * med]
    return keep, [s["cuda_ms"] for s in st if s["cuda_ms"] > OUTLIER_X_MEDIAN * med]


def analyze(rows):
    report = {}
    for budget, series in SERIES.items():
        steps, excluded = {}, {}
        for cell, label in series:
            steps[label], excluded[label] = drop_outliers(full_budget_steps(rows, cell, budget))
        table = []
        for lo, hi in BINS:
            row = {"bin": f"{lo // 1024}-{hi // 1024}k"}
            for label, st in steps.items():
                v = [s["cuda_ms"] for s in st if lo <= s["agg_kv"] < hi]
                row[label] = {"median_ms": statistics.median(v), "n": len(v)} if v else None
            first, last = series[0][1], series[-1][1]
            if row[first] and row[last]:
                row["ratio"] = row[first]["median_ms"] / row[last]["median_ms"]
            table.append(row)
        fits = {}
        for label, st in steps.items():
            if len(st) >= 3:
                a, b = ols([s["proxy_m"] for s in st], [s["cuda_ms"] for s in st])
                fits[label] = {"intercept_ms": a, "ms_per_M": b, "n": len(st)}
        report[budget] = {"table": table, "fits": fits,
                          "excluded_ms": {k: v for k, v in excluded.items() if v}}
    return report


def fmt(report, name):
    lines = [f"## {name}"]
    for budget, rep in report.items():
        labels = [l for _, l in SERIES[budget]]
        lines.append(f"\nbudget {budget}: median step CUDA ms by aggregate KV bin (n)")
        lines.append("| aggregate KV | " + " | ".join(labels) + f" | {labels[0]}/{labels[-1]} |")
        lines.append("| --- |" + " ---: |" * (len(labels) + 1))
        for row in rep["table"]:
            cells = [f"{row[l]['median_ms']:.1f} ({row[l]['n']})" if row[l] else "—" for l in labels]
            ratio = f"{row['ratio']:.2f}x" if "ratio" in row else "—"
            lines.append(f"| {row['bin']} | " + " | ".join(cells) + f" | {ratio} |")
        lines.append("fits on the attention-work proxy alone: " + ", ".join(
            f"{l}: {f['intercept_ms']:.1f} + {f['ms_per_M']:.2f}·M (n={f['n']})" for l, f in rep["fits"].items()))
        if rep["excluded_ms"]:
            lines.append("excluded (>10x cell median): " + ", ".join(
                f"{l}: {[round(v, 1) for v in vs]}" for l, vs in rep["excluded_ms"].items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, nargs="+", required=True, help="NAME=path/to/steps.csv")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    out = {}
    for spec in args.steps:
        name, path = str(spec).split("=", 1) if "=" in str(spec) else (Path(spec).parent.name, spec)
        out[name] = analyze(load(Path(path)))
        print(fmt(out[name], name) + "\n")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
