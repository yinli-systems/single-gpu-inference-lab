#!/usr/bin/env python3
"""Per-step rows with per-request (chunk, depth) geometry for the eleven shape-campaign datasets.

`steps.csv` (analyze_step_cost_v2.py --csv) keeps each step's chunks but not the per-request KV
depths. This module joins every filtered prefill row of a dataset's steps.csv, by (cell, i), to the
iteration trace of that cell, which has `ctx_depths`, and checks the join row by row: the chunks must
be equal and `ctx_kv_sum` must equal the sum of the depths. Filters and splits are those of
analyze_m2_variants.py (first engine iteration dropped, steps > 10x their cell median dropped).

  python scripts/geometry_dataset.py            # prints the join check for every dataset
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import analyze_m2_variants as V

R = Path(__file__).resolve().parents[1] / "benchmarks/results"
A = R / "a100-prefill-cost-geometry/raw"
S = R / "prefill-geometry-attention-shape/raw"
H = R / "h100-prefill-cost-geometry/raw"
L20 = R / "l20-prefill-cost-geometry"

DATASETS = {
    "L20 Qwen3-4B": (L20 / "steps.csv", [L20 / "raw" / c / "trace" for c in ("campaign18", "campaign19")]),
    "L20 Qwen2.5-1.5B": (S / "L20-Qwen2.5-1.5B-Instruct/steps.csv", [S / "L20-Qwen2.5-1.5B-Instruct/trace"]),
    "L20 Qwen2.5-7B": (S / "L20-Qwen2.5-7B-Instruct/steps.csv", [S / "L20-Qwen2.5-7B-Instruct/trace"]),
    "A100 Qwen3-4B": (A / "Qwen3-4B/steps.csv", [A / "Qwen3-4B/trace"]),
    "A100 Qwen3-8B": (A / "Qwen3-8B/steps.csv", [A / "Qwen3-8B/trace"]),
    "A100 Qwen2.5-1.5B": (S / "A100-Qwen2.5-1.5B-Instruct/steps.csv", [S / "A100-Qwen2.5-1.5B-Instruct/trace"]),
    "A100 Qwen2.5-7B": (S / "A100-Qwen2.5-7B-Instruct/steps.csv", [S / "A100-Qwen2.5-7B-Instruct/trace"]),
    "H100 Qwen3-4B": (H / "Qwen3-4B/steps.csv", [H / "Qwen3-4B/trace"]),
    "H100 Qwen3-8B": (H / "Qwen3-8B/steps.csv", [H / "Qwen3-8B/trace"]),
    "H100 Qwen2.5-1.5B": (H / "Qwen2.5-1.5B-Instruct/steps.csv", [H / "Qwen2.5-1.5B-Instruct/trace"]),
    "H100 Qwen2.5-7B": (H / "Qwen2.5-7B-Instruct/steps.csv", [H / "Qwen2.5-7B-Instruct/trace"]),
}


def _iter_trace(dirs, cell):
    for d in dirs:
        for p in (d / f"{cell}.jsonl", d / f"{cell}.jsonl.gz"):
            if p.exists():
                fh = gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
                return {j["i"]: j for j in map(json.loads, fh)}
    return None


def load(name, filtered=True):
    """-> (cells, check): cells as analyze_m2_variants.load_cells, each row with `ctx_depths` added."""
    steps, dirs = DATASETS[name]
    cells = V.load_cells(steps, filtered)
    check = {"rows": 0, "joined": 0, "chunk_mismatch": 0, "kvsum_mismatch": 0, "missing_trace": []}
    for cell, (rows, meta) in cells.items():
        it = _iter_trace(dirs, cell)
        if it is None:
            check["missing_trace"].append(cell)
            continue
        for r in rows:
            check["rows"] += 1
            j = it.get(r["i"])
            if j is None:
                continue
            if list(j["ctx_chunks"]) != r["ctx_chunks"]:
                check["chunk_mismatch"] += 1
                continue
            if sum(j["ctx_depths"]) != r["ctx_kv_sum"]:
                check["kvsum_mismatch"] += 1
                continue
            r["ctx_depths"] = list(j["ctx_depths"])
            check["joined"] += 1
    for cell in list(cells):
        rows, meta = cells[cell]
        cells[cell] = ([r for r in rows if "ctx_depths" in r], meta)
    return cells, check


def main():
    for name in DATASETS:
        _, c = load(name)
        print(f"{name:20s} rows {c['rows']:5d} joined {c['joined']:5d} chunk-mismatch {c['chunk_mismatch']} "
              f"kvsum-mismatch {c['kvsum_mismatch']} missing-trace-cells {len(c['missing_trace'])} {c['missing_trace'][:3]}")


if __name__ == "__main__":
    main()
