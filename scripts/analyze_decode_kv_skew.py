#!/usr/bin/env python3
"""Section 2.2 of l20-prefill-cost-geometry, from any per-step CSV.

Decode-KV distribution at equal aggregate: 8 decoders whose prompt KV sums to 32,768 either way,
balanced 8x4096 (`skew-bal`) vs skewed 4x7936 + 4x256 (`skew-skew`). Reports, per cell, the
decode-only step (8 decoders, no prefill) and the 512-chunk prefill step at 8k / 12k prefill KV
depth (8-9 decoders alongside), as p50 / p95 CUDA ms, plus the decode p50 per third of the cell
(repeats run in order, so a real difference is stable across thirds).

Input: the per-step CSV written by `analyze_step_cost_v2.py --csv`. Depth bins here are
|ctx_kv_sum - depth| < 512 over steps carrying >= 500 prefill tokens, which is close to but not
the exact selection behind the published L20 table (L20 8k balanced: 90.8 here vs 91.2 there).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CELLS = ("skew-bal-chunk512", "skew-skew-chunk512")
DEPTHS = (8192, 12288)


def summarize(v):
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "n": len(v)} if v else None


def analyze(path: Path):
    rows = [r for r in csv.DictReader(open(path)) if r["cell"] in CELLS and r["cuda_ms"] not in ("", "nan")]
    out = {}
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell]
        dec = sorted((r for r in rs if r["ctx_tokens"] == "0" and r["gen_reqs"] == "8"), key=lambda r: int(r["i"]))
        k = len(dec) // 3
        rep = {"decode": summarize([float(r["cuda_ms"]) for r in dec]),
               "decode_gen_kv_sum_p50": float(np.median([int(r["gen_kv_sum"]) for r in dec])) if dec else None,
               "decode_p50_by_third": [float(np.median([float(r["cuda_ms"]) for r in dec[j * k:(j + 1) * k]])) for j in range(3)] if k else []}
        pre = [r for r in rs if r["ctx_tokens"] != "0" and r["gen_reqs"] in ("8", "9") and int(r["ctx_tokens"]) >= 500]
        for d in DEPTHS:
            rep[f"prefill512_at_{d // 1024}k"] = summarize([float(r["cuda_ms"]) for r in pre if abs(int(r["ctx_kv_sum"]) - d) < 512])
        out[cell] = rep
    b, s = (out[c]["decode"] for c in CELLS)
    if b and s:
        out["decode_skewed_over_balanced_p50"] = s["p50_ms"] / b["p50_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True, help="NAME=path/to/steps.csv")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    res = {}
    for spec in args.steps:
        name, path = spec.split("=", 1)
        res[name] = r = analyze(Path(path))
        print(f"## {name}")
        for cell in CELLS:
            c = r[cell]
            parts = [f"decode {c['decode']['p50_ms']:.2f}/{c['decode']['p95_ms']:.2f} (thirds {', '.join(f'{x:.2f}' for x in c['decode_p50_by_third'])})"]
            for d in DEPTHS:
                p = c[f"prefill512_at_{d // 1024}k"]
                parts.append(f"512@{d // 1024}k {p['p50_ms']:.1f}/{p['p95_ms']:.1f} (n={p['n']})" if p else f"512@{d // 1024}k -")
            print(f"  {cell:20s} " + " | ".join(parts))
        if "decode_skewed_over_balanced_p50" in r:
            print(f"  decode skewed/balanced p50: {r['decode_skewed_over_balanced_p50']:.3f}")
    if args.output:
        args.output.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
