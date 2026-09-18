#!/usr/bin/env python3
"""Measurement contract for the step-cost traces (tracer v2).

Answers, from raw files only:
  1. coverage   -- engine iterations are contiguous and every completed one
                   has one runner step (CUDA event pair) joined by sequence;
  2. overhead   -- trace on vs trace off, interleaved repeats: background ITL
                   during injection, injected-request TTFT, tok/s;
  3. agreement  -- result-ready gap (host clock) vs direct CUDA elapsed time,
                   by step class (decode-only / prefill chunk size).

Inputs: a directory with off-rep{k}.json, on-rep{k}.json and the on-rep{k}-<cond>.jsonl
/ .steps.jsonl traces produced by measure_prefill_interference.py --trace-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from step_trace_join import load_joined


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def align(iter_path: Path):
    rows, info = load_joined(iter_path)
    cov = {**info, "iterations_with_tokens": sum(1 for r in rows if r["total_tokens"] > 0),
           "aligned": sum(1 for r in rows if r["runner_step"] is not None),
           "unaligned_iterations": sum(1 for r in rows if r["runner_step"] is None)}
    return cov, [(r, r if r["runner_step"] is not None else None, r["gap_ms"]) for r in rows if r["total_tokens"] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rep = {"coverage": {}, "overhead": {}, "agreement": {}}

    # 1. coverage + 3. agreement
    classes = {}
    print("## Coverage (per trace file)\n")
    print("| trace | engine iters | index gaps | runner steps | offset | token-count matches | unaligned iters |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for f in sorted(args.dir.glob("on-rep*-*.jsonl")):
        if f.name.endswith(".steps.jsonl"):
            continue
        cov, rows = align(f)
        rep["coverage"][f.stem] = cov
        print(f"| {f.stem} | {cov['iterations']} | {cov['index_gaps']} | {cov['runner_steps']} | {cov['offset']} | {cov['matched']}/{cov['compared']} | {cov['unaligned_iterations']} |")
        for r, s, g in rows:
            if s is None or not np.isfinite(g):
                continue
            cls = "decode-only" if r["ctx_tokens"] == 0 else f"prefill chunk~{int(round(r['ctx_tokens']/256)*256)}"
            classes.setdefault(cls, []).append((g, s["cuda_ms"], r["ms"]))
    print("\n## Host result-ready gap vs direct CUDA elapsed (aligned steps)\n")
    print("| step class | n | CUDA p50 | gap p50 | (gap−CUDA) p5 / p50 / p95 | corr | engine-wait `ms` p50 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cls in sorted(classes):
        a = np.array(classes[cls]); g, c, w = a[:, 0], a[:, 1], a[:, 2]
        d = g - c
        corr = float(np.corrcoef(g, c)[0, 1]) if len(a) > 2 else float("nan")
        rep["agreement"][cls] = {"n": int(len(a)), "cuda_p50": q(c, .5), "gap_p50": q(g, .5), "diff_p5": q(d, .05), "diff_p50": q(d, .5), "diff_p95": q(d, .95), "corr": corr, "engine_wait_ms_p50": q(w, .5)}
        print(f"| {cls} | {len(a)} | {q(c,.5):.1f} | {q(g,.5):.1f} | {q(d,.05):+.1f} / {q(d,.5):+.1f} / {q(d,.95):+.1f} | {corr:.3f} | {q(w,.5):.1f} |")

    # 2. overhead
    print("\n## Trace on/off overhead (interleaved repeats)\n")
    print("| condition | metric | off (3 reps) | on (3 reps) | on/off |")
    print("| --- | --- | --- | --- | ---: |")
    for cond in ("chunk512", "chunk2048"):
        vals = {"off": {}, "on": {}}
        for mode in vals:
            for k in (1, 2, 3):
                p = args.dir / f"{mode}-rep{k}.json"
                if not p.exists():
                    continue
                r = json.load(open(p))["conditions"][cond]["repeats"][0]
                vals[mode].setdefault("during ITL p50 ms", []).append(r["background"]["during"]["p50_ms"])
                vals[mode].setdefault("during ITL p95 ms", []).append(r["background"]["during"]["p95_ms"])
                vals[mode].setdefault("TTFT mean s", []).append(float(np.mean([x["ttft_s"] for x in r["long_requests"]])))
                vals[mode].setdefault("bg tok/s during", []).append(r["background_tok_per_s_during"])
        rep["overhead"][cond] = vals
        for m in vals["off"]:
            off, on = np.array(vals["off"][m]), np.array(vals["on"][m])
            print(f"| {cond} | {m} | {' / '.join(f'{x:.1f}' for x in off)} | {' / '.join(f'{x:.1f}' for x in on)} | {on.mean()/off.mean():.3f} |")
    args.output.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
