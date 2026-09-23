#!/usr/bin/env python3
"""Live A/B of the env-gated deadline controller (campaign20).

Per run (one fresh server, one injection of N long prefills into B decoders):
  prefill-step violations : traced prefill-containing steps whose CUDA time > deadline
  safe prefill tok/s      : prefill tokens of non-violating steps / wall time of the prefill phase
  TTFT                    : mean time-to-first-token of the injected requests
  decode ITL during       : background decoders' p95 / p99 inter-token latency and 100 ms SLO
Runs are grouped by condition (name without the -r<k> suffix); mean and min–max over repeats.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from step_trace_join import load_joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--deadline-ms", type=float, default=100.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runs = defaultdict(list)
    for jf in sorted(args.dir.glob("*.json")):
        m = re.match(r"(.+)-r(\d+)$", jf.stem)
        if not m:
            continue
        cond, rep = m.group(1), int(m.group(2))
        d = json.load(open(jf))
        (budget, c), = d["conditions"].items()
        r = c["repeats"][0]
        tr = args.dir / "trace" / f"{jf.stem}-{budget}.jsonl"
        rows, info = load_joined(tr)
        long_ids = None
        pre = [x for x in rows if x["ctx_tokens"] > 0 and np.isfinite(x["cuda_ms"]) and (x["ctx_kv_max"] >= 512 or x["ctx_tokens"] >= 256)]
        # prefill phase = from the first to the last step touching an injected (long) prefill
        # injected-prefill phase: all background decoders are already running (gen_reqs >= 8),
        # which excludes the background prompt prefills at server start
        long_steps = [x for x in pre if x["gen_reqs"] >= 8 and (x["ctx_kv_max"] >= 512 or x["ctx_tokens"] >= 256)]
        if not long_steps:
            continue
        t0, t1 = long_steps[0]["t"], long_steps[-1]["t"] + long_steps[-1]["cuda_ms"] / 1e3
        phase = [x for x in rows if t0 <= x["t"] <= t1 and np.isfinite(x["cuda_ms"])]
        ppre = [x for x in phase if x["ctx_tokens"] > 0]
        cuda = np.array([x["cuda_ms"] for x in ppre])
        viol = cuda > args.deadline_ms
        safe_tokens = sum(x["ctx_tokens"] for x, v in zip(ppre, viol) if not v)
        budgets = [x["exp"]["budget"] for x in ppre if x.get("exp")]
        runs[cond].append({
            "rep": rep, "join_match": info["match_frac"], "prefill_steps": len(ppre), "violation_rate": float(viol.mean()),
            "prefill_cuda_p50": float(np.median(cuda)), "prefill_cuda_p95": float(np.quantile(cuda, .95)), "prefill_cuda_max": float(cuda.max()),
            "safe_tok_per_s": safe_tokens / (t1 - t0), "phase_s": t1 - t0,
            "ttft_mean_s": float(np.mean([x["ttft_s"] for x in r["long_requests"]])), "ttft_max_s": float(max(x["ttft_s"] for x in r["long_requests"])),
            "itl_during_p50": r["background"]["during"]["p50_ms"], "itl_during_p95": r["background"]["during"]["p95_ms"], "itl_during_p99": r["background"]["during"]["p99_ms"],
            "itl_max": r["background"]["during"]["max_ms"], "bg_tok_per_s_during": r["background_tok_per_s_during"],
            "budget_p50": float(np.median(budgets)) if budgets else None,
            # realised prefill tokens per prefill step (the chosen budget as executed; available
            # even when the trace carries no controller decision records)
            "prefill_tokens_p50": float(np.median([x["ctx_tokens"] for x in ppre])),
            "margin_last_ms": next((x["exp"].get("margin_ms") for x in reversed(ppre) if x.get("exp")), None),
            "online_frac": float(np.mean([bool(x["exp"].get("online")) for x in ppre if x.get("exp")])) if budgets else None, "budget_hist": {str(k): int(v) for k, v in zip(*np.unique(budgets, return_counts=True))} if budgets else None,
        })
    print(f"| condition | reps | prefill steps | violations >{args.deadline_ms:.0f} ms | prefill step p95 / max | safe prefill tok/s | TTFT mean / max | bg ITL p95 / p99 / max | budget p50 | prefill tokens/step p50 | final margin |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    summary = {}
    def mr(xs, f="{:.1f}"):
        return f.format(np.mean(xs)) + " (" + f.format(min(xs)) + "–" + f.format(max(xs)) + ")"
    for cond in sorted(runs, key=lambda c: (re.split(r"-", c)[0], c)):
        rs = runs[cond]
        g = lambda k: [x[k] for x in rs]
        summary[cond] = {"runs": rs, "mean": {k: float(np.mean(g(k))) for k in rs[0] if isinstance(rs[0][k], (int, float)) and rs[0][k] is not None}}
        print(f"| {cond} | {len(rs)} | {mr(g('prefill_steps'), '{:.0f}')} | {mr([100*v for v in g('violation_rate')])}% | {mr(g('prefill_cuda_p95'))} / {mr(g('prefill_cuda_max'), '{:.0f}')} | "
              f"{mr(g('safe_tok_per_s'), '{:.0f}')} | {mr(g('ttft_mean_s'), '{:.2f}')} / {mr(g('ttft_max_s'), '{:.2f}')} | {mr(g('itl_during_p95'), '{:.0f}')} / {mr(g('itl_during_p99'), '{:.0f}')} / {mr(g('itl_max'), '{:.0f}')} | "
              f"{np.mean([x['budget_p50'] for x in rs if x['budget_p50']]) if any(x['budget_p50'] for x in rs) else '—'} | {mr(g('prefill_tokens_p50'), '{:.0f}')} | "
              f"{mr([x['margin_last_ms'] for x in rs]) if all(x['margin_last_ms'] is not None for x in rs) else '—'} |")
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
