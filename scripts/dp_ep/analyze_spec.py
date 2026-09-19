#!/usr/bin/env python3
"""Task 27 analysis: does spec-decode acceptance skew across DP/EP ranks cost the low-acceptance rank?

Input: a results dir with spec.json (measure_spec.py) and trace/iter.jsonl.dp{0,1} (engine
iteration trace: `t` monotonic host clock, `total_tokens` = tokens scheduled in the step =
Σ_requests (1 + drafted), `ms` = engine iteration wall time). The runner step trace is not
written under speculative decoding (no padded_tokens), so the padded width is *estimated* as the
cross-rank max of `total_tokens` rounded up to the next multiple of 8 (vLLM's capture-size grid).

Per cell and rank: generation rate (engine counter over the window), acceptance rate, drafts per
request-step, ITL p50/p95, iteration period p50, width p50 = total_tokens per step.
Pooled by (kind0, kind1, B), median over repeats. Questions scored at each B:
  Q1 low-acceptance rank: rate/period of the rnd rank in rnd|rep and rep|rnd vs rnd|rnd
  Q2 high-acceptance rank: rate/period of the rep rank in rep|rnd and rnd|rep vs rep|rep
  Q3 width skew: rep-rank vs rnd-rank total_tokens per step, and the estimated padded width the
     rnd rank executes in skewed cells vs its own width in rnd|rnd
Gate (pre-registered): rank generation rate / ITL change < 5 % kill; > 10 % continue.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, p)) if len(a) else float("nan")


def fmt(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def pad8(n):
    return int(max(8, 8 * np.ceil(n / 8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rec = json.load(open(args.dir / "spec.json"))
    it = {r: load_jsonl(args.dir / "trace" / f"iter.jsonl.dp{r}") for r in (0, 1)}
    tt = {r: np.array([x["t"] for x in it[r]]) for r in (0, 1)}
    rows = []
    print("| cell | B | rep | rank | kind | gen tok/s | accept | drafts/req-step | ITL p50/p95 | iter period p50 | width p50 (tokens/step) | est. padded width p50 |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for c in rec["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        bases = list(c["spec"].keys())
        sel = {r: [it[r][i] for i in np.where((tt[r] >= t0) & (tt[r] <= t1))[0]] for r in (0, 1)}
        # cross-rank padded width estimate: for each rank-0 iteration, the nearest-in-time rank-1 iteration (lockstep => within a few ms)
        w1 = np.array([s["total_tokens"] for s in sel[1]]) if sel[1] else np.array([])
        t1s = np.array([s["t"] for s in sel[1]]) if sel[1] else np.array([])
        padded = []
        for s in sel[0]:
            if len(t1s):
                j = int(np.argmin(np.abs(t1s - s["t"])))
                padded.append(pad8(max(s["total_tokens"], w1[j])) if abs(t1s[j] - s["t"]) < 0.05 else pad8(s["total_tokens"]))
        row = {"kind0": c["kind0"], "kind1": c["kind1"], "B": c["B"], "rep": c["repeat"], "ranks": {}}
        for r, base in enumerate(bases):
            sp, itl = c["spec"][base], c["itl_window"][base]
            steps = max(len(sel[r]), 1)
            d = {"kind": c[f"kind{r}"], "gen_tok_s": sp["gen_tok_s"], "accept": sp.get("accept_rate"),
                 "drafts_per_req_step": sp.get("vllm:spec_decode_num_drafts_total", 0.0) / (steps * c["B"]),
                 "draft_tokens_per_draft": (sp.get("vllm:spec_decode_num_draft_tokens_total", 0.0) / sp["vllm:spec_decode_num_drafts_total"]) if sp.get("vllm:spec_decode_num_drafts_total") else None,
                 "itl_p50": itl.get("p50"), "itl_p95": itl.get("p95"), "errors": itl.get("errors"),
                 "iters": len(sel[r]), "period_p50": q(np.diff([s["t"] for s in sel[r]]) * 1e3, .5) if len(sel[r]) > 1 else float("nan"),
                 "iter_ms_p50": q([s["ms"] for s in sel[r]], .5), "width_p50": q([s["total_tokens"] for s in sel[r]], .5), "width_mean": float(np.mean([s["total_tokens"] for s in sel[r]])) if sel[r] else float("nan"),
                 "gen_per_step": sp.get("vllm:generation_tokens_total", 0.0) / steps / c["B"]}
            d["padded_width_p50"] = q(padded, .5)
            row["ranks"][r] = d
            print(f"| {c['kind0']}\\|{c['kind1']} | {c['B']} | {c['repeat']} | {r} | {d['kind']} | {d['gen_tok_s']:.0f} | {fmt(d['accept'], 2)} | {fmt(d['drafts_per_req_step'], 2)} | {fmt(d['itl_p50'])}/{fmt(d['itl_p95'])} | {fmt(d['period_p50'])} | {fmt(d['width_p50'], 0)} | {fmt(d['padded_width_p50'], 0)} |")
        rows.append(row)
    # pooled by cell type: median over repeats, per rank
    g = defaultdict(list)
    for x in rows:
        g[(x["kind0"], x["kind1"], x["B"])].append(x)
    keys = ("gen_tok_s", "accept", "drafts_per_req_step", "itl_p50", "itl_p95", "period_p50", "width_p50", "padded_width_p50", "gen_per_step")
    pooled = {k: {r: {m: float(np.nanmedian([x["ranks"][r][m] if x["ranks"][r][m] is not None else np.nan for x in xs])) for m in keys} for r in (0, 1)} for k, xs in g.items()}
    print("\n### pooled (median over repeats) per rank\n")
    print("| cell | B | n | rank | kind | gen tok/s | gen/req/step | accept | drafts/req-step | ITL p50/p95 | period p50 | width p50 | est. padded |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for (k0, k1, B), v in sorted(pooled.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        for r in (0, 1):
            d = v[r]
            print(f"| {k0}\\|{k1} | {B} | {len(g[(k0, k1, B)])} | {r} | {(k0, k1)[r]} | {d['gen_tok_s']:.0f} | {fmt(d['gen_per_step'], 2)} | {fmt(d['accept'], 2)} | {fmt(d['drafts_per_req_step'], 2)} | {fmt(d['itl_p50'])}/{fmt(d['itl_p95'])} | {fmt(d['period_p50'])} | {fmt(d['width_p50'], 0)} | {fmt(d['padded_width_p50'], 0)} |")
    # skew scoring: a rank of kind K in a skewed cell vs the same kind in its homogeneous cell (both ranks of the homogeneous cell pooled)
    print("\n### skew effect on each kind (skewed cell vs homogeneous cell of the same kind; ratio > 1 = worse latency / higher rate)\n")
    print("| B | kind | placement | gen tok/s homo → skew (ratio) | ITL p50 homo → skew (ratio) | ITL p95 homo → skew (ratio) | period homo → skew (ratio) | width homo → skew | est. padded homo → skew |")
    print("| ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
    summary = []
    Bs = sorted({k[2] for k in pooled})
    for B in Bs:
        for kind, other in (("rnd", "rep"), ("rep", "rnd")):
            homo = pooled.get((kind, kind, B))
            if homo is None:
                continue
            h = {m: float(np.nanmean([homo[0][m], homo[1][m]])) for m in keys}
            for (k0, k1), r in (((kind, other), 0), ((other, kind), 1)):
                sk = pooled.get((k0, k1, B))
                if sk is None:
                    continue
                s = sk[r]
                e = {"B": B, "kind": kind, "cell": f"{k0}|{k1}", "rank": r}
                for m in ("gen_tok_s", "itl_p50", "itl_p95", "period_p50", "width_p50", "padded_width_p50"):
                    e[f"{m}_homo"], e[f"{m}_skew"] = h[m], s[m]
                    e[f"{m}_ratio"] = s[m] / h[m] if h[m] else float("nan")
                summary.append(e)
                print(f"| {B} | {kind} | {k0}\\|{k1} rank {r} | {h['gen_tok_s']:.0f} → {s['gen_tok_s']:.0f} ({fmt(e['gen_tok_s_ratio'], 2)}) | {fmt(h['itl_p50'])} → {fmt(s['itl_p50'])} ({fmt(e['itl_p50_ratio'], 2)}) | "
                      f"{fmt(h['itl_p95'])} → {fmt(s['itl_p95'])} ({fmt(e['itl_p95_ratio'], 2)}) | {fmt(h['period_p50'])} → {fmt(s['period_p50'])} ({fmt(e['period_p50_ratio'], 2)}) | "
                      f"{fmt(h['width_p50'], 0)} → {fmt(s['width_p50'], 0)} | {fmt(h['padded_width_p50'], 0)} → {fmt(s['padded_width_p50'], 0)} |")
    if summary:
        worst = max(summary, key=lambda e: abs(np.log(e["gen_tok_s_ratio"])) if e["gen_tok_s_ratio"] > 0 else 0)
        print(f"\nlargest rate effect: {worst['kind']} rank in {worst['cell']} B={worst['B']}: gen ×{worst['gen_tok_s_ratio']:.2f}, ITL p50 ×{worst['itl_p50_ratio']:.2f}, period ×{worst['period_p50_ratio']:.2f}; gate: < 5 % kill, > 10 % continue")
    args.output.write_text(json.dumps({"rows": rows, "pooled": {f"{k[0]}|{k[1]}|B{k[2]}": v for k, v in pooled.items()}, "summary": summary}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
