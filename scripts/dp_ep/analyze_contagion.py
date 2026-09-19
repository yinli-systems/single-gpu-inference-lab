#!/usr/bin/env python3
"""Task 21 analysis: does a feature on one DP/EP rank raise the ITL of plain requests on the other?

Input: a results dir with contagion.json (from measure_contagion.py) and, optionally,
trace/step.jsonl.dp{0,1} (per-rank step period / GPU time inside each window).

For every (feat0, feat1, B) the per-repeat values are pooled by median. The contagion table
compares the *plain* rank's ITL in F|plain (rank 1 plain) and plain|F (rank 0 plain) with the
same rank in plain|plain: ratio > 1 is latency imposed on an unrelated client on the other GPU.
Gate (pre-registered): <5% kill, 10% alive, 20% strong, 50% priority upstream problem.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rec = json.load(open(args.dir / "contagion.json"))
    step = {r: load_jsonl(args.dir / "trace" / f"step.jsonl.dp{r}") for r in (0, 1)}
    st = {r: np.array([x["t"] for x in step[r]]) for r in (0, 1)}
    rows = []
    print("| feat0 | feat1 | B | rep | ITL r0 p50/p95 | ITL r1 p50/p95 | err r0/r1 | step r0 period p50 / cuda p50 | step r1 period / cuda | tokens r0/r1 p50 | train reqs / ttft p50 |")
    print("| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |")
    for c in rec["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        itl = {}
        for r, (base, streams) in enumerate(c["decode"].items()):
            gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if t0 <= a <= t1]
            itl[r] = {"p50": q(gaps, .5), "p95": q(gaps, .95), "n": len(gaps), "errors": sum(1 for s in streams if s.get("error"))}
        sp = {}
        for r in (0, 1):
            sel = [step[r][i] for i in np.where((st[r] >= t0) & (st[r] <= t1))[0]]
            iv = np.diff(np.array([s["t"] for s in sel])) * 1e3 if len(sel) > 1 else np.array([])
            sp[r] = {"n": len(sel), "period_p50": q(iv, .5), "period_p95": q(iv, .95), "cuda_p50": q([s["cuda_ms"] for s in sel], .5), "cuda_p95": q([s["cuda_ms"] for s in sel], .95),
                     "tokens_p50": q([s["num_tokens"] for s in sel], .5), "exec_p50": q([s["padded_tokens"] for s in sel], .5)}
        train = [s for v in c.get("train", {}).values() for s in v]
        tr = {"n": len(train), "ttft_p50": q([s.get("ttft_ms", np.nan) for s in train], .5) if train else float("nan"), "errors": sum(1 for s in train if s.get("error"))}
        row = {"feat0": c["feat0"], "feat1": c["feat1"], "B": c["B"], "rep": c["repeat"], "itl": itl, "step": sp, "train": tr}
        rows.append(row)
        print(f"| {c['feat0']} | {c['feat1']} | {c['B']} | {c['repeat']} | {fmt(itl[0]['p50'])}/{fmt(itl[0]['p95'])} | {fmt(itl[1]['p50'])}/{fmt(itl[1]['p95'])} | {itl[0]['errors']}/{itl[1]['errors']} | "
              f"{fmt(sp[0]['period_p50'])} / {fmt(sp[0]['cuda_p50'])} | {fmt(sp[1]['period_p50'])} / {fmt(sp[1]['cuda_p50'])} | {fmt(sp[0]['tokens_p50'], 0)}/{fmt(sp[1]['tokens_p50'], 0)} | {tr['n']} / {fmt(tr['ttft_p50'], 0)}{' err ' + str(tr['errors']) if tr['errors'] else ''} |")
    # pool by (feat0, feat1, B)
    g = defaultdict(list)
    for x in rows:
        g[(x["feat0"], x["feat1"], x["B"])].append(x)
    pooled = {k: {f"itl{r}_{p}": float(np.nanmedian([x["itl"][r][p] for x in xs])) for r in (0, 1) for p in ("p50", "p95")} for k, xs in g.items()}
    for k, xs in g.items():
        pooled[k]["period0"] = float(np.nanmedian([x["step"][0]["period_p50"] for x in xs])); pooled[k]["period1"] = float(np.nanmedian([x["step"][1]["period_p50"] for x in xs]))
    print("\n### contagion on the plain rank (pooled medians over repeats; ratio vs plain|plain on the same rank)\n")
    print("| feature | B | placement | plain rank | plain-rank ITL p50 base → with feature (ratio) | p95 base → feature (ratio) | feature-rank ITL p50 | feature-rank p95 | step period plain rank base → feature |")
    print("| --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- |")
    summary = []
    for (f0, f1, B), v in sorted(pooled.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        if f0 == "plain" and f1 == "plain":
            continue
        base = pooled.get(("plain", "plain", B))
        if base is None:
            continue
        feat, plain_rank, feat_rank = (f0, 1, 0) if f1 == "plain" else (f1, 0, 1)
        b50, b95 = base[f"itl{plain_rank}_p50"], base[f"itl{plain_rank}_p95"]
        w50, w95 = v[f"itl{plain_rank}_p50"], v[f"itl{plain_rank}_p95"]
        r50, r95 = w50 / b50, w95 / b95
        summary.append({"feature": feat, "B": B, "placement": f"{f0}|{f1}", "plain_rank": plain_rank, "base_p50": b50, "with_p50": w50, "ratio_p50": r50, "base_p95": b95, "with_p95": w95, "ratio_p95": r95,
                        "feat_rank_p50": v[f"itl{feat_rank}_p50"], "feat_rank_p95": v[f"itl{feat_rank}_p95"]})
        print(f"| {feat} | {B} | {f0}\\|{f1} | r{plain_rank} | {fmt(b50)} → {fmt(w50)} ({fmt(r50, 2)}) | {fmt(b95)} → {fmt(w95)} ({fmt(r95, 2)}) | {fmt(v[f'itl{feat_rank}_p50'])} | {fmt(v[f'itl{feat_rank}_p95'])} | "
              f"{fmt(base[f'period{plain_rank}'])} → {fmt(v[f'period{plain_rank}'])} |")
    if summary:
        worst = max(summary, key=lambda s: s["ratio_p50"] if not np.isnan(s["ratio_p50"]) else 0)
        print(f"\nlargest plain-rank ITL p50 contagion: {worst['feature']} B={worst['B']} {worst['placement']}: ×{worst['ratio_p50']:.2f} (p95 ×{worst['ratio_p95']:.2f}); gate <5% kill / 10% alive / 20% strong / 50% priority")
    args.output.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
