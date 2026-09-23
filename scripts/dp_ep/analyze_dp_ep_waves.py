#!/usr/bin/env python3
"""M1a analysis: does a heterogeneous DP/EP wave produce rank arrival skew that
lands on the collective critical path?

Inputs: a results dir with waves.json (client) and trace/ep.{0,1}.jsonl,
trace/step.jsonl.dp{0,1}, trace/iter.jsonl.dp{0,1}.

Per cell window [t_window0, t_window1] (client monotonic clock; the EP trace's
t_host_ns is the same node's monotonic clock in ns, and the client runs on the
same node inside the Slurm job):
  waves      EP collective calls in the window, joined across ranks by seq
  sizes      tokens each rank brought (from dp_metadata), classified as
             prefill-ish (>= 64 tokens) / decode-ish
  arrival    host arrival skew per collective = t_host(r0) - t_host(r1)
  coll       per-rank collective CUDA ms (includes waiting for the peer)
  step       per-rank model-step CUDA ms from the runner trace (rank-suffixed)
  itl        decoder inter-token latency per rank from waves.json
Reports per cell: skew p50/p95 (abs), per-rank collective ms, per-rank step ms,
decoder ITL p50/p95, prefill TTFT; then the pooled relation between |skew| and
the slower rank's collective time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    waves = json.load(open(args.dir / "waves.json"))
    ep = {r: load_jsonl(args.dir / "trace" / f"ep.{r}.jsonl") for r in (0, 1)}
    step = {r: load_jsonl(args.dir / "trace" / f"step.jsonl.dp{r}") for r in (0, 1)}
    by_seq = {r: {x["seq"]: x for x in ep[r]} for r in (0, 1)}
    ep_t = {r: np.array([x["t_host_ns"] for x in ep[r]]) / 1e9 for r in (0, 1)}
    step_t = {r: np.array([x["t"] for x in step[r]]) for r in (0, 1)}
    print("| mode | cell | B | L | rep | collectives | wave sizes (r0,r1) top | |arrival skew| p50 / p95 ms | coll r0 p50 / p95 | coll r1 | step r0 p50 | step r1 p50 | ITL r0 p50 / p95 | ITL r1 | prefill TTFT |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    out = []
    pooled = []
    for c in waves["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        idx = {r: np.where((ep_t[r] >= t0) & (ep_t[r] <= t1))[0] for r in (0, 1)}
        rows0 = [ep[0][i] for i in idx[0]]
        skews, coll0, coll1, sizes = [], [], [], defaultdict(int)
        for x in rows0:
            y = by_seq[1].get(x["seq"])
            if y is None or y["op"] != x["op"]:
                continue
            sk = (x["t_host_ns"] - y["t_host_ns"]) / 1e6
            skews.append(sk); coll0.append(x["cuda_ms"]); coll1.append(y["cuda_ms"]); sizes[tuple(x["sizes"])] += 1
            pooled.append((abs(sk), max(x["cuda_ms"], y["cuda_ms"]), x["op"]))
        skews = np.array(skews); coll0 = np.array(coll0); coll1 = np.array(coll1)
        st = {}
        for r in (0, 1):
            sel = [step[r][i] for i in np.where((step_t[r] >= t0) & (step_t[r] <= t1))[0]]
            st[r] = np.array([s["cuda_ms"] for s in sel]) if sel else np.array([np.nan])
        itl = {}
        for r, (base, streams) in enumerate(c["decode"].items()):
            gaps = [(b - a) * 1e3 for s in streams for a, b in zip(s["token_times"], s["token_times"][1:]) if t0 <= a <= t1]
            itl[r] = np.array(gaps) if gaps else np.array([np.nan])
        ttft = c["prefill"][0]["ttft_ms"] if c["prefill"] else float("nan")
        top = ", ".join(f"{k}:{v}" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:3])
        rec = {"mode": waves["mode"], "cell": c["cell"], "B": c["B"], "L": c["L"], "rep": c["repeat"], "n_coll": int(len(skews)),
               "skew_abs_p50": float(np.median(np.abs(skews))) if len(skews) else None, "skew_abs_p95": float(np.quantile(np.abs(skews), .95)) if len(skews) else None,
               "coll0_p50": float(np.median(coll0)) if len(coll0) else None, "coll0_p95": float(np.quantile(coll0, .95)) if len(coll0) else None,
               "coll1_p50": float(np.median(coll1)) if len(coll1) else None, "coll1_p95": float(np.quantile(coll1, .95)) if len(coll1) else None,
               "step0_p50": float(np.nanmedian(st[0])), "step1_p50": float(np.nanmedian(st[1])),
               "itl0_p50": float(np.nanmedian(itl.get(0, np.array([np.nan])))), "itl0_p95": float(np.nanquantile(itl.get(0, np.array([np.nan])), .95)),
               "itl1_p50": float(np.nanmedian(itl.get(1, np.array([np.nan])))), "itl1_p95": float(np.nanquantile(itl.get(1, np.array([np.nan])), .95)),
               "ttft_ms": ttft, "sizes_top": top}
        out.append(rec)
        f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
        print(f"| {rec['mode']} | {rec['cell']} | {rec['B']} | {rec['L']} | {rec['rep']} | {rec['n_coll']} | {top} | {f(rec['skew_abs_p50'])} / {f(rec['skew_abs_p95'])} | {f(rec['coll0_p50'])} / {f(rec['coll0_p95'])} | {f(rec['coll1_p50'])} / {f(rec['coll1_p95'])} | {f(rec['step0_p50'])} | {f(rec['step1_p50'])} | {f(rec['itl0_p50'])} / {f(rec['itl0_p95'])} | {f(rec['itl1_p50'])} / {f(rec['itl1_p95'])} | {f(ttft)} |")
    if pooled:
        a = np.array([(s, m) for s, m, _ in pooled])
        bins = [0, 0.5, 1, 2, 5, 10, 20, 50, 1e9]
        print("\n## pooled: |arrival skew| bin → slower rank's collective CUDA ms (p50 / p95), n")
        for lo, hi in zip(bins, bins[1:]):
            sel = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
            if len(sel):
                print(f"| {lo}–{hi if hi < 1e8 else '∞'} ms | {np.median(sel[:, 1]):.2f} / {np.quantile(sel[:, 1], .95):.2f} | {len(sel)} |")
        print(f"corr(|skew|, max collective ms) = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.3f}")
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
