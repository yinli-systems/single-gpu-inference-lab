#!/usr/bin/env python3
"""Task 26 real-mover analysis: vLLM's native CPU offloading connector as the KV mover on rank 0.

Input: two results dirs from sbatch_c26kv.sh (measure_kvoffload.py) — the connector-ON arm
(OFFLOAD_GIB>0) and the OFF arm (OFFLOAD_GIB=0, same workload: rank 0's store/load trains then run
as plain prefills / recomputes). Each holds kvoffload.json and trace/step.jsonl.dp{0,1} (runner
step trace: `t` host clock, `num_tokens`, `cuda_ms`).

Per cell: rank-1 (no KV movement) ITL p50/p95 in the window, rank-0 train count / TTFT / GiB moved,
per-rank step period and CUDA time, and rank 1's CUDA time split by what rank 0 ran in the same
lockstep step (prefill chunk = rank-0 num_tokens >= chunk_min, else decode-only). The split is the
decomposition: the synchronized prefill chunk costs the peer in BOTH arms (G1/24 mechanism); the
connector's D2H/H2D copies are the ON-minus-OFF difference on the decode-only steps.
Pooled by (kind, B): median over repeats; ratios ON/OFF per kind and kind/plain within each arm.
Gate (pre-registered): rank-1 p95 ITL +10 % with the connector on (vs off, same workload) = strong.
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


def analyze_arm(d: Path, chunk_min: int):
    rec = json.load(open(d / "kvoffload.json"))
    step = {r: load_jsonl(d / "trace" / f"step.jsonl.dp{r}") for r in (0, 1)}
    tt = {r: np.array([s["t"] for s in step[r]]) for r in (0, 1)}
    rows = []
    for c in rec["cells"]:
        t0, t1 = c["t_window0"], c["t_window1"]
        sel = {r: [step[r][i] for i in np.where((tt[r] >= t0) & (tt[r] <= t1))[0]] for r in (0, 1)}
        bases = sorted(c["itl_window"].keys())  # 8300 = rank 0, 8301 = rank 1
        itl = {r: c["itl_window"][bases[r]] for r in (0, 1)}
        st = {}
        for r in (0, 1):
            s = sel[r]
            st[r] = {"n": len(s), "period_p50": q(np.diff([x["t"] for x in s]) * 1e3, .5) if len(s) > 1 else float("nan"),
                     "cuda_p50": q([x["cuda_ms"] for x in s], .5), "cuda_p95": q([x["cuda_ms"] for x in s], .95),
                     "tokens_p50": q([x["num_tokens"] for x in s], .5)}
        # rank 1's CUDA time split by the rank-0 step that ran in lockstep with it (nearest in host time)
        t0s = np.array([x["t"] for x in sel[0]]) if sel[0] else np.array([])
        n0 = np.array([x["num_tokens"] for x in sel[0]]) if sel[0] else np.array([])
        chunk, dec = [], []
        for x in sel[1]:
            if len(t0s):
                j = int(np.argmin(np.abs(t0s - x["t"])))
                (chunk if n0[j] >= chunk_min else dec).append(x["cuda_ms"])
        st[1]["cuda_p50_r0chunk"] = q(chunk, .5)
        st[1]["cuda_p50_r0dec"] = q(dec, .5)
        st[1]["frac_r0chunk"] = len(chunk) / max(len(chunk) + len(dec), 1)
        st[0]["chunk_steps"] = int((n0 >= chunk_min).sum()) if len(n0) else 0
        tr = c["train"]
        rows.append({"kind": c["kind"], "B": c["B"], "rep": c["repeat"], "r1_p50": itl[1].get("p50"), "r1_p95": itl[1].get("p95"), "r1_err": itl[1].get("errors"),
                     "r0_p50": itl[0].get("p50"), "train_n": tr["n"], "train_err": tr["errors"], "ttft_p50": tr["ttft_p50_ms"], "kv_GiB": tr["kv_GiB_moved_est"],
                     "kv_GBps": tr["kv_GiB_moved_est"] * 2**30 / 1e9 / (t1 - t0), "step": st})
    return rec, rows


def pool(rows):
    g = defaultdict(list)
    for x in rows:
        g[(x["kind"], x["B"])].append(x)
    out = {}
    for k, xs in g.items():
        med = lambda f: float(np.nanmedian([f(x) if f(x) is not None else np.nan for x in xs]))  # noqa: E731
        out[k] = {"n": len(xs), "r1_p50": med(lambda x: x["r1_p50"]), "r1_p95": med(lambda x: x["r1_p95"]), "train_n": med(lambda x: x["train_n"]),
                  "ttft_p50": med(lambda x: x["ttft_p50"]), "kv_GiB": med(lambda x: x["kv_GiB"]), "kv_GBps": med(lambda x: x["kv_GBps"]),
                  "period1": med(lambda x: x["step"][1]["period_p50"]), "cuda1": med(lambda x: x["step"][1]["cuda_p50"]), "cuda1_p95": med(lambda x: x["step"][1]["cuda_p95"]),
                  "cuda1_r0chunk": med(lambda x: x["step"][1]["cuda_p50_r0chunk"]), "cuda1_r0dec": med(lambda x: x["step"][1]["cuda_p50_r0dec"]),
                  "frac_r0chunk": med(lambda x: x["step"][1]["frac_r0chunk"]), "cuda0": med(lambda x: x["step"][0]["cuda_p50"]), "tokens0": med(lambda x: x["step"][0]["tokens_p50"]),
                  "chunk_steps0": med(lambda x: x["step"][0]["chunk_steps"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=Path, required=True, help="results dir of the connector-ON arm")
    ap.add_argument("--off", type=Path, required=True, help="results dir of the connector-OFF arm")
    ap.add_argument("--chunk-min", type=int, default=256, help="rank-0 num_tokens at/above which a step counts as a prefill chunk")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    arms = {}
    for name, d in (("on", args.on), ("off", args.off)):
        rec, rows = analyze_arm(d, args.chunk_min)
        arms[name] = {"dir": str(d), "args": rec["args"], "rows": rows, "pooled": pool(rows)}
        print(f"\n### arm {name}: {d.name}\n")
        print("| kind | B | rep | r1 ITL p50/p95 | r0 ITL p50 | train n / err | TTFT p50 | KV GiB (GB/s) | r0 tokens p50 / chunk steps | r1 period / cuda p50 / cuda p95 | r1 cuda p50 when r0 chunk / r0 decode (frac chunk) |")
        print("| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- | --- | --- |")
        for x in rows:
            s = x["step"]
            print(f"| {x['kind']} | {x['B']} | {x['rep']} | {fmt(x['r1_p50'])}/{fmt(x['r1_p95'])} | {fmt(x['r0_p50'])} | {x['train_n']} / {x['train_err']} | {fmt(x['ttft_p50'], 0)} | "
                  f"{x['kv_GiB']:.1f} ({x['kv_GBps']:.2f}) | {fmt(s[0]['tokens_p50'], 0)} / {s[0]['chunk_steps']} | {fmt(s[1]['period_p50'])} / {fmt(s[1]['cuda_p50'])} / {fmt(s[1]['cuda_p95'])} | "
                  f"{fmt(s[1]['cuda_p50_r0chunk'])} / {fmt(s[1]['cuda_p50_r0dec'])} ({s[1]['frac_r0chunk']:.2f}) |")
    on, off = arms["on"]["pooled"], arms["off"]["pooled"]
    print("\n### pooled (median over repeats): connector ON vs OFF, same workload on rank 0; rank 1 runs B plain decode streams and moves no KV\n")
    print("| kind | B | n | r1 ITL p50 off → on (ratio) | r1 ITL p95 off → on (ratio) | r1 cuda p50 off → on | r1 cuda p50 on r0-decode steps off → on (ratio) | r1 cuda p50 on r0-chunk steps off → on | frac chunk off → on | train n off → on | TTFT off → on | KV GB/s on |")
    print("| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: |")
    summary = []
    for k in sorted(set(on) & set(off), key=lambda k: (k[1], k[0])):
        a, b = off[k], on[k]
        e = {"kind": k[0], "B": k[1], "n": min(a["n"], b["n"])}
        for m in ("r1_p50", "r1_p95", "cuda1", "cuda1_r0dec", "cuda1_r0chunk", "frac_r0chunk", "train_n", "ttft_p50", "period1"):
            e[f"{m}_off"], e[f"{m}_on"] = a[m], b[m]
            e[f"{m}_ratio"] = b[m] / a[m] if a[m] else float("nan")
        e["kv_GBps_on"] = b["kv_GBps"]
        summary.append(e)
        print(f"| {k[0]} | {k[1]} | {e['n']} | {fmt(a['r1_p50'])} → {fmt(b['r1_p50'])} ({fmt(e['r1_p50_ratio'], 2)}) | {fmt(a['r1_p95'])} → {fmt(b['r1_p95'])} ({fmt(e['r1_p95_ratio'], 2)}) | "
              f"{fmt(a['cuda1'])} → {fmt(b['cuda1'])} | {fmt(a['cuda1_r0dec'])} → {fmt(b['cuda1_r0dec'])} ({fmt(e['cuda1_r0dec_ratio'], 2)}) | {fmt(a['cuda1_r0chunk'])} → {fmt(b['cuda1_r0chunk'])} | "
              f"{a['frac_r0chunk']:.2f} → {b['frac_r0chunk']:.2f} | {a['train_n']:.0f} → {b['train_n']:.0f} | {fmt(a['ttft_p50'], 0)} → {fmt(b['ttft_p50'], 0)} | {b['kv_GBps']:.2f} |")
    print("\n### within-arm: kind vs plain (same arm, same B) — the cost of rank 0's train on rank 1, connector on or off\n")
    print("| arm | kind | B | r1 ITL p50 plain → kind (ratio) | r1 ITL p95 plain → kind (ratio) | r1 cuda p50 plain → kind |")
    print("| --- | --- | ---: | --- | --- | --- |")
    within = []
    for name, p in (("off", off), ("on", on)):
        for k in sorted(p, key=lambda k: (k[1], k[0])):
            if k[0] == "plain" or ("plain", k[1]) not in p:
                continue
            b, a = p[k], p[("plain", k[1])]
            w = {"arm": name, "kind": k[0], "B": k[1], "r1_p50_ratio": b["r1_p50"] / a["r1_p50"], "r1_p95_ratio": b["r1_p95"] / a["r1_p95"], "cuda1_ratio": b["cuda1"] / a["cuda1"]}
            within.append(w)
            print(f"| {name} | {k[0]} | {k[1]} | {fmt(a['r1_p50'])} → {fmt(b['r1_p50'])} ({fmt(w['r1_p50_ratio'], 2)}) | {fmt(a['r1_p95'])} → {fmt(b['r1_p95'])} ({fmt(w['r1_p95_ratio'], 2)}) | {fmt(a['cuda1'])} → {fmt(b['cuda1'])} ({fmt(w['cuda1_ratio'], 2)}) |")
    st = [e for e in summary if e["kind"] == "store"]
    if st:
        worst = max(st, key=lambda e: e["r1_p95_ratio"] if not np.isnan(e["r1_p95_ratio"]) else 0)
        print(f"\nstore (D2H) cells, connector on vs off: largest rank-1 p95 ratio ×{worst['r1_p95_ratio']:.2f} at B={worst['B']} (p50 ×{worst['r1_p50_ratio']:.2f}; "
              f"r1 cuda on r0-decode steps ×{worst['cuda1_r0dec_ratio']:.2f}); gate: p95 +10 % strong, < 5 % kill — read with the train-count / TTFT change (the connector's own prefill-side cost)")
    args.output.write_text(json.dumps({"arms": {k: {kk: vv for kk, vv in v.items() if kk != "pooled"} | {"pooled": {f"{kk[0]}|B{kk[1]}": vv for kk, vv in v["pooled"].items()}} for k, v in arms.items()},
                                       "summary": summary, "within_arm": within}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
