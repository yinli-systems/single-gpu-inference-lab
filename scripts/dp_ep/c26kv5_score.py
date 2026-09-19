#!/usr/bin/env python3
"""Score the pre-registered Q1–Q4 of the DP2 placement job (sbatch_c26kv5.sh; README "DP2 production-shaped
placement follow-up"). Works on any arm dir that holds kvoffload.json + trace/step.jsonl.dp0 (also the single-arm
dirs of sbatch_c26kv.sh, e.g. results/c26kv-1602608-*), or on a c26kv5 job dir (every sub-dir with kvoffload.json).

Per arm: rank-1 (port 8301) ITL p50/p95 per cell from the token times inside the window; rank-0 chunk steps
(cg NONE, num_tokens >= chunk_min) per store cell (train + window), their cuda p50, the slow fraction (> thresh),
sustained slow runs (analyze_sidecar.eager_runs rule); pooled medians over repeats and the store/plain ratios.
  Q1 (A) slow reproduced : >= 1/5 store cells with chunk p50 > thresh or >= 10 % slow chunk steps, and rank-1 store p95 >= 1.3x
                           the OFF arm's store p95 (same workload: the OFF arm's train is the recompute; 1602608/1602609 = x1.80)
  Q2 (M) / Q3 (B) cured  : < 5 % slow chunk steps and rank-1 store p95 <= 1.2x the OFF arm's store p95
  Q4 (C) baseline clean  : < 5 % slow chunk steps
The store/plain ratio inside one arm is NOT the slow-state signal (a store cell's rank-1 stream runs against a prefill
train on rank 0 = the synchronized-chunk cost, x2.2-3.9 even with the connector off); it is printed for completeness.
usage: c26kv5_score.py <job-or-arm dir> [--thresh 65] [--chunk-min 256] [--json out.json]
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from analyze_sidecar import eager_runs
except ImportError:  # standalone copy
    def eager_runs(steps, th, chunk_min):
        eager = [s for s in steps if str(s["cg_mode"]).endswith("NONE") and s["num_tokens"] >= chunk_min]
        runs = []
        for s in eager:
            if runs and s["t"] - runs[-1][-1]["t"] < 0.5:
                runs[-1].append(s)
            else:
                runs.append([s])
        return [dict(t0=r[0]["t"], t1=r[-1]["t"], n=len(r), cuda_p50=float(np.median([s["cuda_ms"] for s in r])),
                     state="slow" if np.median([s["cuda_ms"] for s in r]) > th else "fast") for r in runs]


def load_jsonl(p):
    try:
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]
    except OSError:
        return []


def q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, p)) if len(a) else float("nan")


def itl_ms(cell, port_suffix):
    """inter-token gaps (ms) of every stream on the port whose both tokens fall inside the window"""
    out = []
    for url, streams in cell["decode"].items():
        if not url.endswith(port_suffix):
            continue
        for st in streams:
            tt = [t for t in st.get("token_times", []) if cell["t_window0"] <= t <= cell["t_window1"]]
            out += [1e3 * (b - a) for a, b in zip(tt, tt[1:])]
    return out


def score_arm(d, thresh, chunk_min):
    kv = json.load(open(os.path.join(d, "kvoffload.json")))
    cells = kv["cells"]
    steps0 = load_jsonl(os.path.join(d, "trace", "step.jsonl.dp0")) or load_jsonl(os.path.join(d, "trace", "step.jsonl"))
    steps0.sort(key=lambda s: s["t"])
    chunks = [s for s in steps0 if str(s["cg_mode"]).endswith("NONE") and s["num_tokens"] >= chunk_min]
    runs = eager_runs(steps0, thresh, chunk_min)
    rows = []
    for c in cells:
        cs = [s for s in chunks if c["t_window0"] <= s["t"] <= c["t_window1"] + 1.0]  # the train runs inside the window
        cm = [s["cuda_ms"] for s in cs]
        rows.append(dict(kind=c["kind"], B=c["B"], repeat=c.get("repeat"), n_chunk=len(cs), chunk_p50=q(cm, 0.5),
                         chunk_slow=float(np.mean([x > thresh for x in cm])) if cm else float("nan"),
                         r1_p50=q(itl_ms(c, ":8301"), 0.5), r1_p95=q(itl_ms(c, ":8301"), 0.95),
                         r0_p50=q(itl_ms(c, ":8300"), 0.5), ttft=c["train"].get("ttft_p50_ms"), train_n=c["train"].get("n")))
    pooled = {}
    for kind in sorted({r["kind"] for r in rows}):
        for B in sorted({r["B"] for r in rows}):
            rr = [r for r in rows if r["kind"] == kind and r["B"] == B]
            if rr:
                pooled[(kind, B)] = dict(n=len(rr), r1_p50=float(np.median([r["r1_p50"] for r in rr])),
                                         r1_p95=float(np.median([r["r1_p95"] for r in rr])),
                                         chunk_p50=float(np.nanmedian([r["chunk_p50"] for r in rr])) if any(r["n_chunk"] for r in rr) else float("nan"),
                                         cells_slow=sum(1 for r in rr if r["chunk_p50"] > thresh))
    all_chunk = [s["cuda_ms"] for s in chunks]
    slow_frac = float(np.mean([x > thresh for x in all_chunk])) if all_chunk else float("nan")
    slow_runs = [r for r in runs if r["state"] == "slow"]
    return dict(rows=rows, pooled=pooled, n_chunk=len(all_chunk), chunk_p50=q(all_chunk, 0.5), slow_frac=slow_frac,
                n_runs=len(runs), n_slow_runs=len(slow_runs), slow_run_s=sum(r["t1"] - r["t0"] for r in slow_runs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--thresh", type=float, default=65.0)
    ap.add_argument("--chunk-min", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = [a.dir] if os.path.exists(os.path.join(a.dir, "kvoffload.json")) else sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(a.dir, "*", "kvoffload.json")))
    out = {}
    for d in arms:
        name = os.path.basename(d.rstrip("/"))
        s = score_arm(d, a.thresh, a.chunk_min)
        out[name] = {k: v for k, v in s.items() if k != "pooled"} | {"pooled": {f"{k[0]} B={k[1]}": v for k, v in s["pooled"].items()}}
        print(f"\n== {name}: rank-0 chunk steps n={s['n_chunk']} cuda p50 {s['chunk_p50']:.1f} ms, slow(> {a.thresh:.0f} ms) fraction {s['slow_frac']:.2f}, "
              f"sustained runs {s['n_runs']} of which slow {s['n_slow_runs']} ({s['slow_run_s']:.0f} s)")
        print(f"   {'cell':14s} {'n':>3s} {'chunk n':>7s} {'chunk p50':>9s} {'slow':>5s} {'r1 p50':>7s} {'r1 p95':>7s} {'r0 p50':>7s} {'ttft':>6s}")
        for r in s["rows"]:
            print(f"   {r['kind']+' B='+str(r['B'])+' r'+str(r['repeat']):14s} {'':>3s} {r['n_chunk']:7d} {r['chunk_p50']:9.1f} {r['chunk_slow']:5.2f} {r['r1_p50']:7.1f} {r['r1_p95']:7.1f} {r['r0_p50']:7.1f} {str(r['ttft'])[:6]:>6s}")
        for k, v in s["pooled"].items():
            print(f"   pooled {k[0]:6s} B={k[1]:<3d} n={v['n']} r1 p50 {v['r1_p50']:.1f} p95 {v['r1_p95']:.1f} chunk p50 {v['chunk_p50']:.1f} slow cells {v['cells_slow']}/{v['n']}")
        verdict = {}
        for B in sorted({k[1] for k in s["pooled"]}):
            st, pl = s["pooled"].get(("store", B)), s["pooled"].get(("plain", B))
            if st and pl:
                verdict[B] = dict(store_over_plain_p95=st["r1_p95"] / pl["r1_p95"], store_over_plain_p50=st["r1_p50"] / pl["r1_p50"],
                                  slow_cells=f"{st['cells_slow']}/{st['n']}", slow_frac=s["slow_frac"])
                print(f"   B={B}: store/plain rank-1 p95 x{verdict[B]['store_over_plain_p95']:.2f} p50 x{verdict[B]['store_over_plain_p50']:.2f} (chunk contagion, not the signal)")
        out[name]["verdict"] = {str(k): v for k, v in verdict.items()}
        out[name]["_pooled_raw"] = s["pooled"]
    # cross-arm scoring against the connector-OFF arm (same node, same cpuset, same workload)
    ref = next((n for n in out if "off" in n.lower()), None)
    if ref and len(out) > 1:
        print(f"\n== Q1-Q4 against the OFF arm {ref} (rank-1 store p95 ratio; chunk p50 ratio; slow fraction)")
        for name, o in out.items():
            for B in sorted({k[1] for k in o["_pooled_raw"]}):
                st, rf = o["_pooled_raw"].get(("store", B)), out[ref]["_pooled_raw"].get(("store", B))
                if not (st and rf):
                    continue
                r95, r50, rc = st["r1_p95"] / rf["r1_p95"], st["r1_p50"] / rf["r1_p50"], st["chunk_p50"] / rf["chunk_p50"]
                sf = o["slow_frac"]
                q1 = (st["cells_slow"] >= max(1, round(st["n"] / 5)) or sf >= 0.10) and r95 >= 1.3
                cured = sf < 0.05 and r95 <= 1.2
                o["verdict"].setdefault(str(B), {}).update(store_p95_vs_off=r95, store_p50_vs_off=r50, chunk_p50_vs_off=rc, Q1_slow_reproduced=bool(q1), Q2Q3_cured=bool(cured), Q4_clean=bool(sf < 0.05))
                print(f"   {name:16s} B={B:<3d} store p95 x{r95:.2f} p50 x{r50:.2f} chunk p50 x{rc:.2f} slow frac {sf:.2f} slow store cells {st['cells_slow']}/{st['n']} | "
                      f"Q1 reproduced {q1} | Q2/Q3 cured {cured} | Q4 clean {sf < 0.05}")
    for o in out.values():
        o.pop("_pooled_raw", None)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
