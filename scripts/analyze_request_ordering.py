#!/usr/bin/env python3
"""Request-ordering arms on live traces (addenda 3 and 5 of the 2026-09-27 pre-registration).

Reads the replay JSONs written by replay_trace_serving.py, named <trace>-x<scale>-<arm>-r<rep>.json,
and reports per cell and arm (averaged over repeats): mean / p50 / p99 TTFT, goodput at
TTFT <= 5 s and TPOT <= 100 ms, and the verdicts O1-O4 against fcfs.

  python scripts/analyze_request_ordering.py --runs DIR [DIR ...] --output OUT.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

NAME = re.compile(r"^(?P<trace>[a-z]+)-x(?P<scale>[0-9.]+)-(?P<arm>fcfs|prompt|uncached|cost)-r(?P<rep>\d+)$")
GOOD = "ttft<=5s,tpot<=100ms"


def run_stats(path: Path):
    d = json.load(open(path))
    ok = [r for r in d["requests"] if "error" not in r]
    ttft = sorted(r["ttft_s"] for r in ok)
    q = lambda p: ttft[min(len(ttft) - 1, int(round(p * (len(ttft) - 1))))]
    return {"requests": len(d["requests"]), "errors": len(d["requests"]) - len(ok), "ttft_mean": statistics.mean(ttft),
            "ttft_p50": q(.5), "ttft_p99": q(.99), "goodput": d["summary"]["goodput_rps"][GOOD],
            "offered": d["summary"]["offered_rps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in args.runs:
        for p in sorted(d.glob("*.json")):
            m = NAME.match(p.stem)
            if m:
                cells[(m["trace"], float(m["scale"]))][m["arm"]].append(run_stats(p))
    out = {}
    for (trace, scale), arms in sorted(cells.items()):
        mean = {a: {k: statistics.mean(r[k] for r in rs) for k in rs[0]} | {"repeats": len(rs)} for a, rs in arms.items()}
        cell = {"arms": mean}
        if "fcfs" in mean:
            f = mean["fcfs"]
            rel = {a: {"ttft_mean_vs_fcfs": v["ttft_mean"] / f["ttft_mean"], "goodput_vs_fcfs": v["goodput"] / f["goodput"] if f["goodput"] else None,
                       "ttft_p99_vs_fcfs": v["ttft_p99"] / f["ttft_p99"]} for a, v in mean.items()}
            cell["relative"] = rel
            if trace == "mooncake" and {"uncached", "cost", "prompt"} <= mean.keys():
                cell["O1"] = rel["uncached"]["ttft_mean_vs_fcfs"] <= 0.8 and rel["cost"]["ttft_mean_vs_fcfs"] <= 0.8
                cell["O2"] = abs(mean["cost"]["ttft_mean"] / mean["uncached"]["ttft_mean"] - 1) <= 0.05
                cell["O3"] = mean["prompt"]["ttft_mean"] > mean["uncached"]["ttft_mean"]
                cell["O4_best_goodput_vs_fcfs"] = max(rel[a]["goodput_vs_fcfs"] or 0 for a in ("prompt", "uncached", "cost"))
        out[f"{trace}-x{scale:g}"] = cell
        print(f"## {trace} x{scale:g}")
        for a in ("fcfs", "prompt", "uncached", "cost"):
            if a in mean:
                v = mean[a]
                r = cell.get("relative", {}).get(a, {})
                print(f"   {a:9s} n{v['repeats']} req {v['requests']:.0f} err {v['errors']:.0f} | TTFT mean {v['ttft_mean']:7.3f} p50 {v['ttft_p50']:6.3f} "
                      f"p99 {v['ttft_p99']:7.3f} s | goodput {v['goodput']:.3f}/{v['offered']:.3f} rps"
                      + (f" | vs fcfs: TTFT mean {r['ttft_mean_vs_fcfs']:.2f}x, p99 {r['ttft_p99_vs_fcfs']:.2f}x, goodput {r['goodput_vs_fcfs']:.3f}x" if r else ""))
        for k in ("O1", "O2", "O3", "O4_best_goodput_vs_fcfs"):
            if k in cell:
                print(f"   {k}: {cell[k]}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
