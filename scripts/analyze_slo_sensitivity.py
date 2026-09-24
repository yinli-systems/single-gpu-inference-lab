#!/usr/bin/env python3
"""SLO sensitivity of live trace-replay goodput (pre-registration addendum 8 F).

For every condition (mean over repeats): goodput at TTFT in {2, 5, 10} s x TPOT in {50, 100, 200} ms,
recomputed from the per-request records of replay_trace_serving.py; and, per run set, the best arm
and the rank of each controller arm under each SLO pair. The primary SLO stays TTFT <= 5 s,
TPOT <= 100 ms; everything else is sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TTFT = (2.0, 5.0, 10.0)
TPOT = (0.05, 0.1, 0.2)


def goodput(reqs, window_s, a, b):
    ok = [r for r in reqs if "error" not in r]
    return sum(1 for r in ok if r["ttft_s"] <= a and (r["tpot_s"] or 0) <= b) / window_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for d in args.dir:
        conds = defaultdict(list)
        for jf in sorted(d.glob("*-r*.json")):
            m = re.match(r"(.+)-r(\d+)$", jf.stem)
            if not m:
                continue
            rep = json.load(open(jf))
            if rep.get("result_type") != "trace_replay_serving":
                continue
            window = rep["args"]["window_s"]
            conds[m.group(1)].append({f"{a:g}s/{b * 1000:g}ms": goodput(rep["requests"], window, a, b) for a in TTFT for b in TPOT})
        table = {c: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for c, rs in conds.items()}
        out[d.name] = table
        keys = list(next(iter(table.values())))
        print(f"## {d.name}: best arm per SLO pair (goodput req/s)")
        for k in keys:
            ranked = sorted(table, key=lambda c: -table[c][k])
            print(f"  {k:12s} best {ranked[0]:26s} {table[ranked[0]][k]:.2f} | " +
                  ", ".join(f"{c}={table[c][k]:.2f}" for c in ranked if "ctl" in c or "ppas" in c))
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
