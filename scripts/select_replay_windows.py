#!/usr/bin/env python3
"""Window-selection rule of pre-registration addendum 8 D, applied mechanically.

A window is a contiguous WINDOW_S of replayed (rate-scaled) time with >= MIN_REQ requests, no
arrival gap > MAX_GAP_S, simulated `default` (2048, no cap) TTFT p50 < 1 s and < 10% of simulated
steps outside the clock range. No geometry statistic or controller outcome is consulted. Scans
start offsets in steps of STEP_S (trace time) and returns the earliest window plus the next
non-overlapping ones, and checks the primary windows chosen by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_geometry_prevalence as S  # noqa: E402

WINDOW_S, MIN_REQ, MAX_GAP_S = 240.0, 100, 20.0


def check(reqs, clock, rng, off, scale, max_gap=MAX_GAP_S):
    seg = [((t - off) / scale, p, o, h) for t, p, o, h in reqs if 0 <= (t - off) / scale < WINDOW_S]
    if len(seg) < MIN_REQ:
        return False, {"requests": len(seg)}
    ts = sorted(x[0] for x in seg)
    gap = max([ts[0]] + [b - a for a, b in zip(ts, ts[1:])] + [WINDOW_S - ts[-1]])
    if gap > max_gap:
        return False, {"requests": len(seg), "max_gap_s": gap}
    st = S.simulate(seg, budget=2048, threshold=0, max_seqs=256, kv_capacity=400000, slope=3.3,
                    clock=clock, clock_range=rng, rate_scale=1)
    sm = S.summarize(st)
    ok = sm["ttft_s_p50_p99"][0] < 1.0 and sm["beyond_clock_share"] < 0.10
    return ok, {"requests": len(seg), "max_gap_s": gap, "sim_ttft_p50": sm["ttft_s_p50_p99"][0], "beyond_clock": sm["beyond_clock_share"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--cell", nargs="+", required=True, help="NAME=KIND:PATH[@JITTER]:SCALE:PRIMARY_OFFSET_S[:MAX_GAP_S]")
    ap.add_argument("--extra", type=int, default=2)
    ap.add_argument("--step-s", type=float, default=30.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pre, dec, rng = S.build_clock(args.steps)
    out = {}
    for spec in args.cell:
        name, rest = spec.split("=", 1)
        kind, rest = rest.split(":", 1)
        parts = rest.split(":")
        mg = MAX_GAP_S
        if len(parts) >= 4 and parts[-3].replace(".", "").isdigit():   # optional MAX_GAP_S (relaxed rule, reported)
            mg = float(parts[-1]); parts = parts[:-1]
        path, scale, prim = ":".join(parts[:-2]), float(parts[-2]), float(parts[-1])
        path, _, jit = path.partition("@")
        reqs, _ = S.load_trace(kind, Path(path), 20000 if kind == "burstgpt" else 10**9, 32768, float(jit or 0))
        span = reqs[-1][0]
        ok_p, st_p = check(reqs, (pre, dec), rng, prim, scale, mg)
        W = WINDOW_S * scale
        earliest, robust, off = None, [], 0.0
        while off + W <= span and len(robust) < args.extra:
            clash = abs(off - prim) < W or any(abs(off - o) < W for o, _ in robust)
            if not clash or earliest is None:
                ok, st = check(reqs, (pre, dec), rng, off, scale, mg)
                if ok:
                    earliest = off if earliest is None else earliest
                    if not clash:
                        robust.append((off, st))
            off += args.step_s
        found = [(earliest, None)] if earliest is not None else []
        out[name] = {"scale": scale, "max_gap_s": mg, "primary_offset_s": prim, "primary_meets_rule": ok_p, "primary_stats": st_p,
                     "earliest_rule_window_offset_s": found[0][0] if found else None,
                     "robustness_windows": [{"offset_s": o, **s} for o, s in robust]}
        print(f"{name}: primary offset {prim:g}s meets rule: {ok_p} {st_p} | earliest rule window {found[0][0] if found else None} | "
              f"robustness offsets {[o for o, _ in robust]}", flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
