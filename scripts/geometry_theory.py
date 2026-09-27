#!/usr/bin/env python3
"""Marginal-only representations, the coupling range, and machine checks of the §3 statements.

A prefill step is a set of request-level pairs {(q_i, k_i)} (chunk, cached depth) plus a decode state
and an execution context. Its marginals are the multiset of chunks and the multiset of depths,
separately. A representation phi is *marginal-only* if phi(s) = phi(s') whenever s and s' have the
same marginals, decode state and context, i.e. it is invariant when the depths are permuted
independently of the chunks. The coupling term C = sum_i q_i k_i is not: by the rearrangement
inequality it ranges exactly over [C_min, C_max], sorted-opposite to sorted-same.

Checks (all offline, on checked-in data):
  1. rearrangement: on random integer instances (n <= 7) the brute-force min/max over all
     permutations equals coupling_range(); C_min < C_max unless all chunks or all depths are equal;
  2. every observed multi-prefill step of the eleven shape datasets has C_min <= C_obs <= C_max;
  3. each representation used in the repository is tested for invariance under re-pairing of the observed
     depths (the same-order and opposite-order pairings, which change C whenever the step is not
     degenerate, plus random permutations), and the result is compared with its declared class;
  4. the two states of every pairing-swap configuration have identical marginals.

  python scripts/geometry_theory.py [--perms 5] [--output out.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def coupling(q, k):
    return sum(a * b for a, b in zip(q, k))


def coupling_range(q, k):
    """(C_min, C_max) over all pairings of the chunk multiset q with the depth multiset k."""
    qs, ks = sorted(q), sorted(k)
    return coupling(qs, ks[::-1]), coupling(qs, ks)


def g_pair(q, k):
    """Normalized pairing ambiguity (C_max - C_min) / mean(C_max, C_min); 0 when C_max + C_min = 0."""
    lo, hi = coupling_range(q, k)
    return 0.0 if hi + lo == 0 else (hi - lo) / ((hi + lo) / 2)


def self_work(q):
    return sum(c * (c + 1) / 2 for c in q)


def permute_row(r, perm):
    """The same step with depths re-paired to chunks by `perm`; every marginal field is unchanged and
    attn_proxy (= sum q(k + q/2) + decode KV) is recomputed for the new pairing."""
    q, k = r["ctx_chunks"], r["ctx_depths"]
    k2 = [k[j] for j in perm]
    s = dict(r)
    s["ctx_depths"] = k2
    s["attn_proxy"] = r["attn_proxy"] - coupling(q, k) + coupling(q, k2)
    return s


def representations():
    import analyze_learned_baseline as LB
    import analyze_published_predictors as PP
    import analyze_step_cost_v2 as A
    import analyze_split_term as ST
    m2n = lambda r: [x for i, x in enumerate(A.features(r, 2)) if i != 2]
    vidur = lambda r: [sum(r["ctx_depths"]), round(sum(c * c for c in r["ctx_chunks"]) ** 0.5) ** 2]
    # name -> (feature function, declared marginal-only)
    return {
        "M0 (aggregate ridge)": (lambda r: A.features(r, 0), True),
        "M1 (M0 + geometry statistics)": (lambda r: A.features(r, 1), True),
        "LPRS-style MLP, 16 features": (LB.mlp_feats, True),
        "LLMVisor formula": (PP.llmvisor, True),
        "Vidur prefill-attention key": (vidur, True),
        "M2 (published)": (lambda r: A.features(r, 2), False),
        "M2n": (m2n, False),
        "M2s (split cross/self)": (lambda r: ST.m2s(ST.add_split(dict(r))), False),
    }


def check_rearrangement(trials=2000, seed=0):
    rnd = random.Random(seed)
    worst = 0
    for _ in range(trials):
        n = rnd.randint(2, 7)
        q = [rnd.randint(1, 20) for _ in range(n)]
        k = [rnd.randint(0, 20) for _ in range(n)]
        vals = [coupling(q, [k[j] for j in p]) for p in itertools.permutations(range(n))]
        lo, hi = coupling_range(q, k)
        assert (min(vals), max(vals)) == (lo, hi), (q, k)
        degenerate = len(set(q)) == 1 or len(set(k)) == 1
        assert (lo == hi) == degenerate, (q, k)
        worst = max(worst, n)
    return {"trials": trials, "max_n": worst, "ok": True}


def main():
    import geometry_dataset as G
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=5, help="random re-pairings per multi-prefill step")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    out = {"rearrangement": check_rearrangement()}
    print("1. rearrangement bounds exact on", out["rearrangement"]["trials"], "brute-force instances")

    reps = representations()
    inv = {name: {"declared_marginal_only": dm, "steps": 0, "changed": 0} for name, (f, dm) in reps.items()}
    bounds = {"steps": 0, "violations": 0}
    rnd = random.Random(1)
    for ds in G.DATASETS:
        cells, _ = G.load(ds)
        for rows, _meta in cells.values():
            for r in rows:
                q, k = r["ctx_chunks"], r["ctx_depths"]
                if len(q) < 2:
                    continue
                lo, hi = coupling_range(q, k)
                bounds["steps"] += 1
                bounds["violations"] += not (lo <= coupling(q, k) <= hi)
                order = sorted(range(len(q)), key=lambda i: q[i])
                by_k = sorted(range(len(k)), key=lambda i: k[i])
                same, opposite = [0] * len(q), [0] * len(q)
                for rank, i in enumerate(order):     # re-pair depths with chunks in the same / opposite order
                    same[i], opposite[i] = by_k[rank], by_k[len(k) - 1 - rank]
                perms = [same, opposite] + [rnd.sample(range(len(k)), len(k)) for _ in range(args.perms)]
                degenerate = len(set(q)) == 1 or len(set(k)) == 1
                bounds["degenerate"] = bounds.get("degenerate", 0) + degenerate
                for name, (f, _dm) in reps.items():
                    base = f(r)
                    changed = any(f(permute_row(r, p)) != base for p in perms)
                    inv[name]["steps"] += 1
                    inv[name]["changed"] += changed
                    inv[name]["changed_on_degenerate"] = inv[name].get("changed_on_degenerate", 0) + (changed and degenerate)
                    inv[name]["unchanged_non_degenerate"] = inv[name].get("unchanged_non_degenerate", 0) + (not changed and not degenerate)
    out["observed_bounds"] = bounds
    print(f"2. {bounds['steps']} observed multi-prefill steps ({bounds.get('degenerate', 0)} with all chunks or all depths equal), "
          f"C outside [C_min, C_max]: {bounds['violations']}")
    print("3. invariance under independent re-pairing of observed depths (a representation is marginal-only iff it never changes)")
    for name, v in inv.items():
        v["measured_marginal_only"] = v["changed"] == 0
        v["agrees_with_declaration"] = v["measured_marginal_only"] == v["declared_marginal_only"]
        print(f"   {name:32s} declared {'marginal-only' if v['declared_marginal_only'] else 'pairing-aware':14s} "
              f"changed on {v['changed']:5d}/{v['steps']} steps (unchanged non-degenerate {v['unchanged_non_degenerate']}, "
              f"changed degenerate {v['changed_on_degenerate']}) -> {'agrees' if v['agrees_with_declaration'] else 'DISAGREES'}")
    out["invariance"] = inv

    swaps = []
    for plan in sorted((Path(__file__).resolve().parents[1] / "benchmarks/results").glob("*/raw/**/pairswap-*.plan.json")):
        p = json.load(open(plan))
        (ka, kb), (qa, qb) = p["k"], p["q"]
        a, b = ([qa, qb], [ka, kb]), ([qb, qa], [ka, kb])
        same = sorted(a[0]) == sorted(b[0]) and sorted(a[1]) == sorted(b[1])
        swaps.append({"plan": str(plan.relative_to(plan.parents[3])), "same_marginals": same,
                      "dC_M": (coupling(*a) - coupling(*b)) / 1e6, "range_M": [x / 1e6 for x in coupling_range([qa, qb], [ka, kb])]})
    out["pairing_swaps"] = swaps
    print(f"4. pairing-swap configurations with identical marginals: {sum(s['same_marginals'] for s in swaps)}/{len(swaps)}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")
    ok = out["rearrangement"]["ok"] and bounds["violations"] == 0 and all(v["agrees_with_declaration"] for v in inv.values()) \
        and all(s["same_marginals"] for s in swaps)
    print("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
