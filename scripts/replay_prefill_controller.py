#!/usr/bin/env python3
"""Offline trace-replay of deadline-aware prefill budget controllers.

Closed-loop simulation of N long prefill requests (length L, arriving together)
sharing the engine with B decoders, where each step a controller picks a
prefill budget C from {0,64,128,...,8192}; C is split equally over the active
prefill requests (the measured partition geometry; --partition fcfs gives the
vLLM default of filling requests in order).

Counterfactual step cost = an explicitly evaluated "truth" model fit on every
measured prefill step (features: prefill tokens, attention FLOP proxy
sum_i q_i(kv_i+(q_i+1)/2), decode batch, aggregate decode KV, aggregate prefill
KV, padded tokens) plus a residual drawn from that model's empirical residual
pool for the same chunk-size class (bootstrap, several seeds). Every decision
is labelled measured (a traced step with the same prefill count, chunk within
10% and FLOP proxy within 10% exists), interpolated (inside the measured range
of tokens, FLOP proxy and decode KV, and no per-request chunk smaller than any
measured against a deep KV), or extrapolated.

Controllers (all deadline-aware ones pick the largest C whose predicted cost
plus the training residual q95 fits the deadline D):
  fixed-C            every candidate C, reported as a frontier
  ppas-style         pressure rule: C shrinks with aggregate decode KV load,
                     no cost model (P-PAS-like)
  m0-one / m0-multi  aggregate-coordinate ridge (decode batch, decode KV sum,
                     prefill KV sum, prefill tokens, tokens x prefill KV sum)
                     fit on one-prefill / multi-prefill traced steps
  m1-one / m1-multi  M0 + geometry statistics (prefill count, KV max/var,
                     largest chunk, tokens x KV max, decode KV max/var)
  m2-one / m2-multi  physics-structured geometry model (aggregates, prefill
                     count, eager flag, padded tokens, per-request FLOP proxy)
Outputs per (scenario, deadline, controller): prefill-step violation rate,
safe prefill progress (tokens prefilled in non-violating steps per second of
engine time), time to finish all prefills, and decision coverage.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from step_trace_join import load_joined

CANDIDATES = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
LAMBDA = 1e-2


def flops(chunks, depths):
    return sum(q * (kv + (q + 1) / 2) for q, kv in zip(chunks, depths)) / 1e6


def state_row(chunks, depths, gen_reqs, gen_kvs):
    """Build a trace-like row for a hypothetical step."""
    chunks = [c for c in chunks if c > 0]
    depths = depths[: len(chunks)]
    tok = sum(chunks)
    return {"ctx_reqs": len(chunks), "ctx_tokens": tok, "ctx_chunks": chunks, "ctx_depths": depths,
            "ctx_kv_sum": sum(depths), "ctx_kv_max": max(depths, default=0), "ctx_kv_var": float(np.var(depths)) if depths else 0.0,
            "gen_reqs": gen_reqs, "gen_kv_sum": sum(gen_kvs), "gen_kv_max": max(gen_kvs, default=0), "gen_kv_var": float(np.var(gen_kvs)) if gen_kvs else 0.0,
            "total_tokens": tok + gen_reqs, "padded_tokens": tok + gen_reqs if tok + gen_reqs > 128 else int(2 ** np.ceil(np.log2(max(tok + gen_reqs, 1)))),
            "cg_mode": "NONE" if tok + gen_reqs > 128 else "PIECEWISE", "attn_proxy": flops(chunks, depths) * 1e6 + sum(gen_kvs)}


def truth_feats(r):
    return [1, r["ctx_tokens"], flops(r["ctx_chunks"], r["ctx_depths"]), r["gen_reqs"], r["gen_kv_sum"] / 1e3, r["ctx_kv_sum"] / 1e3, r["padded_tokens"]]


def m0_feats(r):
    return [r["gen_reqs"], r["gen_kv_sum"] / 1e4, r["ctx_kv_sum"] / 1e4, r["ctx_tokens"] / 1e3, r["ctx_tokens"] * r["ctx_kv_sum"] / 1e7]


def m1_feats(r):
    return m0_feats(r) + [r["ctx_reqs"], r["ctx_kv_max"] / 1e4, r["ctx_kv_var"] / 1e8, max(r["ctx_chunks"], default=0) / 1e3,
                          r["ctx_tokens"] * r["ctx_kv_max"] / 1e7, r["gen_kv_max"] / 1e4, r["gen_kv_var"] / 1e8]


def m2_feats(r):
    """Physics-structured: aggregates without the tokens x KV-sum interaction,
    prefill count, execution flags, and the per-request attention-work proxy."""
    return m0_feats(r)[:4] + [r["ctx_reqs"], 1.0 if r["cg_mode"] == "NONE" else 0.0, r["padded_tokens"] / 1e3, r["attn_proxy"] / 1e6]


class Ridge:
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); self.mu, self.sd = X.mean(0), X.std(0); self.sd[self.sd == 0] = 1
        Z = (X - self.mu) / self.sd
        self.w = np.linalg.solve(Z.T @ Z + LAMBDA * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean())); self.b = float(y.mean())
        self.q95 = float(np.quantile(y - self.predict(X), .95))
        return self

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.w + self.b


class Truth:
    def __init__(self, rows):
        X = np.array([truth_feats(r) for r in rows]); y = np.array([r["cuda_ms"] for r in rows])
        self.w, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ self.w
        self.pools = {}
        for r, e in zip(rows, res):
            self.pools.setdefault(self.cls(r["ctx_tokens"]), []).append(e)
        self.pools = {k: np.array(v) for k, v in self.pools.items()}
        self.resid_p95 = float(np.quantile(res, .95)); self.mae = float(np.abs(res).mean())
        self.rows = rows
        self.tok_rng = (min(r["ctx_tokens"] for r in rows), max(r["ctx_tokens"] for r in rows))
        self.fl_rng = (0.0, max(flops(r["ctx_chunks"], r["ctx_depths"]) for r in rows))
        self.gkv_rng = (min(r["gen_kv_sum"] for r in rows), max(r["gen_kv_sum"] for r in rows))
        # smallest per-request chunk ever measured against a deep (>=1k) KV: smaller
        # per-request chunks are extrapolations even when the step total is in range
        deep = [max(r["ctx_chunks"]) for r in rows if r["ctx_kv_max"] >= 1024]
        self.min_deep_chunk = float(np.quantile(deep, .05)) if deep else 0.0  # 5th pct: ignores request-tail chunks
        self.index = {}
        for r in rows:
            self.index.setdefault(r["ctx_reqs"], []).append((r["ctx_tokens"], flops(r["ctx_chunks"], r["ctx_depths"])))
        self.index = {k: np.array(v) for k, v in self.index.items()}

    @staticmethod
    def cls(tok):
        return 0 if tok < 384 else 1 if tok < 768 else 2 if tok < 1536 else 3

    def cost(self, r, rng):
        base = float(np.dot(self.w, truth_feats(r)))
        pool = self.pools.get(self.cls(r["ctx_tokens"]), np.concatenate(list(self.pools.values())))
        return base + float(rng.choice(pool))

    def coverage(self, r):
        fl = flops(r["ctx_chunks"], r["ctx_depths"])
        idx = self.index.get(r["ctx_reqs"])
        if idx is not None:
            close = (np.abs(idx[:, 0] - r["ctx_tokens"]) <= 0.1 * max(r["ctx_tokens"], 1)) & (np.abs(idx[:, 1] - fl) <= 0.1 * max(fl, 1e-6))
            if close.any():
                return "measured"
        if (self.tok_rng[0] <= r["ctx_tokens"] <= self.tok_rng[1] and fl <= self.fl_rng[1] and self.gkv_rng[0] <= r["gen_kv_sum"] <= self.gkv_rng[1]
                and (r["ctx_kv_max"] < 1024 or max(r["ctx_chunks"]) >= self.min_deep_chunk)):
            return "interpolated"
        return "extrapolated"


def partition(C, remaining, mode):
    n = sum(1 for x in remaining if x > 0)
    if C == 0 or n == 0:
        return [0] * len(remaining)
    if mode == "equal":
        share = C // n
        return [min(x, share) for x in remaining]
    out, left = [], C
    for x in remaining:
        take = min(x, left); out.append(take); left -= take
    return out


def simulate(controller, truth, N, L, B, gen_kv0, D, mode, rng, max_steps=20000):
    remaining = [L] * N; depths = [0] * N; gen_kvs = list(gen_kv0)
    t_ms = 0.0; viol = 0; safe_tokens = 0; pre_steps = 0; cov = {"measured": 0, "interpolated": 0, "extrapolated": 0}
    steps = 0
    while any(x > 0 for x in remaining) and steps < max_steps:
        steps += 1
        C = controller(remaining, depths, B, gen_kvs, D)
        chunks = partition(C, remaining, mode)
        active = [i for i, c in enumerate(chunks) if c > 0]
        r = state_row([chunks[i] for i in active], [depths[i] for i in active], B, gen_kvs)
        if r["ctx_tokens"] == 0:
            c = 13.2 + 0.2 * sum(gen_kvs) / 1e3  # decode-only step (measured: 13.2 ms @2k, 19.6 ms @35k aggregate KV)
            t_ms += c
        else:
            c = truth.cost(r, rng); t_ms += c; pre_steps += 1
            cov[truth.coverage(r)] += 1
            if c > D:
                viol += 1
            else:
                safe_tokens += r["ctx_tokens"]
            for i in active:
                depths[i] += chunks[i]; remaining[i] -= chunks[i]
        gen_kvs = [k + 1 for k in gen_kvs]
    return {"steps": steps, "prefill_steps": pre_steps, "violations": viol, "violation_rate": viol / max(pre_steps, 1),
            "safe_tokens_per_s": safe_tokens / (t_ms / 1e3), "finish_ms": t_ms, "coverage": cov, "finished": all(x <= 0 for x in remaining)}


def make_controllers(truth_rows_one, truth_rows_multi, mode):
    def fit(rows, feats):
        return Ridge().fit([feats(r) for r in rows], [r["cuda_ms"] for r in rows])
    models = {"m0-one": (fit(truth_rows_one, m0_feats), m0_feats), "m1-one": (fit(truth_rows_one, m1_feats), m1_feats), "m2-one": (fit(truth_rows_one, m2_feats), m2_feats),
              "m0-multi": (fit(truth_rows_multi, m0_feats), m0_feats), "m1-multi": (fit(truth_rows_multi, m1_feats), m1_feats), "m2-multi": (fit(truth_rows_multi, m2_feats), m2_feats)}

    def deadline_ctl(model, feats):
        def ctl(remaining, depths, B, gen_kvs, D):
            best = 0
            for C in CANDIDATES:
                chunks = partition(C, remaining, mode)
                act = [i for i, c in enumerate(chunks) if c > 0]
                if not act:
                    continue
                r = state_row([chunks[i] for i in act], [depths[i] for i in act], B, gen_kvs)
                if model.predict([feats(r)])[0] + model.q95 <= D:
                    best = C
            return best
        return ctl

    ctls = {f"fixed-{C}": (lambda C: (lambda remaining, depths, B, gen_kvs, D: C))(C) for C in CANDIDATES if C > 0}

    def ppas(remaining, depths, B, gen_kvs, D):
        load = sum(gen_kvs) / 65536.0  # aggregate decode KV as a fraction of a nominal 64k budget
        cap = 8192 * max(0.0, 1.0 - load)
        return max([C for C in CANDIDATES if C <= cap], default=64)
    ctls["ppas-style"] = ppas
    for name, (model, feats) in models.items():
        ctls[name] = deadline_ctl(model, feats)
    export = {k: {"features": "m0" if feats is m0_feats else "m1" if feats is m1_feats else "m2", "mu": m.mu.tolist(), "sd": m.sd.tolist(),
                  "w": m.w.tolist(), "b": m.b, "q95": m.q95} for k, (m, feats) in models.items()}
    return ctls, export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--partition", default="equal", choices=["equal", "fcfs"])
    ap.add_argument("--deadlines", default="50,100,200")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--export-models", type=Path, help="directory: write <name>.json per fitted controller model (for the live env-gated scheduler)")
    args = ap.parse_args()

    rows = []
    for d in args.trace_dir:
        for f in sorted(glob.glob(str(d / "*.jsonl"))):
            if f.endswith(".steps.jsonl") or "graph-" in f:
                continue
            rs, _ = load_joined(Path(f))
            rows += [dict(cell=Path(f).stem, **r) for r in rs if r["ctx_tokens"] > 0 and np.isfinite(r["cuda_ms"])]
    truth = Truth(rows)
    one = [r for r in rows if r["ctx_reqs"] == 1 and not (r["cell"].startswith("part") and "x" in r["cell"] and not r["cell"].startswith(("part-1x", "part2-1x")))]
    multi = [r for r in rows if r["ctx_reqs"] >= 2 and r["cell"].startswith("part")]
    ctls, minfo = make_controllers(one, multi, args.partition)
    print(f"truth model on {len(rows)} prefill steps: w={np.round(truth.w, 3).tolist()} MAE {truth.mae:.1f} ms, resid q95 {truth.resid_p95:.1f} ms; "
          f"controllers fit on {len(one)} one-prefill / {len(multi)} multi-prefill steps; margins { {k: round(v['q95'], 2) for k, v in minfo.items()} }")
    report = {"truth": {"w": truth.w.tolist(), "mae": truth.mae, "resid_p95": truth.resid_p95, "n": len(rows)}, "models": minfo, "partition": args.partition, "scenarios": {}}
    if args.export_models:
        args.export_models.mkdir(parents=True, exist_ok=True)
        for k, v in minfo.items():
            (args.export_models / f"{k}.json").write_text(json.dumps(v) + "\n")
    gen_kv0 = [2048 + 256 * i for i in range(args.B)]
    for N in (1, 2, 4, 8):
        for D in [float(x) for x in args.deadlines.split(",")]:
            key = f"N{N}-L{args.L}-B{args.B}-D{int(D)}"
            print(f"\n## {key}  ({N} prefill requests of {args.L} tokens, {args.B} decoders, deadline {int(D)} ms, partition {args.partition})")
            print("| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |")
            print("| --- | ---: | ---: | ---: | --- |")
            res = {}
            for name, ctl in ctls.items():
                runs = [simulate(ctl, truth, N, args.L, args.B, gen_kv0, D, args.partition, np.random.default_rng(s)) for s in range(args.seeds)]
                cov = {k: sum(r["coverage"][k] for r in runs) for k in ("measured", "interpolated", "extrapolated")}
                tot = max(sum(cov.values()), 1)
                agg = {"violation_rate": float(np.mean([r["violation_rate"] for r in runs])), "safe_tokens_per_s": float(np.mean([r["safe_tokens_per_s"] for r in runs])),
                       "finish_ms": float(np.mean([r["finish_ms"] for r in runs])), "coverage": {k: v / tot for k, v in cov.items()}, "finished": all(r["finished"] for r in runs)}
                res[name] = agg
                print(f"| {name} | {agg['violation_rate']*100:.1f}% | {agg['safe_tokens_per_s']:.0f} | {agg['finish_ms']:.0f} | "
                      f"{agg['coverage']['measured']*100:.0f} / {agg['coverage']['interpolated']*100:.0f} / {agg['coverage']['extrapolated']*100:.0f} |")
            report["scenarios"][key] = res
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
