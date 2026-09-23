#!/usr/bin/env python3
"""How often do real request traces produce the prefill geometry that an aggregate step-cost
coordinate misprices?  A trace-driven replay of the vLLM 0.29 V1 scheduler's token-budget rules.

Scheduler rules replayed (vllm/v1/core/sched/scheduler.py, 0.29.0):
  * running requests first, in order: each gets min(remaining, long_prefill_token_threshold if > 0,
    token budget); a decode is 1 token;
  * then waiting requests FCFS while budget > 0 and running < max_num_seqs: prompt minus
    prefix-cache hit, capped the same way (chunked prefill on);
  * a new request is admitted only if its full KV (prompt + output) fits in free KV capacity
    (no preemption modeled: admission simply waits).
Prefix caching: only for traces that carry block hashes (Mooncake `hash_ids`, 512-token blocks);
an LRU over blocks bounded by the KV capacity not held by running requests.

Step clock (to advance simulated time): the M2-noaggkv form fitted on every measured prefill step of
one campaign (`--steps` CSV; in-sample, this is a clock, not a test), and a linear decode-only
model (decode batch, decode KV) fitted on its decode-only steps. The measured range is decode batch
<= 32 and at most 8 prefills; beyond it the clock extrapolates (reported).

Geometry error of a prefill step (the quantity of interest): an aggregate coordinate that is exact
on one-prefill steps charges slope * (sum q)(sum kv) where the engine does slope * sum_i q_i kv_i
(kv_i = KV depth before request i's chunk), so
    geo_err_ms = slope * ((sum q)(sum kv) - sum_i q_i kv_i) / 1e6
with slope the measured attention-work slope (ms per million, from the partition fits). It is zero
for one-prefill steps and for steps whose prefills all start at depth 0.

Outputs per (trace, config, load): share of prefill steps with >= 2 prefills, share with
geo_err > 5 ms and > 10 % of the step, the time-weighted share, and P50/P95/P99 geo_err.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- traces


def load_trace(kind: str, path: Path, max_requests: int, max_prompt: int, jitter_s: float = 0.0, seed: int = 0):
    """-> list of (arrival_s, prompt_tokens, output_tokens, block_hashes or None), arrival from 0.

    jitter_s > 0 spreads each arrival uniformly over [t, t + jitter_s): the Mooncake conversation and
    tool-agent traces record arrivals in 3 s slots (1,180 distinct timestamps in 59 min, up to 47
    requests on one), so replayed as-is every slot is a simultaneous burst."""
    reqs = []
    if kind == "azure":  # TIMESTAMP,ContextTokens,GeneratedTokens
        import datetime as dt
        for r in csv.DictReader(open(path)):
            t = dt.datetime.fromisoformat(r["TIMESTAMP"][:26]).timestamp()
            reqs.append((t, int(r["ContextTokens"]), int(r["GeneratedTokens"]), None))
    elif kind == "burstgpt":  # Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
        for r in csv.DictReader(open(path)):
            if int(r["Response tokens"]) == 0:
                continue  # failed requests
            reqs.append((float(r["Timestamp"]), int(r["Request tokens"]), int(r["Response tokens"]), None))
            if len(reqs) >= max_requests:
                break
    elif kind == "mooncake":  # {"timestamp" ms, "input_length", "output_length", "hash_ids"}
        for line in open(path):
            j = json.loads(line)
            reqs.append((j["timestamp"] / 1000.0, j["input_length"], j["output_length"], j["hash_ids"]))
    else:
        raise SystemExit(f"unknown trace kind {kind}")
    if jitter_s > 0:
        import random
        rnd = random.Random(seed)
        reqs = [(t + rnd.uniform(0.0, jitter_s), p, o, h) for t, p, o, h in reqs]
    reqs.sort(key=lambda x: x[0])
    reqs = reqs[:max_requests]
    t0 = reqs[0][0]
    kept = [(t - t0, p, max(o, 1), h) for t, p, o, h in reqs if 0 < p <= max_prompt]
    return kept, len(reqs) - len(kept)


# ---------------------------------------------------------------- clock


def ridge(X, y, lam=1e-2):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    b = float(y.mean())
    return lambda x: float(((np.asarray(x) - mu) / sd) @ w + b)


def build_clock(steps_csv: Path):
    pre_X, pre_y, dec_X, dec_y = [], [], [], []
    for r in csv.DictReader(open(steps_csv)):
        if r["cuda_ms"] in ("", "nan") or r["i"] == "0":
            continue
        y = float(r["cuda_ms"])
        g, gkv, ct = int(r["gen_reqs"]), int(r["gen_kv_sum"]), int(r["ctx_tokens"])
        if ct == 0:
            if g > 0:
                dec_X.append([g, gkv / 1e4]); dec_y.append(y)
        else:
            pre_X.append([g, gkv / 1e4, ct / 1e3, int(r["ctx_reqs"]), float(r["attn_proxy"]) / 1e6]); pre_y.append(y)
    pre_X, pre_y = np.array(pre_X), np.array(pre_y)
    # drop one-off compile steps the same way the analysis does (> 10x the median)
    keep = pre_y <= 10 * np.median(pre_y)
    pre = ridge(pre_X[keep], pre_y[keep])
    dec = ridge(np.array(dec_X), np.array(dec_y))
    return pre, dec, {"prefill_steps": int(keep.sum()), "decode_steps": len(dec_y),
                      "max_decode_batch_seen": int(max(x[0] for x in dec_X)),
                      "max_prefills_seen": int(pre_X[:, 3].max())}


# ---------------------------------------------------------------- simulator


@dataclass
class Req:
    arrival: float
    prompt: int
    output: int
    hashes: list | None
    computed: int = 0       # prompt tokens with KV (cache hit + prefilled)
    generated: int = 0
    first_token: float | None = None


@dataclass
class Stats:
    prefill_steps: int = 0
    multi_steps: int = 0
    over5: int = 0
    over10pct: int = 0
    prefill_time: float = 0.0
    over5_time: float = 0.0
    total_time: float = 0.0
    geo: list = field(default_factory=list)
    nprefill: collections.Counter = field(default_factory=collections.Counter)
    ttft: list = field(default_factory=list)
    beyond_clock: int = 0
    steps: int = 0


def simulate(reqs, *, budget, threshold, max_seqs, kv_capacity, slope, clock, clock_range, rate_scale, block=512):
    pre_clock, dec_clock = clock
    arrivals = collections.deque(Req(t / rate_scale, p, o, h) for t, p, o, h in reqs)
    waiting, running = collections.deque(), []
    lru = collections.OrderedDict()   # block hash -> None (cached, reusable)
    held = 0                          # KV tokens reserved by running requests
    now, st = 0.0, Stats()
    while arrivals or waiting or running:
        while arrivals and arrivals[0].arrival <= now:
            waiting.append(arrivals.popleft())
        if not waiting and not running:
            now = arrivals[0].arrival
            continue
        tok = budget
        chunks = []                   # (q, kv_before) of prefill chunks this step
        n_dec = dec_kv = 0
        for r in running:
            if tok <= 0:
                break
            if r.computed < r.prompt:
                q = r.prompt - r.computed
                if 0 < threshold < q:
                    q = threshold
                q = min(q, tok)
                chunks.append((q, r.computed)); r._q = q; tok -= q
            else:
                n_dec += 1; dec_kv += r.prompt + r.generated; r._q = 1; tok -= 1
        while waiting and tok > 0 and len(running) < max_seqs:
            r = waiting[0]
            need = r.prompt + r.output
            hit = 0
            if r.hashes is not None:
                for h in r.hashes:
                    if h in lru:
                        hit += 1
                    else:
                        break
                hit = min(hit * block, r.prompt - 1)
            if held + need > kv_capacity:
                break
            waiting.popleft()
            for h in (r.hashes or [])[: hit // block]:
                lru.move_to_end(h)
            r.computed = hit
            q = r.prompt - r.computed
            if 0 < threshold < q:
                q = threshold
            q = min(q, tok)
            chunks.append((q, r.computed)); r._q = q; tok -= q
            running.append(r); held += need
        # step cost
        if chunks:
            ct = sum(q for q, _ in chunks)
            proxy = sum(q * (kv + (q + 1) / 2) for q, kv in chunks)
            ms = pre_clock([n_dec, dec_kv / 1e4, ct / 1e3, len(chunks), proxy / 1e6])
            geo = slope * (ct * sum(kv for _, kv in chunks) - sum(q * kv for q, kv in chunks)) / 1e6
            st.prefill_steps += 1; st.prefill_time += ms; st.nprefill[len(chunks)] += 1; st.geo.append(geo)
            if len(chunks) >= 2:
                st.multi_steps += 1
            if geo > 5:
                st.over5 += 1; st.over5_time += ms
            if geo > 0.1 * ms:
                st.over10pct += 1
            if n_dec > clock_range["max_decode_batch_seen"] or len(chunks) > clock_range["max_prefills_seen"]:
                st.beyond_clock += 1
        else:
            ms = dec_clock([n_dec, dec_kv / 1e4])
            if n_dec > clock_range["max_decode_batch_seen"]:
                st.beyond_clock += 1
        ms = max(ms, 1.0)
        now += ms / 1000.0; st.total_time += ms; st.steps += 1
        # advance requests
        still = []
        for r in running:
            q = getattr(r, "_q", 0); r._q = 0
            if r.computed < r.prompt:
                r.computed += q
                if r.computed >= r.prompt:
                    r.generated = 1; r.first_token = now; st.ttft.append(now - r.arrival)
            elif q:
                r.generated += 1
            if r.computed >= r.prompt and r.generated >= r.output:
                held -= r.prompt + r.output
                for h in r.hashes or []:
                    lru[h] = None; lru.move_to_end(h)
            else:
                still.append(r)
        running = still
        # LRU bounded by capacity not held by running requests
        while lru and (len(lru) * block) > max(kv_capacity - held, 0):
            lru.popitem(last=False)
    return st


def summarize(st: Stats):
    g = np.array(st.geo) if st.geo else np.zeros(1)
    n = max(st.prefill_steps, 1)
    tt = np.array(st.ttft) if st.ttft else np.zeros(1)
    return {"steps": st.steps, "prefill_steps": st.prefill_steps,
            "multi_prefill_share": st.multi_steps / n,
            "geo_err_gt5ms_share": st.over5 / n, "geo_err_gt10pct_share": st.over10pct / n,
            "geo_err_gt5ms_time_share_of_prefill_time": st.over5_time / max(st.prefill_time, 1e-9),
            "prefill_time_share": st.prefill_time / max(st.total_time, 1e-9),
            "geo_err_ms_p50_p95_p99": [float(np.quantile(g, x)) for x in (.5, .95, .99)],
            "prefills_per_step": dict(sorted(st.nprefill.items())),
            "ttft_s_p50_p99": [float(np.quantile(tt, .5)), float(np.quantile(tt, .99))],
            "beyond_clock_share": st.beyond_clock / max(st.steps, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True, help="NAME=KIND:PATH[@JITTER_S] (KIND azure|burstgpt|mooncake)")
    ap.add_argument("--steps", type=Path, required=True, help="per-step CSV of the campaign that sets the clock")
    ap.add_argument("--slope", type=float, required=True, help="attention-work slope, ms per million (partition fits)")
    ap.add_argument("--kv-capacity", type=int, required=True, help="KV cache capacity in tokens")
    ap.add_argument("--configs", default="default:2048:0,budget8192:8192:0,thr512:2048:512",
                    help="NAME:max_num_batched_tokens:long_prefill_token_threshold,...")
    ap.add_argument("--rate-scales", default="1,2,4,8", help="arrival-rate multipliers (time compression)")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--max-requests", type=int, default=20000)
    ap.add_argument("--max-prompt", type=int, default=32768)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pre, dec, rng = build_clock(args.steps)
    out = {"clock": rng, "slope_ms_per_M": args.slope, "kv_capacity_tokens": args.kv_capacity,
           "max_seqs": args.max_seqs, "results": {}}
    print(f"clock from {args.steps}: {rng}")
    for spec in args.trace:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        path, _, jit = path.partition("@")
        reqs, dropped = load_trace(kind, Path(path), args.max_requests, args.max_prompt, float(jit or 0))
        span = reqs[-1][0]
        print(f"\n## {name}: {len(reqs)} requests over {span / 60:.1f} min (dropped {dropped} over {args.max_prompt} tokens or empty); "
              f"prompt mean {np.mean([r[1] for r in reqs]):.0f}, output mean {np.mean([r[2] for r in reqs]):.0f}"
              + (", prefix hashes" if reqs[0][3] is not None else ""))
        out["results"][name] = {"requests": len(reqs), "dropped": dropped, "span_s": span, "jitter_s": float(jit or 0), "runs": {}}
        for c in args.configs.split(","):
            cn, b, thr = c.split(":")
            for s in map(float, args.rate_scales.split(",")):
                st = simulate(reqs, budget=int(b), threshold=int(thr), max_seqs=args.max_seqs, kv_capacity=args.kv_capacity,
                              slope=args.slope, clock=(pre, dec), clock_range=rng, rate_scale=s)
                sm = summarize(st)
                out["results"][name]["runs"][f"{cn}@x{s:g}"] = sm
                print(f"  {cn:10s} x{s:<4g} TTFT p50/p99 {sm['ttft_s_p50_p99'][0]:6.2f}/{sm['ttft_s_p50_p99'][1]:7.2f} s | prefill steps {sm['prefill_steps']:6d} "
                      f"(time share {sm['prefill_time_share']*100:4.1f}%) multi {sm['multi_prefill_share']*100:5.1f}% | geo>5ms {sm['geo_err_gt5ms_share']*100:5.1f}% "
                      f"(of prefill time {sm['geo_err_gt5ms_time_share_of_prefill_time']*100:5.1f}%) >10% {sm['geo_err_gt10pct_share']*100:5.1f}% | "
                      f"geo p95/p99 {sm['geo_err_ms_p50_p95_p99'][1]:6.1f}/{sm['geo_err_ms_p50_p95_p99'][2]:6.1f} ms | beyond clock {sm['beyond_clock_share']*100:4.1f}%", flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
