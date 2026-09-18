#!/usr/bin/env python3
"""Oracle request-free KV prefetch upper bound (agentic two-turn sessions).

One vllm server (prefix caching + CPU KV offloading). Per trial:
  turn 1   : prefix of P tokens + short generation (session KV now in GPU cache and,
             on finish, stored to the CPU tier)
  tool wait: filler traffic of F distinct tokens evicts the session's blocks from
             the GPU cache (pressure); then a synthetic wait
  turn 2   : the same prefix + a short append, TTFT measured
Arms:
  A retained  : no filler -> blocks still in GPU cache (latency oracle, capacity cost)
  B reactive  : filler, resume triggers CPU->GPU load (shipped behaviour)
  C oracle    : filler, a prefix-only max_tokens=1 probe is issued `lead` ms before the
                resume so the load completes off the critical path (block-level emulation
                of a request-free prefetch; the probe's own decode step is its overhead)
Records resume TTFT / e2e, probe timing, filler timing, and /metrics deltas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import httpx


def rand_tokens(rng, n, vocab=151000):
    return [rng.randrange(1000, vocab) for _ in range(n)]


async def completion(client, model, prompt, max_tokens, stream=True):
    t0 = time.monotonic(); first = None; n = 0
    async with client.stream("POST", "/v1/completions", json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])
                if first is None and d.get("choices") and d["choices"][0].get("text"):
                    first = time.monotonic()
                if d.get("usage"):
                    n = d["usage"]["completion_tokens"]
    return {"ttft_ms": ((first or time.monotonic()) - t0) * 1e3, "e2e_ms": (time.monotonic() - t0) * 1e3, "tokens": n}


def metrics(base):
    try:
        t = httpx.get(base + "/metrics", timeout=5).text
        out = {}
        for l in t.splitlines():
            if l.startswith("vllm:") and ("offload" in l or "prefix_cache" in l or "kv_cache" in l) and " " in l and not l.startswith("#"):
                k, v = l.rsplit(" ", 1)
                try: out[k] = float(v)
                except ValueError: pass
        return out
    except Exception:
        return {}


async def trial(client, base, model, rng, P, arm, lead_ms, filler_tokens, filler_concurrency, append_tokens, gen_tokens):
    prefix = rand_tokens(rng, P)
    rec_t0 = time.monotonic()
    r1 = await completion(client, model, prefix, gen_tokens)
    await asyncio.sleep(0.3)  # let the finished request's blocks be stored to CPU
    rec = {"P": P, "arm": arm, "lead_ms": lead_ms, "turn1": r1, "t_trial_start": rec_t0}
    if arm != "A":
        f0 = time.monotonic()
        per = max(filler_tokens // filler_concurrency, 256)
        await asyncio.gather(*(completion(client, model, rand_tokens(rng, per), 1) for _ in range(filler_concurrency)))
        rec["filler_ms"] = (time.monotonic() - f0) * 1e3
        await asyncio.sleep(0.5)
    m0 = metrics(base)
    resume_prompt = prefix + rand_tokens(rng, append_tokens)
    rec["t_prefetch_start"] = time.monotonic()
    if arm == "C":
        probe = asyncio.create_task(completion(client, model, prefix, 1))
        await asyncio.sleep(lead_ms / 1e3)
        r2 = await completion(client, model, resume_prompt, gen_tokens)
        rec["probe"] = await probe
    else:
        r2 = await completion(client, model, resume_prompt, gen_tokens)
    rec["resume"] = r2
    rec["t_resume_end"] = time.monotonic()
    m1 = metrics(base)
    rec["metrics_delta"] = {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] != m0.get(k, 0.0)}
    return rec


KNEE_MS = {4096: 50, 8192: 100, 16384: 250}  # campaign26: smallest lead within 5% of retained (unloaded)


async def trial_policy(client, base, model, rng, P, policy, wait_ms, pred_ms, filler_tokens, filler_concurrency, append_tokens, gen_tokens):
    """Two-turn session with a stochastic tool wait; the policy decides when to prefetch.
    reactive: never; immediate: at wait start; fixed: knee ms before the *predicted* resume
    using a fixed prediction = median wait; oracle: knee ms before the true resume;
    ewma: knee ms before the EWMA-predicted resume. Residency = ms the blocks sit in GPU before the resume."""
    prefix = rand_tokens(rng, P)
    r1 = await completion(client, model, prefix, gen_tokens)
    await asyncio.sleep(0.3)
    per = max(filler_tokens // filler_concurrency, 256)
    f0 = time.monotonic()
    await asyncio.gather(*(completion(client, model, rand_tokens(rng, per), 1) for _ in range(filler_concurrency)))
    filler_ms = (time.monotonic() - f0) * 1e3
    knee = KNEE_MS[P]
    if policy == "reactive":
        t_pref = None
    elif policy == "immediate":
        t_pref = 0.0
    elif policy == "oracle":
        t_pref = max(0.0, wait_ms - knee)
    else:  # fixed / ewma: prefetch at predicted resume - knee (may be late)
        t_pref = max(0.0, pred_ms - knee)
    t_wait0 = time.monotonic()
    probe = None
    if t_pref is not None and t_pref < wait_ms:
        await asyncio.sleep(t_pref / 1e3)
        probe = asyncio.create_task(completion(client, model, prefix, 1))
        await asyncio.sleep(max(0.0, wait_ms - t_pref) / 1e3)
    else:
        await asyncio.sleep(wait_ms / 1e3)
    resume_prompt = prefix + rand_tokens(rng, append_tokens)
    t_res0 = time.monotonic()
    r2 = await completion(client, model, resume_prompt, gen_tokens)
    rec = {"P": P, "policy": policy, "wait_ms": wait_ms, "pred_ms": pred_ms, "t_prefetch_rel_ms": t_pref, "filler_ms": filler_ms,
           "residency_ms": (wait_ms - t_pref) if t_pref is not None and t_pref < wait_ms else 0.0, "turn1": r1, "resume": r2}
    if probe is not None:
        rec["probe"] = await probe
    return rec


async def trial_contention(client, base, model, rng, sizes, policy, cap_tokens, wait_median, wait_sigma, filler_tokens, filler_concurrency, append_tokens, gen_tokens):
    """N paused sessions whose KV does not fit in the free GPU cache. All sessions run turn 1
    (stored to CPU), filler evicts them, then every session starts a tool wait W_i and resumes.
    Policy = which sessions get an immediate prefetch (probe at wait start):
      reactive        none
      immediate-all   all (oversubscribed)
      oracle-admit    earliest true resume first, while sum(prefix) <= cap_tokens
      random-admit    random order, same capacity cap (isolates the value of knowing the order)
      smallest-admit  smallest prefixes first, same cap (most sessions per capacity)"""
    import math
    sessions = [{"i": i, "P": P, "prefix": rand_tokens(rng, P), "wait_ms": wait_median * math.exp(rng.gauss(0.0, wait_sigma))} for i, P in enumerate(sizes)]
    for s_ in sessions:
        s_["turn1"] = await completion(client, model, s_["prefix"], gen_tokens)
    await asyncio.sleep(0.3)
    per = max(filler_tokens // filler_concurrency, 256)
    await asyncio.gather(*(completion(client, model, rand_tokens(rng, per), 1) for _ in range(filler_concurrency)))
    await asyncio.sleep(0.5)
    if policy == "reactive":
        admitted = set()
    elif policy == "immediate-all":
        admitted = {s_["i"] for s_ in sessions}
    else:
        order = {"oracle-admit": sorted(sessions, key=lambda x: x["wait_ms"]), "random-admit": random.Random(rng.random()).sample(sessions, len(sessions)),
                 "smallest-admit": sorted(sessions, key=lambda x: x["P"])}[policy]
        admitted, used = set(), 0
        for s_ in order:
            if used + s_["P"] <= cap_tokens:
                admitted.add(s_["i"]); used += s_["P"]
    m0 = metrics(base)
    t_wait0 = time.monotonic()

    async def one(s_):
        probe = asyncio.create_task(completion(client, model, s_["prefix"], 1)) if s_["i"] in admitted else None
        await asyncio.sleep(s_["wait_ms"] / 1e3)
        r = await completion(client, model, s_["prefix"] + rand_tokens(rng, append_tokens), gen_tokens)
        out = {"i": s_["i"], "P": s_["P"], "wait_ms": s_["wait_ms"], "admitted": s_["i"] in admitted, "resume": r}
        if probe is not None:
            out["probe"] = await probe
        return out
    results = await asyncio.gather(*(one(s_) for s_ in sessions))
    m1 = metrics(base)
    return {"policy": policy, "cap_tokens": cap_tokens, "sizes": sizes, "t_wait0": t_wait0, "t_end": time.monotonic(), "sessions": results,
            "metrics_delta": {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] != m0.get(k, 0.0)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--server-args", default="--max-model-len 20480 --max-num-seqs 64 --kv-offloading-size 7 --num-gpu-blocks-override 1536 --gpu-memory-utilization 0.9")
    ap.add_argument("--prefixes", default="4096,8192,16384")
    ap.add_argument("--leads-ms", default="0,10,25,50,100,250,500")
    ap.add_argument("--filler-tokens", type=int, default=32768, help="distinct tokens pushed through between turns (evicts the session from GPU)")
    ap.add_argument("--filler-concurrency", type=int, default=16)
    ap.add_argument("--append-tokens", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--contention-mode", action="store_true", help="campaign28: N paused sessions vs free GPU capacity")
    ap.add_argument("--session-sizes", default="4096,4096,4096,4096,8192,8192,8192,16384,16384")
    ap.add_argument("--cap-tokens", type=int, default=16384, help="prefetch admission capacity (tokens)")
    ap.add_argument("--contention-policies", default="reactive,immediate-all,oracle-admit,random-admit,smallest-admit")
    ap.add_argument("--policy-mode", action="store_true", help="campaign27: stochastic tool waits and prefetch policies instead of fixed leads")
    ap.add_argument("--policies", default="reactive,immediate,fixed,oracle,ewma")
    ap.add_argument("--wait-median-ms", type=float, default=600.0)
    ap.add_argument("--wait-sigma", type=float, default=0.6, help="lognormal sigma of the tool wait")
    ap.add_argument("--ewma-alpha", type=float, default=0.3)
    ap.add_argument("--background", type=int, default=0, help="concurrent background decoders (128-token prompts, long generations) whose per-token arrival times are recorded")
    ap.add_argument("--background-tokens", type=int, default=2048)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"
    cmd = [args.vllm_bin, "serve", args.model, "--port", str(args.port)] + args.server_args.split()
    log = open(args.output.with_suffix(".server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError("timeout")
            time.sleep(2)
        rec = {"schema_version": 1, "result_type": "kv_prefetch_oracle", "cmd": cmd, "args": {k: str(v) for k, v in vars(args).items()}, "trials": []}
        rng = random.Random(args.seed)

        async def background(client, idx, store):
            while not store["stop"]:
                ts = []
                async with client.stream("POST", "/v1/completions", json={"model": args.model, "prompt": rand_tokens(random.Random(idx * 7919 + len(store["runs"])), 128), "max_tokens": args.background_tokens, "temperature": 0, "ignore_eos": True, "stream": True}) as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            ts.append(time.monotonic())
                            if store["stop"]:
                                break
                store["runs"].append(ts)

        async def run():
            async with httpx.AsyncClient(base_url=base, timeout=900) as client:
                await completion(client, args.model, rand_tokens(rng, 256), 4)
                bg_store = {"stop": False, "runs": []}
                bg_tasks = [asyncio.create_task(background(client, i, bg_store)) for i in range(args.background)]
                if bg_tasks:
                    await asyncio.sleep(3.0)
                if args.contention_mode:
                    sizes = [int(x) for x in args.session_sizes.split(",")]
                    for rep in range(args.repeats):
                        plan = args.contention_policies.split(","); random.Random(rep).shuffle(plan)
                        for pol in plan:
                            t = await trial_contention(client, base, args.model, rng, sizes, pol, args.cap_tokens, args.wait_median_ms, args.wait_sigma, args.filler_tokens, args.filler_concurrency, args.append_tokens, args.gen_tokens)
                            t["repeat"] = rep; rec["trials"].append(t)
                            tt = sorted(x["resume"]["ttft_ms"] for x in t["sessions"])
                            print(f"[r{rep}] {pol:14s}: resume TTFT p50 {tt[len(tt)//2]:.0f} max {tt[-1]:.0f} ms, admitted {sum(x['admitted'] for x in t['sessions'])}/{len(tt)}", flush=True)
                            args.output.write_text(json.dumps(rec, indent=1) + "\n")
                elif args.policy_mode:
                    import math
                    ewma = None
                    waits_seen = []
                    for rep in range(args.repeats):
                        for P in [int(x) for x in args.prefixes.split(",")]:
                            wait_ms = args.wait_median_ms * math.exp(rng.gauss(0.0, args.wait_sigma))
                            plan = args.policies.split(","); random.Random(rep * 100 + P).shuffle(plan)
                            for pol in plan:
                                pred = {"fixed": args.wait_median_ms, "ewma": (ewma if ewma is not None else args.wait_median_ms)}.get(pol, wait_ms)
                                t = await trial_policy(client, base, args.model, rng, P, pol, wait_ms, pred, args.filler_tokens, args.filler_concurrency, args.append_tokens, args.gen_tokens)
                                t["repeat"] = rep; rec["trials"].append(t)
                                print(f"[r{rep}] P={P:6d} {pol:9s} wait {wait_ms:5.0f} pred {pred:5.0f}: resume TTFT {t['resume']['ttft_ms']:.0f} ms, residency {t['residency_ms']:.0f} ms", flush=True)
                                args.output.write_text(json.dumps(rec, indent=1) + "\n")
                            waits_seen.append(wait_ms)
                            ewma = wait_ms if ewma is None else (args.ewma_alpha * wait_ms + (1 - args.ewma_alpha) * ewma)
                    plan_leads = []
                else:
                    plan_leads = None
                for rep in (range(args.repeats) if not args.policy_mode else []):
                    for P in [int(x) for x in args.prefixes.split(",")]:
                        plan = [("A", 0), ("B", 0)] + [("C", l) for l in [int(x) for x in args.leads_ms.split(",")]]
                        random.Random(rep * 100 + P).shuffle(plan)
                        for arm, lead in plan:
                            t = await trial(client, base, args.model, rng, P, arm, lead, args.filler_tokens, args.filler_concurrency, args.append_tokens, args.gen_tokens)
                            t["repeat"] = rep
                            rec["trials"].append(t)
                            print(f"[r{rep}] P={P:6d} arm={arm} lead={lead:4d}: resume TTFT {t['resume']['ttft_ms']:.0f} ms" + (f" (probe ttft {t['probe']['ttft_ms']:.0f})" if "probe" in t else "") + (f" filler {t['filler_ms']:.0f} ms" if "filler_ms" in t else ""), flush=True)
                            args.output.write_text(json.dumps(rec, indent=1) + "\n")
                bg_store["stop"] = True
                for t in bg_tasks:
                    t.cancel()
                rec["background_token_times"] = bg_store["runs"]
                args.output.write_text(json.dumps(rec, indent=1) + "\n")
        asyncio.run(run())
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        log.close()


if __name__ == "__main__":
    main()
