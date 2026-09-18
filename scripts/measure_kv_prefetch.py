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
    r1 = await completion(client, model, prefix, gen_tokens)
    await asyncio.sleep(0.3)  # let the finished request's blocks be stored to CPU
    rec = {"P": P, "arm": arm, "lead_ms": lead_ms, "turn1": r1}
    if arm != "A":
        f0 = time.monotonic()
        per = max(filler_tokens // filler_concurrency, 256)
        await asyncio.gather(*(completion(client, model, rand_tokens(rng, per), 1) for _ in range(filler_concurrency)))
        rec["filler_ms"] = (time.monotonic() - f0) * 1e3
        await asyncio.sleep(0.5)
    m0 = metrics(base)
    resume_prompt = prefix + rand_tokens(rng, append_tokens)
    if arm == "C":
        probe = asyncio.create_task(completion(client, model, prefix, 1))
        await asyncio.sleep(lead_ms / 1e3)
        r2 = await completion(client, model, resume_prompt, gen_tokens)
        rec["probe"] = await probe
    else:
        r2 = await completion(client, model, resume_prompt, gen_tokens)
    rec["resume"] = r2
    m1 = metrics(base)
    rec["metrics_delta"] = {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] != m0.get(k, 0.0)}
    return rec


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

        async def run():
            async with httpx.AsyncClient(base_url=base, timeout=900) as client:
                await completion(client, args.model, rand_tokens(rng, 256), 4)
                for rep in range(args.repeats):
                    for P in [int(x) for x in args.prefixes.split(",")]:
                        plan = [("A", 0), ("B", 0)] + [("C", l) for l in [int(x) for x in args.leads_ms.split(",")]]
                        random.Random(rep * 100 + P).shuffle(plan)
                        for arm, lead in plan:
                            t = await trial(client, base, args.model, rng, P, arm, lead, args.filler_tokens, args.filler_concurrency, args.append_tokens, args.gen_tokens)
                            t["repeat"] = rep
                            rec["trials"].append(t)
                            print(f"[r{rep}] P={P:6d} arm={arm} lead={lead:4d}: resume TTFT {t['resume']['ttft_ms']:.0f} ms" + (f" (probe ttft {t['probe']['ttft_ms']:.0f})" if "probe" in t else "") + (f" filler {t['filler_ms']:.0f} ms" if "filler_ms" in t else ""), flush=True)
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
