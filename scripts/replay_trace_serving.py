#!/usr/bin/env python3
"""Replay a public request trace against a live vLLM server and measure TTFT / TPOT per request.

One `vllm serve` per run (the same Server helper as the other harnesses). Arrivals follow the
trace, time-scaled by --rate-scale; only requests arriving within --window-s (scaled seconds) are
sent, then the run drains. Prompts are synthetic token ids with the trace's lengths; for traces
with block hashes (Mooncake, 512-token blocks) every hash id maps to one fixed 512-token block, so
requests that share hashes share a real prefix and hit vLLM's prefix cache (prefix caching is on
unless --no-prefix-caching). Outputs: max_tokens = the trace's output length, ignore_eos.

Per request: submit time, TTFT, end, output tokens, TPOT = (end - first token) / (tokens - 1).
Goodput at (TTFT <= a, TPOT <= b) = requests meeting both / arrival window; several SLO pairs are
reported. With --trace-dir the tracer v2 records every engine iteration and runner step.

Non-stationary workloads: --phases "NAME=KIND:PATH[@JITTER]:SCALE:OFFSET_S:DUR_S;..." concatenates
segments of several traces (OFFSET_S in the trace's own time, DUR_S in replayed seconds, arrivals
time-scaled by SCALE); every request records its phase and the summary is also given per phase.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_vllm_feature_cost import Server, git_provenance, nvidia_smi  # noqa: E402
from simulate_geometry_prevalence import load_trace  # noqa: E402

BLOCK = 512
SLOS = [(1.0, 0.05), (2.0, 0.1), (5.0, 0.1), (10.0, 0.2)]


def build_prompts(reqs, vocab: int, seed: int):
    blocks = {}
    out = []
    for i, (_, p, _, hashes) in enumerate(reqs):
        if hashes is None:
            rnd = random.Random(seed * 1_000_003 + i)
            out.append([rnd.randrange(1000, vocab) for _ in range(p)])
            continue
        ids = []
        for h in hashes:
            if h not in blocks:
                rnd = random.Random(seed * 7_919 + int(h))
                blocks[h] = [rnd.randrange(1000, vocab) for _ in range(BLOCK)]
            ids += blocks[h]
            if len(ids) >= p:
                break
        if len(ids) < p:  # the trace can list fewer blocks than input_length covers
            rnd = random.Random(seed * 104_729 + i)
            ids += [rnd.randrange(1000, vocab) for _ in range(p - len(ids))]
        out.append(ids[:p])
    return out


async def one(session, url, model, ids, max_tokens, delay, t_start, rec):
    await asyncio.sleep(max(0.0, t_start + delay - time.perf_counter()))
    payload = {"model": model, "token_ids": ids, "stream": True,
               "sampling_params": {"max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0}}
    t0 = time.perf_counter(); first = None; n = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec.update(error=f"HTTP {resp.status}: {(await resp.text())[:200]}"); return
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                k = sum(len(c.get("token_ids") or []) for c in json.loads(data).get("choices", []))
                if k and first is None:
                    first = time.perf_counter()
                n += k
    except Exception as e:  # noqa: BLE001
        rec.update(error=repr(e)[:200]); return
    end = time.perf_counter()
    rec.update(submit=t0 - t_start, ttft_s=(first or end) - t0, end=end - t_start, output_tokens=n,
               tpot_s=(end - first) / (n - 1) if first and n > 1 else None)


async def replay(base_url, model, reqs, prompts, t_zero_lag):
    import aiohttp
    url = f"{base_url}/inference/v1/generate"
    recs = [{"i": i, "arrival_s": r[0], "prompt_tokens": r[1], "max_tokens": r[2]} for i, r in enumerate(reqs)]
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        t_start = time.perf_counter() + t_zero_lag
        await asyncio.gather(*(one(s, url, model, prompts[i], r[2], r[0], t_start, recs[i]) for i, r in enumerate(reqs)))
    return recs


def summarize(recs, window_s):
    ok = [r for r in recs if "error" not in r]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None
    ttft = [r["ttft_s"] for r in ok]; tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    good = {f"ttft<={a:g}s,tpot<={b * 1000:g}ms": sum(1 for r in ok if r["ttft_s"] <= a and (r["tpot_s"] or 0) <= b) / window_s
            for a, b in SLOS}
    return {"requests": len(recs), "errors": len(recs) - len(ok), "offered_rps": len(recs) / window_s,
            "ttft_p50_p90_p99": [q(ttft, .5), q(ttft, .9), q(ttft, .99)], "tpot_p50_p90_p99": [q(tpot, .5), q(tpot, .9), q(tpot, .99)],
            "goodput_rps": good, "makespan_s": max((r["end"] for r in ok), default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="m")
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--port", type=int, default=8125)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--long-prefill-token-threshold", type=int, default=0)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--trace", help="KIND:PATH[@JITTER_S] (see simulate_geometry_prevalence.py)")
    ap.add_argument("--rate-scale", type=float)
    ap.add_argument("--phases", help="NAME=KIND:PATH[@JITTER]:SCALE:OFFSET_S:DUR_S;... (replaces --trace/--rate-scale/--window-s)")
    ap.add_argument("--window-s", type=float, default=240.0)
    ap.add_argument("--max-prompt", type=int, default=32768)
    ap.add_argument("--vocab-size", type=int, default=151_000)
    ap.add_argument("--seed", type=int, default=113)
    ap.add_argument("--jitter-seed", type=int, default=0, help="seed of the within-slot arrival spread (@JITTER traces)")
    ap.add_argument("--trace-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    phase_of, phase_bounds = [], []
    if args.phases:
        reqs, dropped, t0 = [], 0, 0.0
        for spec in args.phases.split(";"):
            name, rest = spec.split("=", 1)
            kind, rest = rest.split(":", 1)
            path, scale, off, dur = rest.rsplit(":", 3)
            path, _, jit = path.partition("@")
            allr, d = load_trace(kind, Path(path), 10**9, args.max_prompt, float(jit or 0), args.jitter_seed)
            dropped += d
            seg = [((t - float(off)) / float(scale) + t0, p, o, h) for t, p, o, h in allr
                   if 0 <= (t - float(off)) / float(scale) < float(dur)]
            reqs += seg; phase_of += [name] * len(seg); phase_bounds.append((name, t0, t0 + float(dur)))
            t0 += float(dur)
        args.window_s = t0
    else:
        kind, path = args.trace.split(":", 1)
        path, _, jit = path.partition("@")
        allreqs, dropped = load_trace(kind, Path(path), 10**9, args.max_prompt, float(jit or 0), args.jitter_seed)
        reqs = [(t / args.rate_scale, p, o, h) for t, p, o, h in allreqs if t / args.rate_scale < args.window_s]
    prompts = build_prompts(reqs, args.vocab_size, args.seed)

    flags = ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.long_prefill_token_threshold:
        flags += ["--long-prefill-token-threshold", str(args.long_prefill_token_threshold)]
    if args.no_prefix_caching:
        flags += ["--no-enable-prefix-caching"]
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        flags.append("--enable-logging-iteration-details")
        os.environ["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{args.output.stem}.jsonl")
        os.environ["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{args.output.stem}.steps.jsonl")
    log_path = args.output.with_suffix("") / "server.log"
    report = {"schema_version": 1, "result_type": "trace_replay_serving", "provenance": git_provenance(),
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_EXP_")},
              "dropped_over_max_prompt": dropped, "nvidia_smi_before": nvidia_smi()}
    with Server(args, flags, log_path) as server:
        report["command"] = server.cmd
        warm = [(0.0, 256, 8, None)] * 2   # compile / warm-up, not measured
        asyncio.run(replay(server.base_url, args.served_model_name, warm, build_prompts(warm, args.vocab_size, args.seed + 1), 0.1))
        recs = asyncio.run(replay(server.base_url, args.served_model_name, reqs, prompts, 1.0))
    for r, ph in zip(recs, phase_of):
        r["phase"] = ph
    report["requests"] = recs
    report["summary"] = summarize(recs, args.window_s)
    if phase_bounds:
        report["phases"] = [{"name": n, "start_s": a, "end_s": b,
                             "summary": summarize([r for r in recs if a <= r["arrival_s"] < b], b - a)}
                            for n, a, b in phase_bounds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n")
    s = report["summary"]
    print(f"{args.output.stem}: {s['requests']} req ({s['offered_rps']:.2f}/s), errors {s['errors']} | TTFT p50/p90/p99 "
          + "/".join(f"{x:.2f}" for x in s["ttft_p50_p90_p99"]) + " s | TPOT p50/p90/p99 "
          + "/".join(f"{x * 1000:.0f}" for x in s["tpot_p50_p90_p99"]) + " ms | goodput "
          + ", ".join(f"{k} {v:.2f}" for k, v in s["goodput_rps"].items()), flush=True)


if __name__ == "__main__":
    main()
