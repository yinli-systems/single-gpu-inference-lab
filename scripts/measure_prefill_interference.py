#!/usr/bin/env python3
"""Decode-latency interference from long prefills under a fixed token budget.

Reproduces the setting behind "token budget != wall-clock budget": background
requests are decoding at a steady state when long-prefill requests arrive.
vLLM's chunked prefill limits the *tokens* a step may prefill
(``--max-num-batched-tokens``), but the wall-clock time of a step with a
2048-token chunk of a 32k-context prompt is not the time of a 2048-token chunk
of a 2k prompt, so the decoders' inter-token latency stalls by an amount the
budget does not control.

One ``vllm serve`` per condition (chunk budget). Per run:

  1. start ``--background`` streaming decode requests (short prompt, long
     ``max_tokens``, ``ignore_eos``) on ``/inference/v1/generate`` and record
     every token's arrival time;
  2. once they are all past ``--settle-tokens`` tokens, inject ``--inject``
     long-prefill requests (random token ids of ``--long-tokens``) at once,
     non-streaming, and record their TTFT/e2e;
  3. keep the decoders running until the injected requests finish plus a
     tail, then stop.

Reported: background ITL p50/p95/p99 before / during / after the injection
window, fraction of background tokens within TPOT SLOs (10/25/50 ms), the
largest single stall, long-request TTFT and e2e, and total output tok/s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_vllm_feature_cost import Server, git_provenance, nvidia_smi  # noqa: E402


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


async def background_stream(session, url, model, token_ids, max_tokens, seed, arrivals: list[float], stop: asyncio.Event):
    payload = {
        "model": model,
        "token_ids": token_ids,
        "sampling_params": {"max_tokens": max_tokens, "ignore_eos": True, "temperature": 1.0, "top_k": 50, "top_p": 0.95, "seed": seed},
        "stream": True,
    }
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            n = sum(len(c.get("token_ids") or []) for c in obj.get("choices", []))
            now = time.perf_counter()
            arrivals.extend([now] * max(n, 0))
            if stop.is_set():
                break


async def long_request(session, url, model, token_ids, max_tokens, seed):
    payload = {
        "model": model,
        "token_ids": token_ids,
        "sampling_params": {"max_tokens": max_tokens, "ignore_eos": True, "temperature": 1.0, "top_k": 50, "top_p": 0.95, "seed": seed},
        "stream": True,
    }
    t0 = time.perf_counter()
    first = None
    n_out = 0
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            n = sum(len(c.get("token_ids") or []) for c in obj.get("choices", []))
            if n and first is None:
                first = time.perf_counter()
            n_out += n
    end = time.perf_counter()
    return {"submit": t0, "ttft_s": (first or end) - t0, "e2e_s": end - t0, "first": first, "end": end, "output_tokens": n_out}


async def run_condition(base_url, model, args, bg_prompts, long_prompts) -> dict[str, Any]:
    url = f"{base_url}/inference/v1/generate"
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        stop = asyncio.Event()
        arrivals: list[list[float]] = [[] for _ in bg_prompts]
        bg_tasks = [
            asyncio.create_task(background_stream(session, url, model, p, args.background_tokens, args.seed + i, arrivals[i], stop))
            for i, p in enumerate(bg_prompts)
        ]
        # settle
        while min(len(a) for a in arrivals) < args.settle_tokens:
            await asyncio.sleep(0.05)
            if any(t.done() and t.exception() for t in bg_tasks):
                raise RuntimeError("background stream failed during settle")
        t_inject = time.perf_counter()
        if long_prompts:
            long_results = await asyncio.gather(
                *(long_request(session, url, model, p, args.long_output_tokens, args.seed + 1000 + i) for i, p in enumerate(long_prompts))
            )
        else:
            # decode-only cell: hold the background for a fixed window
            await asyncio.sleep(args.decode_only_window_s)
            long_results = []
        t_long_done = time.perf_counter()
        await asyncio.sleep(args.tail_s)
        t_end = time.perf_counter()
        stop.set()
        for t in bg_tasks:
            t.cancel()
        await asyncio.gather(*bg_tasks, return_exceptions=True)

    # windows: before = [settle..inject), during = [inject..long_done], after = (long_done..end]
    def itls_in(lo, hi):
        out = []
        for a in arrivals:
            for x, y in zip(a, a[1:]):
                if lo <= y <= hi:
                    out.append(y - x)
        return out

    before = itls_in(t_inject - args.before_window_s, t_inject)
    during = itls_in(t_inject, t_long_done)
    after = itls_in(t_long_done, t_end)
    all_tokens = sum(len(a) for a in arrivals)
    span = t_end - min(a[0] for a in arrivals if a)

    def summ(itls):
        return {
            "count": len(itls),
            "p50_ms": 1e3 * pct(itls, 0.5),
            "p95_ms": 1e3 * pct(itls, 0.95),
            "p99_ms": 1e3 * pct(itls, 0.99),
            "max_ms": 1e3 * (max(itls) if itls else float("nan")),
            "slo_attainment": {f"{slo}ms": (sum(1 for x in itls if x <= slo / 1e3) / len(itls) if itls else float("nan")) for slo in (10, 25, 50)},
        }

    return {
        "background": {"before": summ(before), "during": summ(during), "after": summ(after)},
        "long_requests": [
            {"ttft_s": r["ttft_s"], "e2e_s": r["e2e_s"], "output_tokens": r["output_tokens"]} for r in long_results
        ],
        "injection_window_s": t_long_done - t_inject,
        "background_output_tokens_total": all_tokens,
        "background_tok_per_s_overall": all_tokens / span,
        "background_tok_per_s_during": len(during) / (t_long_done - t_inject) if during else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="m")
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--chunk-budgets", default="512,1024,2048,4096,8192", help="max-num-batched-tokens values, one server each")
    ap.add_argument("--extra-server-args", default="")
    ap.add_argument("--background", type=int, default=8)
    ap.add_argument("--background-prompt-tokens", type=int, default=128)
    ap.add_argument(
        "--background-prompt-tokens-list",
        default="",
        help="comma-separated per-decoder prompt lengths, cycled over --background "
        "(decode KV-distribution control: balanced vs skewed at equal aggregate KV)",
    )
    ap.add_argument("--background-tokens", type=int, default=4096)
    ap.add_argument("--settle-tokens", type=int, default=64)
    ap.add_argument("--before-window-s", type=float, default=3.0)
    ap.add_argument("--inject", type=int, default=2)
    ap.add_argument("--long-tokens", type=int, default=32768)
    ap.add_argument("--long-output-tokens", type=int, default=32)
    ap.add_argument("--tail-s", type=float, default=3.0)
    ap.add_argument("--decode-only-window-s", type=float, default=10.0, help="with --inject 0: measured decode-only window")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=113)
    ap.add_argument("--vocab-size", type=int, default=151_000, help="random token id range for synthetic prompts")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--long-prefill-token-threshold",
        type=int,
        default=0,
        help="cap per-request prefill chunk so several injected prefills share a step (partition experiments)",
    )
    ap.add_argument(
        "--trace-dir",
        type=Path,
        help="if set, start servers with --enable-logging-iteration-details and "
        "VLLM_EXP_ITER_TRACE=<trace-dir>/<condition>.jsonl (needs the experiment patch)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bg_lens = [int(x) for x in args.background_prompt_tokens_list.split(",") if x] or [args.background_prompt_tokens]
    bg_prompts = [[rng.randrange(1000, args.vocab_size) for _ in range(bg_lens[i % len(bg_lens)])] for i in range(args.background)]
    long_prompts = [[rng.randrange(1000, args.vocab_size) for _ in range(args.long_tokens)] for _ in range(args.inject)]

    record: dict[str, Any] = {
        "schema_version": 1,
        "result_type": "prefill_interference",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": git_provenance(),
        "model": args.model,
        "workload": {k: getattr(args, k) for k in ("background", "background_prompt_tokens", "background_prompt_tokens_list", "background_tokens", "settle_tokens", "inject", "long_tokens", "long_output_tokens", "repeats", "seed", "max_model_len", "max_num_seqs", "long_prefill_token_threshold")},
        "trace": args.trace_dir is not None,
        "conditions": {},
        "nvidia_smi_before": nvidia_smi(),
    }
    args.extra_server_args_list = args.extra_server_args.split() if args.extra_server_args else []
    for budget in [int(b) for b in args.chunk_budgets.split(",")]:
        flags = ["--max-num-batched-tokens", str(budget), "--no-enable-prefix-caching", *args.extra_server_args_list]
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            flags += ["--enable-logging-iteration-details"]
            os.environ["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{args.output.stem}-chunk{budget}.jsonl")
            os.environ["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{args.output.stem}-chunk{budget}.steps.jsonl")
        else:
            os.environ.pop("VLLM_EXP_ITER_TRACE", None)
            os.environ.pop("VLLM_EXP_STEP_TRACE", None)
        if args.long_prefill_token_threshold:
            flags += ["--long-prefill-token-threshold", str(args.long_prefill_token_threshold)]
        log_path = args.output.with_suffix("") / f"server-chunk{budget}.log"
        # Server expects these attribute names
        args.served_model_name = args.served_model_name
        cond: dict[str, Any] = {"flags": flags, "repeats": []}
        record["conditions"][f"chunk{budget}"] = cond
        with Server(args, flags, log_path) as server:
            cond["command"] = server.cmd
            # warm-up: one short background-only pass
            warm_long = [long_prompts[0][:4096]] if long_prompts else []
            asyncio.run(run_condition(server.base_url, args.served_model_name, argparse.Namespace(**{**vars(args), "inject": len(warm_long), "long_tokens": 4096, "tail_s": 0.5, "decode_only_window_s": 2.0}), bg_prompts[:2], warm_long))
            for r in range(args.repeats):
                res = asyncio.run(run_condition(server.base_url, args.served_model_name, args, bg_prompts, long_prompts))
                cond["repeats"].append(res)
                d = res["background"]["during"]
                print(
                    f"[chunk{budget} r{r}] during-injection ITL p50 {d['p50_ms']:.1f} p99 {d['p99_ms']:.1f} max {d['max_ms']:.0f} ms | "
                    f"SLO25 {d['slo_attainment']['25ms']:.2f} SLO50 {d['slo_attainment']['50ms']:.2f} | "
                    f"long TTFT {statistics.median([x['ttft_s'] for x in res['long_requests']] or [float('nan')]):.2f}s window {res['injection_window_s']:.1f}s | "
                    f"bg tok/s during {res['background_tok_per_s_during']:.0f} overall {res['background_tok_per_s_overall']:.0f}",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
