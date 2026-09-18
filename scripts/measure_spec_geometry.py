#!/usr/bin/env python3
"""Speculative-decoding cost geometry: draft vs verify time and per-request
acceptance across batch size, context depth and prompt class.

One ``vllm serve`` per condition (speculative config); inside it, closed batches
of B requests (all sent together, greedy, fixed output length) for several
prompt classes and 50/50 mixes. Traces (tracer v3): engine iteration trace,
runner step trace with ``draft_ms``, and the scheduler's per-request accepted
tokens per step. Output JSON: per (condition, class, B) batch wall time, output
tokens, tok/s, per-request TTFT/e2e, and the trace file names.

Prompt classes (synthetic but realistic text, deterministic per seed):
  code-short   ~60-token coding task            -> code output, high acceptance
  prose-short  ~40-token story prompt           -> prose output, lower acceptance
  code-ctxN / prose-ctxN   ~N tokens of context (any N)
  code-mid     ~3k-token code file + task       (long_chars // 2)
  code-long    ~6k-token code file + task       -> code output at deep context
  prose-long   ~6k-token story + continuation   -> prose output at deep context
  mix-sl       half code-short + half prose-long (the "mixed utility" batch)
  mix-ls       half code-long  + half prose-short
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

TOPICS = ["a lighthouse keeper", "a retired detective", "a lost satellite", "an old bakery", "a border town in winter",
          "a chess prodigy", "a deep-sea diver", "a cartographer", "a night-shift nurse", "a beekeeper", "a glassblower",
          "a subway musician", "an archivist", "a tugboat captain", "a mountain guide", "a violin maker"]
TASKS = ["parses a CSV file and returns the rows as dictionaries", "implements an LRU cache with a maximum size",
         "computes the Levenshtein distance between two strings", "merges overlapping intervals", "validates an IPv4 address",
         "converts a nested dictionary to dotted keys", "implements binary search on a sorted list", "counts word frequencies in a text",
         "serialises a binary tree to a string and back", "finds the longest palindromic substring", "implements a rate limiter",
         "topologically sorts a DAG", "computes a moving average over a stream", "flattens an arbitrarily nested list",
         "implements Dijkstra's shortest path", "checks whether two strings are anagrams"]


def code_prompt(i: int) -> str:
    return f"Write a Python function that {TASKS[i % len(TASKS)]}. Include type hints, a docstring, and a few unit tests.\n\n```python\n"


def prose_prompt(i: int) -> str:
    return f"Write a short, surprising story about {TOPICS[i % len(TOPICS)]} who discovers something unexpected one morning.\n\n"


def long_code_body(rng: random.Random, target_chars: int) -> str:
    out = ["# utility module\nimport math\nfrom dataclasses import dataclass\n\n"]
    k = 0
    while sum(len(x) for x in out) < target_chars:
        name = f"helper_{k}"
        out.append(f"def {name}(values: list[float], scale: float = {rng.uniform(0.5, 2.0):.3f}) -> float:\n"
                   f"    \"\"\"Scaled aggregate number {k} used by the pipeline.\"\"\"\n"
                   f"    total = 0.0\n    for v in values:\n        total += (v * scale) ** {rng.randint(1, 3)}\n"
                   f"    return math.sqrt(abs(total)) + {rng.randint(0, 99)}\n\n")
        k += 1
    return "".join(out)


def long_prose_body(rng: random.Random, target_chars: int) -> str:
    sentences = ["The harbour was quiet that morning, and the gulls had not yet come in from the rocks.",
                 "She kept the ledger in a drawer that stuck in wet weather, which was most weather.",
                 "Nobody in town remembered when the ferry had last run on a Sunday.",
                 "The letter had been folded so many times that the creases had become the message.",
                 "He measured the tide by the water line on the third step, as his father had.",
                 "Far out, a single light blinked twice and then held steady.",
                 "The clock in the square had been wrong for years and everyone had adjusted.",
                 "Rain arrived in the afternoon as if it had an appointment."]
    out = []
    while sum(len(x) for x in out) < target_chars:
        para = " ".join(rng.choice(sentences) for _ in range(rng.randint(5, 9)))
        out.append(para + "\n\n")
    return "".join(out)


def build_prompts(cls: str, n: int, seed: int, long_chars: int) -> list[str]:
    rng = random.Random(seed)
    if cls == "code-short":
        return [code_prompt(i) for i in range(n)]
    if cls == "prose-short":
        return [prose_prompt(i) for i in range(n)]
    m = re.match(r"(code|prose)-ctx(\d+)$", cls)
    if m:  # ~N tokens of context (code ~3.2 chars/token, prose ~4.6 chars/token)
        n_tok = int(m.group(2))
        if m.group(1) == "code":
            return [long_code_body(random.Random(seed * 1000 + i), int(n_tok * 3.2)) + f"\n# Task: {TASKS[i % len(TASKS)]}. Add the function below with tests.\n" for i in range(n)]
        return [long_prose_body(random.Random(seed * 1000 + i), int(n_tok * 4.6)) + f"Continue the story, now about {TOPICS[i % len(TOPICS)]}.\n\n" for i in range(n)]
    if cls == "code-mid":
        return [long_code_body(random.Random(seed * 1000 + i), long_chars // 2) + f"\n# Task: {TASKS[i % len(TASKS)]}. Add the function below with tests.\n" for i in range(n)]
    if cls == "prose-mid":
        return [long_prose_body(random.Random(seed * 1000 + i), long_chars // 2) + f"Continue the story, now about {TOPICS[i % len(TOPICS)]}.\n\n" for i in range(n)]
    if cls == "code-long":
        return [long_code_body(random.Random(seed * 1000 + i), long_chars) + f"\n# Task: {TASKS[i % len(TASKS)]}. Add the function below with tests.\n" for i in range(n)]
    if cls == "prose-long":
        return [long_prose_body(random.Random(seed * 1000 + i), long_chars) + f"Continue the story, now about {TOPICS[i % len(TOPICS)]}.\n\n" for i in range(n)]
    raise ValueError(cls)


def build_workload(cls: str, n: int, seed: int, long_chars: int) -> tuple[list[str], list[int]]:
    """Prompts and their release wave: long prompts first, short ones once the
    long ones have all started decoding."""
    if cls == "mix-sl":
        long_, short = build_prompts("prose-long", n // 2, seed, long_chars), build_prompts("code-short", n - n // 2, seed, long_chars)
        return long_ + short, [0] * len(long_) + [1] * len(short)
    if cls == "mix-ls":
        long_, short = build_prompts("code-long", n // 2, seed, long_chars), build_prompts("prose-short", n - n // 2, seed, long_chars)
        return long_ + short, [0] * len(long_) + [1] * len(short)
    p = build_prompts(cls, n, seed, long_chars)
    return p, [0] * len(p)


async def run_batch(base: str, model: str, prompts: list[str], max_tokens: int, waves: list[int] | None = None) -> dict[str, Any]:
    """Send all prompts of wave 0 at once; a later wave is released when every
    request of the previous wave has produced its first token (so a mixed
    batch of long and short prompts actually decodes together)."""
    waves = waves or [0] * len(prompts)
    n_waves = max(waves) + 1
    gates = [asyncio.Event() for _ in range(n_waves)]
    gates[0].set()
    pending = [sum(1 for w in waves if w == k) for k in range(n_waves)]

    async def one(client, p, wave):
        await gates[wave].wait()
        t0 = time.monotonic()
        first = None; n = 0; np_ = 0
        async with client.stream("POST", "/v1/completions", json={"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0,
                                                                  "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    d = json.loads(line[6:])
                    if first is None and d.get("choices") and d["choices"][0].get("text"):
                        first = time.monotonic()
                        pending[wave] -= 1
                        if pending[wave] == 0 and wave + 1 < n_waves:
                            gates[wave + 1].set()
                    if d.get("usage"):
                        n = d["usage"]["completion_tokens"]; np_ = d["usage"]["prompt_tokens"]
        return {"wave": wave, "t_send": t0, "ttft_s": (first or time.monotonic()) - t0, "t_first": first, "e2e_s": time.monotonic() - t0, "tokens": n, "prompt_tokens": np_}

    async with httpx.AsyncClient(base_url=base, timeout=900) as client:
        t0 = time.monotonic()
        res = await asyncio.gather(*(one(client, p, w) for p, w in zip(prompts, waves)))
        wall = time.monotonic() - t0
    toks = sum(r["tokens"] for r in res)
    t_all_first = max(r["t_first"] or t0 for r in res)
    t_end = time.monotonic()
    return {"wall_s": wall, "output_tokens": toks, "tok_per_s": toks / wall, "t_all_first": t_all_first, "requests": res}


def wait_ready(base: str, proc: subprocess.Popen, timeout: int) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            if httpx.get(base + "/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server start timeout")


def git_provenance() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--conditions", required=True, help="JSON: {name: speculative-config-json-or-empty}")
    ap.add_argument("--classes", default="code-short,prose-short,code-long,prose-long,mix-sl,mix-ls")
    ap.add_argument("--batch-sizes", default="8,16,32,64")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long-chars", type=int, default=20000, help="~5-6k tokens of synthetic code/prose")
    ap.add_argument("--max-long-batch", type=int, default=32, help="(unused; see --kv-token-capacity)")
    ap.add_argument("--kv-token-capacity", type=int, default=220000, help="skip (class, B) whose B*(ctx+max_tokens) exceeds this")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extra-server-args", default="--attention-backend TRITON_ATTN --max-model-len 16384 --max-num-seqs 64 --no-enable-prefix-caching")
    ap.add_argument("--trace-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    conditions = json.loads(args.conditions)
    classes = args.classes.split(",")
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    base = f"http://localhost:{args.port}"
    try:
        httpx.get(base + "/health", timeout=1); print("port busy", file=sys.stderr); sys.exit(2)
    except Exception:
        pass
    record = {"schema_version": 1, "result_type": "spec_geometry", "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "provenance": git_provenance(), "model": args.model, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "conditions": {}}
    for rep in range(args.repeats):
        for name, spec in conditions.items():
            env = dict(os.environ)
            if args.trace_dir:
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{args.output.stem}-{name}-r{rep}"
                env["VLLM_EXP_ITER_TRACE"] = str(args.trace_dir / f"{stem}.jsonl")
                env["VLLM_EXP_STEP_TRACE"] = str(args.trace_dir / f"{stem}.steps.jsonl")
                env["VLLM_EXP_SPEC_TRACE"] = str(args.trace_dir / f"{stem}.spec.jsonl")
            cmd = [args.vllm_bin, "serve", args.model, "--port", str(args.port)] + args.extra_server_args.split()
            if spec:
                cmd += ["--speculative-config", json.dumps(spec)]
            if args.trace_dir:
                cmd += ["--enable-logging-iteration-details"]
            log = open(args.output.with_name(f"{args.output.stem}-{name}-r{rep}.server.log"), "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                wait_ready(base, proc, 900)
                # warm-up
                asyncio.run(run_batch(base, args.model, build_prompts("code-short", 4, args.seed, args.long_chars), 16))
                def sizes_for(cls):
                    m = re.match(r"(code|prose)-ctx(\d+)$", cls)
                    ctx = int(m.group(2)) if m else (6000 if cls.endswith("long") or cls.startswith("mix") else 3000 if cls.endswith("mid") else 100)
                    return [b for b in sizes if b * (ctx + args.max_tokens) <= args.kv_token_capacity]
                cond = record["conditions"].setdefault(name, {"spec": spec, "cmd": cmd, "batches": []})
                for cls in classes:
                    for B in sizes_for(cls):
                        prompts, waves = build_workload(cls, B, args.seed, args.long_chars)
                        t_start = time.monotonic()
                        r = asyncio.run(run_batch(base, args.model, prompts, args.max_tokens, waves))
                        r.update({"class": cls, "B": B, "repeat": rep, "t_start": t_start, "t_end": time.monotonic()})
                        cond["batches"].append(r)
                        print(f"[{name} r{rep}] {cls:12s} B={B:3d}: {r['output_tokens']} tok in {r['wall_s']:.2f}s = {r['tok_per_s']:.0f} tok/s; "
                              f"ttft p50 {sorted(x['ttft_s'] for x in r['requests'])[B // 2]:.2f}s", flush=True)
                        args.output.write_text(json.dumps(record, indent=1) + "\n")
                        time.sleep(1.0)
                try:
                    m = httpx.get(base + "/metrics", timeout=10).text
                    cond["metrics_tail"] = [l for l in m.splitlines() if l.startswith("vllm:spec_decode_num_")]
                except Exception:
                    pass
            finally:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                log.close()
                time.sleep(3)
    args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
