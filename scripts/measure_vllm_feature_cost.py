#!/usr/bin/env python3
"""Measure the incremental serving cost of probability/support output features.

Stage-1 baseline for the output-contract work. Each *server condition* starts
its own ``vllm serve`` process (the sampling mask is an engine-level flag), and
each *request condition* replays the identical rollout-shaped workload against
it. Rounds are interleaved so drift affects every condition equally.

Server conditions
  native              plain engine
  mask                --return-sampling-mask --logprobs-mode processed_logprobs

Request conditions (per server)
  gen                 no logprobs
  logprobs            logprobs=1 (sampled-token logprob)
  (the mask server additionally returns ``sampling_mask`` on logprobs requests)

The workload uses ``ignore_eos`` with a fixed ``max_tokens`` so every condition
generates exactly the same number of tokens; latency differences are then
attributable to the output path, not to different completion lengths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def build_prompts(count: int, prompt_tokens: int, seed: int) -> list[str]:
    """Deterministic pseudo-natural prompts of roughly ``prompt_tokens`` words."""

    import random

    rng = random.Random(seed)
    vocab = (
        "the model should explain why a tensor parallel rank can compute its local "
        "maximum and partial softmax denominator before exchanging compact statistics "
        "with peers so that no vocabulary sized gather is needed for the sampled token "
        "probability in reinforcement learning rollouts where support information is "
        "returned alongside each generated token id and later replayed by the trainer"
    ).split()
    prompts = []
    for i in range(count):
        words = [rng.choice(vocab) for _ in range(prompt_tokens)]
        prompts.append(f"Prompt {i}: " + " ".join(words) + "\nContinue:")
    return prompts


async def one_generate_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    token_ids: list[int],
    *,
    max_tokens: int,
    sampling: dict[str, Any],
    logprobs: int | None,
    seed: int,
) -> dict[str, Any]:
    """Non-streaming ``/inference/v1/generate`` request (token in, token out).

    This is the endpoint that carries ``sampling_mask`` in vLLM v0.29.0 and the
    shape an RL rollout collector uses, so every condition is measured on it.
    """

    params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "seed": seed,
        **sampling,
    }
    if isinstance(logprobs, dict):
        params.update(logprobs)
    elif logprobs is not None:
        params["logprobs"] = logprobs
    payload = {"model": model, "token_ids": token_ids, "sampling_params": params, "stream": False}
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        raw = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:400]!r}")
    end = time.perf_counter()
    obj = json.loads(raw)
    choice = obj["choices"][0]
    out_tokens = len(choice["token_ids"])
    usage = obj.get("usage") or {}
    if usage.get("completion_tokens") is not None:
        out_tokens = int(usage["completion_tokens"])
    mask = choice.get("sampling_mask")
    mask_sizes = [len(m) for m in mask] if mask else []
    lp = choice.get("logprobs")
    flat = choice.get("token_logprobs")
    got_logprobs = bool(lp and (lp.get("content") or lp.get("token_logprobs"))) or bool(flat)
    if flat is not None and len(flat) != out_tokens:
        raise RuntimeError(f"token_logprobs length {len(flat)} != {out_tokens} tokens")
    return {
        "ttft_s": end - start,
        "flat_token_logprobs": flat is not None,
        "e2e_s": end - start,
        "itl_s": [],
        "chunks": 1,
        "output_tokens": out_tokens,
        "text_chunks": 0,
        "got_logprobs": got_logprobs,
        "mask_entries": len(mask_sizes),
        "mask_mean_size": statistics.fmean(mask_sizes) if mask_sizes else None,
        "mask_max_size": max(mask_sizes) if mask_sizes else None,
        "body_bytes": len(raw),
        "token_ids": list(choice["token_ids"]),
        "sampling_mask": mask,
    }


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    sampling: dict[str, Any],
    logprobs: int | None,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
        **sampling,
    }
    if isinstance(logprobs, dict):
        payload.update(logprobs)
    elif logprobs is not None:
        payload["logprobs"] = logprobs
    start = time.perf_counter()
    first_token = None
    chunk_times: list[float] = []
    text_chunks = 0
    got_logprobs = False
    mask_sizes: list[int] = []
    body_bytes = 0
    completion_tokens = None
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            body_bytes += len(raw)
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter()
            if first_token is None:
                first_token = now
            chunk_times.append(now)
            obj = json.loads(data)
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            if not obj.get("choices"):
                continue
            choice = obj["choices"][0]
            if choice.get("text"):
                text_chunks += 1
            lp = choice.get("logprobs")
            if lp and lp.get("token_logprobs"):
                got_logprobs = True
            mask = choice.get("sampling_mask")
            if mask:
                # HTTP shape: list of per-token candidate lists (or dict with token_ids)
                if isinstance(mask, dict):
                    mask = mask.get("token_ids", [])
                for entry in mask:
                    mask_sizes.append(len(entry) if isinstance(entry, list) else 0)
    end = time.perf_counter()
    itls = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    if completion_tokens is None:
        raise RuntimeError("server did not report usage.completion_tokens")
    return {
        "ttft_s": (first_token or end) - start,
        "e2e_s": end - start,
        "itl_s": itls,
        "chunks": len(chunk_times),
        "output_tokens": completion_tokens,
        "text_chunks": text_chunks,
        "got_logprobs": got_logprobs,
        "mask_entries": len(mask_sizes),
        "mask_mean_size": statistics.fmean(mask_sizes) if mask_sizes else None,
        "mask_max_size": max(mask_sizes) if mask_sizes else None,
        "body_bytes": body_bytes,
    }


async def run_round(
    base_url: str,
    model: str,
    prompts: list[str],
    *,
    concurrency: int,
    max_tokens: int,
    sampling: dict[str, Any],
    logprobs: int | None,
    seed: int,
    prompt_token_ids: list[list[int]] | None = None,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    generate_api = prompt_token_ids is not None
    url = f"{base_url}/inference/v1/generate" if generate_api else f"{base_url}/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def guarded(i: int, prompt: str):
            async with sem:
                if generate_api:
                    return await one_generate_request(
                        session,
                        url,
                        model,
                        prompt_token_ids[i],
                        max_tokens=max_tokens,
                        sampling=sampling,
                        logprobs=logprobs,
                        seed=seed + i,
                    )
                return await one_request(
                    session,
                    url,
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    sampling=sampling,
                    logprobs=logprobs,
                    seed=seed + i,
                )

        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(i, p) for i, p in enumerate(prompts)))
        wall = time.perf_counter() - start
    total_tokens = sum(r["output_tokens"] for r in results)
    total_chunks = sum(r["chunks"] for r in results)
    itls = [x for r in results for x in r["itl_s"]] or [float("nan")]
    ttfts = [r["ttft_s"] for r in results]
    e2es = [r["e2e_s"] for r in results]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "wall_s": wall,
        "requests": len(results),
        "output_tokens": total_tokens,
        "stream_chunks": total_chunks,
        "tokens_per_chunk": total_tokens / max(total_chunks, 1),
        "output_tokens_per_s": total_tokens / wall,
        "ttft_ms": {"median": 1e3 * statistics.median(ttfts), "p99": 1e3 * pct(ttfts, 0.99)},
        "itl_ms": {
            "mean": 1e3 * statistics.fmean(itls),
            "median": 1e3 * statistics.median(itls),
            "p99": 1e3 * pct(itls, 0.99),
        },
        "e2e_ms": {"median": 1e3 * statistics.median(e2es), "p99": 1e3 * pct(e2es, 0.99)},
        "all_got_logprobs": all(r["got_logprobs"] for r in results),
        "all_flat_token_logprobs": all(r.get("flat_token_logprobs", False) for r in results),
        "mask_entries_total": sum(r["mask_entries"] for r in results),
        "mask_mean_size": statistics.fmean(
            [r["mask_mean_size"] for r in results if r["mask_mean_size"] is not None]
        )
        if any(r["mask_mean_size"] is not None for r in results)
        else None,
        "mask_max_size": max([r["mask_max_size"] or 0 for r in results]),
        "body_bytes_total": sum(r["body_bytes"] for r in results),
        "outputs": [
            {"token_ids": r.get("token_ids"), "sampling_mask": r.get("sampling_mask")}
            for r in results
        ]
        if collect_outputs
        else None,
    }


class Server:
    def __init__(self, args: argparse.Namespace, extra: list[str], log_path: Path):
        self.args = args
        self.extra = extra
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{args.port}"

    def __enter__(self):
        import socket

        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError(
                    f"port {self.args.port} already answers; a stale server would be "
                    "measured instead of the intended condition"
                )
        cmd = [
            self.args.vllm_bin,
            "serve",
            self.args.model,
            "--port",
            str(self.args.port),
            "--max-model-len",
            str(self.args.max_model_len),
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.args.max_num_seqs),
            "--seed",
            "0",
            *self.extra,
        ]
        if self.args.served_model_name:
            cmd += ["--served-model-name", self.args.served_model_name]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.log.write(" ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=os.environ)
        self.cmd = cmd
        deadline = time.time() + self.args.startup_timeout
        import urllib.request

        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except Exception:
                time.sleep(2)
        raise RuntimeError("server did not become healthy")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        return False


def nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    q = "name,driver_version,clocks.current.sm,memory.used,temperature.gpu,power.draw"
    out = subprocess.check_output([exe, f"--query-gpu={q}", "--format=csv,noheader"], text=True)
    return {"available": True, "fields": q.split(","), "rows": [r.split(", ") for r in out.strip().splitlines()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="m")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--server-conditions",
        default="native,mask",
        help="comma list from {native,native_fi_off,native_processed,mask,mask_fi_off,"
        "mask_compact,mask_bitmap}; "
        "each starts its own vllm serve",
    )
    parser.add_argument("--request-conditions", default="gen,logprobs")
    parser.add_argument("--extra-server-args", default="", help="appended to every vllm serve")
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable set for every server (e.g. VLLM_BATCH_INVARIANT=1)",
    )
    parser.add_argument(
        "--api",
        choices=("generate", "completions"),
        default="generate",
        help="generate = non-streaming /inference/v1/generate (carries sampling_mask); "
        "completions = streaming /v1/completions (ITL/TTFT, no mask)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sampling = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p}
    prompts = build_prompts(args.prompts, args.prompt_tokens, args.seed)
    warm = build_prompts(args.warmup_prompts, args.prompt_tokens, args.seed + 1)
    prompt_ids = warm_ids = None
    if args.api == "generate":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        prompt_ids = [tok(p)["input_ids"] for p in prompts]
        warm_ids = [tok(p)["input_ids"] for p in warm]
    server_flags = {
        "native": [],
        "mask": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        "mask_fi_off": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        # same flags; selected by env on a vLLM carrying the compact-mask patch
        "mask_compact": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        "mask_bitmap": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        # same flags on a vLLM that carries upstream #54901 (0.29.1+): the
        # engine's own compact layout, no env needed
        "mask_upstream": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        # native engine with the FlashInfer sampler disabled: isolates the cost
        # of the top-k/top-p + Gumbel fallback that the mask server also uses
        "native_fi_off": [],
        # native engine in processed-logprobs mode without the mask
        "native_processed": ["--logprobs-mode", "processed_logprobs"],
        # token-in/token-out server without a tokenizer: removes per-token
        # logprob detokenization on the API side
        "native_skip_tokenizer": ["--skip-tokenizer-init"],
        # upper-bound experiment: upstream compact ids without producing or
        # copying the bitmask (raises on tie overflow; not exact)
        "mask_nobitmap": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        # experiment: generate endpoint emits flat token_logprobs (env-gated patch)
        "native_flat": [],
        "mask_upstream_flat": ["--return-sampling-mask", "--logprobs-mode", "processed_logprobs"],
        "mask_upstream_skip_tokenizer": [
            "--return-sampling-mask", "--logprobs-mode", "processed_logprobs", "--skip-tokenizer-init",
        ],
    }
    server_env = {
        "native_fi_off": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "mask_nobitmap": {"VLLM_MASK_SKIP_BITMAP": "1"},
        "native_flat": {"VLLM_GENERATE_FLAT_TOKEN_LOGPROBS": "1"},
        "mask_upstream_flat": {"VLLM_GENERATE_FLAT_TOKEN_LOGPROBS": "1"},
        "mask_fi_off": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "mask_compact": {"VLLM_SAMPLING_MASK_COMPACT": "1"},
        "mask_bitmap": {"VLLM_SAMPLING_MASK_COMPACT": "0"},
    }
    request_logprobs = {
        "gen": None,
        "logprobs": 1,  # sampled token + top-1 (what the earlier artifacts measured)
        "logprobs0": 0,  # sampled-token logprob only: the RL-correct request
        "logprobs0_flat": {"logprobs": 0, "flat_logprobs": True},  # + existing engine knob
    }
    extra = args.extra_server_args.split() if args.extra_server_args else []

    record: dict[str, Any] = {
        "schema_version": 1,
        "result_type": "vllm_feature_cost_baseline",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": git_provenance(),
        "model": args.model,
        "workload": {
            "api": args.api,
            "prompts": args.prompts,
            "prompt_tokens_approx": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "concurrency": args.concurrency,
            "rounds": args.rounds,
            "sampling": sampling,
            "seed": args.seed,
        },
        "environment": {},
        "servers": {},
        "nvidia_smi_before": nvidia_smi(),
    }
    try:
        ver = subprocess.check_output([args.vllm_bin, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover
        ver = f"unavailable: {exc}"
    record["environment"]["vllm_version_output"] = ver
    record["environment"]["python"] = sys.version.split()[0]

    for server_name in args.server_conditions.split(","):
        flags = server_flags[server_name] + extra
        env_backup = dict(os.environ)
        shared_env = dict(kv.split("=", 1) for kv in args.server_env)
        os.environ.update(shared_env)
        os.environ.update(server_env.get(server_name, {}))
        log_path = args.output.with_suffix("") / f"server-{server_name}.log"
        entry: dict[str, Any] = {
            "flags": flags,
            "env": {**shared_env, **server_env.get(server_name, {})},
            "log": str(log_path),
            "rounds": [],
        }
        record["servers"][server_name] = entry
        with Server(args, flags, log_path) as server:
            entry["command"] = server.cmd
            # warmup: every request condition once, small
            for rc in args.request_conditions.split(","):
                asyncio.run(
                    run_round(
                        server.base_url,
                        args.served_model_name,
                        warm,
                        concurrency=args.concurrency,
                        max_tokens=args.max_tokens,
                        sampling=sampling,
                        logprobs=request_logprobs[rc],
                        seed=args.seed,
                        prompt_token_ids=warm_ids,
                    )
                )
            for r in range(args.rounds):
                order = args.request_conditions.split(",")
                if r % 2 == 1:
                    order = order[::-1]
                for rc in order:
                    result = asyncio.run(
                        run_round(
                            server.base_url,
                            args.served_model_name,
                            prompts,
                            concurrency=args.concurrency,
                            max_tokens=args.max_tokens,
                            sampling=sampling,
                            logprobs=request_logprobs[rc],
                            seed=args.seed,
                            prompt_token_ids=prompt_ids,
                            collect_outputs=(r == 0 and args.api == "generate"),
                        )
                    )
                    outputs = result.pop("outputs", None)
                    if outputs is not None:
                        dump = args.output.with_suffix("") / f"outputs-{server_name}-{rc}-r0.json"
                        dump.parent.mkdir(parents=True, exist_ok=True)
                        dump.write_text(json.dumps(outputs))
                        result["outputs_file"] = str(dump)
                    entry["rounds"].append({"round": r, "request_condition": rc, **result})
                    print(
                        f"[{server_name}/{rc} r{r}] {result['output_tokens_per_s']:.0f} tok/s "
                        f"ITL med {result['itl_ms']['median']:.2f} ms  e2e med {result['e2e_ms']['median']:.0f} ms "
                        f"tok/chunk {result['tokens_per_chunk']:.2f} mask_entries={result['mask_entries_total']} mask_mean={result['mask_mean_size']}",
                        flush=True,
                    )
            # capture sampler selection evidence from the server log
            try:
                text = log_path.read_text(errors="replace")
                lines = [ln.strip() for ln in text.splitlines() if "Initializing a V1 LLM engine" not in ln]
                entry["log_markers"] = {
                    "sampler_backend": [ln for ln in lines if "topk_topp_sampler" in ln][:5],
                    "flashinfer_sampler_selected": any("Using FlashInfer for top-p" in ln for ln in lines),
                    "cuda_graph_capture_seen": any("Capturing CUDA graphs" in ln for ln in lines),
                    "warnings": [ln for ln in lines if "WARNING" in ln and "sampl" in ln.lower()][:10],
                }
            except Exception:
                pass
        os.environ.clear()
        os.environ.update(env_backup)

    record["nvidia_smi_after"] = nvidia_smi()

    # aggregate: median across rounds per (server, request condition)
    summary: dict[str, Any] = {}
    for server_name, entry in record["servers"].items():
        for rc in args.request_conditions.split(","):
            rows = [x for x in entry["rounds"] if x["request_condition"] == rc]
            if not rows:
                continue
            summary[f"{server_name}/{rc}"] = {
                "output_tokens_per_s_median": statistics.median(x["output_tokens_per_s"] for x in rows),
                "output_tokens_per_s_all": [x["output_tokens_per_s"] for x in rows],
                "itl_ms_median_of_medians": statistics.median(x["itl_ms"]["median"] for x in rows),
                "itl_ms_mean_median": statistics.median(x["itl_ms"]["mean"] for x in rows),
                "e2e_ms_median_of_medians": statistics.median(x["e2e_ms"]["median"] for x in rows),
                "ttft_ms_median_of_medians": statistics.median(x["ttft_ms"]["median"] for x in rows),
                "mask_entries_total": rows[0]["mask_entries_total"],
                "mask_mean_size": rows[0]["mask_mean_size"],
                "body_bytes_total": rows[0]["body_bytes_total"],
                "tokens_per_chunk": rows[0]["tokens_per_chunk"],
            }
    base = summary.get("native/gen")
    if base:
        for key, val in summary.items():
            val["throughput_vs_native_gen"] = val["output_tokens_per_s_median"] / base["output_tokens_per_s_median"]
            val["itl_mean_vs_native_gen"] = val["itl_ms_mean_median"] / base["itl_ms_mean_median"]
    record["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
