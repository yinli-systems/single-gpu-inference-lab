#!/usr/bin/env python3
"""Run a real vLLM OpenAI server smoke for the L20 GEMM epilogue hook."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or local snapshot path.")
    parser.add_argument("--served-model-name", default="qwen-smoke")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/l20-vllm-gemm-smoke"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--repo-src", type=Path, default=Path("src"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--trace-limit", type=int, default=200)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-flashinfer-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set VLLM_USE_FLASHINFER_SAMPLER=0 to avoid FlashInfer JIT in smoke runs.",
    )
    return parser.parse_args()


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _pythonpath(args: argparse.Namespace) -> str:
    parts = [str(args.repo_src.resolve())]
    if args.vllm_source:
        parts.append(str(args.vllm_source.expanduser().resolve()))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def start_server(args: argparse.Namespace, trace_path: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(args)
    env["VLLM_L20_GEMM_EPILOGUE_TRACE"] = str(trace_path)
    env["VLLM_L20_GEMM_EPILOGUE_ENABLE"] = "1"
    env["VLLM_L20_GEMM_EPILOGUE_TRACE_LIMIT"] = str(args.trace_limit)
    env.setdefault("VLLM_L20_FLASHSAMPLING_CANDIDATE", "0")
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if args.disable_flashinfer_sampler:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    command = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--no-enable-log-requests",
    ]
    if args.vllm_source:
        cwd = args.vllm_source.expanduser().resolve()
    else:
        cwd = None
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited before ready with code {process.returncode}")
        try:
            return _request_json(f"{base_url}/v1/models", None, timeout=5.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    raise RuntimeError(f"vLLM server did not become ready: {last_error}")


def summarize_trace(path: Path, expected_events: int | None) -> dict[str, Any]:
    events = []
    if path.exists():
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    good = 0
    bad_reasons: list[dict[str, Any]] = []
    token_pairs: list[dict[str, Any]] = []
    for event in events:
        metadata = event.get("metadata", {})
        api = metadata.get("api", {})
        epilogue = metadata.get("epilogue", {})
        correctness = epilogue.get("correctness", {})
        checks = {
            "eligible": event.get("eligible") is True,
            "mutates_outputs": metadata.get("mutates_outputs") is True,
            "api_called": api.get("api_called") is True,
            "returned_output": epilogue.get("returned_output") is True,
            "uses_full_logits": epilogue.get("uses_full_logits") is False,
            "fallback_to_compute_logits": epilogue.get("fallback_to_compute_logits") is False,
            "correctness_checked": correctness.get("checked") is True,
            "matches_baseline_argmax": correctness.get("matches_baseline_argmax") is True,
        }
        if all(checks.values()):
            good += 1
        else:
            bad_reasons.append({"checks": checks, "reasons": event.get("reasons", [])})
        token_pairs.append(
            {
                "expected": correctness.get("expected_tokens"),
                "actual": correctness.get("actual_tokens"),
            }
        )
    trace_count_matches_completion = (
        isinstance(expected_events, int)
        and expected_events > 0
        and len(events) == expected_events
    )
    return {
        "trace_events": len(events),
        "expected_trace_events": expected_events,
        "trace_count_matches_completion": trace_count_matches_completion,
        "good_gemm_epilogue_events": good,
        "all_gemm_epilogue_events_ok": (
            bool(events) and good == len(events) and trace_count_matches_completion
        ),
        "bad_events": bad_reasons,
        "token_pairs": token_pairs,
    }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def terminate(process: subprocess.Popen) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 15
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            if process.poll() is None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(0.1)
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        if process.poll() is None:
            process.wait(timeout=15)
        return

    if process.poll() is not None:  # pragma: no cover - the GPU runner is Linux.
        return
    process.terminate()  # pragma: no cover - the GPU runner is Linux.
    try:  # pragma: no cover - the GPU runner is Linux.
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover - the GPU runner is Linux.
        process.kill()
        process.wait(timeout=15)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "gemm_epilogue_trace.jsonl"
    log_path = output_dir / "server.log"
    models_path = output_dir / "models.json"
    completion_path = output_dir / "completion.json"
    summary_path = args.summary_output or output_dir / "summary.json"
    for path in (trace_path, log_path, models_path, completion_path, summary_path):
        path.unlink(missing_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    process = start_server(args, trace_path, log_path)
    try:
        models = wait_ready(base_url, process, args.startup_timeout)
        models_path.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
        completion = _request_json(
            f"{base_url}/v1/completions",
            {
                "model": args.served_model_name,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "top_p": 1,
            },
            timeout=args.request_timeout,
        )
        completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
        completion_tokens = completion.get("usage", {}).get("completion_tokens")
        trace_summary = summarize_trace(trace_path, completion_tokens)
        summary = {
            "schema_version": 1,
            "server_ready": True,
            "model": args.model,
            "served_model_name": args.served_model_name,
            "completion_finish_reason": completion.get("choices", [{}])[0].get("finish_reason"),
            "completion_text": completion.get("choices", [{}])[0].get("text", ""),
            "artifacts": {
                "models": str(models_path),
                "completion": str(completion_path),
                "trace": str(trace_path),
                "server_log": str(log_path),
            },
            **trace_summary,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["all_gemm_epilogue_events_ok"] else 1
    finally:
        if not args.keep_server:
            terminate(process)


if __name__ == "__main__":
    raise SystemExit(main())
