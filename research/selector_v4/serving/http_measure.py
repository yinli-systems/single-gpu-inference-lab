"""Prospective full-model HTTP measurement after complete formal kernel PASS.

No action on import. All five workloads, four blocks and full failures remain.
Ordinary serving and deterministic token/logprob trials are separate processes.
This draft has not yet run a real model or granted HTTP qualification.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from research.release_qualification.serving.workloads import prefix_tokens, suffix, work_specs
from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.http_client import collect_block
from research.selector_v4.serving.metric_gate import WORKLOADS
from research.selector_v4.serving.model_binding import verify_model
from research.selector_v4.serving.training_session import authorize_http_training


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def server_command(model, port, memory, role, stage):
    command = [sys.executable, "-m", "research.selector_v4.serving.prefix_training_server"]
    if role == "pristine":
        command = [sys.executable, "-m", "sglang.launch_server"]
    command += [
        "--model-path",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--attention-backend",
        "flashinfer",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        str(memory),
        "--context-length",
        "12288",
        "--chunked-prefill-size",
        "1024",
        "--max-running-requests",
        "16",
        "--cuda-graph-max-bs-decode",
        "16",
        "--page-size",
        "16",
        "--skip-tokenizer-init",
        "--random-seed",
        "42",
    ]
    if stage == "parity":
        command.append("--enable-deterministic-inference")
    return command


async def post(session, url, path, payload):
    async with session.post(url + path, json=payload) as response:
        body = await response.text()
        need(response.status == 200, f"{path}: HTTP {response.status}: {body[:2000]}")
        return body


async def ready(session, url, process):
    import aiohttp

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        need(process.poll() is None, "Owned server exited during startup")
        try:
            async with session.get(url + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        # aiohttp's connector errors subclass OSError; no failed measurement is
        # retried here, only the explicit startup readiness endpoint.
        await asyncio.sleep(0.5)
    raise TimeoutError("Finite full-model startup deadline")


async def phase(session, url, output, command):
    deadline = time.monotonic() + 60
    while True:
        values = json.loads(
            await post(
                session, url, "/set_internal_state", {"server_args": {"sgi_prefix_phase": command}}
            )
        )
        need(isinstance(values, list) and values, "Actual scheduler acknowledgements required")
        if all(value is True for value in values):
            return
        need(
            not list(output.glob("control-failure-*.json")),
            "Worker phase failed; keep first attempt",
        )
        need(time.monotonic() < deadline, "Scheduler did not reach a safe idle phase boundary")
        await asyncio.sleep(0.2)


async def seed(session, url, prefix, output, tag):
    await post(session, url, "/flush_cache", {})
    work = {
        "name": "seed",
        "concurrency": 1,
        "cells": [{"id": "seed", "input_ids": prefix, "output_tokens": 2, "expect_cached": 0}],
    }
    result = await collect_block(session, url, work, tag)
    save_new(output / (tag + ".json"), result)
    need(result["complete"], "Failed prefix seed; no measurement retry")


async def workloads(session, url, output, tag, *, logprobs):
    prefix = prefix_tokens()
    for block in range(4):
        specs = work_specs(prefix, block)
        order = WORKLOADS[block % 5 :] + WORKLOADS[: block % 5]
        for name in order:
            work = specs[name]
            if work["seed"]:
                await seed(session, url, prefix, output, f"{tag}-seed-{name}-b{block}")
            result = await collect_block(session, url, work, f"{tag}-b{block}", logprobs=logprobs)
            save_new(output / f"{name}-b{block}.json", result)
            need(result["complete"], "Full HTTP workload failed; all outcomes retained")


async def profile(session, url, root, output, tag):
    traces = root / "profile"
    traces.mkdir()
    await post(
        session,
        url,
        "/start_profile",
        {"output_dir": str(traces), "activities": ["CPU", "GPU"], "record_shapes": True},
    )
    try:
        await seed(session, url, prefix_tokens(), output, tag + "-seed")
        result = await collect_block(
            session, url, work_specs(prefix_tokens(), 99)["guarded_prefix"], tag
        )
        save_new(output / "profile-workload.json", result)
        need(result["complete"], "Independent profile workload failed")
    finally:
        await post(session, url, "/stop_profile", {})
    save_new(
        output / "profile-timing-scope.json",
        {
            "timing_excluded": True,
            "trace_files": {
                str(f.relative_to(root)): sha(f) for f in traces.rglob("*") if f.is_file()
            },
        },
    )


def verify_sources(arguments, binding, kernel):
    need(
        binding["candidate_commit"] == kernel["candidate_commit"],
        "HTTP/kernel candidate source mismatch",
    )
    need(
        binding["pristine_commit"] == kernel["pristine_commit"],
        "HTTP/kernel pristine source mismatch",
    )
    need(
        sha(arguments.kernel_campaign / "frozen-binding.json") == binding["kernel_binding_sha256"],
        "Kernel binding changed",
    )
    for name, digest in binding["helper_files"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "Unsafe HTTP helper path")
        need(sha(arguments.harness / relative) == digest, "HTTP helper changed: " + name)
    for name, digest in binding["sglang_files"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "Unsafe SGLang source path")
        need(sha(arguments.sglang_source / relative) == digest, "SGLang source changed: " + name)
    role = arguments.role
    source = arguments.flashinfer_source
    need(str(source) == kernel[role + "_source"], "Use the actual qualified normal package")
    need(
        sha(source / "source.sha256") == kernel["source_ledger_sha256"][role],
        "Normal package ledger changed",
    )
    subprocess.run(
        ["sha256sum", "-c", "source.sha256"],
        cwd=source,
        stdout=subprocess.DEVNULL,
        check=True,
        timeout=120,
    )


async def run(arguments):
    need(not arguments.out.exists(), "Preserve every earlier HTTP attempt")
    binding = json.loads(arguments.binding.read_text())
    kernel = json.loads((arguments.kernel_campaign / "frozen-binding.json").read_text())
    authorization = authorize_http_training(arguments.kernel_campaign, kernel["candidate_commit"])
    verify_sources(arguments, binding, kernel)
    import aiohttp
    import torch
    from flashinfer._build_meta import __git_commit__, __version__

    need(
        (__git_commit__, __version__) == (kernel[arguments.role + "_commit"], "0.7.1"),
        "Actual loaded normal package identity",
    )
    model = json.loads(arguments.model_binding.read_text())
    verify_model(model)
    need(
        sha(arguments.model_binding) == binding["models"][arguments.model_id]["binding_sha256"],
        "Complete model binding changed",
    )
    properties = torch.cuda.get_device_properties(0)
    gpu = os.environ["SLURM_JOB_PARTITION"]
    need(
        gpu in ("gpu_4090", "gpu_5090") and gpu.removeprefix("gpu_") in properties.name,
        "Actual GPU family and allocation required",
    )
    raw_uuid = str(properties.uuid)
    gpu_uuid = raw_uuid if raw_uuid.startswith("GPU-") else "GPU-" + raw_uuid
    arguments.out.mkdir(parents=True)
    training = arguments.out / "training"
    observations = arguments.out / "observations"
    observations.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = server_command(
        model["model_path"], port, arguments.mem_fraction, arguments.role, arguments.stage
    )
    env = dict(
        os.environ,
        PYTHONPATH=":".join(
            map(
                str,
                (
                    arguments.harness,
                    arguments.sglang_source / "python",
                    arguments.flashinfer_source,
                ),
            )
        ),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SGI_HTTP_TRAINING_OUTPUT=str(training),
    )
    env.pop("SGI_FORMAL_KERNEL_CAMPAIGN", None)
    if arguments.role == "candidate":
        env["SGI_FORMAL_KERNEL_CAMPAIGN"] = str(arguments.kernel_campaign)
    save_new(
        arguments.out / "environment.json",
        {
            "command": command,
            "authorization": authorization,
            "role": arguments.role,
            "stage": arguments.stage,
            "allocation": arguments.allocation,
            "model_id": arguments.model_id,
            "model_binding_sha256": sha(arguments.model_binding),
            "http_binding_sha256": sha(arguments.binding),
            "actual_flashinfer_commit": __git_commit__,
            "actual_flashinfer_version": __version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "full_http_qualified": False,
        },
    )
    log = (arguments.out / "server.log").open("x")
    process = subprocess.Popen(  # noqa: ASYNC220 - premeasurement owned server launch
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    save_new(arguments.out / "owned-process.json", {"pid": process.pid, "port": port})
    url = f"http://127.0.0.1:{port}"
    tag = f"{arguments.model_id}-{arguments.role}-{arguments.stage}-a{arguments.allocation}"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=7200), connector=aiohttp.TCPConnector(limit=16)
        ) as session:
            await ready(session, url, process)
            async with session.get(url + "/get_server_info") as response:
                info = await response.json()
                need(response.status == 200, "Actual resolved server configuration required")
            save_new(arguments.out / "actual-server-info.json", info)
            need(
                all(info[key] == 1 for key in ("tp_size", "pp_size", "dp_size")),
                "This qualification requires the actual single-GPU topology",
            )
            need(
                info["attention_backend"] == "flashinfer" and info["dtype"] == "bfloat16",
                "Actual serving backend/dtype changed",
            )
            if arguments.stage != "parity":
                need(
                    not info["disable_cuda_graph"] and not info["disable_overlap_schedule"],
                    "Ordinary Graph/overlap serving required",
                )
            # Both arms receive identical complete premeasurement workloads.
            # Only candidate enables separate explicit training during this phase.
            if arguments.role == "candidate":
                await phase(session, url, training, 1)
            preparation = arguments.out / "premeasurement"
            preparation.mkdir()
            await workloads(session, url, preparation, tag + "-calibration", logprobs=False)
            if arguments.role == "candidate":
                await phase(session, url, training, 2)
            frozen_warmup = arguments.out / "frozen-warmup"
            frozen_warmup.mkdir()
            await workloads(session, url, frozen_warmup, tag + "-frozen", logprobs=False)
            warm = {
                "name": "conditioning",
                "concurrency": 2,
                "cells": [
                    {
                        "id": str(i),
                        "input_ids": suffix(1024, 20261001 + i),
                        "output_tokens": 16,
                        "expect_cached": 0,
                    }
                    for i in range(2)
                ],
            }
            warm_result = await collect_block(session, url, warm, tag + "-conditioning")
            save_new(arguments.out / "conditioning.json", warm_result)
            need(warm_result["complete"], "Conditioning failed")
            await workloads(session, url, observations, tag, logprobs=arguments.stage == "parity")
            if arguments.role == "candidate":
                await phase(session, url, training, 3)
            await profile(session, url, arguments.out, observations, tag + "-profile")
            if arguments.role == "candidate":
                await phase(session, url, training, 3)
                need(
                    not list(training.rglob("failure.json"))
                    and not list(training.glob("control-failure-*.json")),
                    "Incomplete first calibration blocks HTTP completion",
                )
        verify_sources(arguments, binding, kernel)
        verify_model(model)
        save_new(
            arguments.out / "complete.json",
            {
                "complete": True,
                "role": arguments.role,
                "stage": arguments.stage,
                "allocation": arguments.allocation,
                "model_id": arguments.model_id,
                "files": {
                    str(f.relative_to(arguments.out)): sha(f)
                    for f in arguments.out.rglob("*")
                    if f.is_file() and f.name != "server.log"
                },
                "full_http_qualified": False,
                "historical_token_divergence_resolved": False,
            },
        )
    except BaseException as error:
        save_new(
            arguments.out / "failure.json",
            {"type": type(error).__name__, "message": str(error), "server_exit": process.poll()},
        )
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "binding",
        "kernel-campaign",
        "model-binding",
        "harness",
        "sglang-source",
        "flashinfer-source",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mem-fraction", type=float, required=True)
    parser.add_argument("--role", choices=("pristine", "candidate"), required=True)
    parser.add_argument("--stage", choices=("functional", "performance", "parity"), required=True)
    parser.add_argument("--allocation", type=int, choices=range(3), required=True)
    asyncio.run(run(parser.parse_args()))
