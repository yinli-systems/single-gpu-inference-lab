"""Finite full-model Native HTTP/SLO baseline plus separate Graph driver audit.

Instrumentation belongs only to the diagnostic process. Both complete HTTP
matrices and every failure remain. No Resource certificate or promotion occurs.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

from research.release_qualification.serving.workloads import prefix_tokens, work_specs
from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.http_measure import (
    profile,
    ready,
    save_new,
    server_command,
    workloads,
)
from research.selector_v4.serving.metric_gate import WORKLOADS
from research.selector_v4.serving.model_binding import verify_model
from research.selector_v4.serving.parity_gate import compare_blocks
from research.selector_v4.serving.slo_report import summarize


async def arm(arguments, role, model, memory):
    import aiohttp

    out = arguments.out / role
    out.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = server_command(model["model_path"], port, memory, "pristine", "functional")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("SGI_FORMAL_KERNEL_CAMPAIGN", None)
    env.pop("SGI_GRAPH_SERVING_DIAGNOSTIC", None)
    if role == "graph_diagnostic":
        command[2] = "research.selector_v4.serving.graph_serving_server"
        env["SGI_GRAPH_SERVING_DIAGNOSTIC"] = str(out / "graph-epochs")
    env["FLASHINFER_WORKSPACE_BASE"] = str(out / "jit-cache")
    save_new(
        out / "command.json",
        {
            "command": command,
            "instrumented": role == "graph_diagnostic",
            "timing_valid": role == "native_baseline",
            "resource_enabled": False,
        },
    )
    log = (out / "server.log").open("x")
    process = subprocess.Popen(  # noqa: ASYNC220 - owned premeasurement startup
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    save_new(out / "owned-process.json", {"pid": process.pid, "port": port})
    url = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            await ready(session, url, process)
            async with session.get(url + "/get_server_info") as response:
                info = await response.json()
                need(response.status == 200, "Actual server config")
            save_new(out / "server-info.json", info)
            need(
                not info["disable_cuda_graph"] and not info["disable_overlap_schedule"],
                "Ordinary Graph/overlap serving required",
            )
            need(
                info["attention_backend"] == "flashinfer" and info["dtype"] == "bfloat16",
                "Actual backend and dtype",
            )
            warm = out / "warmup"
            warm.mkdir()
            await workloads(
                session, url, warm, arguments.model_id + "-warm-" + role, logprobs=False
            )
            observations = out / "observations"
            observations.mkdir()
            await workloads(
                session, url, observations, arguments.model_id + "-" + role, logprobs=False
            )
            await profile(
                session,
                url,
                out,
                observations,
                arguments.model_id + "-" + role + "-independent-profile",
            )
        reports = {}
        for block in range(4):
            specs = work_specs(prefix_tokens(), block)
            for name in WORKLOADS:
                key = f"{name}-b{block}.json"
                reports[key] = summarize(json.loads((observations / key).read_text()), specs[name])
        save_new(
            out / "descriptive-slo.json",
            {
                "blocks": reports,
                "timing_valid": role == "native_baseline",
                "independent_allocations": 1,
                "confidence_or_gain_claim": False,
                "full_http_qualified": False,
            },
        )
        save_new(
            out / "complete.json",
            {
                "complete": True,
                "role": role,
                "full_http_qualified": False,
                "files": {
                    str(p.relative_to(out)): sha(p)
                    for p in out.rglob("*")
                    if p.is_file() and "jit-cache" not in p.parts and p.name != "server.log"
                },
            },
        )
    except BaseException as error:
        save_new(
            out / "failure.json",
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


async def run(arguments):
    import torch
    from flashinfer._build_meta import __git_commit__

    need(not arguments.out.exists(), "Preserve all earlier diagnostic attempts")
    model = json.loads(arguments.model_binding.read_text())
    verify_model(model)
    need(__git_commit__ == "75544a17ce0019ca877f50354d95451ee089f859", "Pinned real normal package")
    props = torch.cuda.get_device_properties(0)
    arguments.out.mkdir()
    save_new(
        arguments.out / "environment.json",
        {
            "model_id": arguments.model_id,
            "model_binding_sha256": sha(arguments.model_binding),
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "job": os.environ["SLURM_JOB_ID"],
            "flashinfer_commit": __git_commit__,
            "torch_version": torch.__version__,
            "full_http_qualified": False,
            "default_promotion": False,
            "serving_promotion": False,
            "historical_token_divergence_resolved": False,
        },
    )
    for role in ("native_baseline", "graph_diagnostic"):
        await arm(arguments, role, model, arguments.memory)
    parity = []
    for block in range(4):
        for name in WORKLOADS:
            key = f"{name}-b{block}.json"
            a = json.loads((arguments.out / "native_baseline/observations" / key).read_text())
            b = json.loads((arguments.out / "graph_diagnostic/observations" / key).read_text())
            parity.append({"block": key, **compare_blocks(a, b, require_logprobs=False)})
    verify_model(model)
    save_new(
        arguments.out / "complete.json",
        {
            "complete": True,
            "native_instrumentation_token_parity": parity,
            "full_http_qualified": False,
            "qualification_authority": False,
            "resource_graph_serving_qualified": False,
            "default_promotion": False,
            "serving_promotion": False,
            "historical_token_divergence_resolved": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-binding", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--memory", type=float, required=True)
    asyncio.run(run(p.parse_args()))
