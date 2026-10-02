"""Finite full-model diagnostic: real captured Resource, exact HTTP parity.

Source-bound and gated before allocation. All timings are invalid: duplicate
Native computation, CPU readbacks and CUPTI are intentional correctness probes.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.http_client import collect_block
from research.selector_v4.serving.http_measure import (
    post,
    ready,
    save_new,
    server_command,
)
from research.selector_v4.serving.http_pipeline import authorize
from research.selector_v4.serving.model_binding import verify_model
from research.selector_v4.serving.parity_gate import compare_blocks
from research.selector_v4.serving.resource_graph_contract import resource_trace
from research.selector_v4.serving.workload_modes import verify_configuration

MODELS = {
    "qwen25-coder-1.5b": 0.70,
    "qwen3-4b": 0.70,
    "qwen25-7b": 0.76,
    "qwen3-8b": 0.78,
}
SCOPE = "FULL_MODEL_FIXED_128_PREFILL_FUNCTIONAL_DIAGNOSTIC"


def protocol():
    """Expose exact cold shape, padded shapes and a cached-prefix fallback."""
    rows = []
    for i, (length, prime) in enumerate(
        ((128, 0), (128, 256), (128, 512), (127, 0), (129, 0), (256, 0))
    ):
        ids = [1000 + i * 1000 + j % 431 for j in range(length)]
        rows.append(
            {
                "name": f"case-{i}",
                "concurrency": 1,
                "prime_tokens": prime,
                "seed_ids": ids[:128] if i == 5 else [],
                "expect_resource": i < 3,
                "cells": [
                    {
                        "id": f"case-{i}",
                        "input_ids": ids,
                        "output_tokens": 8,
                        "expect_cached": 128 if i == 5 else 0,
                    }
                ],
            }
        )
    return rows


def command(model, port, memory, role):
    cmd = server_command(model, port, memory, "pristine", "fixed_parity")
    cmd[2] = (
        "research.selector_v4.serving.resource_graph_server"
        if role == "resource"
        else "sglang.launch_server"
    )
    cmd[cmd.index("--page-size") + 1] = "1"
    cmd[cmd.index("--context-length") + 1] = "2048"
    cmd[cmd.index("--cuda-graph-max-bs-decode") + 1] = "1"
    cmd += [
        "--cuda-graph-config",
        json.dumps(
            {
                "decode": {"backend": "full", "bs": [1]},
                "prefill": {
                    "backend": "full",
                    "bs": [128, 256],
                    "full_prefill_max_req": 1,
                },
            }
        ),
    ]
    return cmd


def events(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and (
            path.name.endswith(".json") or path.name.endswith(".json.gz")
        ):
            raw = (
                gzip.decompress(path.read_bytes())
                if path.suffix == ".gz"
                else path.read_bytes()
            )
            value = json.loads(raw)
            if isinstance(value, dict) and "traceEvents" in value:
                result.extend(value["traceEvents"])
    return result


def records(root):
    rows = []
    for path in sorted(root.glob("resource-graph-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def validate_case(work, trace, rows, *, layers, resource):
    need(
        any(e.get("cat") == "kernel" for e in trace),
        "Actual GPU trace required, including Native fallbacks",
    )
    replays = [r for r in rows if r["event"] == "resource_replay"]
    need(
        not any(r["event"] == "replay_failure" for r in rows),
        "No concealed replay failure",
    )
    kernels = [
        e
        for e in trace
        if e.get("cat") == "kernel" and "ResourceKernel" in e.get("name", "")
    ]
    if not resource or not work["expect_resource"]:
        need(
            not kernels and not replays,
            "Native/fallback must execute zero Resource kernels",
        )
        return {"pass": True, "resource_replays": 0, "resource_kernels": 0}
    need(len(replays) == 1, "One real Resource replay per exact request")
    row = replays[0]
    need(row["full_model_output_exact"] is True, "Native/Resource model body equality")
    need(
        len(row["layers"]) == layers
        and all(v["out_lse_exact"] is True for v in row["layers"].values()),
        "All model layers exact O/LSE",
    )
    proof = resource_trace(trace)
    need(
        proof["markers"] == 1 and proof["actual_resource_graph_kernels"] == layers,
        "CUPTI must prove exactly one captured Resource kernel per real model layer",
    )
    need(
        len(kernels) == layers, "No additional uncorrelated or eager Resource execution"
    )
    return {
        "pass": True,
        "resource_replays": 1,
        "resource_kernels": len(kernels),
        "trace": proof,
        "epoch": row,
    }


async def auxiliary(session, url, ids, output, label):
    work = {
        "name": label,
        "concurrency": 1,
        "cells": [
            {"id": label, "input_ids": ids, "output_tokens": 2, "expect_cached": 0}
        ],
    }
    value = await collect_block(session, url, work, label, logprobs=True)
    save_new(output / (label + ".json"), value)
    need(value["complete"], "Failed priming/seed request retained; no retry")


async def arm(a, model, role):
    import aiohttp

    out = a.out / role
    out.mkdir()
    trace_rows = out / "epochs"
    trace_rows.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = command(model["model_path"], port, MODELS[a.model_id], role)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("SGI_RESOURCE_GRAPH_DIAGNOSTIC", "SGI_GRAPH_SERVING_DIAGNOSTIC"):
        env.pop(name, None)
    env["SGI_FORMAL_KERNEL_CAMPAIGN"] = str(a.kernel)
    if role == "resource":
        env["SGI_RESOURCE_GRAPH_DIAGNOSTIC"] = str(trace_rows)
    env["FLASHINFER_WORKSPACE_BASE"] = str(out / "jit-cache")
    save_new(out / "command.json", {"argv": cmd, "role": role, "timing_valid": False})
    log = (out / "server.log").open("x")
    process = subprocess.Popen(  # noqa: ASYNC220 - owned startup before diagnostic requests
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    save_new(out / "owned-process.json", {"pid": process.pid, "port": port})
    proofs = []
    try:
        url = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            await ready(session, url, process)
            async with session.get(url + "/get_server_info") as response:
                info = await response.json()
                need(response.status == 200, "Actual resolved server info")
            save_new(out / "server-info.json", info)
            verify_configuration(info, "fixed_parity")
            need(
                info["page_size"] == 1
                and info["dtype"] == "bfloat16"
                and info["attention_backend"] == "flashinfer",
                "Pinned actual NHD page1 BF16 backend",
            )
            need(
                info["cuda_graph_config"]["prefill"]["backend"] == "full",
                "Actual full prefill backend required",
            )
            for work in protocol():
                case = out / work["name"]
                case.mkdir()
                await post(session, url, "/flush_cache", {})
                if work["prime_tokens"]:
                    await auxiliary(
                        session,
                        url,
                        [19000 + j % 521 for j in range(work["prime_tokens"])],
                        case,
                        "prime",
                    )
                if work["seed_ids"]:
                    await auxiliary(session, url, work["seed_ids"], case, "seed")
                before = records(trace_rows)
                trace = case / "profile"
                trace.mkdir()
                await post(
                    session,
                    url,
                    "/start_profile",
                    {
                        "output_dir": str(trace),
                        "activities": ["CPU", "GPU"],
                        "record_shapes": True,
                    },
                )
                try:
                    value = await collect_block(session, url, work, role, logprobs=True)
                    save_new(case / "http.json", value)
                    need(value["complete"], "Failed full HTTP request retained")
                finally:
                    await post(session, url, "/stop_profile", {})
                after = records(trace_rows)
                need(after[: len(before)] == before, "Epoch history append-only")
                save_new(case / "epochs.json", after[len(before) :])
                proof = validate_case(
                    work,
                    events(trace),
                    after[len(before) :],
                    layers=model["layers"],
                    resource=role == "resource",
                )
                save_new(case / "proof.json", proof)
                proofs.append(proof)
        if role == "resource":
            epochs = [r["epoch"] for r in proofs if r["resource_replays"]]
            need(
                len(epochs) == 3 and len({e["graph_object"] for e in epochs}) == 1,
                "Three HTTP requests must reuse the same actual captured model graph",
            )
            need(
                len({e["epoch"] for e in epochs}) == 3, "Distinct announced load epochs"
            )
            for layer in epochs[0]["layers"]:
                for key in ("q_sha256", "k_sha256", "v_sha256", "output_sha256"):
                    need(
                        len({e["layers"][layer][key] for e in epochs}) == 3,
                        "Actual per-layer payload must change across all three HTTP epochs",
                    )
            need(
                len(
                    {
                        e["layers"][next(iter(e["layers"]))]["page_indices_sha256"]
                        for e in epochs
                    }
                )
                >= 2,
                "Actual page allocation change must be observed",
            )
        save_new(
            out / "complete.json",
            {"complete": True, "proofs": proofs, "timing_valid": False},
        )
    except BaseException as error:
        save_new(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "server_exit": process.poll(),
            },
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


def audit_run(root, model):
    """Recompute from raw HTTP, trace and append-only worker epochs."""
    comparisons, proofs = [], []
    all_epochs = records(root / "resource/epochs")
    need(
        not any(r["event"] == "replay_failure" for r in all_epochs),
        "No hidden worker failure",
    )
    for work in protocol():
        pair = []
        for role in ("native", "resource"):
            case = root / role / work["name"]
            raw = json.loads((case / "http.json").read_text())
            expected_hash = hashlib.sha256(
                json.dumps(work, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            need(
                raw["workload_sha256"] == expected_hash and len(raw["requests"]) == 1,
                "Exact preregistered workload population",
            )
            row = raw["requests"][0]
            need(
                row["id"] == work["cells"][0]["id"]
                and row["payload"]["input_ids"] == work["cells"][0]["input_ids"]
                and row["payload"]["sampling_params"]["max_new_tokens"] == 8,
                "Actual expected prompt and output length",
            )
            epoch_rows = json.loads((case / "epochs.json").read_text())
            need(
                role == "native" or all(e in all_epochs for e in epoch_rows),
                "Case epochs must exist in original append-only worker log",
            )
            proof = validate_case(
                work,
                events(case / "profile"),
                epoch_rows,
                layers=model["layers"],
                resource=role == "resource",
            )
            proofs.append(proof)
            pair.append(raw)
        comparisons.append(compare_blocks(*pair, require_logprobs=True))
    need(all(c["parity_pass"] for c in comparisons), "Raw complete token/top5 parity")
    return {
        "pass": True,
        "comparisons": comparisons,
        "proofs": proofs,
        "timing_valid": False,
        "full_http_qualified": False,
    }


async def run(a):
    authorization = authorize(a.kernel)  # Before Torch import or server allocation.
    import torch
    from flashinfer._build_meta import __git_commit__

    need(
        __git_commit__ == "75544a17ce0019ca877f50354d95451ee089f859",
        "Exact candidate package",
    )
    model = json.loads(a.model_binding.read_text())
    verify_model(model)
    need(
        0 < model["layers"] <= 40 and model["head_dim"] == 128,
        "Reviewed model geometry",
    )
    a.out.mkdir()
    save_new(
        a.out / "binding.json",
        {
            "scope": SCOPE,
            "protocol": protocol(),
            "kernel_authorization": authorization,
            "model_binding_sha256": sha(a.model_binding),
            "gpu_name": torch.cuda.get_device_properties(0).name,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "job": os.environ["SLURM_JOB_ID"],
            "timing_valid": False,
        },
    )
    for role in ("native", "resource"):
        await arm(a, model, role)
    comparisons = []
    for work in protocol():
        pair = [
            json.loads((a.out / role / work["name"] / "http.json").read_text())
            for role in ("native", "resource")
        ]
        comparisons.append(compare_blocks(*pair, require_logprobs=True))
    verify_model(model)
    passed = all(r["parity_pass"] for r in comparisons)
    save_new(
        a.out / "complete.json",
        {
            "pass": passed,
            "scope": SCOPE,
            "comparisons": comparisons,
            "full_http_qualified": False,
            "resource_graph_serving_qualified": False,
            "historical_token_divergence_resolved": False,
            "original_natural_parity_failure_closed": False,
            "timing_valid": False,
            "default_promotion": False,
            "serving_promotion": False,
        },
    )
    need(passed, "Complete HTTP token/top5 mismatch; original outcomes retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--model-binding", type=Path, required=True)
    parser.add_argument("--model-id", choices=MODELS, required=True)
    asyncio.run(run(parser.parse_args()))
