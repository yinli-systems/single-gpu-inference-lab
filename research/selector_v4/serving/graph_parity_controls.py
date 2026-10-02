"""Finite new diagnostic controls; original natural parity failures stay HOLD.

Two uninstrumented ordinary Native processes test repeatability independently.
Two deterministic processes compare ordinary Native to the readback observer,
including full output/top5 logprobs. Each mode keeps all five workloads and
four blocks. No scored retry, Resource activation, gain or historical closure.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.graph_serving_probe import arm
from research.selector_v4.serving.http_measure import save_new
from research.selector_v4.serving.metric_gate import WORKLOADS
from research.selector_v4.serving.model_binding import verify_model
from research.selector_v4.serving.parity_gate import compare_blocks

MODES = ("ordinary_native_repeat", "deterministic_observer")


def comparison(root, *, right, logprobs):
    results = []
    for block in range(4):
        for name in WORKLOADS:
            key = f"{name}-b{block}.json"
            arms = [
                json.loads((root / role / "observations" / key).read_text())
                for role in ("native_baseline", right)
            ]
            results.append({"block": key, **compare_blocks(*arms, require_logprobs=logprobs)})
    return results


async def run(a):
    import torch
    from flashinfer._build_meta import __git_commit__

    need(not a.out.exists(), "Preserve every original diagnostic attempt")
    model = json.loads(a.model_binding.read_text())
    verify_model(model)
    need(__git_commit__ == "75544a17ce0019ca877f50354d95451ee089f859", "Pinned normal package")
    a.out.mkdir()
    original_out = a.out
    save_new(
        original_out / "environment.json",
        {
            "gpu_name": torch.cuda.get_device_properties(0).name,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "job": os.environ["SLURM_JOB_ID"],
            "model_binding_sha256": sha(a.model_binding),
            "modes": list(MODES),
            "original_natural_failure_retained": True,
            "qualification_authority": False,
            "full_http_qualified": False,
            "default_promotion": False,
            "serving_promotion": False,
        },
    )
    for mode in MODES:
        a.out = original_out / mode
        a.out.mkdir()
        right = "native_repeat" if mode == "ordinary_native_repeat" else "graph_diagnostic"
        deterministic = mode == "deterministic_observer"
        stage = "parity" if deterministic else "functional"
        for role in ("native_baseline", right):
            await arm(a, role, model, a.memory, stage=stage, logprobs=deterministic)
        results = comparison(a.out, right=right, logprobs=deterministic)
        save_new(
            a.out / "complete.json",
            {
                "complete": True,
                "mode": mode,
                "all_twenty_blocks_parity_pass": all(r["parity_pass"] for r in results),
                "comparison": results,
                "qualification_authority": False,
                "ordinary_natural_parity_failure_closed": False,
                "historical_token_divergence_resolved": False,
            },
        )
    verify_model(model)
    save_new(original_out / "complete.json", {"complete": True, "qualification_authority": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-binding", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--memory", type=float, required=True)
    asyncio.run(run(parser.parse_args()))
