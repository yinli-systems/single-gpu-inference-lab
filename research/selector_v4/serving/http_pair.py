"""Sequential full-model arms on one Slurm GPU/CPU allocation, no automatic retry."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.training_session import authorize_http_training


def run(root, kernel_campaign, model_id, stage, allocation):
    root, kernel_campaign = Path(root), Path(kernel_campaign)
    binding = json.loads((root / "binding.json").read_text())
    kernel = json.loads((kernel_campaign / "frozen-binding.json").read_text())
    authorize_http_training(kernel_campaign, kernel["candidate_commit"])
    need(
        sha(kernel_campaign / "frozen-binding.json") == binding["kernel_binding_sha256"],
        "Qualified kernel changed",
    )
    need(
        stage in ("functional", "performance", "parity") and allocation in range(3),
        "Frozen HTTP stage/allocation",
    )
    model = binding["models"][model_id]
    model_binding = root / "models" / (model_id + ".json")
    need(sha(model_binding) == model["binding_sha256"], "Complete model binding changed")
    job = os.environ["SLURM_JOB_ID"]
    out = root / "paired" / job
    out.mkdir()  # A repeated allocation ID cannot replace earlier measurements.
    for name in ("logs", "receipts", "cache"):
        (out / name).mkdir()
    roles = ("candidate", "pristine") if allocation == 1 else ("pristine", "candidate")
    code = root / "harness/research/selector_v4/serving/http_measure.py"
    need(
        sha(code) == binding["helper_files"][str(code.relative_to(root / "harness"))],
        "Frozen measurement helper changed",
    )
    try:
        for role in roles:
            source = Path(kernel[role + "_source"])
            phase = role + "-" + stage
            cache = root / "cache" / job / phase
            cccl = source / "flashinfer/data/cccl"
            sdk = root / "sdk-libraries"
            env = dict(
                os.environ,
                PYTHONPATH=":".join(map(str, (root / "harness", Path(binding["sglang_source"]) / "python", source))),
                FLASHINFER_WORKSPACE_BASE=str(cache),
                XDG_CACHE_HOME=str(root / "cache" / ("xdg-" + job) / phase),
                CPATH=":".join(
                    map(str, (sdk, root / "sdk-cccl", cccl / "libcudacxx/include", cccl / "cub", cccl / "thrust"))
                ),
                NVCC_PREPEND_FLAGS=f"-I{sdk} -I{root / 'sdk-cccl'} -I{cccl / 'libcudacxx/include'} -I{cccl / 'cub'} -I{cccl / 'thrust'}",
            )
            command = [
                sys.executable,
                str(code),
                "--binding",
                str(root / "binding.json"),
                "--kernel-campaign",
                str(kernel_campaign),
                "--model-binding",
                str(model_binding),
                "--harness",
                str(root / "harness"),
                "--sglang-source",
                binding["sglang_source"],
                "--flashinfer-source",
                str(source),
                "--out",
                str(out / role),
                "--model-id",
                model_id,
                "--mem-fraction",
                str(model["mem_fraction"]),
                "--role",
                role,
                "--stage",
                stage,
                "--allocation",
                str(allocation),
            ]
            with (out / "logs" / (role + ".log")).open("x") as log:
                subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=24000,
                )
        audit = root / "harness/research/selector_v4/serving/http_sass_audit.py"
        need(
            sha(audit) == binding["helper_files"][str(audit.relative_to(root / "harness"))],
            "Frozen independent SASS helper changed",
        )
        audit_env = dict(os.environ, PYTHONPATH=str(root / "harness"))
        with (out / "logs/independent-sass.log").open("x") as log:
            subprocess.run(
                [sys.executable, str(audit), str(root), job, str(out / "independent-sass")],
                env=audit_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=5400,
            )
        (out / "exit.txt").write_text("0\n")
    except BaseException as error:
        (out / "failure.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "all_started_results_retained": True,
                }
            )
            + "\n"
        )
        (out / "exit.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("kernel_campaign", type=Path)
    parser.add_argument("model_id")
    parser.add_argument("stage", choices=("functional", "performance", "parity"))
    parser.add_argument("allocation", type=int, choices=range(3))
    args = parser.parse_args()
    run(args.root, args.kernel_campaign, args.model_id, args.stage, args.allocation)
