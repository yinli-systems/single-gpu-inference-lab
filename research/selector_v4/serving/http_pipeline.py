"""Prospective finite 72-allocation HTTP campaign, gated by all kernel stages.

No action on import; no retry, default activation or historical parity closure.
One complete global stage must pass before any allocation of its successor.
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.public_qualification.pipeline import (
    GPUS,
    ledger,
    now,
    save_new,
    terminal,
)
from research.selector_v4.serving.cccl_namespace import prepare as prepare_cccl
from research.selector_v4.serving.http_analysis import analyze
from research.selector_v4.serving.model_binding import verify_model
from research.selector_v4.serving.training_session import authorize_http_training

MODELS = ("qwen25-coder-1.5b", "qwen3-4b", "qwen25-7b", "qwen3-8b")
STAGES = ("functional", "performance", "parity")


def authorize(kernel):
    binding = json.loads((kernel / "frozen-binding.json").read_text())
    return binding, authorize_http_training(kernel, binding["candidate_commit"])


def initialize(root, kernel, archive, model_evidence, sglang_source):
    # Kernel HOLD must fail before even creating a campaign or copying a file.
    qualified, authorization = authorize(kernel)
    need(not root.exists(), "Never replace an HTTP campaign")
    complete = json.loads((model_evidence / "complete.json").read_text())
    config = json.loads((model_evidence / "models_6754cc8.json").read_text())
    need(complete["complete"] and set(complete["models"]) == set(MODELS), "All four models")
    for name in MODELS:
        path = model_evidence / (name + ".json")
        need(sha(path) == complete["models"][name]["binding_sha256"], "Model evidence changed")
        verify_model(json.loads(path.read_text()))
    root.mkdir()
    for name in ("harness", "models", "logs", "receipts", "paired", "analysis", "sdk-libraries"):
        (root / name).mkdir()
    shutil.copyfile(archive, root / "harness.tar.gz")
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            path = Path(member.name)
            need(
                member.isfile() and not path.is_absolute() and ".." not in path.parts,
                "Committed regular helper files only",
            )
        stream.extractall(root / "harness")
    metadata = json.loads((root / "harness/harness-commit.json").read_text())
    for name, digest in metadata["git_blob_files_sha256"].items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts, "Safe committed helper path")
        need(sha(root / "harness" / path) == digest, "Committed HTTP helper changed")
    sglang_files = ledger(sglang_source)
    need(
        sglang_files["python/sglang/srt/layers/attention/flashinfer_backend.py"]
        == "5c8baba0d14eeca4c9d4bef8674318be1697b0160f105224282cd4aa27575810"
        and sglang_files["python/sglang/srt/managers/scheduler.py"]
        == "13f93b9f21f0ef09a5a951e837db45b6790b6b7f054193ac806d9d8322dcecac",
        "Reviewed SGLang source required",
    )
    models = {}
    for name in MODELS:
        shutil.copyfile(model_evidence / (name + ".json"), root / "models" / (name + ".json"))
        models[name] = {
            "binding_sha256": sha(root / "models" / (name + ".json")),
            "mem_fraction": config[name]["mem_fraction"],
        }
    for source in (kernel / "sdk-libraries").iterdir():
        shutil.copyfile(source, root / "sdk-libraries" / source.name)
    cccl = prepare_cccl(root / "sdk-cccl")
    binding = {
        "scope": "FULL_QWEN_PAIRED_HTTP_PROSPECTIVE",
        "kernel_campaign": str(kernel),
        "kernel_binding_sha256": sha(kernel / "frozen-binding.json"),
        "candidate_commit": qualified["candidate_commit"],
        "pristine_commit": qualified["pristine_commit"],
        "harness_commit": metadata["harness_commit"],
        "harness_archive_sha256": sha(root / "harness.tar.gz"),
        "helper_files": metadata["git_blob_files_sha256"],
        "sglang_source": str(sglang_source),
        "sglang_files": sglang_files,
        "cuda13_cccl_namespace_sha256": sha(root / "sdk-cccl/binding.json"),
        "cuda13_cccl_header_files": cccl["headers"],
        "models": models,
        "stages": list(STAGES),
        "gpus": list(GPUS),
        "allocations_per_model_gpu_stage": 3,
        "maximum_inflight_allocations": 4,
        "finite_controller_days": 14,
        "all_metric_floor": 0.99,
        "full_http_qualified": False,
        "default_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    save_new(root / "binding.json", binding)
    save_new(root / "receipts/kernel-authorization.json", authorization)
    files = [root / "binding.json", root / "harness.tar.gz"]
    files += [root / "harness" / n for n in binding["helper_files"]]
    files += [sglang_source / n for n in sglang_files]
    files += list((root / "models").iterdir()) + list((root / "sdk-libraries").iterdir())
    files += [p for p in (root / "sdk-cccl").rglob("*") if p.is_file()]
    (root / "validation.sha256").write_text("".join(f"{sha(p)}  {p}\n" for p in files))
    save_new(root / "receipts/frozen-ledger.json", {"sha256": sha(root / "validation.sha256")})
    return binding


def verify_frozen(root, binding):
    authorize(Path(binding["kernel_campaign"]))
    need(
        sha(Path(binding["kernel_campaign"]) / "frozen-binding.json")
        == binding["kernel_binding_sha256"],
        "Qualified kernel binding changed",
    )
    frozen = json.loads((root / "receipts/frozen-ledger.json").read_text())
    need(sha(root / "validation.sha256") == frozen["sha256"], "Frozen source ledger changed")
    subprocess.run(
        ["sha256sum", "-c", str(root / "validation.sha256")],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=300,
    )


def allocation_terminal(root, job):
    if not terminal(root, job):
        return False
    path = root / f"receipts/allocation-terminal-{job}.json"
    if not path.exists():
        raw = subprocess.check_output(
            ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed", "-n", "-P"],
            text=True,
            timeout=60,
        ).strip()
        cells = raw.split("|")
        need(len(cells) == 4 and cells[:3] == [job, "COMPLETED", "0:0"], "Actual Slurm completion")
        paths = [
            root / f"receipts/source-{job}.txt",
            root / f"receipts/source-post-{job}.txt",
            root / f"receipts/exit-{job}.txt",
        ]
        save_new(
            path,
            {
                "job": job,
                "sacct": raw,
                "utc": now(),
                "files": {str(p.relative_to(root)): sha(p) for p in paths},
            },
        )
    return True


def run(root):
    binding = json.loads((root / "binding.json").read_text())
    authorize(Path(binding["kernel_campaign"]))
    save_new(root / "receipts/controller-start.json", {"pid": os.getpid(), "utc": now()})
    deadline = time.monotonic() + binding["finite_controller_days"] * 86400
    result = {
        "terminal": False,
        "state": "FULL_HTTP_VALIDATING",
        "binding_sha256": sha(root / "binding.json"),
        "default_promotion": False,
        "historical_token_divergence_resolved": False,
        "full_http_qualified": False,
        "analyses": {},
    }
    try:
        verify_frozen(root, binding)
        script = root / "harness/research/selector_v4/serving/http_pair.sbatch"
        for stage in STAGES:
            jobs, active = {}, []
            for model in MODELS:
                for gpu in GPUS:
                    jobs[model + ":" + gpu] = []
                    for allocation in range(3):
                        active = [j for j in active if not allocation_terminal(root, j)]
                        need(time.monotonic() < deadline, "Finite HTTP controller deadline")
                        while len(active) >= binding["maximum_inflight_allocations"]:
                            active = [j for j in active if not allocation_terminal(root, j)]
                            need(time.monotonic() < deadline, "Finite HTTP controller deadline")
                            if len(active) >= binding["maximum_inflight_allocations"]:
                                time.sleep(15)
                        label = f"{stage}-{model}-{gpu}-{allocation}"
                        save_new(root / f"receipts/intent-{label}.json", {"utc": now()})
                        job = (
                            subprocess.check_output(
                                [
                                    "sbatch",
                                    "--parsable",
                                    "-p",
                                    gpu,
                                    "-o",
                                    str(root / "logs/%j.out"),
                                    "-e",
                                    str(root / "logs/%j.err"),
                                    str(script),
                                    str(root),
                                    binding["kernel_campaign"],
                                    model,
                                    stage,
                                    str(allocation),
                                ],
                                text=True,
                                timeout=60,
                            )
                            .strip()
                            .split(";")[0]
                        )
                        need(job.isdigit(), "Unambiguous actual Slurm allocation")
                        save_new(
                            root / f"receipts/dispatch-{label}.json",
                            {
                                "job": job,
                                "stage": stage,
                                "model": model,
                                "gpu": gpu,
                                "allocation": allocation,
                                "utc": now(),
                            },
                        )
                        jobs[model + ":" + gpu].append(job)
                        active.append(job)
            while active:
                active = [j for j in active if not allocation_terminal(root, j)]
                need(time.monotonic() < deadline, "Finite HTTP controller deadline")
                if active:
                    time.sleep(15)
            verify_frozen(root, binding)
            for model in MODELS:
                for gpu in GPUS:
                    label = stage + "-" + model + "-" + gpu
                    output = root / "analysis" / label
                    summary = analyze(
                        root,
                        output,
                        jobs=jobs[model + ":" + gpu],
                        gpu=gpu,
                        model_id=model,
                        stage=stage,
                    )
                    result["analyses"][label] = sha(output / "summary.json")
                    need(summary["pass"], "Complete HTTP stage HOLD: " + label)
        need(len(result["analyses"]) == 24, "All four models, two GPUs, three stages")
        result.update(
            state="FULL_HTTP_QUALIFICATION_PASS_HISTORY_REQUIRED", full_http_qualified=True
        )
    except BaseException:
        result.update(state="FULL_HTTP_QUALIFICATION_HOLD", traceback=traceback.format_exc())
        raise
    finally:
        result.update(terminal=True, finished_utc=now())
        save_new(root / "receipts/controller-terminal.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--model-evidence", type=Path)
    parser.add_argument("--sglang-source", type=Path)
    args = parser.parse_args()
    if args.initialize:
        need(all((args.kernel, args.archive, args.model_evidence, args.sglang_source)), "Inputs")
        initialize(args.root, args.kernel, args.archive, args.model_evidence, args.sglang_source)
    else:
        run(args.root)
