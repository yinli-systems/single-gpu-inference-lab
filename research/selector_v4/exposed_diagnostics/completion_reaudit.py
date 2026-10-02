"""Separate read-only-data reanalysis after the reference-directory checker bug.

Original terminal HOLD/archive stay immutable. No measurements, new jobs,
changed choices, thresholds, omitted records or automatic stage dispatch.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(campaign, original_archive, corrected_helper, output):
    assert not output.exists()
    output.mkdir()
    spec = importlib.util.spec_from_file_location("corrected_completeness", corrected_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old_controller = json.loads((campaign / "receipts/controller.json").read_text())
    assert (
        old_controller["terminal"]
        and old_controller["error"]["message"] == "All nine original phases"
    )
    old = helper.verify_archive(original_archive)  # Every complete original byte.
    assert not old["pass_public_path_development"]
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())
    for name, digest in old["files"].items():
        assert sha(campaign / name) == digest, name
    helper.verify_sources(campaign)
    references = {}
    import torch

    def tensor_hash(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    for gpu, group in jobs.items():
        for case, job in group.items():
            assert helper.status(job)["completed"]
            helper.completeness(campaign, gpu, case, job)
            root = campaign / f"runs/{gpu}-{job}"
            for path in sorted((root / "references").glob("*.pt")):
                value = torch.load(path, map_location="cpu", weights_only=True)
                assert set(value["windows"]) == {
                    "eager_full_call",
                    "graph1_replay",
                    "graph16_replay",
                }
                hashes = [tensor_hash(value[k]) for k in ("out", "lse")]
                for phase in helper.PHASES:
                    for execution in value["windows"]:
                        cell = json.loads(
                            (
                                root / phase / (path.stem + "-" + execution) / "complete.json"
                            ).read_text()
                        )
                        assert hashes == cell["native_output_lse_sha256"]
                        assert value["inputs_sha256"] == cell["input_hashes"]
                        assert value["plan"] == cell["actual_plan"]
                        assert cell["exact"] and cell["binding_failures"] == 0
                references[str(path.relative_to(campaign))] = {
                    "sha256": sha(path),
                    "out_lse_sha256": hashes,
                }
    assert len(references) == 32
    save(
        output / "input-integrity.json",
        {
            "original_archive_sha256": old["archive_sha256"],
            "original_archive_members_verified": len(old["files"]),
            "references": references,
            "references_checked_against_all_864_cells": True,
            "corrected_helper_sha256": sha(corrected_helper),
            "old_controller_sha256": sha(campaign / "receipts/controller.json"),
            "old_controller_retained_hold": True,
            "measurement_or_threshold_changes": False,
        },
    )
    shadow = output / "verified-inputs"
    # Hardlinks only immutable input bytes. New analyses go to new directories;
    # do not overwrite any linked original receipt, helper or measurement file.
    shutil.copytree(
        campaign,
        shadow,
        copy_function=os.link,
        ignore=shutil.ignore_patterns("cache", "sdk-libraries", "__pycache__", ".pytest_cache"),
    )
    (shadow / "cache").symlink_to(campaign / "cache", target_is_directory=True)
    (shadow / "sdk-libraries").symlink_to(campaign / "sdk-libraries", target_is_directory=True)
    try:
        for group in jobs.values():
            for job in group.values():
                target = shadow / f"receipts/independent-sass-{job}"
                with (output / f"sass-{job}.log").open("x") as log:
                    subprocess.run(
                        [
                            helper.E,
                            str(shadow / "audit_public_path_sass.py"),
                            str(shadow),
                            job,
                            str(target),
                        ],
                        env={
                            **os.environ,
                            "PYTHONPATH": str(shadow) + ":" + str(shadow / "harness"),
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=5400,
                    )
                assert json.loads((target / "receipt.json").read_text())["pass"]
        summaries = {}
        for gpu in jobs:
            target = shadow / f"analysis/{gpu}"
            target.parent.mkdir(exist_ok=True)
            with (output / f"analysis-{gpu}.log").open("x") as log:
                subprocess.run(
                    [
                        helper.E,
                        str(shadow / "analyze_public_path_prequal.py"),
                        str(shadow),
                        gpu,
                        str(target),
                    ],
                    env={**os.environ, "PYTHONPATH": str(shadow) + ":" + str(shadow / "harness")},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=3600,
                )
            summaries[gpu] = json.loads((target / "summary.json").read_text())
        helper.verify_sources(campaign)
        for name, digest in old["files"].items():
            assert sha(campaign / name) == digest, name
        save(
            output / "result.json",
            {
                "pass_complete_data_development_reaudit": all(
                    v["pass"] for v in summaries.values()
                ),
                "summaries": {
                    g: {
                        "sha256": sha(shadow / f"analysis/{g}/summary.json"),
                        "pass": s["pass"],
                        "metrics": s["metrics"],
                        "failed_requirements": [k for k, v in s["requirements"].items() if not v],
                    }
                    for g, s in summaries.items()
                },
                "original_controller_and_archive_remain_hold": True,
                "measurement_or_threshold_changes": False,
                "gpu_jobs_dispatched": 0,
                "fresh_cases_consumed": 0,
                "qualification_authority": False,
                "full_http_qualified": False,
                "default_promotion": False,
                "serving_promotion": False,
                "historical_token_divergence_resolved": False,
            },
        )
    except BaseException:
        save(
            output / "failure.json",
            {
                "traceback": traceback.format_exc(),
                "old_hold_preserved": True,
                "full_http_qualified": False,
                "default_promotion": False,
                "serving_promotion": False,
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("campaign", "original-archive", "corrected-helper", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.campaign, a.original_archive, a.corrected_helper, a.output)
