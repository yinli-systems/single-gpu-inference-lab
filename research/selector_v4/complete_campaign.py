"""Finite sequential qualification, retaining HOLD and stopping all successors.

Runs existing source-bound controllers, never retries, publishes, activates
defaults, or claims that new parity resolves missing historical states.
"""

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

from research.selector_v4.exposed_diagnostics import complete_public as exposed
from research.selector_v4.public_qualification import archive as kernel_archive
from research.selector_v4.public_qualification import pipeline as kernel
from research.selector_v4.serving import http_archive, http_pipeline


def initialize(root, public, source, archive, models, sglang):
    exposed.need(not root.exists(), "Never replace a workflow")
    exposed.need(
        not (public / "receipts/controller.json").exists(),
        "Public controller must not already be running",
    )
    metadata = json.loads((source / "harness-commit.json").read_text())
    for name, expected in metadata["git_blob_files_sha256"].items():
        exposed.need(exposed.sha(source / name) == expected, "Committed future helper: " + name)
    with tarfile.open(archive) as bundle:
        exposed.need(
            json.load(bundle.extractfile("harness-commit.json")) == metadata,
            "Future archive identity",
        )
    for name in (
        "complete_public.py",
        "public_path_prequal.py",
        "public_path_contract.py",
        "analyze_public_path_prequal.py",
    ):
        relative = Path("research/selector_v4/exposed_diagnostics") / name
        exposed.need(
            exposed.sha(source / relative) == exposed.sha(public / "harness" / relative),
            "Public execution helpers must remain byte-identical",
        )
    root.mkdir()
    binding = {
        "public_campaign": str(public),
        "public_binding_sha256": exposed.sha(public / "binding.json"),
        "source": str(source),
        "source_archive": str(archive),
        "source_archive_sha256": exposed.sha(archive),
        "harness_commit": metadata["harness_commit"],
        "helper_files": metadata["git_blob_files_sha256"],
        "model_evidence": str(models),
        "sglang_source": str(sglang),
        "finite_days": 30,
        "no_retries_or_resampling": True,
        "fresh_generation_requires_archived_dual_development_pass": True,
        "http_requires_all_eight_kernel_stages": True,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    exposed.save(root / "binding.json", binding)
    return binding


def verify(root, binding):
    source = Path(binding["source"])
    exposed.need(
        exposed.sha(Path(binding["source_archive"])) == binding["source_archive_sha256"],
        "Prepared source archive changed",
    )
    for name, expected in binding["helper_files"].items():
        exposed.need(
            exposed.sha(source / name) == expected, "Prepared committed source changed: " + name
        )
    public = Path(binding["public_campaign"])
    exposed.need(
        exposed.sha(public / "binding.json") == binding["public_binding_sha256"],
        "Exposed campaign binding changed",
    )


def wait_terminal(jobs, deadline):
    while jobs:
        jobs = [job for job in jobs if not exposed.status(job)["terminal"]]
        exposed.need(time.monotonic() < deadline, "Finite archive wait; known jobs remain recorded")
        if jobs:
            time.sleep(30)


def child_run(runner, campaign):
    # Child controllers persist their own terminal receipt even on error.
    try:
        runner(campaign)
        return None
    except BaseException as error:  # noqa: BLE001 - retain terminal evidence, never resume
        return {"type": type(error).__name__, "message": str(error)}


def run(root):
    binding = json.loads((root / "binding.json").read_text())
    exposed.need(not (root / "controller.json").exists(), "One finite workflow; no retries")
    verify(root, binding)
    state = {
        "state": "EXPOSED_COMPLETE_REPETITION",
        "terminal": False,
        "stages": {},
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
        "full_http_qualified": False,
    }
    deadline = time.monotonic() + binding["finite_days"] * 86400

    def stage(label):
        exposed.need(time.monotonic() < deadline, "Finite workflow deadline")
        state["state"] = label
        exposed.save(root / "controller.json", state)
        verify(root, binding)

    exposed.save(root / "controller.json", state)
    try:
        public = Path(binding["public_campaign"])
        state["stages"]["public_controller"] = exposed.run(public)
        jobs = json.loads((public / "receipts/jobs.json").read_text())
        wait_terminal(
            [j for group in jobs.values() for j in group.values()],
            min(deadline, time.monotonic() + 13 * 3600),
        )
        receipt = exposed.archive(public, root / "public-archive")
        state["stages"]["public_archive"] = {
            "sha256": receipt["archive_sha256"],
            "pass": receipt["pass_public_path_development"],
        }
        exposed.need(
            receipt["pass_public_path_development"],
            "Complete dual public development HOLD; no new canary generation",
        )

        stage("FORMAL_KERNEL_SMOKE_DEV_CANARY_RELEASE_STRESS")
        formal = root / "kernel"
        # This source/protocol binding and new manifest happen only after the
        # complete development archive passed. Prepared source stays immutable.
        qualified_archive = root / "qualified-kernel-harness.tar.gz"
        shutil.copyfile(binding["source_archive"], qualified_archive)
        kernel.initialize(
            Path(binding["source"]),
            formal,
            root / "public-archive/receipt.json",
            qualified_archive,
            binding["harness_commit"],
        )
        error = child_run(kernel.run, formal)
        wait_terminal(
            list(kernel_archive.job_records(formal)), min(deadline, time.monotonic() + 13 * 3600)
        )
        receipt = kernel_archive.archive(formal, root / "kernel-archive")
        state["stages"]["kernel_archive"] = {
            "sha256": receipt["archive_sha256"],
            "pass": receipt["pass_formal_kernel_qualification"],
            "controller_error": error,
        }
        exposed.need(
            error is None and receipt["pass_formal_kernel_qualification"],
            "Formal kernel HOLD; no HTTP initializer",
        )

        stage("FULL_MODEL_PAIRED_HTTP_FUNCTIONAL_PERFORMANCE_PARITY")
        serving = root / "http"
        http_pipeline.initialize(
            serving,
            formal,
            qualified_archive,
            Path(binding["model_evidence"]),
            Path(binding["sglang_source"]),
        )
        error = child_run(http_pipeline.run, serving)
        ids = [
            json.loads(p.read_text())["job"] for p in (serving / "receipts").glob("dispatch-*.json")
        ]
        wait_terminal(ids, min(deadline, time.monotonic() + 9 * 3600))
        receipt = http_archive.archive(serving, root / "http-archive")
        state["stages"]["http_archive"] = {
            "sha256": receipt["archive_sha256"],
            "pass": receipt["full_http_qualified"],
            "controller_error": error,
        }
        exposed.need(error is None and receipt["full_http_qualified"], "Full model HTTP HOLD")
        state.update(
            state="KERNEL_HTTP_PASS_HISTORY_AND_UPSTREAM_REVIEW_REQUIRED", full_http_qualified=True
        )
    except BaseException as error:  # noqa: BLE001 - retain all child evidence and halt successors
        state.update(
            state="QUALIFICATION_WORKFLOW_HOLD",
            error={"type": type(error).__name__, "message": str(error)},
        )
    state["terminal"] = True
    exposed.save(root / "controller.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    for name in ("public", "source", "archive", "models", "sglang"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.initialize:
        exposed.need(
            all((args.public, args.source, args.archive, args.models, args.sglang)),
            "All frozen inputs",
        )
        initialize(args.root, args.public, args.source, args.archive, args.models, args.sglang)
    else:
        result = run(args.root)
        print(json.dumps({k: result[k] for k in ("state", "terminal", "full_http_qualified")}))
