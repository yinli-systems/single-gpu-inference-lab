"""New finite kernel-to-HTTP workflow after independently archived correction.

Original workflows/controllers remain terminal HOLD. No original allocation is
retried and no gate, case geometry, threshold or choice is edited. Resource HTTP
still requires all eight exact-source formal kernel stage verdicts. The default
and serving switches stay off even after later experimental qualification.
"""

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

from research.selector_v4.complete_campaign import child_run, wait_terminal
from research.selector_v4.exposed_diagnostics import complete_public as exposed
from research.selector_v4.public_qualification import archive as kernel_archive
from research.selector_v4.public_qualification import pipeline as kernel
from research.selector_v4.serving import http_archive, http_pipeline


def initialize(root, source, archive, prerequisite, models, sglang):
    exposed.need(not root.exists(), "New workflow; original terminal HOLD cannot resume")
    receipt = exposed.verify_archive(prerequisite.parent)
    exposed.need(
        receipt["pass_public_path_development"]
        and receipt["controller"]["original_terminal_hold_retained"] is True,
        "Separately accepted complete development correction required",
    )
    metadata = json.loads((source / "harness-commit.json").read_text())
    for name, digest in metadata["git_blob_files_sha256"].items():
        exposed.need(exposed.sha(source / name) == digest, "Exact new committed helper")
    with tarfile.open(archive) as bundle:
        exposed.need(
            json.load(bundle.extractfile("harness-commit.json")) == metadata,
            "Frozen archive and extracted source identity",
        )
    root.mkdir()
    binding = {
        "source": str(source),
        "source_archive": str(archive),
        "source_archive_sha256": exposed.sha(archive),
        "harness_commit": metadata["harness_commit"],
        "helper_files": metadata["git_blob_files_sha256"],
        "prerequisite": str(prerequisite),
        "prerequisite_receipt_sha256": exposed.sha(prerequisite),
        "prerequisite_archive_sha256": receipt["archive_sha256"],
        "original_terminal_hold_retained": True,
        "model_evidence": str(models),
        "sglang_source": str(sglang),
        "finite_days": 30,
        "no_scored_retries_or_resampling": True,
        "http_requires_all_eight_exact_source_kernel_stages": True,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    exposed.save(root / "binding.json", binding)
    return binding


def verify(binding):
    source = Path(binding["source"])
    exposed.need(
        exposed.sha(Path(binding["source_archive"])) == binding["source_archive_sha256"],
        "New committed archive unchanged",
    )
    for name, digest in binding["helper_files"].items():
        exposed.need(exposed.sha(source / name) == digest, "Frozen continuation helper changed")
    prerequisite = Path(binding["prerequisite"])
    exposed.need(
        exposed.sha(prerequisite) == binding["prerequisite_receipt_sha256"]
        and exposed.sha(prerequisite.parent / "raw-public-path-development.tar.gz")
        == binding["prerequisite_archive_sha256"],
        "Independent corrective prerequisite unchanged",
    )


def run(root):
    binding = json.loads((root / "binding.json").read_text())
    exposed.need(
        not (root / "controller.json").exists(), "No controller retries or resumed dispatch"
    )
    verify(binding)
    deadline = time.monotonic() + binding["finite_days"] * 86400
    state = {
        "state": "NEW_FORMAL_KERNEL_AFTER_ARCHIVED_DEVELOPMENT_CORRECTION",
        "terminal": False,
        "stages": {},
        "full_http_qualified": False,
        "original_workflow_remains_hold": True,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    exposed.save(root / "controller.json", state)
    try:
        formal = root / "kernel"
        archive = root / "qualified-harness.tar.gz"
        shutil.copyfile(binding["source_archive"], archive)
        kernel.initialize(
            Path(binding["source"]),
            formal,
            Path(binding["prerequisite"]),
            archive,
            binding["harness_commit"],
        )
        error = child_run(kernel.run, formal)
        wait_terminal(
            list(kernel_archive.job_records(formal)), min(deadline, time.monotonic() + 13 * 3600)
        )
        receipt = kernel_archive.archive(formal, root / "kernel-archive")
        state["stages"]["kernel"] = {
            "pass": receipt["pass_formal_kernel_qualification"],
            "archive_sha256": receipt["archive_sha256"],
            "error": error,
        }
        exposed.need(
            error is None and receipt["pass_formal_kernel_qualification"],
            "Formal kernel HOLD; stop HTTP",
        )
        exposed.need(time.monotonic() < deadline, "Finite continuation deadline")
        verify(binding)
        state["state"] = "FULL_PAIRED_FOUR_MODEL_HTTP_AFTER_ALL_EIGHT_KERNEL_PASS"
        exposed.save(root / "controller.json", state)
        serving = root / "http"
        http_pipeline.initialize(
            serving,
            formal,
            archive,
            Path(binding["model_evidence"]),
            Path(binding["sglang_source"]),
        )
        error = child_run(http_pipeline.run, serving)
        jobs = [
            json.loads(p.read_text())["job"] for p in (serving / "receipts").glob("dispatch-*.json")
        ]
        wait_terminal(jobs, min(deadline, time.monotonic() + 9 * 3600))
        receipt = http_archive.archive(serving, root / "http-archive")
        state["stages"]["http"] = {
            "pass": receipt["full_http_qualified"],
            "archive_sha256": receipt["archive_sha256"],
            "error": error,
        }
        exposed.need(error is None and receipt["full_http_qualified"], "Complete paired HTTP HOLD")
        state.update(state="EXPERIMENTAL_HTTP_PASS_HISTORY_HOLD_REMAINS", full_http_qualified=True)
    except BaseException as error:  # noqa: BLE001 - retain terminal failure and stop successors
        state.update(
            state="CORRECTIVE_CONTINUATION_HOLD",
            error={"type": type(error).__name__, "message": str(error)},
        )
    state["terminal"] = True
    exposed.save(root / "controller.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    for name in ("source", "archive", "prerequisite", "models", "sglang"):
        parser.add_argument("--" + name, type=Path)
    a = parser.parse_args()
    if a.initialize:
        print(
            json.dumps(
                initialize(a.root, a.source, a.archive, a.prerequisite, a.models, a.sglang),
                indent=2,
            )
        )
    else:
        print(json.dumps(run(a.root), indent=2))
