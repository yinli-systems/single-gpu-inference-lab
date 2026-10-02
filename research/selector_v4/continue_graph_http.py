"""Finite continuation of existing kernel jobs into fully gated HTTP experiments.

Never restarts the original workflow or any scored job. Independent cached
Native/observer parity is a protocol preflight, with its own evidence scope.
The full HTTP campaign still checks ordinary concurrent parity and all metrics.
"""

import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from research.selector_v4.complete_campaign import child_run, wait_terminal
from research.selector_v4.public_qualification import archive as kernel_archive
from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.public_qualification.pipeline import GPUS, now, save_new
from research.selector_v4.serving import http_archive, http_pipeline
from research.selector_v4.serving.graph_serving_audit import observe
from research.selector_v4.serving.training_session import authorize_http_training

ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED")


def read(path):
    return json.loads(path.read_text())


def check_native_verdicts(verdicts):
    need(set(verdicts) == set(GPUS), "Both real GPU preflight verdicts required")
    for verdict in verdicts.values():
        need(
            verdict["state"] == "PASS_FIXED_SCHEDULE_DIAGNOSTIC_ONLY"
            and verdict["fixed_schedule_full_token_logprob_parity_pass"]
            and verdict["actual_decode_graph_replay_and_input_boundary_verified"]
            and verdict["historical_token_divergence_resolved"] is False,
            "Complete cached Native/observer parity and actual Graph proof required",
        )


def native_jobs(native):
    jobs = read(native / "receipts/jobs.json")
    need(set(jobs) == set(GPUS) and len(set(jobs.values())) == 2, "Two unique real preflight jobs")
    for gpu, job in jobs.items():
        need(job.isdigit(), "Unambiguous preflight job")
        confirmed = read(native / f"receipts/submit-confirmed-{gpu}.json")
        intent = read(native / f"receipts/submit-intent-{gpu}.json")
        need(confirmed["job"] == job, "Actual confirmed Native allocation")
        need(intent["command"][-1] == "fixed_schedule_observer", "Exactly the new fixed mode")
    return jobs


def initialize(root, source, archive, kernel, native, models, sglang):
    need(not root.exists(), "New continuation only; never replace an earlier attempt")
    metadata = read(source / "harness-commit.json")
    for name, digest in metadata["git_blob_files_sha256"].items():
        need(sha(source / name) == digest, "Frozen continuation source")
    with tarfile.open(archive) as bundle:
        need(
            json.load(bundle.extractfile("harness-commit.json")) == metadata,
            "Archive/source identity",
        )
    formal = read(kernel / "frozen-binding.json")
    need(
        (kernel / "receipts/controller-start.json").exists(),
        "Existing owned kernel controller required",
    )
    need(
        formal["default_promotion"] is False and formal["serving_promotion"] is False,
        "Switches stay off",
    )
    native_binding = read(native / "binding.json")
    need(native_binding["resource_enabled"] is False, "Native-only protocol preflight")
    root.mkdir()
    binding = {
        "source": str(source),
        "source_archive": str(archive),
        "source_archive_sha256": sha(archive),
        "harness_commit": metadata["harness_commit"],
        "helper_files": metadata["git_blob_files_sha256"],
        "kernel": str(kernel),
        "kernel_binding_sha256": sha(kernel / "frozen-binding.json"),
        "native": str(native),
        "native_binding_sha256": sha(native / "binding.json"),
        "native_jobs": native_jobs(native),
        "model_evidence": str(models),
        "sglang_source": str(sglang),
        "finite_days": 30,
        "http_requires_all_eight_exact_source_kernel_stages": True,
        "no_scored_retries": True,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    save_new(root / "binding.json", binding)
    return binding


def verify(binding):
    source = Path(binding["source"])
    need(
        sha(Path(binding["source_archive"])) == binding["source_archive_sha256"],
        "Frozen HTTP bundle",
    )
    for name, digest in binding["helper_files"].items():
        need(sha(source / name) == digest, "Frozen continuation helper: " + name)
    for field in ("kernel", "native"):
        filename = "frozen-binding.json" if field == "kernel" else "binding.json"
        need(
            sha(Path(binding[field]) / filename) == binding[field + "_binding_sha256"],
            "Binding changed",
        )


def native_status(job):
    raw = subprocess.check_output(
        ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed", "-n", "-P"],
        text=True,
        timeout=60,
    ).strip()
    fields = raw.split("|")
    need(len(fields) == 4 and fields[0] == job, "Known Native allocation status")
    return raw, fields


def archive_native(native, output, statuses):
    """Keep every raw arm/failure/trace/log; compilation caches are regenerable."""
    need(not output.exists(), "Never replace raw Native evidence")
    files = {}
    for parent, folders, names in os.walk(native):
        folders[:] = sorted(
            n for n in folders if n not in ("cache", "jit-cache", "__pycache__", ".pytest_cache")
        )
        for name in sorted(names):
            path = Path(parent) / name
            need(not path.is_symlink(), "Actual regular Native evidence")
            if path.is_file():
                files[str(path.relative_to(native))] = sha(path)
    output.mkdir()
    bundle = output / "raw-native-controls.tar.gz"
    with tarfile.open(bundle, "w:gz") as stream:
        for name in files:
            stream.add(native / name, arcname=name, recursive=False)
    with tarfile.open(bundle) as stream:
        seen = set()
        for member in stream:
            need(
                member.isfile() and member.name in files and member.name not in seen,
                "Complete unique raw archive",
            )
            seen.add(member.name)
            need(
                http_archive.stream_sha(stream.extractfile(member)) == files[member.name],
                "Raw archive byte mismatch",
            )
        need(seen == set(files), "Every raw Native file archived")
    receipt = {
        "archive_sha256": sha(bundle),
        "archive_bytes": bundle.stat().st_size,
        "files": files,
        "statuses": statuses,
        "compiled_caches_excluded": True,
        "full_http_qualified": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    save_new(output / "receipt.json", receipt)
    return receipt


def run(root):
    binding = read(root / "binding.json")
    save_new(root / "controller-start.json", {"pid": os.getpid(), "utc": now()})
    state = {
        "state": "WAIT_CACHED_GRAPH_PROTOCOL_PREFLIGHT",
        "terminal": False,
        "full_http_qualified": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
        "original_natural_parity_failure_closed": False,
    }
    save_new(root / "controller.json", state)
    deadline = time.monotonic() + binding["finite_days"] * 86400
    try:
        verify(binding)
        native, formal = Path(binding["native"]), Path(binding["kernel"])
        statuses = {}
        while len(statuses) != 2:
            for gpu, job in binding["native_jobs"].items():
                raw, fields = native_status(job)
                if fields[1] not in ACTIVE:
                    statuses[gpu] = raw
            need(time.monotonic() < deadline, "Finite Native preflight wait")
            if len(statuses) != 2:
                time.sleep(15)
        state["native_archive"] = archive_native(native, root / "native-archive", statuses)[
            "archive_sha256"
        ]
        need(
            all(raw.split("|")[1:3] == ["COMPLETED", "0:0"] for raw in statuses.values()),
            "Native protocol preflight HOLD",
        )
        verdicts = {}
        for gpu, job in binding["native_jobs"].items():
            need(
                (native / f"receipts/exit-{job}.txt").read_text().strip() == "0",
                "Completed owned Native command",
            )
            verdicts[gpu] = observe(
                native / f"runs/{gpu}-{job}/fixed_schedule_observer", fixed_schedule=True
            )
            save_new(root / f"native-verdict-{gpu}.json", verdicts[gpu])
        check_native_verdicts(verdicts)
        state["state"] = "WAIT_ALL_EIGHT_FORMAL_KERNEL_VERDICTS"
        state["native_protocol_preflight_pass"] = True
        save_new(
            root / "native-preflight-pass.json",
            {"verdict_sha256": {gpu: sha(root / f"native-verdict-{gpu}.json") for gpu in GPUS}},
        )
        (root / "controller.json").write_text(json.dumps(state, indent=2) + "\n")
        while not (formal / "receipts/controller-terminal.json").exists():
            need(time.monotonic() < deadline, "Finite formal kernel wait")
            time.sleep(15)
        wait_terminal(
            list(kernel_archive.job_records(formal)), min(deadline, time.monotonic() + 13 * 3600)
        )
        receipt = kernel_archive.archive(formal, root / "kernel-archive")
        state["kernel_archive"] = receipt["archive_sha256"]
        need(receipt["pass_formal_kernel_qualification"], "Formal kernel HOLD; stop Resource HTTP")
        formal_binding = read(formal / "frozen-binding.json")
        authorize_http_training(formal, formal_binding["candidate_commit"])
        verify(binding)
        need(time.monotonic() < deadline, "Finite continuation deadline")
        state["state"] = "FULL_FOUR_MODEL_PAIRED_HTTP"
        (root / "controller.json").write_text(json.dumps(state, indent=2) + "\n")
        serving = root / "http"
        http_pipeline.initialize(
            serving,
            formal,
            Path(binding["source_archive"]),
            Path(binding["model_evidence"]),
            Path(binding["sglang_source"]),
        )
        error = child_run(http_pipeline.run, serving)
        jobs = [read(p)["job"] for p in (serving / "receipts").glob("dispatch-*.json")]
        wait_terminal(jobs, min(deadline, time.monotonic() + 9 * 3600))
        receipt = http_archive.archive(serving, root / "http-archive")
        state["http_archive"] = receipt["archive_sha256"]
        need(error is None and receipt["full_http_qualified"], "Complete HTTP/SLO HOLD")
        state.update(state="EXPERIMENTAL_HTTP_PASS_HISTORY_HOLD_REMAINS", full_http_qualified=True)
    except BaseException as error:  # noqa: BLE001 - retain terminal failure; never dispatch a successor
        state.update(
            state="GRAPH_HTTP_CONTINUATION_HOLD",
            error={"type": type(error).__name__, "message": str(error)},
        )
    state.update(terminal=True, finished_utc=now())
    (root / "controller.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    for name in ("source", "archive", "kernel", "native", "models", "sglang"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.initialize:
        result = initialize(
            args.root, args.source, args.archive, args.kernel, args.native, args.models, args.sglang
        )
    else:
        result = run(args.root)
    print(json.dumps(result, indent=2))
