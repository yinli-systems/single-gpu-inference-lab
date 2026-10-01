"""Archive terminal formal campaigns, preserving failed and partial evidence.

Wait for every known allocation to terminate, even if the controller stopped
earlier. Retain all raw responses/references/cache receipts and committed helper
blobs; exclude only rebuildable JIT caches, SDK copies and wheel extraction.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha


def job_records(campaign):
    records = {}
    candidates = [campaign / "smoke/receipts"]
    candidates += sorted((campaign / "stages").glob("*/receipts"))
    for root in candidates:
        for receipt in sorted(root.glob("*.json")):
            value = json.loads(receipt.read_text())
            if isinstance(value, dict) and "job" in value:
                job = value["job"]
                need(isinstance(job, str) and re.fullmatch(r"[0-9]+", job), "Actual Slurm job ID")
                records.setdefault(job, []).append(str(receipt.relative_to(campaign)))
    return records


def archive(campaign, output, *, status_reader=None):
    campaign, output = Path(campaign), Path(output)
    need(not output.exists(), "Never replace an existing campaign archive")
    terminal = json.loads((campaign / "receipts/controller-terminal.json").read_text())
    need(terminal.get("terminal") is True, "Finite controller must terminate before archive")
    jobs = job_records(campaign)
    if status_reader is None:
        status_reader = lambda job: subprocess.check_output(
            ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed", "-n"],
            text=True,
            timeout=60,
        )
    statuses = {}
    active = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED")
    for job in jobs:
        status = status_reader(job)
        need(
            status.strip() and not any(state in status for state in active),
            "Known allocation still active: " + job,
        )
        statuses[job] = status
    binding = json.loads((campaign / "frozen-binding.json").read_text())
    integrity = sha(campaign / "manifest.json") == binding["manifest_sha256"]
    integrity &= sha(campaign / "harness.tar.gz") == binding["harness_archive_sha256"]
    integrity &= all(sha(campaign / "harness" / n) == h for n, h in binding["helper_files"].items())
    passed = terminal.get("state") == "FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED" and integrity
    if passed:
        # The full HTTP gate verifies smoke and all eight stage summaries with
        # raw evidence hashes; a controller state alone cannot produce a PASS.
        from research.selector_v4.serving.training_session import authorize_http_training

        authorize_http_training(campaign, binding["candidate_commit"])
        need(len(jobs) == 146, "Dual smoke plus all72 dual-GPU cases required")
        need(
            all("COMPLETED" in s and "0:0" in s for s in statuses.values()),
            "Every formal allocation must complete",
        )
    paths = {}
    exclude = {"cache", "sdk-libraries", "pristine-overlay", "__pycache__", ".pytest_cache"}
    for current, folders, names in os.walk(campaign):
        folders[:] = sorted(
            d
            for d in folders
            if d not in exclude
            and not d.startswith("xdg-")
            and not (Path(current) / d).is_symlink()
        )
        for name in sorted(names):
            path = Path(current) / name
            if path.is_file() and not path.is_symlink():
                paths[str(path.relative_to(campaign))] = path
    paths["archive-helper.py"] = Path(__file__)
    hashes = {name: sha(path) for name, path in paths.items()}
    output.mkdir()
    target = output / "raw-formal-kernel-campaign.tar.gz"
    with tarfile.open(target, "w:gz") as bundle:
        for name, path in sorted(paths.items()):
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(target) as bundle:
        need(set(bundle.getnames()) == set(hashes), "Complete archive member set")
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with bundle.extractfile(name) as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
            need(digest.hexdigest() == expected, "Archived member changed: " + name)
    receipt = {
        "pass_formal_kernel_qualification": passed,
        "controller": terminal,
        "binding": binding,
        "source_integrity_at_archive": bool(integrity),
        "known_jobs": jobs,
        "statuses": statuses,
        "files": hashes,
        "archive_sha256": sha(target),
        "archive_bytes": target.stat().st_size,
        "raw_references_included": True,
        "trimmed_windows": 0,
        "full_http_qualified": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = archive(arguments.campaign, arguments.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("pass_formal_kernel_qualification", "archive_sha256", "archive_bytes")
            }
        )
    )
