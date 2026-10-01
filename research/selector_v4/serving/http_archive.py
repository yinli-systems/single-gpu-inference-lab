"""Archive every terminal HTTP arm, failed request and managed decision byte."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha


def stream_sha(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(8 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def archive(root, output):
    need(not output.exists(), "Never replace an HTTP evidence archive")
    terminal = json.loads((root / "receipts/controller-terminal.json").read_text())
    need(terminal["terminal"], "Wait for HTTP controller termination")
    intents = {p.name.removeprefix("intent-") for p in (root / "receipts").glob("intent-*.json")}
    dispatch_paths = sorted((root / "receipts").glob("dispatch-*.json"))
    dispatches = {p.name.removeprefix("dispatch-") for p in dispatch_paths}
    need(intents == dispatches, "Reconcile every uncertain Slurm submission before archiving")
    jobs = {}
    for path in dispatch_paths:
        record = json.loads(path.read_text())
        job = record["job"]
        need(job.isdigit() and job not in jobs, "Unique actual submitted allocation")
        status = subprocess.check_output(
            ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed", "-n", "-P"],
            text=True,
            timeout=60,
        ).strip()
        fields = status.split("|")
        need(len(fields) == 4 and fields[0] == job, "Actual known allocation status")
        need(
            fields[1]
            not in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED"),
            "Wait for every actual submitted HTTP job to stop",
        )
        jobs[job] = {"dispatch": record, "sacct": status}
    passed = terminal["state"] == "FULL_HTTP_QUALIFICATION_PASS_HISTORY_REQUIRED"
    if passed:
        need(
            terminal["full_http_qualified"] and len(jobs) == 72 and len(terminal["analyses"]) == 24,
            "Complete 72-allocation / 24-verdict evidence required",
        )
        for label, digest in terminal["analyses"].items():
            path = root / "analysis" / label / "summary.json"
            need(
                sha(path) == digest and json.loads(path.read_text())["pass"], "All full HTTP stages"
            )
        need(
            all(v["sacct"].split("|")[1:3] == ["COMPLETED", "0:0"] for v in jobs.values()),
            "Every actual HTTP allocation must have completed successfully",
        )
    files = {}
    for parent, folders, names in os.walk(root):
        current = Path(parent)
        if current == root:
            folders[:] = [n for n in folders if n not in ("cache", "sdk-libraries")]
        folders[:] = [n for n in folders if n not in ("__pycache__", ".pytest_cache")]
        for name in names:
            path = current / name
            need(not path.is_symlink(), "Archive actual regular HTTP evidence")
            if path.is_file():
                files[str(path.relative_to(root))] = path
    digests = {name: file_sha(path) for name, path in files.items()}
    output.mkdir()
    bundle = output / "raw-full-http.tar.gz"
    with tarfile.open(bundle, "w:gz") as stream:
        for name, path in sorted(files.items()):
            stream.add(path, arcname=name, recursive=False)
    with tarfile.open(bundle) as stream:
        members = stream.getmembers()
        need(
            len(members) == len(files) and {m.name for m in members} == set(files), "All HTTP files"
        )
        for member in members:
            need(
                member.isfile() and stream_sha(stream.extractfile(member)) == digests[member.name],
                "Original full HTTP evidence bytes",
            )
    receipt = {
        "full_http_qualified": passed,
        "controller": terminal,
        "jobs": jobs,
        "files": digests,
        "archive_sha256": file_sha(bundle),
        "archive_bytes": bundle.stat().st_size,
        "all_failed_requests_and_partial_arms_retained": True,
        "managed_decision_cache_retained": True,
        "default_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive(args.root, args.output)
