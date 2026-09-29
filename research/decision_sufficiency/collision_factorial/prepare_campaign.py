"""Prepare an immutable minimal source tree and isolated cap overlay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


FILES = [
    "src/l20_stack/__init__.py",
    "src/l20_stack/decision_sufficiency.py",
    "research/decision_sufficiency/collision_factorial/manifest.py",
    "research/decision_sufficiency/collision_factorial/measure.py",
    "research/decision_sufficiency/collision_factorial/analyze.py",
    "research/decision_sufficiency/collision_factorial/validate_canary.py",
    "research/decision_sufficiency/collision_factorial/PROTOCOL.md",
    "research/decision_sufficiency/collision_factorial/run.sbatch",
    "research/decision_sufficiency/collision_factorial/prepare_campaign.py",
    "research/section6/metadata_adapter.py",
    "research/section6/bindings.json",
    "benchmarks/results/plan-order-mechanism/plan_contract.py",
    "research/resource_generalization/prepare.py",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_resource_prepare(path):
    spec = importlib.util.spec_from_file_location("resource_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare(repo, site_packages, out):
    repo = repo.resolve()
    site_packages = site_packages.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError("campaign root already exists")

    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError("refuse to freeze a dirty source checkout")

    source_tree = out / "source-tree"
    for relative in FILES:
        source = repo / relative
        if not source.exists():
            raise FileNotFoundError(relative)
        target = source_tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest_module_path = (
        source_tree
        / "research"
        / "decision_sufficiency"
        / "collision_factorial"
        / "manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "collision_manifest", manifest_module_path
    )
    manifest_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(manifest_module)
    manifest = manifest_module.build()

    (out / "receipts").mkdir(parents=True)
    (out / "runs").mkdir()
    (out / "logs").mkdir()
    (out / "analysis").mkdir()
    (out / "cache").mkdir()

    resource_prepare = load_resource_prepare(
        source_tree / "research" / "resource_generalization" / "prepare.py"
    )
    cap_binding = resource_prepare.prepare(
        site_packages,
        out / "overlays" / "cap",
        "cap",
    )

    source_hashes = {
        relative: sha256(source_tree / relative) for relative in FILES
    }
    source_manifest = {
        "git_commit": head,
        "git_branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "files": source_hashes,
        "collision_case_hash": manifest["case_hash"],
        "site_packages": str(site_packages),
        "official_flashinfer_header_sha256": resource_prepare.EXPECTED,
        "cap_binding": cap_binding,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (out / "SOURCE.json").write_text(
        json.dumps(source_manifest, indent=2, allow_nan=False) + "\n"
    )
    (out / "SOURCE_COMMIT.txt").write_text(head + "\n")
    with (out / "source.sha256").open("w") as handle:
        for relative, digest in sorted(source_hashes.items()):
            handle.write(digest + "  source-tree/" + relative + "\n")

    receipt = {
        "complete": True,
        "git_commit": head,
        "case_hash": manifest["case_hash"],
        "source_files": len(source_hashes),
        "cap_overlay": str(out / "overlays" / "cap"),
        "cap_header_sha256": cap_binding["modified_sha256"],
        "performance_measured": False,
    }
    (out / "PREPARE.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.repo, args.site_packages, args.out),
            indent=2,
            allow_nan=False,
        )
    )
