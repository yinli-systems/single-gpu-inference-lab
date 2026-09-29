"""Prepare an immutable minimal source tree and isolated cap overlay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


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


def prepare(repo, pristine_source, out):
    repo = repo.resolve()
    pristine_source = pristine_source.resolve()
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
    sys.path.insert(0, str(source_tree / "src"))
    try:
        spec.loader.exec_module(manifest_module)
    finally:
        sys.path.pop(0)
    manifest = manifest_module.build()

    (out / "receipts").mkdir(parents=True)
    (out / "runs").mkdir()
    (out / "logs").mkdir()
    (out / "analysis").mkdir()
    (out / "cache").mkdir()

    resource_prepare = load_resource_prepare(
        source_tree / "research" / "resource_generalization" / "prepare.py"
    )
    pristine_header = (
        pristine_source
        / "flashinfer"
        / "data"
        / "include"
        / "flashinfer"
        / "attention"
        / "prefill.cuh"
    )
    if not pristine_header.exists():
        raise FileNotFoundError("pristine FlashInfer 0.7.0 header missing")
    if sha256(pristine_header) != resource_prepare.EXPECTED:
        raise RuntimeError("pristine 0.7.0 header hash mismatch")

    pristine_out = out / "overlays" / "pristine"
    shutil.copytree(
        pristine_source / "flashinfer",
        pristine_out / "flashinfer",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    dist_infos = list(pristine_source.glob("flashinfer_python-*.dist-info"))
    if len(dist_infos) != 1:
        raise RuntimeError("expected one FlashInfer dist-info directory")
    shutil.copytree(dist_infos[0], pristine_out / dist_infos[0].name)

    cap_binding = resource_prepare.prepare(
        pristine_source,
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
        "pristine_source": str(pristine_source),
        "official_flashinfer_header_sha256": resource_prepare.EXPECTED,
        "pristine_overlay": str(pristine_out),
        "pristine_header_sha256": sha256(
            pristine_out
            / "flashinfer"
            / "data"
            / "include"
            / "flashinfer"
            / "attention"
            / "prefill.cuh"
        ),
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
        "pristine_overlay": str(out / "overlays" / "pristine"),
        "pristine_header_sha256": resource_prepare.EXPECTED,
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
    parser.add_argument("--pristine-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.repo, args.pristine_source, args.out),
            indent=2,
            allow_nan=False,
        )
    )
