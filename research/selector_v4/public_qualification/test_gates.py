"""Synthetic CPU stage evidence only; no GPU cases or real tickets are produced."""

import copy
import json

import pytest

from research.selector_v4.public_qualification.gates import (
    REQUIREMENTS,
    REVISION,
    SCOPE,
    STAGES,
    build_stage_ticket,
    predecessor_stages,
    sha,
    verify_case_ticket,
    verify_manifest,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def campaign(tmp_path):
    root = tmp_path / "synthetic-no-gpu"
    root.mkdir()
    cases = [{"id": stage + "-synthetic", "family": stage} for stage in STAGES]
    import hashlib

    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "cases": cases,
        "case_hash": digest(cases),
        "stage_hashes": {s: digest([c for c in cases if c["family"] == s]) for s in STAGES},
        "families": {s: 1 for s in STAGES},
    }
    save(root / "manifest.json", manifest)
    (root / "harness.tar.gz").write_bytes(b"synthetic-test-source-not-an-engine")
    expected = {
        "scope": SCOPE,
        "qualification_revision": REVISION,
        "manifest_sha256": sha(root / "manifest.json"),
        "harness_archive_sha256": sha(root / "harness.tar.gz"),
        "candidate_commit": "synthetic-candidate",
        "pristine_commit": "synthetic-pristine",
        "analyzer_sha256": "synthetic-analyzer",
        "stage_hashes": manifest["stage_hashes"],
    }
    save(root / "frozen-binding.json", expected)
    (root / "smoke-proof.txt").write_text("synthetic raw evidence")
    smoke = {
        k: expected[k]
        for k in (
            "scope",
            "qualification_revision",
            "manifest_sha256",
            "harness_archive_sha256",
            "candidate_commit",
            "pristine_commit",
        )
    }
    smoke["results"] = {
        g: {"pass": True, "files": {"smoke-proof.txt": sha(root / "smoke-proof.txt")}}
        for g in ("gpu_4090", "gpu_5090")
    }
    save(root / "receipts/dual-smoke.json", smoke)
    for stage in STAGES:
        parent = root / "stages" / stage
        parent.mkdir(parents=True)
        (parent / "raw.txt").write_text("synthetic evidence, no measurements")
        for gpu in ("gpu_4090", "gpu_5090"):
            summary = {
                k: expected[k]
                for k in (
                    "scope",
                    "qualification_revision",
                    "manifest_sha256",
                    "harness_archive_sha256",
                    "candidate_commit",
                    "pristine_commit",
                    "analyzer_sha256",
                )
            }
            summary.update(
                pass_=True,
                stage=stage,
                gpu=gpu,
                stage_hash=expected["stage_hashes"][stage],
                requirements={k: True for k in REQUIREMENTS},
                files={"raw.txt": sha(parent / "raw.txt")},
            )
            summary["pass"] = summary.pop("pass_")
            save(parent / "analysis" / gpu / "summary.json", summary)
    return root, expected, cases


def test_complete_prior_stages_are_required_in_order(tmp_path):
    root, _, _ = campaign(tmp_path)
    for stage, count in zip(STAGES, (0, 2, 4, 6)):
        ticket = build_stage_ticket(root, stage)
        assert len(ticket["predecessors"]) == count and not ticket["serving_promotion"]
    assert predecessor_stages("stress") == ("dev", "canary", "release")


@pytest.mark.parametrize(
    "field,value",
    [
        ("pass", False),
        ("scope", "EXPOSED_PUBLIC_PATH_PREQUAL_2"),
        ("qualification_revision", "4.2.0"),
        ("harness_archive_sha256", "stale"),
        ("stage_hash", "other-shapes"),
        ("gpu", "gpu_4090"),
    ],
)
def test_hold_old_diagnostic_or_wrong_binding_cannot_authorize_canary(tmp_path, field, value):
    root, _, _ = campaign(tmp_path)
    path = root / "stages/dev/analysis/gpu_5090/summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    save(path, summary)
    with pytest.raises(ValueError):
        build_stage_ticket(root, "canary")


def test_missing_gate_or_changed_raw_evidence_blocks_dispatch(tmp_path):
    root, _, _ = campaign(tmp_path)
    path = root / "stages/dev/analysis/gpu_5090/summary.json"
    summary = json.loads(path.read_text())
    original = copy.deepcopy(summary)
    del summary["requirements"]["actual_policy_pristine_joint_lcb_at_least_0_99"]
    save(path, summary)
    with pytest.raises(ValueError, match="requirements"):
        build_stage_ticket(root, "canary")
    save(path, original)
    (root / "stages/dev/raw.txt").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        build_stage_ticket(root, "canary")


def test_case_measurement_requires_ticket_geometry_and_consumption_ledger(tmp_path):
    root, expected, cases = campaign(tmp_path)
    stage = root / "stages/canary"
    binding = dict(
        expected, stage="canary", stage_hash=expected["stage_hashes"]["canary"], cases=[cases[1]]
    )
    save(stage / "binding.json", binding)
    save(stage / "authorization.json", build_stage_ticket(root, "canary"))
    save(stage / "receipts/consumption.json", {"consumed_case_ids": []})
    with pytest.raises(ValueError, match="consumption"):
        verify_case_ticket(stage, 0)
    save(stage / "receipts/consumption.json", {"consumed_case_ids": [cases[1]["id"]]})
    verify_case_ticket(stage, 0)
    with pytest.raises(ValueError, match="routing"):
        verify_case_ticket(stage, 1)
    path = root / "stages/dev/analysis/gpu_5090/summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="verdict changed"):
        verify_case_ticket(stage, 0)


def test_changed_smoke_or_manifest_cannot_authorize_even_dev(tmp_path):
    root, _, _ = campaign(tmp_path)
    (root / "smoke-proof.txt").write_text("changed")
    with pytest.raises(ValueError, match="Smoke evidence"):
        build_stage_ticket(root, "dev")
    (root / "manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="Manifest"):
        build_stage_ticket(root, "dev")


@pytest.mark.parametrize("field", ["case_hash", "stage_hashes", "families"])
def test_declared_file_hash_cannot_hide_wrong_derived_geometry(tmp_path, field):
    root, expected, _ = campaign(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[field] = "wrong" if field == "case_hash" else {}
    save(root / "manifest.json", manifest)
    expected["manifest_sha256"] = sha(root / "manifest.json")
    with pytest.raises(ValueError, match="Derived"):
        verify_manifest(root, expected)
