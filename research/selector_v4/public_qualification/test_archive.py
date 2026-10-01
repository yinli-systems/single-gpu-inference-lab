"""Synthetic partial evidence retention and active-allocation barrier."""

import json
import tarfile

import pytest

from research.selector_v4.public_qualification.archive import archive
from research.selector_v4.public_qualification.gates import sha
from research.selector_v4.public_qualification.pipeline import save_new


def partial(tmp_path):
    root = tmp_path / "synthetic-campaign"
    save_new(
        root / "receipts/controller-terminal.json",
        {"terminal": True, "state": "FORMAL_QUALIFICATION_HOLD"},
    )
    save_new(
        root / "stages/canary/receipts/dispatch-gpu_4090-0.json",
        {"job": "123", "case_id": "consumed"},
    )
    save_new(root / "stages/canary/receipts/consumption.json", {"consumed_case_ids": ["consumed"]})
    save_new(root / "manifest.json", {"synthetic": True})
    save_new(root / "harness/helper.json", {"source": "synthetic"})
    (root / "harness.tar.gz").write_bytes(b"synthetic-source-bundle")
    save_new(
        root / "frozen-binding.json",
        {
            "manifest_sha256": sha(root / "manifest.json"),
            "harness_archive_sha256": sha(root / "harness.tar.gz"),
            "helper_files": {"helper.json": sha(root / "harness/helper.json")},
        },
    )
    save_new(root / "stages/canary/runs/partial/managed-cache/receipt.json", {"actual_tactic": -1})
    save_new(root / "cache/rebuildable.json", {})
    return root


def test_hold_controller_cannot_archive_while_an_existing_allocation_runs(tmp_path):
    root = partial(tmp_path)
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="still active"):
        archive(root, out, status_reader=lambda job: "123 RUNNING 0:0")
    assert not out.exists()


def test_terminal_failed_attempt_keeps_cache_consumption_and_partial_source(tmp_path):
    root = partial(tmp_path)
    out = tmp_path / "archive"
    result = archive(root, out, status_reader=lambda job: "123 TIMEOUT 0:9")
    assert not result["pass_formal_kernel_qualification"]
    assert result["known_jobs"] == {"123": ["stages/canary/receipts/dispatch-gpu_4090-0.json"]}
    with tarfile.open(out / "raw-formal-kernel-campaign.tar.gz") as bundle:
        names = set(bundle.getnames())
        assert "stages/canary/runs/partial/managed-cache/receipt.json" in names
        assert "stages/canary/receipts/consumption.json" in names
        assert "harness/helper.json" in names
        assert "cache/rebuildable.json" not in names
        consumed = json.load(bundle.extractfile("stages/canary/receipts/consumption.json"))
        assert consumed["consumed_case_ids"] == ["consumed"]
    with pytest.raises(ValueError, match="Never replace"):
        archive(root, out, status_reader=lambda job: "123 TIMEOUT 0:9")
