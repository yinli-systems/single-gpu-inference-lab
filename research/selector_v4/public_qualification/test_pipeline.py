"""Synthetic controller failure, consumption and bounded submission proofs."""

import json
import time

import pytest

from research.selector_v4.public_qualification import pipeline


def test_exclusive_receipts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    pipeline.save_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        pipeline.save_new(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


@pytest.mark.parametrize(
    "state", ["FAILED 1:0", "TIMEOUT 0:9", "TIMEOUT 0:0", "CANCELLED 0:0", "OUT_OF_MEMORY 0:9"]
)
def test_failed_allocations_cannot_become_successful_from_exit_file(tmp_path, monkeypatch, state):
    pipeline.save_new(tmp_path / "receipts/unused.json", {})
    (tmp_path / "receipts/exit-synthetic.txt").write_text("0\n")
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *a, **k: state)
    with pytest.raises(ValueError, match="Failed allocation"):
        pipeline.terminal(tmp_path, "synthetic")


def test_completed_allocation_needs_actual_exit_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *a, **k: "COMPLETED 0:0")
    with pytest.raises(ValueError, match="exit receipt"):
        pipeline.terminal(tmp_path, "synthetic")
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts/exit-synthetic.txt").write_text("0\n")
    assert pipeline.terminal(tmp_path, "synthetic")


def test_canary_ticket_failure_prevents_case_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "verify_manifest", lambda *a: None)
    monkeypatch.setattr(pipeline, "verify_helpers", lambda *a: None)
    monkeypatch.setattr(
        pipeline, "build_stage_ticket", lambda *a: (_ for _ in ()).throw(ValueError("dev HOLD"))
    )
    monkeypatch.setattr(pipeline, "submit", lambda *a: pytest.fail("No dispatch under HOLD"))
    with pytest.raises(ValueError, match="dev HOLD"):
        pipeline.dispatch_stage(tmp_path, "canary", {}, time.monotonic() + 1)
    assert not (tmp_path / "stages").exists()


def test_dispatch_intent_is_consumed_before_failed_submission(tmp_path, monkeypatch):
    campaign = tmp_path / "synthetic-no-gpu"
    campaign.mkdir()
    for name in ("stages", "sdk-libraries", "harness"):
        (campaign / name).mkdir()
    pipeline.save_new(
        campaign / "manifest.json", {"cases": [{"id": "synthetic-case", "family": "canary"}]}
    )
    binding = {
        "stage_hashes": {"canary": "synthetic"},
        "manifest_sha256": "synthetic",
        "helper_files": {},
        "maximum_inflight_cases": 2,
        "candidate_source": "unused",
        "pristine_source": "unused",
    }
    monkeypatch.setattr(pipeline, "verify_manifest", lambda *a: None)
    monkeypatch.setattr(pipeline, "verify_helpers", lambda *a: None)
    monkeypatch.setattr(pipeline, "build_stage_ticket", lambda *a: {"synthetic": True})
    monkeypatch.setattr(pipeline, "validation_ledger", lambda *a: None)

    def fail_submission(root, *args):
        recorded = json.loads((root / "receipts/consumption.json").read_text())
        assert recorded["consumed_case_ids"] == ["synthetic-case"]
        raise RuntimeError("synthetic submission failure")

    monkeypatch.setattr(pipeline, "submit", fail_submission)
    with pytest.raises(RuntimeError, match="submission failure"):
        pipeline.dispatch_stage(campaign, "canary", binding, time.monotonic() + 1)
    assert json.loads((campaign / "stages/canary/receipts/consumption.json").read_text())[
        "consumed_case_ids"
    ] == ["synthetic-case"]


def test_stage_hold_stops_successors_and_preserves_terminal_state(tmp_path, monkeypatch):
    pipeline.save_new(
        tmp_path / "frozen-binding.json",
        {"finite_controller_days": 1, "harness_archive_sha256": "synthetic"},
    )
    monkeypatch.setattr(pipeline, "verify_helpers", lambda *a: None)
    monkeypatch.setattr(pipeline, "smoke", lambda *a: None)
    monkeypatch.setattr(pipeline, "verify_manifest", lambda *a: None)
    monkeypatch.setattr(pipeline, "sha", lambda *a: "synthetic")
    stages = []

    def hold(campaign, stage, *args):
        stages.append(stage)
        raise ValueError("synthetic dev HOLD")

    monkeypatch.setattr(pipeline, "dispatch_stage", hold)
    with pytest.raises(ValueError, match="dev HOLD"):
        pipeline.run(tmp_path)
    assert stages == ["dev"]
    terminal = json.loads((tmp_path / "receipts/controller-terminal.json").read_text())
    assert terminal["terminal"] and terminal["state"] == "FORMAL_QUALIFICATION_HOLD"
    assert not terminal["serving_promotion"]
    with pytest.raises(FileExistsError):
        pipeline.run(tmp_path)
