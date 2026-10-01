"""Bounded controller sequencing and source/kernel failures never grant release."""

import json

import pytest

from research.selector_v4.public_qualification.test_gates import campaign, save
from research.selector_v4.serving import http_pipeline as p


def test_kernel_hold_prevents_initializer_files_and_model_readback(tmp_path, monkeypatch):
    kernel, _, _ = campaign(tmp_path)
    save(
        kernel / "receipts/controller-terminal.json",
        {"terminal": True, "state": "FORMAL_QUALIFICATION_HOLD"},
    )
    monkeypatch.setattr(
        p, "verify_model", lambda *a: pytest.fail("Must not hash models under HOLD")
    )
    out = tmp_path / "never-created"
    with pytest.raises(ValueError, match="Complete formal kernel"):
        p.initialize(out, kernel, None, None, None)
    assert not out.exists()


def controller(tmp_path, monkeypatch):
    save(
        tmp_path / "binding.json",
        {
            "kernel_campaign": "/synthetic-not-real",
            "finite_controller_days": 1,
            "maximum_inflight_allocations": 4,
        },
    )
    for name in ("logs", "analysis"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(p, "authorize", lambda *a: ({}, {}))
    monkeypatch.setattr(p, "verify_frozen", lambda *a: None)
    monkeypatch.setattr(p, "allocation_terminal", lambda *a: True)
    events = []

    def submit(command, **kwargs):
        assert command[0] == "sbatch" and kwargs["timeout"] == 60
        events.append(("submit", command[-2], command[-3], command[-1]))
        return str(1000 + len(events))

    monkeypatch.setattr(p.subprocess, "check_output", submit)
    return events


def test_all_72_allocations_and_24_verdicts_precede_full_http_stage_pass(tmp_path, monkeypatch):
    events = controller(tmp_path, monkeypatch)

    def analyze(root, output, **kwargs):
        events.append(("analysis", kwargs["stage"], kwargs["model_id"]))
        assert len(kwargs["jobs"]) == len(set(kwargs["jobs"])) == 3
        output.mkdir()
        save(output / "summary.json", {"pass": True})
        return {"pass": True}

    monkeypatch.setattr(p, "analyze", analyze)
    p.run(tmp_path)
    for index, stage in enumerate(p.STAGES):
        group = events[index * 32 : (index + 1) * 32]
        assert len(group) == 32
        assert all(v[0] == "submit" and v[1] == stage for v in group[:24])
        assert all(v[0] == "analysis" and v[1] == stage for v in group[24:])
    receipt = json.loads((tmp_path / "receipts/controller-terminal.json").read_text())
    assert receipt["full_http_qualified"] and len(receipt["analyses"]) == 24
    assert not receipt["default_promotion"] and not receipt["historical_token_divergence_resolved"]


def test_first_stage_hold_stops_all_successors_and_retains_submissions(tmp_path, monkeypatch):
    events = controller(tmp_path, monkeypatch)

    def analyze(root, output, **kwargs):
        output.mkdir()
        save(output / "summary.json", {"pass": False})
        return {"pass": False}

    monkeypatch.setattr(p, "analyze", analyze)
    with pytest.raises(ValueError, match="stage HOLD"):
        p.run(tmp_path)
    assert len(events) == 24 and all(v[1] == "functional" for v in events)
    receipt = json.loads((tmp_path / "receipts/controller-terminal.json").read_text())
    assert receipt["state"] == "FULL_HTTP_QUALIFICATION_HOLD" and not receipt["full_http_qualified"]
    assert len(list((tmp_path / "receipts").glob("dispatch-*.json"))) == 24
    with pytest.raises(FileExistsError):
        p.run(tmp_path)


def test_changed_source_stops_before_any_submission_and_persists_hold(tmp_path, monkeypatch):
    events = controller(tmp_path, monkeypatch)
    monkeypatch.setattr(
        p, "verify_frozen", lambda *a: (_ for _ in ()).throw(ValueError("changed source"))
    )
    with pytest.raises(ValueError, match="changed source"):
        p.run(tmp_path)
    assert not events
    assert not json.loads((tmp_path / "receipts/controller-terminal.json").read_text())[
        "full_http_qualified"
    ]


def test_actual_slurm_status_and_original_source_receipts_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "terminal", lambda *a: True)
    monkeypatch.setattr(p.subprocess, "check_output", lambda *a, **k: "123|COMPLETED|0:0|01:00:00")
    for name in ("source-123.txt", "source-post-123.txt", "exit-123.txt"):
        save(tmp_path / "receipts" / name, {"original": name})
    assert p.allocation_terminal(tmp_path, "123")
    record = json.loads((tmp_path / "receipts/allocation-terminal-123.json").read_text())
    assert record["job"] == "123" and len(record["files"]) == 3
    assert all(p.sha(tmp_path / n) == h for n, h in record["files"].items())


def test_slurm_timeout_cannot_be_recorded_as_complete_from_local_exit_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "terminal", lambda *a: True)
    monkeypatch.setattr(p.subprocess, "check_output", lambda *a, **k: "123|TIMEOUT|0:9|08:00:00")
    with pytest.raises(ValueError, match="Actual Slurm completion"):
        p.allocation_terminal(tmp_path, "123")
    assert not (tmp_path / "receipts/allocation-terminal-123.json").exists()
