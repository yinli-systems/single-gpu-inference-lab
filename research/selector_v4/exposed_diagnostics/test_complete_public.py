"""Regression tests for real allocation failure and incomplete exposed evidence."""

import io
import json
import tarfile

import pytest

from research.selector_v4.exposed_diagnostics import complete_public as c


@pytest.mark.parametrize(
    "state,exit_code,expected",
    [
        ("COMPLETED", "0:0", True),
        ("TIMEOUT", "0:0", False),
        ("FAILED", "0:0", False),
        ("COMPLETED", "1:0", False),
    ],
)
def test_accounting_never_trusts_zero_exit_alone(state, exit_code, expected):
    result = c.status("123", lambda _: f"123|{state}|{exit_code}|08:00:27|\n")
    assert result["terminal"] and result["completed"] is expected


def fixture_campaign(tmp_path):
    (tmp_path / "receipts").mkdir()
    (tmp_path / "logs").mkdir()
    binding = {
        "cases": [{"id": "original"}],
        "candidate_commit": c.CANDIDATE,
        "pristine_commit": c.PRISTINE,
    }
    (tmp_path / "binding.json").write_text(json.dumps(binding))
    (tmp_path / "receipts/exit-123.txt").write_text("0\n")
    root = tmp_path / "runs/gpu_4090-123"
    for identity in c.PHASES:
        phase = root / identity
        phase.mkdir(parents=True)
        role, rep = identity.rsplit("-", 1)
        (phase / "complete.json").write_text(
            json.dumps({"complete": True, "cells": {str(i): {} for i in range(24)}})
        )
        (phase / "environment.json").write_text(
            json.dumps(
                {
                    "case": binding["cases"][0],
                    "role": role,
                    "rep": int(rep),
                    "source_commit": c.PRISTINE if role == "pristine" else c.CANDIDATE,
                }
            )
        )
    return root


def test_last_phase_thirteen_cells_rejected_despite_exit_zero(tmp_path):
    root = fixture_campaign(tmp_path)
    assert c.completeness(tmp_path, "gpu_4090", "0", "123")
    (root / "policy-2/complete.json").write_text(
        json.dumps({"complete": True, "cells": {str(i): {} for i in range(13)}})
    )
    with pytest.raises(ValueError, match="All 24 cells"):
        c.completeness(tmp_path, "gpu_4090", "0", "123")


def test_source_identity_changed_rejected(tmp_path):
    root = fixture_campaign(tmp_path)
    path = root / "train-0/environment.json"
    value = json.loads(path.read_text())
    value["source_commit"] = "other"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="package identity"):
        c.completeness(tmp_path, "gpu_4090", "0", "123")


def test_controller_timeout_retains_all_four_jobs_and_blocks_analysis(tmp_path, monkeypatch):
    (tmp_path / "receipts").mkdir()
    binding = {"candidate_source": "candidate", "pristine_source": "pristine"}
    (tmp_path / "binding.json").write_text(json.dumps(binding))
    monkeypatch.setattr(c, "verify_sources", lambda _: binding)
    ids = iter(["123", "124", "125", "126"])
    monkeypatch.setattr(c.subprocess, "check_output", lambda *a, **kw: next(ids))
    monkeypatch.setattr(
        c,
        "status",
        lambda job: {
            "terminal": True,
            "completed": job != "124",
            "state": "TIMEOUT" if job == "124" else "COMPLETED",
        },
    )
    monkeypatch.setattr(
        c, "execute", lambda *a, **kw: pytest.fail("No analysis after incomplete allocation")
    )
    result = c.run(tmp_path)
    assert result["terminal"] and result["state"] == "PUBLIC_PATH_DEVELOPMENT_COMPLETION_HOLD"
    assert sum(len(g) for g in result["jobs"].values()) == 4
    assert json.loads((tmp_path / "receipts/jobs.json").read_text()) == result["jobs"]
    with pytest.raises(ValueError, match="No controller retries"):
        c.run(tmp_path)


def test_prior_archive_member_integrity_required(tmp_path):
    source = tmp_path / "raw-public-path-development.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo("raw.json")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"bad"))
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "archive_sha256": c.sha(source),
                "files": {"raw.json": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="Prior archive member"):
        c.verify_archive(tmp_path)
