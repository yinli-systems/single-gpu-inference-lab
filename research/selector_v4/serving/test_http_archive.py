"""Terminal HOLD retains partial HTTP data; active and uncertain jobs block."""

import tarfile

import pytest

from research.selector_v4.public_qualification.test_gates import save
from research.selector_v4.serving import http_archive as a


def campaign(root):
    save(
        root / "receipts/controller-terminal.json",
        {
            "terminal": True,
            "state": "FULL_HTTP_QUALIFICATION_HOLD",
            "full_http_qualified": False,
        },
    )
    save(root / "receipts/intent-functional.json", {})
    save(root / "receipts/dispatch-functional.json", {"job": "123"})


def test_partial_hold_archives_every_failed_response_and_managed_cache(tmp_path, monkeypatch):
    root = tmp_path / "http"
    campaign(root)
    save(
        root / "paired/123/candidate/observations/partial.json", {"tokens": [1], "error": "timeout"}
    )
    save(root / "paired/123/candidate/training/managed-cache/actual.json", {"winner": 1})
    save(root / "cache/123/rebuildable.json", {"jit": True})
    monkeypatch.setattr(a.subprocess, "check_output", lambda *args, **k: "123|TIMEOUT|0:9|08:00:00")
    out = tmp_path / "archive"
    receipt = a.archive(root, out)
    assert not receipt["full_http_qualified"]
    with tarfile.open(out / "raw-full-http.tar.gz") as stream:
        names = stream.getnames()
    assert "paired/123/candidate/observations/partial.json" in names
    assert "paired/123/candidate/training/managed-cache/actual.json" in names
    assert "cache/123/rebuildable.json" not in names


def test_active_allocation_blocks_archive_before_output(tmp_path, monkeypatch):
    root = tmp_path / "http"
    campaign(root)
    monkeypatch.setattr(a.subprocess, "check_output", lambda *args, **k: "123|RUNNING|0:0|00:01:00")
    out = tmp_path / "archive"
    with pytest.raises(ValueError, match="every actual"):
        a.archive(root, out)
    assert not out.exists()


def test_uncertain_submission_must_be_reconciled_before_archive(tmp_path):
    root = tmp_path / "http"
    campaign(root)
    (root / "receipts/dispatch-functional.json").unlink()
    out = tmp_path / "archive"
    with pytest.raises(ValueError, match="uncertain Slurm"):
        a.archive(root, out)
    assert not out.exists()
