"""Verify that failed qualifications cannot launch or initialize successors."""

import json

import pytest

from research.selector_v4 import complete_campaign as workflow


@pytest.mark.parametrize(
    "public_pass,kernel_pass,http_pass,expected",
    [
        (False, True, True, ["public", "public-archive"]),
        (
            True,
            False,
            True,
            ["public", "public-archive", "kernel-init", "kernel", "kernel-archive"],
        ),
        (
            True,
            True,
            False,
            [
                "public",
                "public-archive",
                "kernel-init",
                "kernel",
                "kernel-archive",
                "http-init",
                "http",
                "http-archive",
            ],
        ),
        (
            True,
            True,
            True,
            [
                "public",
                "public-archive",
                "kernel-init",
                "kernel",
                "kernel-archive",
                "http-init",
                "http",
                "http-archive",
            ],
        ),
    ],
)
def test_complete_chain_stops_at_first_hold_and_never_closes_history(
    tmp_path, monkeypatch, public_pass, kernel_pass, http_pass, expected
):
    root = tmp_path / "workflow"
    root.mkdir()
    public = tmp_path / "public"
    (public / "receipts").mkdir(parents=True)
    (public / "receipts/jobs.json").write_text("{}")
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"frozen prepared source")
    binding = {
        "public_campaign": str(public),
        "source": str(tmp_path),
        "source_archive": str(archive),
        "harness_commit": "a" * 40,
        "model_evidence": "models",
        "sglang_source": "sglang",
        "finite_days": 30,
    }
    (root / "binding.json").write_text(json.dumps(binding))
    events = []
    monkeypatch.setattr(workflow, "verify", lambda *args: None)
    monkeypatch.setattr(workflow, "wait_terminal", lambda *args: None)

    def record(label, value=None):
        def function(*args):
            events.append(label)
            return value

        return function

    monkeypatch.setattr(workflow.exposed, "run", record("public", {"terminal": True}))
    monkeypatch.setattr(
        workflow.exposed,
        "archive",
        record(
            "public-archive", {"archive_sha256": "raw", "pass_public_path_development": public_pass}
        ),
    )
    monkeypatch.setattr(workflow.kernel, "initialize", record("kernel-init"))
    monkeypatch.setattr(workflow.kernel, "run", record("kernel"))
    monkeypatch.setattr(workflow.kernel_archive, "job_records", lambda *args: {})
    monkeypatch.setattr(
        workflow.kernel_archive,
        "archive",
        record(
            "kernel-archive",
            {"archive_sha256": "kernelraw", "pass_formal_kernel_qualification": kernel_pass},
        ),
    )

    def http_init(*args):
        events.append("http-init")
        (root / "http/receipts").mkdir(parents=True)

    monkeypatch.setattr(workflow.http_pipeline, "initialize", http_init)
    monkeypatch.setattr(workflow.http_pipeline, "run", record("http"))
    monkeypatch.setattr(
        workflow.http_archive,
        "archive",
        record("http-archive", {"archive_sha256": "httpraw", "full_http_qualified": http_pass}),
    )
    result = workflow.run(root)
    assert events == expected
    assert result["terminal"]
    assert result["historical_token_divergence_resolved"] is False
    assert result["default_promotion"] is False and result["serving_promotion"] is False
    assert result["full_http_qualified"] is (public_pass and kernel_pass and http_pass)
    with pytest.raises(ValueError, match="no retries"):
        workflow.run(root)


def test_uncertain_formal_submission_is_retained_and_cannot_be_archived(tmp_path, monkeypatch):
    campaign = tmp_path / "kernel"
    receipts = campaign / "smoke/receipts"
    receipts.mkdir(parents=True)
    (campaign / "receipts").mkdir()
    (campaign / "receipts/controller-terminal.json").write_text(
        json.dumps({"terminal": True, "state": "FORMAL_QUALIFICATION_HOLD"})
    )
    script = tmp_path / "smoke.sbatch"
    script.write_text("#!/bin/bash\n")

    def uncertain(*args, **kwargs):
        raise TimeoutError("Submission result not delivered")

    monkeypatch.setattr(workflow.kernel.subprocess, "check_output", uncertain)
    with pytest.raises(TimeoutError):
        workflow.kernel.submit(
            campaign / "smoke", script, ["gpu_4090", campaign / "smoke", "source"]
        )
    assert len(list(receipts.glob("submit-intent-*.json"))) == 1
    assert not list(receipts.glob("submit-confirmed-*.json"))
    with pytest.raises(ValueError, match="uncertain formal Slurm"):
        workflow.kernel_archive.archive(campaign, tmp_path / "archive")
