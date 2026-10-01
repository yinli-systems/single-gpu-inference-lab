"""Synthetic premeasurement authorization/lifecycle, no actual GPU training."""

import json
from types import SimpleNamespace

import pytest

from research.selector_v4.public_qualification.test_gates import campaign, save
from research.selector_v4.serving.training_session import (
    PrefixTrainingSession,
    authorize_http_training,
    geometry_key,
)


def complete_kernel(tmp_path):
    root, expected, _ = campaign(tmp_path)
    save(
        root / "receipts/controller-terminal.json",
        {"state": "FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED", "terminal": True},
    )
    return root, expected


def test_all_eight_stage_verdicts_and_same_loaded_source_are_required(tmp_path):
    root, expected = complete_kernel(tmp_path)
    ticket = authorize_http_training(root, expected["candidate_commit"])
    assert len(ticket["summaries"]) == 8
    assert not ticket["request_time_training"] and not ticket["full_http_resource_qualified"]
    with pytest.raises(ValueError, match="Different serving kernel"):
        authorize_http_training(root, "other-source")
    path = root / "stages/stress/analysis/gpu_5090/summary.json"
    result = json.loads(path.read_text())
    result["pass"] = False
    save(path, result)
    with pytest.raises(ValueError, match="stress gpu_5090 HOLD"):
        authorize_http_training(root, expected["candidate_commit"])


def test_kernel_hold_cannot_create_training_output(tmp_path):
    root, expected = complete_kernel(tmp_path)
    save(
        root / "receipts/controller-terminal.json",
        {"state": "FORMAL_QUALIFICATION_HOLD", "terminal": True},
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Complete formal kernel"):
        PrefixTrainingSession(root, output, candidate_commit=expected["candidate_commit"])
    assert not output.exists()


def test_changed_smoke_or_stress_raw_data_blocks_premeasurement_training(tmp_path):
    root, expected = complete_kernel(tmp_path)
    (root / "stages/stress/raw.txt").write_text("changed")
    with pytest.raises(ValueError, match="Raw evidence changed"):
        authorize_http_training(root, expected["candidate_commit"])
    (root / "smoke-proof.txt").write_text("changed")
    with pytest.raises(ValueError, match="Smoke evidence"):
        authorize_http_training(root, expected["candidate_commit"])


def test_bounded_session_seals_once_and_cannot_train_or_update_on_requests(tmp_path):
    root, expected = complete_kernel(tmp_path)
    output = tmp_path / "synthetic-session"
    session = PrefixTrainingSession(
        root, output, candidate_commit=expected["candidate_commit"], maximum_keys=2
    )
    events = []
    session.registries[1] = SimpleNamespace(
        before_metadata_update=lambda: events.append("release"),
        freeze=lambda: events.append("freeze"),
    )
    assert session.seal() == () and events == ["release", "freeze"]
    with pytest.raises(ValueError, match="sealed"):
        session.record_current_plan(None, [], qo_lengths=[], kv_lengths=[], forward_options={})
    with pytest.raises(ValueError, match="serving adapter"):
        session.before_metadata_update()
    with pytest.raises(ValueError, match="only once"):
        session.seal()
    with pytest.raises(FileExistsError):
        PrefixTrainingSession(root, output, candidate_commit=expected["candidate_commit"])
    with pytest.raises(ValueError, match="capacity"):
        PrefixTrainingSession(
            root,
            tmp_path / "invalid",
            candidate_commit=expected["candidate_commit"],
            maximum_keys=257,
        )


def test_geometry_identity_binds_stride_options_and_positive_host_lengths():
    tensor = lambda stride: SimpleNamespace(
        shape=(2, 1, 8, 128), stride=lambda: stride, dtype="bf16", device="cuda:0"
    )
    inputs = [tensor((1024, 1024, 128, 1)) for _ in range(3)]
    key = geometry_key(inputs, (2,), (5,), {"causal": False})
    assert key != geometry_key(inputs, (2,), (5,), {"causal": True})
    inputs[1] = tensor((2048, 1024, 128, 1))
    assert key != geometry_key(inputs, (2,), (5,), {"causal": False})
    with pytest.raises(ValueError, match="Positive"):
        geometry_key(inputs, (0,), (5,), {})
    with pytest.raises(ValueError, match="scalar"):
        geometry_key(inputs, (2,), (5,), {"k_scale": object()})


def test_incomplete_first_calibration_preserves_failure_and_cannot_seal_or_retry(tmp_path):
    root, expected = complete_kernel(tmp_path)
    session = PrefixTrainingSession(
        root, tmp_path / "failed-session", candidate_commit=expected["candidate_commit"]
    )
    attempt = session.output / "entry-0000"
    attempt.mkdir()
    (attempt / "training.json").write_text('{"actual_managed_tactic": -1}')
    released = []
    session.registries[1] = SimpleNamespace(before_metadata_update=lambda: released.append(True))
    with pytest.raises(RuntimeError, match="registration failed"), session._attempt(attempt):
        raise RuntimeError("registration failed")
    assert released == [True]
    assert json.loads((attempt / "training.json").read_text())["actual_managed_tactic"] == -1
    assert json.loads((attempt / "failure.json").read_text())["type"] == "RuntimeError"
    with pytest.raises(ValueError, match="incomplete"):
        session.seal()
    with pytest.raises(ValueError, match="incomplete"):
        session.record_current_plan(None, [], qo_lengths=[], kv_lengths=[], forward_options={})
