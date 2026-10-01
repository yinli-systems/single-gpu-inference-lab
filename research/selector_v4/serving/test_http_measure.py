"""Prospective HTTP stage gate and command contract, no full-model run."""

import asyncio
from types import SimpleNamespace

import pytest

from research.selector_v4.public_qualification.test_gates import campaign, save
from research.selector_v4.serving.http_measure import run, save_new, server_command


def test_kernel_hold_blocks_model_or_server_launch_before_creating_output(tmp_path):
    root, _, _ = campaign(tmp_path)
    save(
        root / "receipts/controller-terminal.json",
        {"terminal": True, "state": "FORMAL_QUALIFICATION_HOLD"},
    )
    binding = tmp_path / "http-binding.json"
    save(binding, {})
    output = tmp_path / "must-not-exist"
    arguments = SimpleNamespace(out=output, binding=binding, kernel_campaign=root)
    with pytest.raises(ValueError, match="Complete formal kernel"):
        asyncio.run(run(arguments))
    assert not output.exists()


@pytest.mark.parametrize("role", ["pristine", "candidate"])
def test_ordinary_and_deterministic_trials_use_explicit_separate_process_modes(role):
    ordinary = server_command("/synthetic/model", 12345, 0.8, role, "performance")
    parity = server_command("/synthetic/model", 12345, 0.8, role, "parity")
    assert "--enable-deterministic-inference" not in ordinary
    assert parity == ordinary + ["--enable-deterministic-inference"]
    assert "--disable-overlap-schedule" not in ordinary and "--disable-cuda-graph" not in ordinary
    assert ordinary[ordinary.index("--context-length") + 1] == "12288"
    assert ordinary[ordinary.index("--attention-backend") + 1] == "flashinfer"


def test_complete_and_failed_http_artifacts_cannot_replace_an_earlier_attempt(tmp_path):
    path = tmp_path / "first.json"
    save_new(path, {"failed": True})
    with pytest.raises(FileExistsError):
        save_new(path, {"failed": False})
