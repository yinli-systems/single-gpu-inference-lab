"""Kernel HOLD prevents paired HTTP execution before any measurement mutation."""

import pytest

from research.selector_v4.public_qualification.test_gates import campaign, save
from research.selector_v4.serving.http_pair import run


def test_kernel_hold_blocks_pair_before_output_or_subprocess(tmp_path, monkeypatch):
    kernel, _, _ = campaign(tmp_path)
    save(
        kernel / "receipts/controller-terminal.json",
        {"terminal": True, "state": "FORMAL_QUALIFICATION_HOLD"},
    )
    root = tmp_path / "http"
    root.mkdir()
    save(root / "binding.json", {})
    monkeypatch.setattr(
        "research.selector_v4.serving.http_pair.subprocess.run",
        lambda *a, **k: pytest.fail("Kernel HOLD must never launch HTTP"),
    )
    with pytest.raises(ValueError, match="Complete formal kernel"):
        run(root, kernel, "Qwen3-4B", "functional", 0)
    assert not (root / "paired").exists()
