import pytest

from research.selector_v4 import continue_after_reaudit as c


@pytest.mark.parametrize("passed, retained", [(False, True), (True, False)])
def test_only_separately_accepted_correction_can_initialize_new_workflow(tmp_path, monkeypatch, passed, retained):
    monkeypatch.setattr(c.exposed, "verify_archive", lambda p: {
        "pass_public_path_development": passed,
        "controller": {"original_terminal_hold_retained": retained},
    })
    out = tmp_path / "never-created"
    with pytest.raises(ValueError):
        c.initialize(out, None, None, tmp_path / "receipt.json", None, None)
    assert not out.exists()
