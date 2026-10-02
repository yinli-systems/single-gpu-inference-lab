"""Adversarial transaction boundaries; real CUDA execution is tested separately."""
import pytest

from research.selector_v4.exposed_diagnostics.graph_epoch import EpochGuard


def test_announced_updates_preserve_epoch_order():
    guard = EpochGuard(("plan0", "pages0"))
    assert guard.permits(guard.frame)
    for epoch in range(1, 4):
        assert guard.begin(guard.frame)
        with pytest.raises(RuntimeError):
            guard.permits(guard.frame)
        assert guard.finish((f"plan{epoch}", f"pages{epoch}"), compatible=True)
        assert guard.permits(guard.frame)
        assert guard.generation == epoch


@pytest.mark.parametrize("changed", ["physical_pages", "plan_generation", "input_pointer", "stream"])
def test_unannounced_changes_retire_permanently(changed):
    guard = EpochGuard("original")
    assert not guard.permits(changed)
    assert guard.state == "RETIRED"
    assert not guard.permits("original")
    assert not guard.begin("original")
    assert not guard.finish("original", compatible=True)


def test_incompatible_new_geometry_cannot_recover():
    guard = EpochGuard("old")
    assert guard.begin("old")
    assert not guard.finish("changed_geometry", compatible=False)
    assert not guard.finish("old", compatible=True)
    assert not guard.permits("old")


def test_finish_without_begin_rejects_even_equal_values():
    guard = EpochGuard("old")
    assert not guard.finish("old", compatible=True)
    assert guard.state == "RETIRED"


def test_nested_update_cannot_silently_reset_epoch():
    guard = EpochGuard("old")
    assert guard.begin("old")
    assert not guard.begin("old")
    assert guard.state == "RETIRED"
    assert not guard.finish("new", compatible=True)
