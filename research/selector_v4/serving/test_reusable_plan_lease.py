"""Ownership violations must not resume a prepared resource call."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from research.selector_v4.serving.reusable_plan_lease import ReusableServingPlanLease


def lease():
    value = ReusableServingPlanLease.__new__(ReusableServingPlanLease)
    value.state = "BOUND"
    value.owner_generation = value.bound_generation = 0
    value.invalidated = False
    value.last_reason = None
    value.options = {"causal": True}
    value.owner_plan_values = tuple(range(15))
    value.owner = SimpleNamespace(_plan_info=list(range(15)))
    value.native = lambda inputs, options: ("owner_native", inputs, options)
    return value


def test_resource_cannot_run_between_release_and_explicit_bind():
    value = lease()
    value.release_before_update()
    assert value.owner_generation == 1 and value.state == "RELEASED"
    assert not value.current([])
    assert value.run([], forward_options=value.options)[0] == "owner_native"
    assert value.state == "RELEASED"


def test_unannounced_bind_retires_the_lease():
    value = lease()
    assert not value.bind_current_plan([], qo_lengths=[1], kv_lengths=[1], forward_options={})
    assert value.state == "RETIRED" and value.last_reason == "update_not_announced"


@pytest.mark.parametrize("field", ["owner_generation", "plan_values"])
def test_generation_or_same_object_plan_mutation_is_sticky(field):
    value = lease()
    if field == "owner_generation":
        value.owner_generation += 1
    else:
        value.owner._plan_info[3] += 1
    assert not value.current([])
    assert value.state == "RETIRED" and value.invalidated
    value.owner_generation = value.bound_generation
    value.owner._plan_info[:] = value.owner_plan_values
    assert not value.current([])


def test_changed_forward_options_never_resume_resource():
    value = lease()
    assert value.run([], forward_options={"causal": False})[0] == "owner_native"
    assert value.state == "RETIRED"
    assert value.run([], forward_options=value.options)[0] == "owner_native"


def test_released_lease_cannot_run_on_changed_options():
    value = lease()
    value.release_before_update()
    assert value.run([], forward_options={"causal": False})[0] == "owner_native"
    assert value.state == "RETIRED"
    with pytest.raises(RuntimeError, match="retired"):
        value.release_before_update()
