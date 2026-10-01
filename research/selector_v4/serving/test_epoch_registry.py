"""Verify amortization and fail-closed scheduler integration without GPU mocks."""

from types import SimpleNamespace

import pytest

from research.selector_v4.serving.epoch_registry import ServingEpochRegistry


class Lease:
    def __init__(self, owner):
        self.owner = owner
        self.runner = SimpleNamespace(
            execution_mode="eager_run",
            _receipt_valid=True,
            _proxy=object(),
            qo_lengths=(3, 5),
            kv_lengths=(10, 20),
        )
        self.options = {"causal": True}
        self.state = "BOUND"
        self.bind_calls = self.run_calls = 0
        self.bind_succeeds = self.still_current = True

    def release_before_update(self):
        self.state = "RELEASED"

    def invalidate(self):
        self.state = "RETIRED"

    def bind_current_plan(self, inputs, **options):
        assert self.state == "RELEASED"
        self.bind_calls += 1
        self.state = "BOUND" if self.bind_succeeds else "RETIRED"
        return self.bind_succeeds

    def current(self, inputs):
        return self.state == "BOUND" and self.still_current

    def run(self, inputs, **options):
        self.run_calls += 1
        return "prepared_eager"


def invoke(registry, key="geometry", **overrides):
    options = {
        "qo_lengths": (3, 5),
        "kv_lengths": (10, 20),
        "forward_options": {"causal": True},
        "native_call": lambda: "owner_native",
    }
    options.update(overrides)
    return registry.run(key, [], **options)


def prepared():
    owner = object()
    registry = ServingEpochRegistry(owner)
    lease = Lease(owner)
    registry.register("geometry", lease, actual_managed_tactic=1)
    registry.freeze()
    return registry, lease


def test_one_metadata_rebind_is_amortized_across_all_36_layers():
    registry, lease = prepared()
    for _ in range(3):
        registry.before_metadata_update()
        for _ in range(36):
            assert invoke(registry) == "prepared_eager"
    assert lease.bind_calls == 3 and lease.run_calls == 108
    assert registry.evidence()["actual_resource_launch_count"] is None


def test_unknown_geometry_stays_native_without_constructor_or_registration():
    registry, lease = prepared()
    before = dict(registry.entries)
    for _ in range(36):
        assert invoke(registry, "unseen") == "owner_native"
    assert registry.entries == before and lease.bind_calls == lease.run_calls == 0


def test_native_choice_has_no_prepared_wrapper_cost():
    registry = ServingEpochRegistry(object())
    registry.register("geometry", None, actual_managed_tactic=-1)
    registry.freeze()
    registry.before_metadata_update()
    assert invoke(registry) == "owner_native"
    assert registry.evidence()["native_keys"] == 1


def test_failed_rebind_is_sticky_and_not_retried_per_layer():
    registry, lease = prepared()
    lease.bind_succeeds = False
    registry.before_metadata_update()
    for _ in range(36):
        assert invoke(registry) == "owner_native"
    registry.before_metadata_update()
    assert invoke(registry) == "owner_native"
    assert lease.bind_calls == 1 and lease.run_calls == 0


def test_graph_capture_keeps_native_and_does_not_consume_an_eager_bind():
    registry, lease = prepared()
    registry.before_metadata_update()
    assert invoke(registry, graph_or_tracing=True) == "owner_native"
    assert lease.bind_calls == lease.run_calls == 0 and lease.state == "RELEASED"
    assert invoke(registry) == "prepared_eager" and lease.bind_calls == 1


@pytest.mark.parametrize("field", ["qo_lengths", "kv_lengths", "forward_options"])
def test_same_key_with_changed_geometry_or_options_retires(field):
    registry, lease = prepared()
    value = {"causal": False} if field == "forward_options" else (4, 4)
    assert invoke(registry, **{field: value}) == "owner_native"
    assert lease.state == "RETIRED" and invoke(registry) == "owner_native"


def test_choices_cannot_be_replaced_or_added_after_freeze():
    registry = ServingEpochRegistry(object())
    registry.register("geometry", None, actual_managed_tactic=-1)
    with pytest.raises(ValueError, match="first"):
        registry.register("geometry", None, actual_managed_tactic=-1)
    registry.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register("new", None, actual_managed_tactic=-1)


def test_uncertified_or_foreign_owner_lease_cannot_be_registered():
    registry = ServingEpochRegistry(object())
    with pytest.raises(ValueError, match="owned"):
        registry.register("geometry", Lease(object()), actual_managed_tactic=1)
    lease = Lease(registry.owner)
    lease.runner._receipt_valid = False
    with pytest.raises(ValueError, match="certified"):
        registry.register("geometry", lease, actual_managed_tactic=1)


def test_execution_before_freeze_is_native():
    owner = object()
    registry = ServingEpochRegistry(owner)
    lease = Lease(owner)
    registry.register("geometry", lease, actual_managed_tactic=1)
    assert invoke(registry) == "owner_native" and lease.run_calls == 0


def test_registry_capacity_is_bounded_before_any_serving_freeze():
    registry = ServingEpochRegistry(object(), maximum_keys=1)
    registry.register("geometry", None, actual_managed_tactic=-1)
    with pytest.raises(ValueError, match="capacity"):
        registry.register("unseen", None, actual_managed_tactic=-1)
    registry.freeze()
    assert invoke(registry, "unseen") == "owner_native"


def test_metadata_can_bind_before_first_layer_without_executing_attention():
    registry, lease = prepared()
    registry.before_metadata_update()
    assert registry.bind_after_metadata_update(
        "geometry", [], qo_lengths=(3, 5), kv_lengths=(10, 20), forward_options=lease.options
    )
    assert lease.bind_calls == 1 and lease.run_calls == 0
    for _ in range(36):
        assert invoke(registry) == "prepared_eager"
    assert lease.bind_calls == 1 and lease.run_calls == 36


def test_early_bind_cache_miss_remains_native_without_preparation():
    registry, lease = prepared()
    registry.before_metadata_update()
    assert not registry.bind_after_metadata_update(
        "unseen", [], qo_lengths=(3, 5), kv_lengths=(10, 20), forward_options=lease.options
    )
    assert lease.bind_calls == lease.run_calls == 0
