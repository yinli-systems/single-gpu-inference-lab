from types import SimpleNamespace

import pytest

from research.selector_v4.serving.graph_load_boundary import wrap_load
from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter


def fixture():
    events = []
    lease = SimpleNamespace(state="BOUND")

    def release():
        events.append("lease-released")
        lease.state = "PLANNING"

    lease.release_before_update = release
    registry = SimpleNamespace()

    def before():
        if lease.state == "BOUND":
            lease.release_before_update()

    registry.before_metadata_update = before
    backend = SimpleNamespace()
    adapter = MetadataEpochAdapter(backend)
    adapter.bindings = [SimpleNamespace(registry=registry)]
    backend._sgi_metadata_adapter = adapter
    runner = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=backend))
    return events, lease, adapter, runner


def test_gpu_only_glue_replay_cannot_skip_lease_release():
    events, lease, adapter, runner = fixture()

    def load(self):
        assert lease.state == "PLANNING"
        events.extend(["input-copy", "gpu-only-metadata-glue-replay"])
        return "native-result"

    assert wrap_load(load)(runner) == "native-result"
    assert events == ["lease-released", "input-copy", "gpu-only-metadata-glue-replay"]
    assert adapter.counters["graph_updates"] == 1
    wrap_load(load)(runner)
    assert runner.model_runner.attn_backend._sgi_decode_load_notifications == 2


def test_failed_native_load_never_keeps_a_bound_lease():
    _, lease, _, runner = fixture()

    def fail(self):
        raise ValueError("original metadata update failed")

    with pytest.raises(ValueError):
        wrap_load(fail)(runner)
    assert lease.state == "PLANNING"


def test_foreign_or_nested_adapter_rejected_before_native_writes():
    _, _, adapter, runner = fixture()
    adapter.depth = 1
    with pytest.raises(RuntimeError):
        wrap_load(lambda self: None)(runner)
    adapter.depth = 0
    adapter.backend = object()
    with pytest.raises(RuntimeError):
        wrap_load(lambda self: None)(runner)


def test_unknown_native_backend_unchanged_and_double_install_rejected():
    runner = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=object()))
    wrapped = wrap_load(lambda self, x: x)
    assert wrapped(runner, "native") == "native"
    with pytest.raises(RuntimeError):
        wrap_load(wrapped)


def test_prefill_static_buffer_writes_follow_lease_release():
    events, lease, _, runner = fixture()

    def load(self):
        assert lease.state == "PLANNING"
        events.extend(["prefill-fill-from", "prefill-metadata-refresh"])

    wrap_load(load, kind="prefill")(runner)
    assert events == ["lease-released", "prefill-fill-from", "prefill-metadata-refresh"]
    assert runner.model_runner.attn_backend._sgi_prefill_load_notifications == 1


def test_sealed_router_current_adapter_wins_over_stale_diagnostic_attribute():
    events, lease, original, runner = fixture()
    backend = runner.model_runner.attn_backend
    current = MetadataEpochAdapter(backend)
    current.bindings = original.bindings
    original.bindings = []
    backend._sgi_training_router = SimpleNamespace(adapter=current)

    def load(self):
        assert lease.state == "PLANNING"

    wrap_load(load)(runner)
    assert events == ["lease-released"]
    assert current.counters["graph_updates"] == 1
    assert original.counters["graph_updates"] == 0
