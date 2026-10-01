"""Scheduler notification ordering, nested graph boundaries and fail-closed scope."""

from types import SimpleNamespace

import pytest

from research.selector_v4.serving.epoch_registry import ServingEpochRegistry
from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter, host_lengths
from research.selector_v4.serving.test_epoch_registry import Lease


def fixture():
    owner = SimpleNamespace(_kv_layout="NHD")
    registry = ServingEpochRegistry(owner)
    lease = Lease(owner)
    registry.register("prefix", lease, actual_managed_tactic=1)
    registry.freeze()
    batch = SimpleNamespace(
        extend_seq_lens_cpu=[3, 5],
        extend_prefix_lens_cpu=[10, 20],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    backend = SimpleNamespace(
        page_size=1,
        prefill_wrappers_paged=[owner],
        forward_metadata=SimpleNamespace(
            use_ragged=True, extend_no_prefix=False, prefill_wrappers=[owner]
        ),
    )
    events = []

    def write(forward_batch):
        events.append("first_foreign_metadata_write")
        assert lease.state == "RELEASED", "Notification must precede non-Torch writes"
        return "original_result"

    backend.init_forward_metadata = write
    backend.init_forward_metadata_out_graph = lambda forward_batch: backend.init_forward_metadata(
        forward_batch
    )
    backend.init_cuda_graph_state = lambda *args, **kwargs: write(batch)
    adapter = MetadataEpochAdapter(backend)
    adapter.attach(
        registry,
        "prefix",
        [object(), SimpleNamespace(shape=(64, 1, 8, 128)), SimpleNamespace(shape=(64, 1, 8, 128))],
        qo_lengths=(3, 5),
        kv_lengths=(10, 20),
        forward_options=lease.options,
    )
    return backend, adapter, registry, lease, batch, events


def test_notification_precedes_first_foreign_write_and_binding_follows_original():
    backend, adapter, registry, lease, batch, events = fixture()
    adapter.install()
    assert backend.init_forward_metadata(batch) == "original_result"
    assert events == ["first_foreign_metadata_write"]
    assert lease.state == "BOUND" and lease.bind_calls == 1 and lease.run_calls == 0
    assert registry.epoch == 1 and adapter.counters["early_binds"] == 1


def test_nested_graph_metadata_releases_once_without_early_bind():
    backend, adapter, registry, lease, batch, _ = fixture()
    adapter.install()
    backend.init_forward_metadata_out_graph(forward_batch=batch)
    assert registry.epoch == 1 and lease.state == "RELEASED" and lease.bind_calls == 0
    assert adapter.depth == 0 and adapter.counters["graph_updates"] == 1


def test_graph_state_initialization_releases_without_binding():
    backend, adapter, registry, lease, _, _ = fixture()
    adapter.install()
    backend.init_cuda_graph_state(32, max_num_tokens=256)
    assert registry.epoch == 1 and lease.state == "RELEASED" and lease.bind_calls == 0


@pytest.mark.parametrize("boundary", ["unknown_lengths", "no_prefix", "paged_only", "decode"])
def test_unqualified_metadata_keeps_released_native_state(boundary):
    backend, adapter, _, lease, batch, _ = fixture()
    if boundary == "unknown_lengths":
        batch.extend_seq_lens_cpu = [4, 4]
    elif boundary == "no_prefix":
        backend.forward_metadata.extend_no_prefix = True
    elif boundary == "paged_only":
        backend.forward_metadata.use_ragged = False
    else:
        batch.forward_mode.is_extend_without_speculative = lambda: False
    adapter.install()
    backend.init_forward_metadata(batch)
    assert lease.state == "RELEASED" and lease.bind_calls == 0


def test_planner_exception_leaves_lease_released_and_resets_nested_depth():
    backend, adapter, _, lease, batch, _ = fixture()

    def broken(forward_batch):
        assert lease.state == "RELEASED"
        raise RuntimeError("planner failed")

    backend.init_forward_metadata = broken
    adapter.install()
    with pytest.raises(RuntimeError, match="planner failed"):
        backend.init_forward_metadata(batch)
    assert adapter.depth == 0 and lease.state == "RELEASED" and lease.bind_calls == 0


def test_restore_releases_resource_before_removing_all_notifications():
    backend, adapter, _, lease, batch, _ = fixture()
    original = backend.init_forward_metadata
    adapter.install()
    backend.init_forward_metadata(batch)
    adapter.restore()
    assert backend.init_forward_metadata is original and lease.state == "RELEASED"


def test_incomplete_backend_is_rejected_before_any_method_mutation():
    backend, adapter, _, _, _, _ = fixture()
    original = backend.init_forward_metadata
    del backend.init_cuda_graph_state
    with pytest.raises(ValueError, match="entry points"):
        adapter.install()
    assert backend.init_forward_metadata is original and not adapter.originals


def test_untrained_foreign_duplicate_and_post_install_attachment_are_rejected():
    backend, adapter, registry, _, _, _ = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.attach(
            registry, "prefix", [], qo_lengths=(3, 5), kv_lengths=(10, 20), forward_options={}
        )
    foreign = ServingEpochRegistry(object())
    foreign.freeze()
    with pytest.raises(ValueError, match="original paged wrapper"):
        adapter.attach(foreign, "prefix", [], qo_lengths=(), kv_lengths=(), forward_options={})
    untrained = ServingEpochRegistry(backend.prefill_wrappers_paged[0])
    with pytest.raises(ValueError, match="Freeze"):
        adapter.attach(untrained, "prefix", [], qo_lengths=(), kv_lengths=(), forward_options={})
    adapter.install()
    with pytest.raises(RuntimeError, match="before installation"):
        adapter.attach(registry, "prefix", [], qo_lengths=(), kv_lengths=(), forward_options={})


@pytest.mark.parametrize("value", [None, [True], [-1], [1.0], SimpleNamespace(is_cuda=True)])
def test_device_or_unknown_lengths_never_trigger_metadata_readback(value):
    assert host_lengths(value) is None


def test_allocator_page16_does_not_override_actual_native_page1_signature():
    backend, adapter, _, lease, batch, _ = fixture()
    backend.page_size = 16
    adapter.install()
    backend.init_forward_metadata(batch)
    assert lease.bind_calls == 1 and lease.state == "BOUND"


def test_actual_native_page16_signature_is_rejected_before_installation():
    backend, _, registry, lease, _, _ = fixture()
    adapter = MetadataEpochAdapter(backend)
    inputs = [
        object(),
        SimpleNamespace(shape=(64, 16, 8, 128)),
        SimpleNamespace(shape=(64, 16, 8, 128)),
    ]
    with pytest.raises(ValueError, match="Actual NHD native page1"):
        adapter.attach(
            registry,
            "prefix",
            inputs,
            qo_lengths=(3, 5),
            kv_lengths=(10, 20),
            forward_options=lease.options,
        )
    assert not adapter.bindings
