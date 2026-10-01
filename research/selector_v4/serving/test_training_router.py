"""CPU control/metadata transition checks; no actual HTTP training claim."""

from types import SimpleNamespace as NS

import pytest

from research.selector_v4.serving.epoch_registry import ServingEpochRegistry
from research.selector_v4.serving.metadata_adapter import Binding, MetadataEpochAdapter
from research.selector_v4.serving.prefix_training_server import CONTROL_KEY, phase_transition
from research.selector_v4.serving.test_epoch_registry import Lease
from research.selector_v4.serving.test_prefix_router import tensor
from research.selector_v4.serving.training_router import TrainingPrefixRouter


def setup():
    events = []
    graph = [False]
    owner = NS(_kv_layout="NHD", forward_return_lse=lambda *a, **k: "native")
    backend = NS(
        prefill_wrappers_paged=[owner],
        forward_metadata=NS(use_ragged=True, extend_no_prefix=False, prefill_wrappers=[owner]),
    )
    registry = ServingEpochRegistry(owner)
    lease = Lease(owner)
    lease.return_lse = True

    def write(batch=None):
        if registry.entries:
            assert lease.state == "RELEASED"
        events.append("write")

    backend.init_forward_metadata = write
    backend.init_forward_metadata_out_graph = write
    backend.init_cuda_graph_state = write
    adapter = MetadataEpochAdapter(backend)
    adapter.install()
    router = TrainingPrefixRouter(
        adapter, graph_or_tracing=lambda: graph[0], unpack_paged_cache=lambda cache, layout: cache
    )
    inputs = [tensor(), tensor(), tensor()]
    binding = Binding(registry, "trained", inputs, (3, 5), (10, 20), lease.options)

    def release():
        events.append("release")
        registry.before_metadata_update()

    def record(owner, values, **options):
        events.append("train")
        if not registry.entries:
            registry.register("trained", lease, actual_managed_tactic=1)

    session = NS(
        sealed=False,
        failed=False,
        bindings=[],
        before_metadata_update=release,
        record_current_plan=record,
    )

    def seal():
        release()
        registry.freeze()
        session.sealed = True
        return (binding,)

    session.seal = seal
    batch = NS(
        extend_seq_lens_cpu=(3, 5),
        extend_prefix_lens_cpu=(10, 20),
        forward_mode=NS(is_extend_without_speculative=lambda: True),
    )
    scheduler = NS(
        is_fully_idle=lambda: True,
        tp_worker=NS(
            model_runner=NS(
                attn_backend=NS(_sgi_training_router=router),
                forward_stream=NS(synchronize=lambda: events.append("barrier")),
            )
        ),
    )
    return router, backend, batch, session, inputs, owner, graph, scheduler, events, lease


def test_startup_training_release_and_frozen_rebind_preserve_order():
    router, backend, batch, session, inputs, owner, _, scheduler, events, lease = setup()
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "native"
    assert phase_transition(scheduler, {CONTROL_KEY: 1}, make_session=lambda: session)
    backend.init_forward_metadata(batch)
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "native"
    backend.init_forward_metadata(batch)
    assert events[:6] == ["barrier", "release", "write", "train", "release", "write"]
    assert phase_transition(scheduler, {CONTROL_KEY: 2}, make_session=lambda: None)
    assert lease.state == "RELEASED"
    backend.init_forward_metadata(batch)
    before = events.count("train")
    for _ in range(36):
        assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "prepared_eager"
    assert events.count("train") == before and lease.bind_calls == 1 and lease.run_calls == 36
    assert not phase_transition(scheduler, {CONTROL_KEY: 1}, make_session=lambda: session)
    assert not phase_transition(scheduler, {CONTROL_KEY: 2}, make_session=lambda: None)


@pytest.mark.parametrize(
    "reason", ["busy", "metadata", "unknown", "mixed", "bool", "seal-before-training"]
)
def test_phase_rejection_happens_before_barrier_or_session_creation(reason):
    router, _, _, _, _, _, _, scheduler, events, _ = setup()
    values = {CONTROL_KEY: 1}
    if reason == "busy":
        scheduler.is_fully_idle = lambda: False
    elif reason == "metadata":
        router.adapter.depth = 1
    elif reason == "unknown":
        values[CONTROL_KEY] = 3
    elif reason == "mixed":
        values["pp_max_micro_batch_size"] = 8
    elif reason == "bool":
        values[CONTROL_KEY] = True
    else:
        values[CONTROL_KEY] = 2

    def forbidden():
        raise AssertionError("Rejected phase must not create a calibration session")

    assert not phase_transition(scheduler, values, make_session=forbidden)
    assert events == []


def test_graph_and_unknown_geometry_cannot_train_and_failed_seal_cannot_route():
    router, backend, batch, session, inputs, owner, graph, _, events, _ = setup()
    router.begin(session)
    backend.init_forward_metadata(batch)
    graph[0] = True
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "native"
    graph[0] = False
    router.adapter.current_prefix_lengths = None
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "native"
    assert "train" not in events

    def broken():
        raise RuntimeError("incomplete first calibration")

    session.seal = broken
    with pytest.raises(RuntimeError, match="incomplete"):
        router.seal()
    assert router.phase == "FAILED" and router.frozen_router is None
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=True) == "native"
