"""CPU routing contract tests, no claims about actual resource kernel dispatch."""

from types import SimpleNamespace as NS

import pytest

from research.selector_v4.serving.prefix_router import PagedPrefixRouter


def tensor():
    return NS(
        shape=(2, 1, 8, 128), stride=lambda: (1024, 1024, 128, 1), dtype="bf16", device="cuda:0"
    )


def setup(*, native_entry=False, return_lse=True):
    inputs = [tensor(), tensor(), tensor()]
    native_calls = []
    owner = NS(
        _kv_layout="NHD",
        forward_return_lse=lambda *a, **k: (
            native_calls.append((a, k)) or ("native-o", "native-lse")
        ),
    )
    calls = []
    lease = NS(return_lse=return_lse, state="BOUND")
    registry = NS(
        owner=owner,
        epoch=3,
        bound_keys={"key"},
        entries={"key": None if native_entry else NS(lease=lease, bound_epoch=3)},
    )

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return "managed-o", "managed-lse"

    registry.run = run
    binding = NS(
        registry=registry,
        key="key",
        signature_inputs=inputs,
        qo_lengths=(2,),
        kv_lengths=(5,),
        forward_options={"causal": False},
    )
    adapter = NS(frozen=True, bindings=[binding], depth=0, current_prefix_lengths=((2,), (5,)))
    graph = [False]
    router = PagedPrefixRouter(
        adapter, graph_or_tracing=lambda: graph[0], unpack_paged_cache=lambda value, layout: value
    )
    return router, adapter, registry, owner, inputs, graph, calls, native_calls


def test_current_early_bound_inputs_use_public_registry_with_native_fallback():
    router, _, _, owner, inputs, _, calls, native = setup()
    assert router.run(owner, inputs[0], tuple(inputs[1:]), causal=False) == (
        "managed-o",
        "managed-lse",
    )
    assert len(calls) == 1 and not native
    args, kwargs = calls[0]
    assert args == ("key", inputs)
    assert kwargs["qo_lengths"] == (2,) and kwargs["kv_lengths"] == (5,)
    assert kwargs["native_call"]() == ("native-o", "native-lse")
    assert len(native) == 1
    assert router.evidence()["actual_resource_launch_count"] is None


def test_unknown_host_geometry_returns_native_before_tensor_unpack_or_signatures():
    router, adapter, _, owner, inputs, _, calls, native = setup()
    adapter.current_prefix_lengths = ((999,), (8888,))

    def forbidden(*args):
        raise AssertionError("Unknown host geometry must avoid KV unpack and tensor inspection")

    router.unpack_paged_cache = forbidden
    inputs[0].stride = forbidden
    assert router.run(owner, inputs[0], object(), causal=False) == ("native-o", "native-lse")
    assert not calls and len(native) == 1


@pytest.mark.parametrize(
    "reason",
    [
        "graph",
        "missing-host-geometry",
        "update-in-progress",
        "wrong-options",
        "unseen-input",
        "unbound-epoch",
        "released",
        "unbound-key",
    ],
)
def test_unsupported_or_unbound_never_late_binds_or_calls_registry(reason):
    router, adapter, registry, owner, inputs, graph, calls, native = setup()
    options = {"causal": False}
    if reason == "graph":
        graph[0] = True
    elif reason == "missing-host-geometry":
        adapter.current_prefix_lengths = None
    elif reason == "update-in-progress":
        adapter.depth = 1
    elif reason == "wrong-options":
        options["causal"] = True
    elif reason == "unseen-input":
        inputs[0].dtype = "fp16"
    elif reason == "unbound-epoch":
        registry.entries["key"].bound_epoch = 2
    elif reason == "released":
        registry.entries["key"].lease.state = "RELEASED"
    elif reason == "unbound-key":
        registry.bound_keys.clear()
    assert router.run(owner, inputs[0], tuple(inputs[1:]), **options) == ("native-o", "native-lse")
    assert len(native) == 1 and not calls


def test_native_training_result_is_never_overridden():
    router, _, _, owner, inputs, _, calls, native = setup(native_entry=True)
    assert not router.bindings
    router.run(owner, inputs[0], tuple(inputs[1:]), causal=False)
    assert len(native) == 1 and not calls


def test_missing_lse_duplicate_geometry_or_unfrozen_adapter_rejected():
    with pytest.raises(ValueError, match="LSE"):
        setup(return_lse=False)
    _, adapter, *_ = setup()
    adapter.bindings.append(adapter.bindings[0])
    with pytest.raises(ValueError, match="one frozen decision"):
        PagedPrefixRouter(adapter, graph_or_tracing=lambda: False)
    adapter.frozen = False
    with pytest.raises(ValueError, match="frozen"):
        PagedPrefixRouter(adapter, graph_or_tracing=lambda: False)


def test_packed_payload_must_go_through_native_unpacker_without_iteration():
    router, _, _, owner, inputs, _, calls, _ = setup()

    class Packed:
        def __iter__(self):
            raise AssertionError("Do not iterate token-major packed KV")

    packed = Packed()
    unpacked = []

    def native_unpack(value, layout):
        assert value is packed and layout == "NHD"
        unpacked.append(True)
        return inputs[1], inputs[2]

    router.unpack_paged_cache = native_unpack
    assert router.run(owner, inputs[0], packed, causal=False) == ("managed-o", "managed-lse")
    assert unpacked == [True] and calls[0][0][1] == inputs


def test_unknown_owner_uses_native_without_touching_payload_or_unpacking():
    router, _, _, _, inputs, _, calls, _ = setup()
    owner = NS(forward_return_lse=lambda *a, **k: "native-other-owner")
    router.unpack_paged_cache = lambda *a: pytest.fail("No unpack on unknown owner")
    assert router.run(owner, inputs[0], object(), causal=False) == "native-other-owner"
    assert not calls
