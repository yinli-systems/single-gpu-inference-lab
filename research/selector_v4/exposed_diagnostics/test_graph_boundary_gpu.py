"""Exposed Graph ownership boundaries; raw functional evidence, no performance claim."""

import functools

import pytest

torch = pytest.importorskip("torch")
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from research.selector_v4.exposed_diagnostics.graph_boundary import GraphBoundaryProbe
from research.selector_v4.serving.plan_lease import ServingPlanLease

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
Q = [3, 39, 107, 175, 243, 311, 411]
KV = [a + b for a, b in zip(Q, [22528, 16512, 11840, 8256, 2624, 672, 80])]


def indptr(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )


def owner(paged):
    q = torch.randn(sum(Q), 32, 128, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    if paged:
        k = torch.randn(sum(KV), 1, 8, 128, dtype=q.dtype, device=q.device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")
        plan = functools.partial(
            wrapper.plan,
            indptr(Q),
            indptr(KV),
            torch.arange(sum(KV), dtype=torch.int32, device="cuda"),
            torch.ones(len(Q), dtype=torch.int32, device="cuda"),
            32,
            8,
            128,
            1,
            causal=False,
            q_data_type=q.dtype,
            disable_split_kv=True,
        )
    else:
        k = torch.randn(sum(KV), 8, 128, dtype=q.dtype, device=q.device)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
        plan = functools.partial(
            wrapper.plan,
            indptr(Q),
            indptr(KV),
            32,
            8,
            128,
            causal=True,
            q_data_type=q.dtype,
            disable_split_kv=True,
        )
    v = torch.randn_like(k)
    plan()
    options = {
        "causal": not paged,
        "sm_scale": 128**-0.5,
        "window_left": -1,
        "logits_soft_cap": 0.0,
    }
    return wrapper, [q, k, v], options, plan


def exact(actual, expected):
    torch.cuda.synchronize()
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class ForbiddenGraph:
    def replay(self):
        raise AssertionError("Stale resource graph was replayed")


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("count", [1, 16])
def test_input_updates_and_every_invalidation_boundary(paged, count):
    torch.manual_seed(20261001)
    with torch.inference_mode():
        wrapper, inputs, options, plan = owner(paged)
        original_module = wrapper._cached_module
        original_epoch = wrapper._plan_info

        def native(bound_inputs, bound_options=options):
            q, k, v = bound_inputs
            args = (q, (k, v)) if paged else (q, k, v)
            return tuple(t.clone() for t in wrapper.forward_return_lse(*args, **bound_options))

        def probe():
            lease = ServingPlanLease(
                wrapper,
                inputs,
                qo_lengths=Q,
                kv_lengths=KV,
                forward_options=options,
                return_lse=True,
            )
            return GraphBoundaryProbe(
                lease, inputs, replays=count, allow_uncertified_diagnostic=True
            )

        graph = probe()
        for epoch in range(3):
            inputs[0].add_(0.015625 * (epoch + 1))
            inputs[1].mul_(0.984375)
            inputs[2].add_(0.03125)
            expected = native(inputs)
            exact(graph.replay(inputs, forward_options=options), expected)
            assert graph.last_execution == "uncertified_resource_graph"
        assert wrapper._cached_module is original_module and wrapper._plan_info is original_epoch

        # A same-shaped replacement buffer must never replay the old Q pointer.
        replacement = [inputs[0].neg(), inputs[1], inputs[2]]
        expected = native(replacement)
        graph._graph = ForbiddenGraph()
        exact(graph.replay(replacement, forward_options=options), expected)
        assert graph.last_execution == "native_eager" and graph.invalidated

        # Mutated tracked snapshot metadata must fall back to the original owner.
        graph = probe()
        graph.lease.prepared._qo_indptr_buf.zero_()
        graph._graph = ForbiddenGraph()
        exact(graph.replay(inputs, forward_options=options), native(inputs))
        assert graph.last_execution == "native_eager"

        # The caller releases before writing inference-mode metadata. No version
        # counter is claimed to detect an untracked foreign write automatically.
        graph = probe()
        graph.invalidate()
        if paged:
            wrapper._paged_kv_indices_buf.copy_(wrapper._paged_kv_indices_buf.roll(1))
        else:
            wrapper._qo_indptr_buf.copy_(wrapper._qo_indptr_buf.clone())
        graph._graph = ForbiddenGraph()
        exact(graph.replay(inputs, forward_options=options), native(inputs))

        # A new native planning epoch invalidates even equal 15-field plan values.
        graph = probe()
        previous = wrapper._plan_info
        plan()
        assert wrapper._plan_info is not previous and wrapper._plan_info == previous
        graph._graph = ForbiddenGraph()
        exact(graph.replay(inputs, forward_options=options), native(inputs))

        graph = probe()
        changed_options = dict(options, sm_scale=0.125)
        graph._graph = ForbiddenGraph()
        exact(
            graph.replay(inputs, forward_options=changed_options), native(inputs, changed_options)
        )

        graph = probe()
        graph._graph = ForbiddenGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            exact(graph.replay(inputs, forward_options=options), native(inputs))
            assert graph.last_execution == "native_eager"
        assert wrapper._cached_module is original_module
        assert not graph.evidence()["serving_qualified"]
        assert not graph.evidence()["managed_certificate_reused"]
