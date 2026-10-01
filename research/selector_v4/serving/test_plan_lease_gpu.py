"""Exposed-geometry functional tests; no serving/performance/freshness verdict."""

import pytest

torch = pytest.importorskip("torch")
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    make_prefill_resource_runner,
)

from research.selector_v4.serving.plan_lease import ServingPlanLease

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
Q = [3, 39, 107, 175, 243, 311, 411]
KV = [a + b for a, b in zip(Q, [22528, 16512, 11840, 8256, 2624, 672, 80])]


def indptr(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )


@pytest.mark.parametrize("paged", [False, True])
def test_inference_metadata_snapshot_and_epoch_fallback(paged):
    torch.manual_seed(42)
    options = {
        "causal": not paged,
        "sm_scale": 128**-0.5,
        "window_left": -1,
        "logits_soft_cap": 0.0,
    }
    with torch.inference_mode():
        q = torch.randn(sum(Q), 32, 128, dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        if paged:
            k = torch.randn(sum(KV), 1, 8, 128, dtype=q.dtype, device=q.device)
            v = torch.randn_like(k)
            w = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")
            w.plan(
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
            v = torch.randn_like(k)
            w = BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
            w.plan(
                indptr(Q),
                indptr(KV),
                32,
                8,
                128,
                causal=True,
                q_data_type=q.dtype,
                disable_split_kv=True,
            )
        inputs = [q, k, v]
        method = w.forward_return_lse
        args = (q, (k, v)) if paged else (q, k, v)
        native = tuple(t.clone() for t in method(*args, **options))
        original_plan = w._plan_info
        original_module = w._cached_module
        rejected = make_prefill_resource_runner(w, inputs, qo_lengths=Q, kv_lengths=KV)
        assert not rejected._eligible, "Untracked original inference metadata must stay unsupported"
        lease = ServingPlanLease(
            w, inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options, return_lse=True
        )
        assert lease.current(inputs) and lease.runner._eligible
        assert lease.prepared._qo_indptr_buf is not w._qo_indptr_buf
        assert lease.prepared._qo_indptr_buf._version == 0
        cap = lease.runner._resource(inputs, **lease.run_options)
        after = method(*args, **options)
        for a, b, c in zip(native, cap, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, c, rtol=0, atol=0)
        assert w._plan_info is original_plan and w._cached_module is original_module
        assert lease.runner._current(inputs)
        # The owner changes the CPU planning epoch even when all15 plan values
        # remain equal. No expired resource path may be invoked.
        w._plan_info = list(original_plan)

        def forbidden(*args, **kwargs):
            raise AssertionError("Expired lease called a resource runner")

        lease.runner.run = forbidden
        fallback = lease.run(inputs, forward_options=options)
        assert lease.invalidated
        for actual, expected in zip(fallback, native):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Untracked metadata writes require explicit release by the exclusive
        # owner. This contract is recorded, never inferred from tensor counters.
        lease.invalidate()
        w._qo_indptr_buf.copy_(w._qo_indptr_buf.clone())
        assert not lease.current(inputs)
    evidence = lease.evidence()
    assert evidence["exclusive_planning_owner_required"]
    assert evidence["source_epoch_invalidated"]
    assert not evidence["dynamic_graph_supported"] and not evidence["serving_qualified"]
