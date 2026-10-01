"""Exposed functional coverage of real serving storage; no selection authority."""

import pytest

torch = pytest.importorskip("torch")
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    make_prefill_resource_runner,
)
from research.selector_v4.serving.geometry_server import unpack_paged_payload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
Q = [3, 39, 107, 175, 243, 311, 411]
KV = [q + n for q, n in zip(Q, [22528, 16512, 11840, 8256, 2624, 672, 80])]


def indptr(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["projection_views", "implicit_page1_tuple", "implicit_page1_packed"]
)
def test_serving_views_exact_and_graph_payload_update(dtype, layout):
    torch.manual_seed(42)
    q = torch.randn(sum(Q), 32, 128, dtype=dtype, device="cuda")
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    if layout == "projection_views":
        # Actual Qwen HTTP ragged K/V signatures: [tokens,8,128], stride6144.
        projection = torch.randn(sum(KV), 48, 128, dtype=dtype, device="cuda")
        k, v = projection[:, 32:40], projection[:, 40:48]
        assert k.stride() == v.stride() == (6144, 128, 1)
        assert not k.is_contiguous() and not v.is_contiguous()
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
        wrapper.plan(
            indptr(Q), indptr(KV), 32, 8, 128, causal=True, q_data_type=dtype, disable_split_kv=True
        )
        native = lambda: wrapper.run(q, k, v, return_lse=True)
        contiguous = lambda: wrapper.run(q, k.contiguous(), v.contiguous(), return_lse=True)
    else:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")
        wrapper.plan(
            indptr(Q),
            indptr(KV),
            torch.arange(sum(KV), dtype=torch.int32, device="cuda"),
            torch.ones(len(Q), dtype=torch.int32, device="cuda"),
            32,
            8,
            128,
            1,
            causal=False,
            q_data_type=dtype,
            disable_split_kv=True,
        )
        if layout == "implicit_page1_tuple":
            payload = tuple(
                torch.randn(sum(KV), 8, 128, dtype=dtype, device="cuda") for _ in range(2)
            )
        else:
            payload = torch.randn(sum(KV), 2, 8, 128, dtype=dtype, device="cuda")
        k, v, page = unpack_paged_payload(payload, "NHD")
        assert page == 1 and k.shape == v.shape == (sum(KV), 1, 8, 128)
        native = lambda: wrapper.run(q, payload, return_lse=True)
        contiguous = lambda: wrapper.run(q, (k.contiguous(), v.contiguous()), return_lse=True)
    inputs = [q, k, v]
    module, plan = wrapper._cached_module, wrapper._plan_info
    before = tuple(t.clone() for t in native())
    runner = make_prefill_resource_runner(wrapper, inputs, qo_lengths=Q, kv_lengths=KV)
    assert runner._eligible and runner._prepare(inputs), getattr(runner, "_failure_reason", None)
    raw = lambda: runner._resource(inputs, return_lse=True)
    for result in (raw(), native(), contiguous()):
        for actual, expected in zip(result, before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert wrapper._cached_module is module and wrapper._plan_info is plan
    assert runner.receipt is None
    # Fixed binding Graph only. No metadata update or eager certificate reuse.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = raw()
    q.mul_(0.75)
    k.add_(0.125)
    v.sub_(0.25)
    expected = native()
    graph.replay()
    for actual, reference in zip(captured, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Even for these real storage layouts, an uncertified tactic1 stays native.
    fallback = runner.forward(inputs, tactic=1, return_lse=True)
    for actual, reference in zip(fallback, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert runner.receipt is None and runner._current(inputs)
