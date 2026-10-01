"""Exposed-only functional checks; no qualification or timing claim."""

from pathlib import Path
import sys
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "selector_v32"))
import measure as base
from research.selector_v4.exposed_diagnostics.native_context import CELLS
from research.selector_v4.exposed_diagnostics.shared_runtime import SharedResourceRuntime


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["ragged", "paged"])
def test_replanned_private_resource_shares_native_storage(dtype, layout):
    import flashinfer

    torch.manual_seed(74001)
    cell = CELLS[0]
    qs = cell["q"]
    lengths = [q + c for q, c in zip(qs, cell["cached"], strict=True)]
    qi = torch.tensor([0, *torch.tensor(qs).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(qs), 32, 128, device="cuda", dtype=dtype)
    bundle = None
    if layout == "ragged":
        k = torch.randn(sum(lengths), 8, 128, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        inputs = [q, k, v]
    else:
        page = 16
        pages = [(n + page - 1) // page for n in lengths]
        pp = torch.tensor(
            [0, *torch.tensor(pages).cumsum(0).tolist()], device="cuda", dtype=torch.int32
        )
        pi = torch.randperm(sum(pages), device="cuda").to(torch.int32)
        last = torch.tensor([(n - 1) % page + 1 for n in lengths], device="cuda", dtype=torch.int32)
        kp = torch.randn(sum(pages), page, 8, 128, device="cuda", dtype=dtype)
        vp = torch.randn_like(kp)
        k, v = kp.flatten(0, 1), vp.flatten(0, 1)
        bundle = (page, pp, pi, last, kp, vp)
        inputs = [q, kp, vp]
    native = base.Runtime(
        flashinfer,
        torch,
        layout,
        qi,
        None,
        None,
        k,
        v,
        lengths,
        dtype,
        "unsplit",
        "off",
        False,
        bundle,
    )
    native.attach(q)
    native.plan()
    native.call(q)
    torch.cuda.synchronize()
    reference = native.out.clone(), native.lse.clone()
    original_module = native.wrapper._cached_module
    shared = SharedResourceRuntime(native, inputs, qo_lengths=qs, kv_lengths=lengths)
    assert shared.ws is native.ws and shared.out is native.out and shared.lse is native.lse
    for _ in range(3):
        previous = native.wrapper._plan_info
        shared.plan()
        assert native.wrapper._plan_info is not previous
        assert shared.proxy._plan_info is native.wrapper._plan_info
        shared.call(q)
        torch.cuda.synchronize()
        torch.testing.assert_close(shared.out, reference[0], atol=0, rtol=0)
        torch.testing.assert_close(shared.lse, reference[1], atol=0, rtol=0)
        assert native.wrapper._cached_module is original_module
        native.plan()
        native.call(q)
        torch.cuda.synchronize()
        torch.testing.assert_close(native.out, reference[0], atol=0, rtol=0)
        torch.testing.assert_close(native.lse, reference[1], atol=0, rtol=0)
    for count in (1, 16):
        shared.capture(q, count)
        shared.graphs[count].replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(shared.out, reference[0], atol=0, rtol=0)
        torch.testing.assert_close(shared.lse, reference[1], atol=0, rtol=0)
    # A same-valued input metadata write invalidates fixed ownership; fail before launch.
    native.wrapper._qo_indptr_buf.add_(0)
    with pytest.raises(RuntimeError, match="metadata/module/configuration changed"):
        shared.plan()
    assert native.wrapper._cached_module is original_module
