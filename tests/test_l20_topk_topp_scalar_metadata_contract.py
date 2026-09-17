"""The guarded top-k/top-p hook must not synchronize the host by accident.

``maybe_l20_topk_topp_sample`` used to fall back to ``torch.all(...)`` and
``.item()`` on the per-row ``k``/``p`` tensors whenever the caller did not
pre-resolve scalar metadata. On a CUDA tensor that is a device-to-host sync in
the middle of the sampler and is not CUDA Graph capturable. The contract is now
explicit: pre-resolved scalars are the fast path, CPU tensors may be inspected,
and CUDA tensors are refused unless the caller opts into ``allow_host_sync``.
"""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_hook():
    path = Path("integrations/vllm/l20_topk_topp_sampling.py")
    spec = importlib.util.spec_from_file_location("l20_topk_topp_sampling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _NoSync:
    """Fail the test if anything calls ``Tensor.item``/``bool`` on a CUDA tensor."""

    def __enter__(self):
        self._item = torch.Tensor.item
        self._bool = torch.Tensor.__bool__

        def guarded_item(tensor):
            if tensor.is_cuda:
                raise AssertionError("host sync via .item() on a CUDA tensor")
            return self._item(tensor)

        def guarded_bool(tensor):
            if tensor.is_cuda:
                raise AssertionError("host sync via bool() on a CUDA tensor")
            return self._bool(tensor)

        torch.Tensor.item = guarded_item
        torch.Tensor.__bool__ = guarded_bool
        return self

    def __exit__(self, *exc):
        torch.Tensor.item = self._item
        torch.Tensor.__bool__ = self._bool
        return False


def test_scalar_from_tensor_refuses_cuda_without_opt_in():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    hook = _load_hook()
    k = torch.full((4,), 40, device="cuda", dtype=torch.int32)
    with _NoSync(), pytest.raises(ValueError, match="unresolved_top_k_requires_host_sync"):
        hook._scalar_from_tensor(k, name="top_k")
    assert hook._scalar_from_tensor(k, name="top_k", allow_host_sync=True) == 40


def test_scalar_from_tensor_reads_cpu_tensors_without_opt_in():
    hook = _load_hook()
    assert hook._scalar_from_tensor(torch.tensor([0.9, 0.9]), name="top_p") == pytest.approx(0.9)
    with pytest.raises(ValueError, match="mixed_top_p"):
        hook._scalar_from_tensor(torch.tensor([0.9, 0.8]), name="top_p")
    assert hook._scalar_from_tensor(None, name="top_p") is None
    assert hook._scalar_from_tensor(torch.empty(0), name="top_p") is None


def test_hook_traces_unresolved_metadata_as_ineligible(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    hook = _load_hook()
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv(hook.ENABLE_ENV, "1")
    monkeypatch.setenv(hook.ALLOW_NON_L20_ENV, "1")
    monkeypatch.setenv(hook.TRACE_ENV, str(trace))
    batch, vocab = 4, 151_936
    logits = torch.randn((batch, vocab), device="cuda")
    k = torch.full((batch,), 40, device="cuda", dtype=torch.int32)
    p = torch.full((batch,), 0.9, device="cuda", dtype=torch.float32)
    rng_state = {
        "expanded_idx_mapping": torch.arange(batch, device="cuda"),
        "seeds": torch.zeros(batch, device="cuda", dtype=torch.int64),
        "positions": torch.zeros(batch, device="cuda", dtype=torch.int64),
    }
    with _NoSync():
        out = hook.maybe_l20_topk_topp_sample(logits, k, p, None, **rng_state)
    assert out is None
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert events, "hook did not trace the ineligible request"
    last = events[-1]
    assert last["eligible"] is False
    assert "unresolved_top_k_requires_host_sync" in last["reasons"]
    assert last["metadata"]["scalar_metadata_source"] == "tensor"
