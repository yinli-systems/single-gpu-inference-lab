import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.selector_v4.serving.resource_graph_contract import (
    ReplayTicket,
    resource_trace,
    supported_batch,
)
from research.selector_v4.serving.resource_graph_server import install, transform


def batch():
    return SimpleNamespace(
        batch_size=1,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        extend_seq_lens_cpu=[128],
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=[128],
        extend_num_tokens=128,
        spec_info=None,
        mm_inputs=None,
        lora_ids=None,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 2),
        ("extend_seq_lens_cpu", [127]),
        ("extend_prefix_lens_cpu", [1]),
        ("seq_lens_cpu", [129]),
        ("extend_num_tokens", 127),
        ("spec_info", object()),
        ("mm_inputs", [object()]),
        ("lora_ids", ["adapter"]),
    ],
)
def test_padding_or_changed_geometry_cannot_enter_unsplit_resource_graph(field, value):
    b = batch()
    assert supported_batch(b, 128)
    setattr(b, field, value)
    assert not supported_batch(b, 128)


def test_stale_metadata_blocks_before_graph_launch_and_permanently_retires():
    ticket = ReplayTicket()
    ticket.begin()
    ticket.finish(("graph", "pages-a"), eligible=True)
    with pytest.raises(RuntimeError, match="Stale"):
        ticket.consume(("graph", "pages-b"))
    with pytest.raises(RuntimeError, match="retired"):
        ticket.begin()


def test_ticket_single_use_and_no_replay_while_loading():
    ticket = ReplayTicket()
    ticket.begin()
    ticket.finish((1, 2), eligible=True)
    assert ticket.consume((1, 2))
    with pytest.raises(RuntimeError):
        ticket.consume((1, 2))
    other = ReplayTicket()
    other.begin()
    with pytest.raises(RuntimeError):
        other.consume(None)


def test_unknown_geometry_native_fallback_cannot_get_resource_ticket():
    ticket = ReplayTicket()
    ticket.begin()
    ticket.finish(None, eligible=False)
    assert not ticket.consume(None)
    ticket.begin()
    ticket.finish("new-valid-epoch", eligible=True)
    assert ticket.consume("new-valid-epoch")


def trace():
    return [
        {
            "name": "sgi_resource_full_model_graph",
            "cat": "user_annotation",
            "ph": "X",
            "pid": 5,
            "tid": 8,
            "ts": 10,
            "dur": 20,
        },
        {
            "name": "cudaGraphLaunch",
            "cat": "cuda_runtime",
            "ph": "X",
            "pid": 5,
            "tid": 8,
            "ts": 15,
            "dur": 1,
            "args": {"correlation": 17},
        },
        {
            "name": "BatchPrefillWithPagedKVCacheResourceKernel",
            "cat": "kernel",
            "ts": 40,
            "args": {"correlation": 17, "graph id": 2, "shared memory": 65536},
        },
    ]


def test_actual_async_graph_correlation_not_cpu_or_gpu_annotation_count():
    events = trace()
    events.append(dict(events[0], cat="gpu_user_annotation", pid=0, tid=0, ts=40))
    assert resource_trace(events)["actual_resource_graph_kernels"] == 1


@pytest.mark.parametrize(
    "fault", ["eager", "memory", "native", "thread", "duplicate", "missing"]
)
def test_eager_resource_and_uncorrelated_profiles_rejected(fault):
    e = copy.deepcopy(trace())
    if fault == "eager":
        e[2]["args"]["graph id"] = 0
    elif fault == "memory":
        e[2]["args"]["shared memory"] = 32768
    elif fault == "native":
        e[2]["name"] = "NativeAttentionKernel"
    elif fault == "thread":
        e[1]["tid"] = 1
    elif fault == "duplicate":
        e.append(copy.deepcopy(e[0]))
    else:
        e.pop(1)
    with pytest.raises(ValueError):
        resource_trace(e)


def test_exact_sglang_ast_only_full_paged_call_is_routed():
    p = Path(
        os.environ.get(
            "SGI_REVIEWED_BACKEND_SOURCE",
            "/Users/kevin/Projects/sglang-http-ownership-current-20261001/python/sglang/srt/layers/attention/flashinfer_backend.py",
        )
    )
    if not p.exists():
        pytest.skip("Pinned SGLang local source absent")
    source = p.read_text()
    changed = transform(source)
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
    )
    original = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"
    )
    routed = [
        n
        for n in ast.walk(changed)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_sgi_full_graph_paged"
    ]
    assert len(routed) == 1
    call = routed[0]
    call.func = ast.Attribute(
        value=ast.Name(id="prefill_wrapper_paged", ctx=ast.Load()),
        attr="forward",
        ctx=ast.Load(),
    )
    call.args = call.args[3:]
    assert ast.dump(original) == ast.dump(changed.body[0])
    with pytest.raises(ValueError, match="Unreviewed"):
        transform(source + "\n")


def test_kernel_hold_prevents_import_allocation_or_hook_install(tmp_path, monkeypatch):
    from research.selector_v4.serving import http_pipeline

    calls = []

    def reject(root):
        calls.append(root)
        raise ValueError("Complete formal kernel qualification required")

    monkeypatch.setattr(http_pipeline, "authorize", reject)
    with pytest.raises(ValueError, match="Complete formal"):
        install(tmp_path / "must-not-exist", tmp_path / "kernel")
    assert len(calls) == 1
    assert not (tmp_path / "must-not-exist").exists()
