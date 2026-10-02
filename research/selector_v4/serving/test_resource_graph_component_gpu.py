"""Actual graph-mode Native owner -> separate fixed unsplit Resource capture.

Component test only: not a full model, HTTP result or serving qualification.
The original native split plan and graph-mode buffers remain untouched.
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real CUDA required"
)

from research.selector_v4.serving.resource_graph_contract import (
    ReplayTicket,
    resource_trace,
)
from research.selector_v4.serving.resource_graph_server import LayerEntry, tensor_digest


def test_actual_native_split_graph_owner_and_physical_page_epochs(tmp_path):
    from flashinfer._build_meta import __git_commit__
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    assert __git_commit__ == "75544a17ce0019ca877f50354d95451ee089f859"
    out = Path(
        os.environ.get("SGI_RESOURCE_GRAPH_COMPONENT_OUT", str(tmp_path / "component"))
    )
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261002)
    with torch.inference_mode():
        q = torch.randn(128, 12, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(384, 1, 2, 128, dtype=q.dtype, device=q.device)
        v = torch.randn_like(k)
        qptr = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
        kptr = qptr.clone()
        pages = torch.arange(128, dtype=torch.int32, device="cuda")
        last = torch.ones(1, dtype=torch.int32, device="cuda")
        owner = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
            "NHD",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=qptr.clone(),
            paged_kv_indptr_buf=kptr.clone(),
            paged_kv_indices_buf=torch.zeros(512, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buf=last.clone(),
        )

        def plan():
            owner.plan(
                qptr,
                kptr,
                pages,
                last,
                12,
                2,
                128,
                1,
                causal=True,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                sm_scale=128**-0.5,
            )

        plan()
        assert owner._plan_info[13] and owner._plan_info[14]
        options = {
            "causal": True,
            "sm_scale": 128**-0.5,
            "window_left": -1,
            "logits_soft_cap": 0.0,
            "k_scale": None,
            "v_scale": None,
        }
        entry = LayerEntry(owner, q, (k, v), options, 128)
        assert not entry.wrapper._plan_info[13] and not entry.wrapper._plan_info[14]
        for _ in range(3):
            entry.run(q, (k, v), options)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = entry.run(q, (k, v), options)
        identity = id(graph), entry.frame(include_payload=False)
        records, ticket = [], ReplayTicket()
        for epoch in range(3):
            ticket.begin()
            pages.copy_(
                torch.arange(
                    epoch * 128, (epoch + 1) * 128, dtype=torch.int32, device="cuda"
                )
            )
            q.add_(0.015625)
            k.mul_(0.984375)
            v.add_(0.03125)
            plan()
            assert owner._plan_info[13] and owner._plan_info[14]
            expected = tuple(
                t.clone() for t in owner.forward_return_lse(q, (k, v), **options)
            )
            entry.refresh()
            frame = (entry.frame(), entry.live_owner_frame())
            ticket.finish(frame, eligible=True)
            assert ticket.consume((entry.frame(), entry.live_owner_frame()))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                with torch.profiler.record_function("sgi_resource_full_model_graph"):
                    graph.replay()
                torch.cuda.synchronize()
            trace = out / f"epoch-{epoch}.json"
            prof.export_chrome_trace(str(trace))
            proof = resource_trace(json.loads(trace.read_text())["traceEvents"])
            assert proof["actual_resource_graph_kernels"] == 1
            assert all(
                torch.equal(a, b) for a, b in zip(expected, entry.resource_result)
            )
            assert torch.equal(result, expected[0])
            assert (id(graph), entry.frame(include_payload=False)) == identity
            records.append(
                {
                    "epoch": epoch,
                    "profile": proof,
                    "payload": entry.verify_output(),
                    "original_native_split_kv": True,
                    "output_sha256": tensor_digest(result),
                }
            )
            (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        assert len({r["payload"]["page_indices_sha256"] for r in records}) == 3
        assert len({r["payload"]["q_sha256"] for r in records}) == 3
        assert len({r["output_sha256"] for r in records}) == 3
        ticket.begin()
        ticket.finish((entry.frame(), entry.live_owner_frame()), eligible=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as negative:
            entry.indices[0] = (entry.indices[0] + 1) % 384
            with pytest.raises(RuntimeError, match="Stale"):
                if ticket.consume((entry.frame(), entry.live_owner_frame())):
                    graph.replay()
            torch.cuda.synchronize()
        negative.export_chrome_trace(str(out / "stale-rejection.json"))
        events = json.loads((out / "stale-rejection.json").read_text())["traceEvents"]
        assert not any(
            e.get("cat") == "kernel" and "ResourceKernel" in e.get("name", "")
            for e in events
        )
        (out / "complete.json").write_text(
            json.dumps(
                {
                    "pass": True,
                    "epochs": records,
                    "actual_physical_stale_update_rejected": True,
                    "resource_kernels_after_stale": 0,
                    "scope": "GPU_COMPONENT_ONLY_NOT_SGLANG_HTTP",
                    "resource_graph_serving_qualified": False,
                    "timing_valid": False,
                    "default_promotion": False,
                    "serving_promotion": False,
                },
                indent=2,
            )
            + "\n"
        )
