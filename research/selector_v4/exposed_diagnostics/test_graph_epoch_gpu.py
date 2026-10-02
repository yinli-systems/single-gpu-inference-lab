"""Real fixed-capture metadata epochs on exposed geometry; never qualification."""
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

from research.selector_v4.exposed_diagnostics.graph_epoch import GraphEpochProbe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
Q = [3, 39, 107, 175, 243, 311, 411]
KV = [22528, 16512, 11840, 8256, 2624, 672, 80]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def exact(actual, expected):
    torch.cuda.synchronize()
    assert len(actual) == len(expected) == 2
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


class ForbiddenGraph:
    def replay(self):
        raise AssertionError("Retired Resource Graph was replayed")


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("replays", [1, 16])
def test_physical_page_epochs_preserve_one_capture(dtype_name, layout, packed, replays, tmp_path):
    from flashinfer._build_meta import __git_commit__

    assert __git_commit__ == "75544a17ce0019ca877f50354d95451ee089f859"
    root = Path(os.environ.get("SGI_GRAPH_EPOCH_EVIDENCE", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    name = f'{dtype_name}-{layout}-{"packed" if packed else "tuple"}-graph{replays}'
    cell = root / name
    cell.mkdir()
    record = {"cell": name, "epochs": [], "graph_replays_per_call": replays, "diagnostic_only": True,
              "qualification_authority": False, "default_promotion": False,
              "serving_promotion": False, "historical_token_divergence_resolved": False}

    def save():
        (cell / "result.json").write_text(json.dumps(record, indent=2)+"\n")

    torch.manual_seed(20261002)
    dtype = getattr(torch, dtype_name)
    ip = lambda lengths: torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    options = {"causal": False, "sm_scale": 128**-0.5, "logits_soft_cap": 0.0}
    with torch.inference_mode():
        q = torch.randn(sum(Q), 32, 128, dtype=dtype, device="cuda")
        shape = (sum(KV), 1, 8, 128) if layout == "NHD" else (sum(KV), 8, 1, 128)
        if packed:
            payload = torch.randn(shape[0], 2, *shape[1:], dtype=dtype, device="cuda")
            k, v = payload.unbind(1)
            cache = payload
        else:
            k = torch.randn(*shape, dtype=dtype, device="cuda")
            v = torch.randn_like(k)
            cache = (k, v)
        inputs = [q, k, v]
        metadata = [ip(Q), ip(KV), torch.arange(sum(KV), dtype=torch.int32, device="cuda"),
                    torch.ones(len(Q), dtype=torch.int32, device="cuda")]
        owner = BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
                                                  kv_layout=layout, backend="fa2")

        def plan():
            owner.plan(*metadata, 32, 8, 128, 1, causal=False, q_data_type=dtype, disable_split_kv=True)

        def native(current=inputs, current_options=options):
            actual_cache = cache if current is inputs else (current[1], current[2])
            return tuple(x.clone() for x in owner.forward_return_lse(current[0], actual_cache, **current_options))

        plan()
        reference = native()
        probe = GraphEpochProbe(owner, inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options,
                                replays=replays, allow_uncertified_diagnostic=True)
        graph_id = id(probe._graph)
        captured_ptrs = probe.buffer_binding(probe.owned)
        record.update(captured_graph_id=graph_id, owned_buffer_binding=captured_ptrs,
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                      gpu_name=torch.cuda.get_device_properties(0).name,
                      input_binding=probe.bind_inputs(inputs))
        save()
        exact(probe.replay(inputs, forward_options=options), reference)
        previous_digest = digest(reference[0])
        for epoch in range(3):
            start = time.perf_counter_ns()
            assert probe.before_metadata_update(inputs)
            with pytest.raises(RuntimeError):
                probe.replay(inputs, forward_options=options)
            old_pages = digest(metadata[2])
            metadata[2].copy_(metadata[2].roll(epoch+1))
            # Payload and physical page ownership both actually change.
            q.add_(0.015625)
            k.mul_(0.984375)
            v.add_(0.03125)
            plan()
            # Establish the current epoch's control BEFORE any Resource replay,
            # so a Native-after-Resource drift cannot corrupt its own reference.
            expected = native()
            assert probe.bind_after_metadata_update(inputs, forward_options=options)
            for _ in range(4):
                result = probe.replay(inputs, forward_options=options)
                assert probe.last_execution == "uncertified_resource_graph"
            torch.cuda.synchronize()
            full_step_ms = (time.perf_counter_ns()-start)/1e6
            exact(result, expected)
            # Native-after-Resource is evaluated against the current plan.
            exact(native(), expected)
            assert digest(metadata[2]) != old_pages
            assert digest(expected[0]) != previous_digest
            previous_digest = digest(expected[0])
            assert id(probe._graph) == graph_id and probe.buffer_binding(probe.owned) == captured_ptrs
            # Every epoch has an independent retained actual-launch trace.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(4):
                    probe.replay(inputs, forward_options=options)
                torch.cuda.synchronize()
            trace_path = cell / f'epoch{epoch+1}-trace.json'
            prof.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel'
                       and 'BatchPrefillWithPagedKV' in e.get('name', '')]
            row = {"epoch": epoch+1, "full_transaction_with_native_reference_plus_four_calls_wall_ms": full_step_ms,
                   "timing_scope": "Metadata value readback, synchronization, copies, Native plan AND pre-Resource Native control, payload updates and four graph calls; diagnostic only, not deployment timing",
                   "physical_pages_sha256": digest(metadata[2]), "output_lse_sha256": [digest(x) for x in expected],
                   "output_lse_exact": True, "native_after_resource_exact": True,
                   "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                   "attention_launches": len(kernels),
                   "actual_shared_memory_bytes": [e['args']['shared memory'] for e in kernels]}
            record["epochs"].append(row)
            save()
            assert len(kernels) == 4*replays and all(e['args']['shared memory'] == 65536 for e in kernels)
        # Mutate inference metadata without notification: GPU readback detects
        # values even though version counters do not exist.
        metadata[2].copy_(metadata[2].roll(1))
        probe._graph = ForbiddenGraph()
        exact(probe.replay(inputs, forward_options=options), native())
        assert probe.last_execution == "native_eager" and probe.guard.state == "RETIRED"
        assert not probe.bind_after_metadata_update(inputs, forward_options=options)
        # A new query partition preserves total Q storage but changes ordered
        # geometry. It must not inherit this fixed-geometry captured graph.
        geometry_probe = GraphEpochProbe(owner, inputs, qo_lengths=Q, kv_lengths=KV,
                                        forward_options=options, replays=replays,
                                        allow_uncertified_diagnostic=True)
        assert geometry_probe.before_metadata_update(inputs)
        new_q = list(Q)
        new_q[0] += 1
        new_q[1] -= 1
        metadata[0].copy_(ip(new_q))
        plan()
        assert not geometry_probe.bind_after_metadata_update(inputs, forward_options=options)
        geometry_probe._graph = ForbiddenGraph()
        exact(geometry_probe.replay(inputs, forward_options=options), native())
        assert geometry_probe.guard.state == "RETIRED"
        record.update(probe=probe.evidence(), stale_graph_never_replayed=True,
                      unchanged_capture_across_three_metadata_epochs=True,
                      unannounced_inference_tensor_write_detected=True,
                      changed_ordered_geometry_native=True,
                      managed_certificate_reused=False, pass_functional=True)
        save()
