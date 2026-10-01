"""Real packed/tuple GPU routing composition; not Resource HTTP qualification.

A minimal metadata driver executes the actual Native plan. The adapter, frozen
registry, real public lease/managed cache and native unpacker compose unchanged.
This driver is explicitly not the production SGLang backend; its Native full
model hook has separate evidence. Keep inconclusive decisions without retries.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flashinfer import MeasurementPolicy, autotune_v2, prefill
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import _unpack_paged_kv_cache

from research.selector_v4.serving.epoch_registry import ServingEpochRegistry
from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter
from research.selector_v4.serving.prefix_router import PagedPrefixRouter
from research.selector_v4.serving.reusable_plan_lease import ReusableServingPlanLease
from research.selector_v4.serving.test_reusable_plan_lease_gpu import KV, Q, assert_exact

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("packed", [True, False])
def test_real_packed_prefix_router_and_announced_epochs(packed, tmp_path, monkeypatch):
    root = Path(os.environ.get("SGI_PREFIX_ROUTER_EVIDENCE", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    label = "packed" if packed else "tuple"
    torch.manual_seed(20261001)
    evidence = {
        "epochs": [],
        "scope": "REAL_GPU_MINIMAL_METADATA_DRIVER_COMPOSITION",
        "full_http_resource_qualified": False,
        "qualification_authority": False,
    }
    with torch.inference_mode():
        q = torch.randn(sum(Q), 32, 128, dtype=torch.bfloat16, device="cuda")
        storage = torch.randn(sum(KV), 2, 8, 128, dtype=q.dtype, device="cuda")
        cache = storage if packed else tuple(storage.unbind(dim=1))
        k, v = _unpack_paged_kv_cache(cache, "NHD")
        inputs = [q, k, v]
        signatures = [q.neg(), v, k]
        ip = lambda lengths: torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )
        metadata = [
            ip(Q),
            ip(KV),
            torch.arange(sum(KV), dtype=torch.int32, device="cuda"),
            torch.ones(len(Q), dtype=torch.int32, device="cuda"),
        ]
        owner = prefill.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 << 20, dtype=torch.uint8, device="cuda"), backend="fa2"
        )
        options = {"causal": False, "sm_scale": 128**-0.5, "logits_soft_cap": 0.0}

        def plan():
            owner.plan(
                *metadata, 32, 8, 128, 1, causal=False, q_data_type=q.dtype, disable_split_kv=True
            )

        plan()
        native = lambda: owner.forward_return_lse(q, cache, **options)
        lease = ReusableServingPlanLease(
            owner, inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options, return_lse=True
        )
        receipt = lease.calibrate(inputs)
        lease.runner.tuning_config.profiling_repeat = 256
        managed_cache = root / (label + "-managed-cache")
        with autotune_v2(
            mode="tune",
            measurement_policy=MeasurementPolicy(execution_mode="eager"),
            cache_root=managed_cache,
        ):
            assert_exact(lease.run(inputs, forward_options=options), native())
            _, tactic = AutoTuner.get().choose_one(
                "experimental_prefill_resource", [lease.runner], lease.runner.tuning_config, inputs
            )
        # Keep the complete first certificate and actual winner before any
        # witness assertion, including failed diagnostic attempts.
        (root / (label + "-training.json")).write_text(
            json.dumps({"certificate": receipt, "actual_managed_tactic": tactic}, indent=2) + "\n"
        )
        registry = ServingEpochRegistry(owner)
        registry.register(label, lease, actual_managed_tactic=tactic)
        registry.freeze()
        # All calls below invoke actual Native planning/GPU metadata. This small
        # driver supplies host scheduler lengths, not fake tensor signatures.
        driver = SimpleNamespace(
            prefill_wrappers_paged=[owner],
            forward_metadata=SimpleNamespace(
                use_ragged=True, extend_no_prefix=False, prefill_wrappers=[owner]
            ),
            init_forward_metadata=lambda batch: plan(),
            init_forward_metadata_out_graph=lambda batch: plan(),
            init_cuda_graph_state=lambda: None,
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            extend_seq_lens_cpu=Q,
            extend_prefix_lens_cpu=KV,
        )
        adapter = MetadataEpochAdapter(driver)
        adapter.attach(
            registry, label, signatures, qo_lengths=Q, kv_lengths=KV, forward_options=options
        )
        adapter.install()
        router = PagedPrefixRouter(
            adapter,
            graph_or_tracing=lambda: (
                torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
            ),
        )
        count = [0]
        resource = lease.runner._resource

        def witness(*args, **kwargs):
            count[0] += 1
            return resource(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Cold preparation/source audit or late metadata bind during layers"
            )

        from flashinfer.experimental.prefill_resource import _jit, _ops

        monkeypatch.setattr(lease.runner, "_resource", witness)
        monkeypatch.setattr(prefill, "make_prefill_resource_runner", forbidden)
        monkeypatch.setattr(_jit, "source_contract", forbidden)
        monkeypatch.setattr(_ops, "resource_module", forbidden)
        for epoch in range(3):
            # Explicit foreign-write announcement precedes both index writes and
            # the driver's native plan. Both announcements are conservative and
            # release each bound lease before the first write.
            adapter.before_update(False)
            metadata[2].copy_(metadata[2].roll(epoch + 1))
            storage.mul_(0.96875)
            q.mul_(0.984375)
            driver.init_forward_metadata(batch)
            reference = tuple(t.clone() for t in native())
            before = count[0]
            hits = []
            with monkeypatch.context() as patch:
                patch.setattr(registry, "bind_after_metadata_update", forbidden)
                for _ in range(36):
                    if tactic == 1:
                        tuner = AutoTuner.get()
                        config = lease.runner.tuning_config
                        policy = tuner._effective_measure_policy
                        if policy is not None:
                            config = tuner._apply_measure_policy(config, policy)
                        hit, index, cached, _ = tuner.search_cache(
                            "experimental_prefill_resource",
                            [lease.runner],
                            tuple(tuner._get_input_sizes(inputs)),
                            config,
                            inputs=inputs,
                        )
                        assert hit and index == 0 and cached == 1
                        hits.append(cached)
                    assert_exact(router.run(owner, q, cache, **options), reference)
            expected = 36 if tactic == 1 else 0
            assert count[0] - before == expected
            evidence["epochs"].append(
                {
                    "epoch": epoch,
                    "actual_resource_calls": count[0] - before,
                    "actual_managed_cache_hits": hits,
                    "full_output_lse_exact": True,
                    "late_metadata_binding_forbidden": True,
                }
            )
        # Real capture/tracing routes directly Native. This is still a Native
        # Graph witness; it cannot authorize Resource graph metadata updates.
        driver.init_forward_metadata_out_graph(batch)
        before = count[0]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = router.run(owner, q, cache, **options)
        storage.mul_(0.984375)
        q.mul_(0.96875)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(captured, native())
        assert count[0] == before and adapter.current_prefix_lengths is None
        adapter.restore()
        assert not registry.bound_keys
        evidence.update(
            actual_managed_tactic=tactic,
            certificate=receipt,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            router=router.evidence(),
            native_graph_payload_update_exact=True,
            native_unpack_used=True,
            actual_input_shapes=[list(t.shape) for t in inputs],
            actual_input_strides=[list(t.stride()) for t in inputs],
        )
        (root / (label + ".json")).write_text(json.dumps(evidence, indent=2) + "\n")
