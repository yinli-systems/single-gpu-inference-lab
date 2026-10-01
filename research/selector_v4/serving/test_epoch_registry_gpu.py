"""Real public early binding and 36-layer reuse on already exposed geometries."""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from flashinfer import MeasurementPolicy, autotune_v2, prefill
from flashinfer.autotuner import AutoTuner

from research.selector_v4.serving.epoch_registry import ServingEpochRegistry
from research.selector_v4.serving.reusable_plan_lease import ReusableServingPlanLease
from research.selector_v4.serving.test_reusable_plan_lease_gpu import KV, Q, assert_exact, digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("paged", [False, True])
def test_actual_early_bind_and_layer_reuse(paged, tmp_path, monkeypatch):
    root = Path(os.environ.get("SGI_REUSABLE_LEASE_EVIDENCE", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    name = "paged" if paged else "ragged"
    evidence = {"epochs": [], "full_http_qualified": False, "qualification_authority": False}
    torch.manual_seed(20261001)

    def ip(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )

    options = {"causal": not paged, "sm_scale": 128**-0.5, "logits_soft_cap": 0.0}
    with torch.inference_mode():
        q = torch.randn(sum(Q), 32, 128, dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        metadata = [ip(Q), ip(KV)]
        if paged:
            k = torch.randn(sum(KV), 1, 8, 128, dtype=q.dtype, device="cuda")
            v = torch.randn_like(k)
            metadata += [
                torch.arange(sum(KV), device="cuda", dtype=torch.int32),
                torch.ones(len(Q), device="cuda", dtype=torch.int32),
            ]
            owner = prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")

            def plan():
                owner.plan(
                    *metadata,
                    32,
                    8,
                    128,
                    1,
                    causal=False,
                    q_data_type=q.dtype,
                    disable_split_kv=True,
                )

            signatures = [q.neg(), v, k]
            native_args = lambda: (q, (k, v))
        else:
            projected = torch.randn(sum(KV), 48, 128, dtype=q.dtype, device="cuda")
            k, v = projected[:, 32:40, :], projected[:, 40:48, :]
            owner = prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")

            def plan():
                owner.plan(
                    *metadata, 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True
                )

            signatures = [q.neg(), projected[:, :8, :], projected[:, 8:16, :]]
            native_args = lambda: (q, k, v)
        inputs = [q, k, v]
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(signatures, inputs, strict=True))
        assert all(
            a.shape == b.shape and a.stride() == b.stride()
            for a, b in zip(signatures, inputs, strict=True)
        )
        plan()
        native = lambda: owner.forward_return_lse(*native_args(), **options)
        value = ReusableServingPlanLease(
            owner, inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options, return_lse=True
        )
        receipt = value.calibrate(inputs)
        value.runner.tuning_config.profiling_repeat = 256
        cache = root / (name + "-managed-cache")
        with autotune_v2(
            mode="tune",
            measurement_policy=MeasurementPolicy(execution_mode="eager"),
            cache_root=cache,
        ):
            assert_exact(value.run(inputs, forward_options=options), native())
            _, tactic = AutoTuner.get().choose_one(
                "experimental_prefill_resource", [value.runner], value.runner.tuning_config, inputs
            )
        registry = ServingEpochRegistry(owner)
        registry.register(name, value, actual_managed_tactic=tactic)
        registry.freeze()
        identity, checksum = value.runner.identity, receipt["checksum"]
        count = 0
        original_resource = value.runner._resource

        def resource(*args, **kwargs):
            nonlocal count
            count += 1
            return original_resource(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("Cold construction, source audit or compile in serving epoch")

        from flashinfer.experimental.prefill_resource import _jit, _ops

        with monkeypatch.context() as patch:
            patch.setattr(value.runner, "_resource", resource)
            patch.setattr(prefill, "make_prefill_resource_runner", forbidden)
            patch.setattr(_jit, "source_contract", forbidden)
            patch.setattr(_ops, "resource_module", forbidden)
            for epoch in range(3):
                registry.before_metadata_update()
                q.mul_(0.984375)
                k.add_(0.015625)
                v.mul_(0.96875)
                if paged:
                    metadata[2].copy_(metadata[2].roll(epoch + 1))
                plan()
                expected = tuple(t.clone() for t in native())
                before = count
                rebound = registry.bind_after_metadata_update(
                    name, signatures, qo_lengths=Q, kv_lengths=KV, forward_options=options
                )
                assert count == before, "Early binding must not execute attention"
                assert rebound == (tactic == 1)
                hits = []
                for _ in range(36):
                    if rebound:
                        tuner = AutoTuner.get()
                        config = value.runner.tuning_config
                        policy = tuner._effective_measure_policy
                        if policy is not None:
                            config = tuner._apply_measure_policy(config, policy)
                        hit, index, cached, _ = tuner.search_cache(
                            "experimental_prefill_resource",
                            [value.runner],
                            tuple(tuner._get_input_sizes(inputs)),
                            config,
                            inputs=inputs,
                        )
                        assert hit and index == 0 and cached == tactic
                        hits.append({"hit": hit, "tactic": cached})
                    actual = registry.run(
                        name,
                        inputs,
                        qo_lengths=Q,
                        kv_lengths=KV,
                        forward_options=options,
                        native_call=native,
                    )
                    assert_exact(actual, expected)
                evidence["epochs"].append(
                    {
                        "epoch": epoch + 1,
                        "early_binding": rebound,
                        "actual_cache_hits": hits,
                        "actual_resource_invocations": count - before,
                        "native_output_lse_sha256": [digest(t) for t in expected],
                        "36_layer_outputs_lse_exact": True,
                    }
                )
                (root / (name + ".json")).write_text(json.dumps(evidence, indent=2) + "\n")
                assert count - before == (36 if rebound else 0)
                assert (value.runner.identity, value.runner.receipt["checksum"]) == (
                    identity,
                    checksum,
                )
            assert registry.counters["public_rebinds"] == (3 if tactic == 1 else 0)
            before = count
            assert_exact(
                registry.run(
                    "unseen",
                    inputs,
                    qo_lengths=Q,
                    kv_lengths=KV,
                    forward_options=options,
                    native_call=native,
                ),
                native(),
            )
            assert_exact(
                registry.run(
                    name,
                    inputs,
                    qo_lengths=Q,
                    kv_lengths=KV,
                    forward_options=options,
                    native_call=native,
                    graph_or_tracing=True,
                ),
                native(),
            )
            assert count == before
        evidence.update(
            actual_managed_tactic=tactic,
            certificate=receipt,
            registry=registry.evidence(),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            signature_inputs_are_real_tensors=True,
            signature_and_run_storage_differ=True,
            graph_and_unseen_geometry_native=True,
        )
        (root / (name + ".json")).write_text(json.dumps(evidence, indent=2) + "\n")
        assert list(cache.glob("v2/*/entries/*.json"))
