"""Actual certificates/v2 choices across explicit serving metadata transitions.

Only the already exposed geometry is used. Inconclusive certificates/winners
remain native and are recorded; no test manufactures a resource authorization.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from flashinfer import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from research.selector_v4.serving.reusable_plan_lease import ReusableServingPlanLease

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
Q = [3, 39, 107, 175, 243, 311, 411]
KV = [q + k for q, k in zip(Q, [22528, 16512, 11840, 8256, 2624, 672, 80])]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def assert_exact(actual, reference):
    torch.cuda.synchronize()
    assert len(actual) == len(reference) == 2
    assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))


@pytest.mark.parametrize("paged", [False, True])
def test_real_managed_decision_and_owned_metadata_rebind(paged, tmp_path, monkeypatch):
    torch.manual_seed(20261001)
    ip = lambda lengths: torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    options = {"causal": not paged, "sm_scale": 128**-0.5, "logits_soft_cap": 0.0}
    evidence = {"paged": paged, "epochs": [], "full_http_qualified": False}
    with torch.inference_mode():
        q = torch.randn(sum(Q), 32, 128, dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        if paged:
            k = torch.randn(sum(KV), 1, 8, 128, dtype=q.dtype, device="cuda")
            v = torch.randn_like(k)
            metadata = [
                ip(Q),
                ip(KV),
                torch.arange(sum(KV), device="cuda", dtype=torch.int32),
                torch.ones(len(Q), device="cuda", dtype=torch.int32),
            ]
            owner = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")
            plan = lambda: owner.plan(
                *metadata, 32, 8, 128, 1, causal=False, q_data_type=q.dtype, disable_split_kv=True
            )
        else:
            # Match actual SGLang fused-QKV projection slices without making
            # the KV inputs contiguous or altering their6144-element stride.
            projected = torch.randn(sum(KV), 48, 128, dtype=q.dtype, device="cuda")
            k, v = projected[:, 32:40, :], projected[:, 40:48, :]
            metadata = [ip(Q), ip(KV)]
            owner = BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
            plan = lambda: owner.plan(
                *metadata, 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True
            )
        plan()
        inputs = [q, k, v]
        native_args = lambda x: (x[0], (x[1], x[2])) if paged else tuple(x)
        native = lambda x: tuple(
            t.clone() for t in owner.forward_return_lse(*native_args(x), **options)
        )
        reference = native(inputs)
        value = ReusableServingPlanLease(
            owner, inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options, return_lse=True
        )
        assert value.runner._eligible
        receipt = value.calibrate(inputs)
        value.runner.tuning_config.profiling_repeat = 256
        policy = MeasurementPolicy(execution_mode="eager")
        cache = tmp_path / "managed-cache"
        with autotune_v2(mode="tune", measurement_policy=policy, cache_root=cache):
            result = value.run(inputs, forward_options=options)
            _, tactic = AutoTuner.get().choose_one(
                "experimental_prefill_resource", [value.runner], value.runner.tuning_config, inputs
            )
        assert_exact(result, reference)
        identity, checksum = value.runner.identity, receipt["checksum"]
        assert tactic != 1 or value.runner._receipt_valid
        evidence.update(
            identity=identity,
            certificate=receipt,
            actual_managed_tactic=tactic,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        )
        original_module = owner._cached_module
        original_runner = value.runner
        # These cold operations must never happen inside an epoch transition.
        from flashinfer import prefill
        from flashinfer.experimental.prefill_resource import _jit, _ops

        def forbidden(*args, **kwargs):
            raise AssertionError("Cold source audit/compile/constructor during rebind")

        with monkeypatch.context() as patch:
            patch.setattr(_jit, "source_contract", forbidden)
            patch.setattr(_ops, "resource_module", forbidden)
            patch.setattr(prefill, "make_prefill_resource_runner", forbidden)
            for epoch in range(3):
                value.release_before_update()
                q.mul_(0.984375)
                k.add_(0.015625)
                v.mul_(0.96875)
                if paged:
                    # Same logical lengths, different physical page ownership.
                    metadata[2].copy_(metadata[2].roll(epoch + 1))
                plan()
                expected = native(inputs)
                # The released lease always uses the current owner until bind.
                assert_exact(value.run(inputs, forward_options=options), expected)
                start = time.perf_counter_ns()
                rebound = value.bind_current_plan(
                    inputs, qo_lengths=Q, kv_lengths=KV, forward_options=options
                )
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                assert rebound == value.runner._receipt_valid
                result = value.run(inputs, forward_options=options)
                assert_exact(result, expected)
                assert owner._cached_module is original_module and value.runner is original_runner
                assert (value.runner.identity, value.runner.receipt["checksum"]) == (
                    identity,
                    checksum,
                )
                evidence["epochs"].append(
                    {
                        "generation": epoch + 1,
                        "public_rebind": rebound,
                        "transition_wall_ms": elapsed,
                        "owner_plan": list(owner._plan_info),
                        "native_output_lse_sha256": [digest(t) for t in expected],
                        "exact": True,
                    }
                )
                if not rebound:
                    assert value.state == "RETIRED"
                    break
            if value.state == "BOUND":
                value.release_before_update()
                changed = list(Q)
                changed[0] += 1
                changed[1] -= 1
                metadata[0].copy_(ip(changed))
                plan()
                assert not value.bind_current_plan(
                    inputs, qo_lengths=changed, kv_lengths=KV, forward_options=options
                )
                assert value.state == "RETIRED"
                assert_exact(value.run(inputs, forward_options=options), native(inputs))
        evidence["lease"] = value.evidence()
    destination = Path(os.environ.get("SGI_REUSABLE_LEASE_EVIDENCE", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ("paged.json" if paged else "ragged.json")).write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
