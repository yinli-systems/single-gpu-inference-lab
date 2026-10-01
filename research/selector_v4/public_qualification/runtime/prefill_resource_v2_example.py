# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Explicit resource-cap calibration for a standard, unsplit ragged FA2 plan."""

import json

import torch

from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer import autotune_v2, MeasurementPolicy
from flashinfer.autotuner import AutoTuner
from flashinfer.prefill import make_prefill_resource_runner

q_lengths = [3, 39, 107, 175, 243, 311, 411]
cached_lengths = [22528, 16512, 11840, 8256, 2624, 672, 80]
kv_lengths = [q + k for q, k in zip(q_lengths, cached_lengths, strict=True)]
qo = torch.tensor(
    [0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
)
kv = torch.tensor(
    [0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
)
q = torch.randn(sum(q_lengths), 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(sum(kv_lengths), 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
wrapper = BatchPrefillWithRaggedKVCacheWrapper(
    torch.empty(128 << 20, dtype=torch.uint8, device="cuda"), backend="fa2"
)
wrapper.plan(
    qo, kv, 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True
)
runner = make_prefill_resource_runner(
    wrapper, [q, k, v], qo_lengths=q_lengths, kv_lengths=kv_lengths
)
receipt = runner.calibrate([q, k, v])
print(json.dumps(receipt, indent=2))
policy = MeasurementPolicy(execution_mode="eager")
with autotune_v2(mode="tune", measurement_policy=policy):
    result = runner.run([q, k, v])
    chosen, tactic = AutoTuner.get().choose_one(
        "experimental_prefill_resource", [runner], runner.tuning_config, [q, k, v]
    )
    assert chosen is runner and tactic in (-1, 0, 1)
    # v2 profiles the default (-1) alongside explicit tactics. A native default
    # winner is valid; an unprofiled fallback after every tactic failed is not.
    assert (
        AutoTuner.get().stats.tuned_op_successful_configs.get(
            "experimental_prefill_resource", 0
        )
        > 0
    ), "Managed profiling did not produce a winner"
    assert not AutoTuner.get().stats.failed_tactics.get(
        "experimental_prefill_resource::ResourceCapRunner"
    ), "An offered tactic failed profiling"
print(json.dumps({"managed_tactic": tactic, "measurement_policy": "eager"}))
with autotune_v2(mode="replay", measurement_policy=policy):
    replayed = runner.run([q, k, v])
torch.testing.assert_close(result, replayed, rtol=0, atol=0)
torch.testing.assert_close(result, wrapper.run(q, k, v), rtol=0, atol=0)

# Optional same-geometry eager replanning under an immutable package/compiler
# envelope and exclusive metadata ownership. Keep a native managed winner on
# the direct native path; only an actually selected resource winner may rebind.
wrapper.plan(
    qo, kv, 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True
)
with autotune_v2(mode="replay", measurement_policy=policy):
    if tactic == 1 and runner.rebind_same_geometry([q, k, v]):
        replanned = runner.run([q, k, v])
    else:
        replanned = wrapper.run(q, k, v)
torch.testing.assert_close(replanned, wrapper.run(q, k, v), rtol=0, atol=0)
