"""Frozen v4 candidate-pool policy.

The static rule only decides whether cap is worth calibrating. It never enables
cap by itself. A cache miss, incomplete identity, or failed calibration returns
native. The 2.4-wave floor is frozen from exposed v3.2.2 development evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

SCHEMA = 1
WAVE_NUMERATOR = 12
WAVE_DENOMINATOR = 5
MIN_DESCRIPTORS = 40
MIN_BATCH = 5
MIN_MAX_CACHED = 8192


def ceil_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise ValueError("ceil_div")
    return (a + b - 1) // b


def descriptor_threshold(num_sms: int, num_kv_heads: int) -> int:
    if num_sms <= 0 or num_kv_heads <= 0:
        raise ValueError("hardware identity")
    return max(MIN_DESCRIPTORS, ceil_div(WAVE_NUMERATOR * num_sms,
                                         WAVE_DENOMINATOR * num_kv_heads))


def anti_monotone(q: Sequence[int], cached: Sequence[int]) -> bool:
    if len(q) != len(cached) or len(q) < MIN_BATCH:
        return False
    concordant = discordant = 0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            product = (q[i] - q[j]) * (cached[i] - cached[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    return concordant == 0 and discordant >= len(q) - 1


@dataclass(frozen=True)
class PoolDecision:
    eligible: bool
    reason: str
    descriptor_threshold: int
    block_waves: float


def candidate_pool(*, q: Sequence[int], cached: Sequence[int],
                   padded_batch_size: int, num_qo_heads: int,
                   num_kv_heads: int, num_sms: int, split_kv: bool,
                   head_dim_qk: int = 128, head_dim_vo: int = 128,
                   causal: bool = True) -> PoolDecision:
    threshold = descriptor_threshold(num_sms, num_kv_heads)
    waves = padded_batch_size * num_kv_heads / num_sms if num_sms else 0.0
    checks = (
        (causal, "noncausal"),
        (head_dim_qk == 128 and head_dim_vo == 128, "head-dim"),
        (num_qo_heads == 32 and num_kv_heads == 8, "head-count"),
        (len(q) >= MIN_BATCH, "small-batch"),
        (len(q) == len(cached), "length-mismatch"),
        (not split_kv, "split-plan"),
        (max(cached, default=0) >= MIN_MAX_CACHED, "short-cache"),
        (anti_monotone(q, cached), "pairing"),
        (padded_batch_size >= threshold, "insufficient-waves"),
    )
    for ok, reason in checks:
        if not ok:
            return PoolDecision(False, reason, threshold, waves)
    return PoolDecision(True, "eligible", threshold, waves)
