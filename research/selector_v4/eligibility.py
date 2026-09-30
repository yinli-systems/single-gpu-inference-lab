from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from .identity import TacticIdentity

@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}

def evaluate_eligibility(identity: TacticIdentity) -> EligibilityDecision:
    identity.validate()
    env: Mapping[str, Any] = identity.environment
    op: Mapping[str, Any] = identity.operation
    reasons: list[str] = []
    q = [int(x) for x in op["q"]]
    cached = [int(x) for x in op["cached"]]
    if len(q) != len(cached) or len(q) < 5:
        reasons.append("batch_lt_5_or_length_mismatch")
    if op["actual_split"] != "unsplit":
        reasons.append("split_plan")
    if max(cached, default=0) < 8192:
        reasons.append("short_cached_prefix")
    if (op["num_qo_heads"], op["num_kv_heads"], op["head_dim_qk"], op["head_dim_vo"]) != (32, 8, 128, 128):
        reasons.append("unsupported_attention_geometry")
    if op["layout"] not in ("ragged", "paged"):
        reasons.append("unsupported_layout")
    if op["page_size"] != (1 if op["layout"] == "ragged" else 16):
        reasons.append("page_size_mismatch")
    if int(env.get("max_smem_per_sm", 102400)) < 102400:
        reasons.append("insufficient_smem_per_sm")
    if int(env.get("max_smem_per_block_optin", 65536)) < 65536:
        reasons.append("insufficient_optin_smem")
    return EligibilityDecision(not reasons, tuple(reasons))
