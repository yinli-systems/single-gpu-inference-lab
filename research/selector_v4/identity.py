from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math
from typing import Any, Mapping
from .schema import QUALIFICATION_REVISION, require_fields

_ENV_FIELDS = (
    "gpu_name", "gpu_uuid", "num_sms", "driver", "cuda", "torch", "flashinfer",
    "nvcc", "backend_source_sha256", "official_overlay_sha256",
    "resource_binding_sha256", "max_smem_per_sm", "max_smem_per_block_optin",
)
_OP_FIELDS = (
    "execution_mode", "backend", "causal", "layout", "dtype", "actual_split",
    "num_qo_heads", "num_kv_heads", "head_dim_qk", "head_dim_vo", "page_size",
    "q", "cached", "plan_signature",
)

def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite identity value")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value)}
    raise TypeError(f"unsupported identity value: {type(value).__name__}")

def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

@dataclass(frozen=True)
class TacticIdentity:
    environment: Mapping[str, Any]
    operation: Mapping[str, Any]
    measurement_policy: Mapping[str, Any]

    def validate(self) -> None:
        require_fields(self.environment, _ENV_FIELDS, "environment")
        require_fields(self.operation, _OP_FIELDS, "operation")
        require_fields(self.measurement_policy, ("execution_mode", "timer", "qualification_revision"), "measurement policy")
        if self.operation["execution_mode"] != self.measurement_policy["execution_mode"]:
            raise ValueError("execution-mode identity alias")
        if self.measurement_policy["qualification_revision"] != QUALIFICATION_REVISION:
            raise ValueError("qualification revision mismatch")
        if self.operation["backend"] != "fa2" or self.operation["causal"] is not True:
            raise ValueError("unsupported operator identity")
        if self.operation["dtype"] not in ("float16", "bfloat16"):
            raise ValueError("unsupported dtype")
        if len(self.operation["plan_signature"]) != 15:
            raise ValueError("operation identity must exclude tactic-specific plan flag")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": _canonical(self.environment),
            "operation": _canonical(self.operation),
            "measurement_policy": _canonical(self.measurement_policy),
        }

    @property
    def key(self) -> str:
        self.validate()
        return digest(self.to_dict())

    @property
    def environment_hash(self) -> str:
        return digest({"environment": self.environment, "measurement_policy": self.measurement_policy})[:24]

    @property
    def operation_hash(self) -> str:
        return digest({"operation": self.operation})[:32]
