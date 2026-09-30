from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA_VERSION = 4
QUALIFICATION_REVISION = "4.1.0"
TACTIC_NATIVE = "native"
TACTIC_CAP = "resource_cap"
VALID_TACTICS = frozenset((TACTIC_NATIVE, TACTIC_CAP))

@dataclass(frozen=True)
class DecisionThresholds:
    train_geomean: float = 1.05
    train_lcb95: float = 1.01
    train_block_min: float = 0.99
    control_equivalence: float = 0.005
    min_training_blocks: int = 32
    bootstrap_draws: int = 20000

    def validate(self) -> None:
        if not (1.0 < self.train_geomean and 1.0 <= self.train_lcb95):
            raise ValueError("unsafe gain thresholds")
        if not (0.0 < self.train_block_min <= 1.0):
            raise ValueError("invalid block floor")
        if not (0.0 < self.control_equivalence < 0.05):
            raise ValueError("invalid control equivalence")
        if self.min_training_blocks < 8 or self.bootstrap_draws < 2000:
            raise ValueError("insufficient evidence budget")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

DEFAULT_THRESHOLDS = DecisionThresholds()

def require_fields(value: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")
