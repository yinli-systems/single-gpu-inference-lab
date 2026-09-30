"""Fail-closed, deployment-mode-specific tactic calibration and persistence."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence
import fcntl, hashlib, json, math, os, random, tempfile, time

SCHEMA = 1
TACTICS = ("native", "cap")
MEASUREMENT_MODES = ("eager_full_call", "graph1_replay", "graph16_replay")
MIN_PROCESSES = 3
MIN_BLOCKS_PER_PROCESS = 8
MIN_POINT_SPEEDUP = 1.02
MIN_CI95_LOWER = 1.01
CONTROL_TOLERANCE = 0.005
BOOTSTRAP_DRAWS = 10000


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("positive finite values required")
    return math.exp(sum(math.log(x) for x in values) / len(values))


def hierarchical_bootstrap(process_blocks: Sequence[Sequence[float]], *,
                           draws: int = BOOTSTRAP_DRAWS,
                           seed: int = 470041) -> list[float]:
    if len(process_blocks) < MIN_PROCESSES:
        raise ValueError("insufficient process repeats")
    if any(len(x) < MIN_BLOCKS_PER_PROCESS for x in process_blocks):
        raise ValueError("insufficient paired blocks")
    if any(not math.isfinite(x) or x <= 0 for row in process_blocks for x in row):
        raise ValueError("invalid ratios")
    rng = random.Random(seed)
    out: list[float] = []
    nproc = len(process_blocks)
    for _ in range(draws):
        logs: list[float] = []
        for _ in range(nproc):
            row = process_blocks[rng.randrange(nproc)]
            for _ in range(len(row)):
                logs.append(math.log(row[rng.randrange(len(row))]))
        out.append(math.exp(sum(logs) / len(logs)))
    return out


def quantile(values: Sequence[float], q: float) -> float:
    if not 0 <= q <= 1 or not values:
        raise ValueError("quantile")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(pos); hi = min(lo + 1, len(xs) - 1); frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


@dataclass(frozen=True)
class CalibrationEvidence:
    tactic: str
    ratio: float
    ci95: tuple[float, float]
    control_ci90: tuple[float, float]
    exact_outputs: bool
    clock_stable: bool
    candidate_pool: bool
    process_repeats: int
    blocks_per_process: int
    evidence_sha256: str
    reason: str


def decide(*, native_over_cap: Sequence[Sequence[float]],
           null_controls: Sequence[Sequence[float]], exact_outputs: bool,
           clock_stable: bool, candidate_pool: bool,
           seed: int = 470041) -> CalibrationEvidence:
    nproc = len(native_over_cap)
    nblocks = min((len(x) for x in native_over_cap), default=0)
    payload = dict(native_over_cap=[list(x) for x in native_over_cap],
                   null_controls=[list(x) for x in null_controls],
                   exact_outputs=exact_outputs, clock_stable=clock_stable,
                   candidate_pool=candidate_pool, seed=seed)
    evidence_hash = canonical_hash(payload)
    if not candidate_pool:
        return CalibrationEvidence("native", 1.0, (1.0, 1.0), (1.0, 1.0),
                                   exact_outputs, clock_stable, False,
                                   nproc, nblocks, evidence_hash, "outside-candidate-pool")
    if not exact_outputs:
        return CalibrationEvidence("native", 1.0, (0.0, 0.0), (0.0, 0.0),
                                   False, clock_stable, True, nproc, nblocks,
                                   evidence_hash, "numerical-mismatch")
    if not clock_stable:
        return CalibrationEvidence("native", 1.0, (0.0, 0.0), (0.0, 0.0),
                                   True, False, True, nproc, nblocks,
                                   evidence_hash, "clock-unstable")
    try:
        draws = hierarchical_bootstrap(native_over_cap, seed=seed)
        controls = hierarchical_bootstrap(null_controls, seed=seed ^ 0x5A5A)
    except ValueError as exc:
        return CalibrationEvidence("native", 1.0, (0.0, 0.0), (0.0, 0.0),
                                   True, True, True, nproc, nblocks,
                                   evidence_hash, "invalid-evidence:" + str(exc))
    flat = [x for row in native_over_cap for x in row]
    ratio = geometric_mean(flat)
    ci = (quantile(draws, .025), quantile(draws, .975))
    control_ci = (quantile(controls, .05), quantile(controls, .95))
    control_ok = control_ci[0] >= 1 / (1 + CONTROL_TOLERANCE) and control_ci[1] <= 1 + CONTROL_TOLERANCE
    choose_cap = control_ok and ratio >= MIN_POINT_SPEEDUP and ci[0] >= MIN_CI95_LOWER
    reason = "calibrated-cap" if choose_cap else ("control-unresolved" if not control_ok else "margin-not-met")
    return CalibrationEvidence("cap" if choose_cap else "native", ratio, ci,
                               control_ci, True, True, True, nproc, nblocks,
                               evidence_hash, reason)


class TacticStore:
    """Atomic persistent store with a fully hydrated, zero-I/O steady-state map."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._hydrated = False

    def hydrate(self) -> None:
        if self._hydrated:
            return
        if not self.path.exists():
            self._records = {}; self._hydrated = True; return
        try:
            raw = json.loads(self.path.read_text())
            if raw.get("schema") != SCHEMA or not isinstance(raw.get("records"), dict):
                raise ValueError("schema")
            for key, record in raw["records"].items():
                if key != canonical_hash(record["identity"]):
                    raise ValueError("identity digest")
                if record.get("tactic") not in TACTICS:
                    raise ValueError("tactic")
            self._records = raw["records"]
        except Exception:
            self._records = {}
        self._hydrated = True

    def lookup(self, identity: dict[str, Any]) -> str:
        self.hydrate()
        record = self._records.get(canonical_hash(identity))
        return record["tactic"] if record else "native"

    def publish(self, identity: dict[str, Any], evidence: CalibrationEvidence) -> None:
        key = canonical_hash(identity)
        record = dict(identity=identity, tactic=evidence.tactic,
                      evidence=asdict(evidence), published_unix_ns=time.time_ns())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records: dict[str, dict[str, Any]] = {}
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text())
                    if raw.get("schema") == SCHEMA and isinstance(raw.get("records"), dict):
                        records = raw["records"]
                except Exception:
                    records = {}
            records[key] = record
            payload = json.dumps({"schema": SCHEMA, "records": records},
                                 indent=2, sort_keys=True) + "\n"
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".",
                                       dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w") as out:
                    out.write(payload); out.flush(); os.fsync(out.fileno())
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            self._records = records
            self._hydrated = True
