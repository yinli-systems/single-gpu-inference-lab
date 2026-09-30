from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from typing import Any
from .eligibility import evaluate_eligibility
from .identity import TacticIdentity
from .manifest_v4 import load
from .schema import QUALIFICATION_REVISION

EXECUTIONS = ("eager_full_call", "graph1_replay", "graph16_replay")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def need(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)

def expected_sequence(mode: str, block: int) -> tuple[tuple[str, str], ...]:
    if mode == "pristine":
        roles = ("a","b","b","a") if block % 2 == 0 else ("b","a","a","b")
        return tuple(("pristine", role) for role in roles)
    arms = ("off","cap","cap","off") if block % 2 == 0 else ("cap","off","off","cap")
    return tuple((arm, "A" if arm == "off" else "B") for arm in arms)

def validate_run(path: Path, *, mode: str, stage: str, rep: int, shard: int, shards: int, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path); complete = json.loads((path / "complete.json").read_text())
    need(complete.get("complete") is True, "incomplete run")
    required = {"environment.json","measurements.json","qualification.json","memory.json","progress.json"}
    need(required <= set(complete.get("files", {})), "missing artifact hash")
    for name, expected in complete["files"].items():
        need(Path(name).name == name and sha(path / name) == expected, "artifact hash mismatch " + name)
    env = json.loads((path / "environment.json").read_text()); rows = json.loads((path / "measurements.json").read_text())
    quals = json.loads((path / "qualification.json").read_text()); memory = json.loads((path / "memory.json").read_text())
    need((env["mode"],env["stage"],env["rep"],env["shard"],env["shards"]) == (mode,stage,rep,shard,shards), "run identity")
    need(env["measurement_contract_revision"] == QUALIFICATION_REVISION and env["profiled"] is False, "measurement identity")
    manifest = manifest or load(); cases = [c for i,c in enumerate([x for x in manifest["cases"] if x["family"] == stage]) if i % shards == shard]
    need(env["case_hash"] == manifest["case_hash"] and env["cases"] == cases, "manifest/shard mismatch")
    basics = {(c["id"],dt,layout,split) for c in cases for dt in manifest["dtypes"] for layout in manifest["layouts"] for split in manifest["requested_splits"]}
    qmap = {}
    arms = {"pristine"} if mode == "pristine" else {"off","cap"}
    for q in quals:
        key = (q["case"],q["dtype"],q["layout"],q["split"])
        need(key in basics and key not in qmap, "qualification coordinate")
        need(q["measurement_contract_revision"] == QUALIFICATION_REVISION and set(q["arms"]) == arms, "qualification contract")
        for arm, item in q["arms"].items():
            need(item["pristine_exact"] is True and item["execution_checks"] == dict.fromkeys(EXECUTIONS, True), "numerical qualification")
            need(len(item["out_sha256"]) == len(item["lse_sha256"]) == 64, "tensor hash")
            for execution in EXECUTIONS:
                raw = q["identities"][arm][execution]
                identity = TacticIdentity(raw["payload"]["environment"], raw["payload"]["operation"], raw["payload"]["measurement_policy"])
                need(identity.key == raw["key"], "identity digest")
                need(evaluate_eligibility(identity).to_dict() == q["eligibility"][execution], "eligibility drift")
                need(raw["payload"]["operation"]["plan_signature"] == (item["plan_info"][:-1] if len(item["plan_info"]) == 16 else item["plan_info"]), "tactic-neutral plan identity")
        if mode == "paired":
            need(q["candidate_plan_core_equal"] is True and q["arms"]["off"]["plan_info"][-1] == 0 and q["arms"]["cap"]["plan_info"][-1] == 1, "tactic plan contract")
        qmap[key] = q
    need(set(qmap) == basics and len(memory) == len(basics), "incomplete qualification/memory matrix")
    expected = {(k,b,e,p) for k in basics for b in range(manifest["blocks"]) for e in EXECUTIONS for p in range(4)}
    observed = set()
    for row in rows:
        k = (row["case"],row["dtype"],row["layout"],row["split"]); coord = (k,row["block"],row["execution_mode"],row["position"])
        need(coord in expected and coord not in observed, "timing coordinate")
        observed.add(coord); arm, role = expected_sequence(mode,row["block"])[row["position"]]
        need((row["arm"],row["role"],row["comparison_group"]) == (arm,role,"position" if mode == "pristine" else "cap"), "ABBA/BAAB contract")
        need(math.isfinite(row["wall_us"]) and row["wall_us"] > 0 and row["window_elapsed_us"] >= 12000, "timing validity")
        need(math.isclose(row["window_elapsed_us"], row["wall_us"] * row["kernel_calls"], rel_tol=1e-10), "timing normalization")
        need(row["tactic_identity"] == qmap[k]["identities"][arm][row["execution_mode"]]["key"], "row identity")
    need(observed == expected and len(rows) == complete["rows"] == complete["expected"], "incomplete timing matrix")
    return {"path":str(path),"environment":env,"complete":complete,"rows":rows,"qualifications":qmap,"memory":memory}
