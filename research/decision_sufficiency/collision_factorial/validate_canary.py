"""Fail-closed infrastructure gate for the collision-factorial canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hardware_identity(env):
    lines = env["hardware"]["out"].strip().splitlines()
    require(len(lines) == 2, "one visible GPU required")
    return lines[1].strip()


def validate(root, partition):
    paths = sorted((root / "runs").glob("canary-" + partition + "-r0-*"))
    require(len(paths) == 2, "expected pristine and cap canary runs")
    runs = {}

    for path in paths:
        require(not (path / "failure.json").exists(), "canary failure retained")
        require((path / "complete.json").exists(), "incomplete canary")
        require((path / "launcher-complete.txt").exists(), "launcher incomplete")
        complete = json.loads((path / "complete.json").read_text())
        env = json.loads((path / "environment.json").read_text())
        require(complete["complete"] is True, "completion false")
        require(env["stage"] == "canary", "wrong stage")
        require(env["partition"] == partition, "wrong partition")
        require(env["rep"] == 0, "wrong canary repeat")
        resource = env["resource_mode"]
        require(resource in ("pristine", "cap"), "bad resource mode")
        require(resource not in runs, "duplicate resource canary")

        for filename, digest in complete["files"].items():
            target = path / filename
            require(target.exists() and sha256(target) == digest, "raw hash mismatch")

        quals = json.loads((path / "qualification.json").read_text())
        require(quals, "missing qualifications")
        for q in quals:
            require(q["eager_graph_exact"] is True, "graph parity missing")
            require(q["FP32"]["vectors"] > 0, "FP32 check missing")
            for action in ("identity", "identity_repeat", "heavy_first"):
                require(q["actions"][action]["exact_output"] is True, "output mismatch")
                require(q["actions"][action]["exact_lse"] is True, "LSE mismatch")

        rows = json.loads((path / "measurements.json").read_text())
        require(rows, "missing timing rows")
        aa_ratios = []
        grouped = {}
        for row in rows:
            for field in ("device_us", "wall_us", "setup_us"):
                value = row[field]
                require(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0,
                    "invalid timing",
                )
            if row["comparison"] != "identity_repeat":
                continue
            key = (
                row["case"],
                row["dtype"],
                row["split"],
                row["execution_mode"],
                row["block"],
            )
            grouped.setdefault(key, {"identity": [], "identity_repeat": []})
            grouped[key][row["arm"]].append(row["device_us"])
        for values in grouped.values():
            require(
                len(values["identity"]) == 2
                and len(values["identity_repeat"]) == 2,
                "A/A placement incomplete",
            )
            native = sum(values["identity"]) / 2
            repeat = sum(values["identity_repeat"]) / 2
            aa_ratios.append(native / repeat)

        runs[resource] = {
            "path": str(path),
            "env": env,
            "complete": complete,
            "aa_ratios": aa_ratios,
        }

    require(set(runs) == {"pristine", "cap"}, "resource canary coverage")
    require(
        runs["pristine"]["env"]["job"] == runs["cap"]["env"]["job"],
        "resource canaries not in same allocation",
    )
    require(
        hardware_identity(runs["pristine"]["env"])
        == hardware_identity(runs["cap"]["env"]),
        "resource canaries changed physical GPU",
    )

    pristine_quals = {
        (q["case"], q["dtype"], q["split"]): q
        for q in json.loads(
            (Path(runs["pristine"]["path"]) / "qualification.json").read_text()
        )
    }
    cap_quals = {
        (q["case"], q["dtype"], q["split"]): q
        for q in json.loads(
            (Path(runs["cap"]["path"]) / "qualification.json").read_text()
        )
    }
    require(set(pristine_quals) == set(cap_quals), "cross-resource qualification set")
    for key in pristine_quals:
        a = pristine_quals[key]
        b = cap_quals[key]
        require(a["input_hashes"] == b["input_hashes"], "input drift")
        require(a["output_hash"] == b["output_hash"], "output drift")
        require(a["lse_hash"] == b["lse_hash"], "LSE drift")

    all_aa = runs["pristine"]["aa_ratios"] + runs["cap"]["aa_ratios"]
    return {
        "passed": True,
        "partition": partition,
        "physical_gpu": hardware_identity(runs["pristine"]["env"]),
        "canary_runs": {k: v["path"] for k, v in runs.items()},
        "AA_point_ratio_min": min(all_aa),
        "AA_point_ratio_max": max(all_aa),
        "performance_gate": False,
        "note": (
            "Canary timing is infrastructure evidence only. Formal thresholds, "
            "cases and endpoints remain unchanged regardless of these point ratios."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=["gpu_4090", "gpu_5090"],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior canary gate")
    result = validate(args.root, args.partition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
