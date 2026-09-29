"""Frozen formal analysis for the fresh collision-factorial campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from manifest import build, cases_for
from l20_stack.decision_sufficiency import (
    Geometry,
    ResolvedDecision,
    feature_with_context,
    find_opposite_action_collisions,
    marginal_moments,
)

DRAWS = 10000
SEED = 2026093007
REPS = 4
BLOCKS = 12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hardware_identity(env):
    lines = env["hardware"]["out"].strip().splitlines()
    require(len(lines) == 2, "one visible GPU required")
    return lines[1].strip()


def load_runs(root, partition):
    manifest = build()
    cases = cases_for("formal")
    paths = sorted((root / "runs").glob("formal-" + partition + "-r*-*"))
    require(len(paths) == REPS * 2, "expected eight formal run directories")
    runs = {}
    sources = set()

    expected_rows = set(itertools.product(
        [c["id"] for c in cases],
        manifest["dtypes"],
        manifest["split_modes"],
        manifest["execution_modes"],
        ("heavy_first", "identity_repeat"),
        range(BLOCKS),
        range(4),
    ))
    expected_quals = set(itertools.product(
        [c["id"] for c in cases],
        manifest["dtypes"],
        manifest["split_modes"],
    ))

    for path in paths:
        require(not (path / "failure.json").exists(), "failed run retained")
        require((path / "complete.json").exists(), "incomplete run")
        require((path / "launcher-complete.txt").exists(), "launcher incomplete")
        complete = json.loads((path / "complete.json").read_text())
        env = json.loads((path / "environment.json").read_text())
        require(complete["complete"] is True, "completion false")
        require(env["stage"] == "formal", "wrong stage")
        require(env["partition"] == partition, "wrong partition")
        require(env["case_hash"] == manifest["case_hash"], "manifest drift")
        require(env["blocks"] == BLOCKS, "block drift")
        key = (env["rep"], env["resource_mode"])
        require(key not in runs, "duplicate run")

        for filename, digest in complete["files"].items():
            target = path / filename
            require(target.exists() and sha256(target) == digest, "raw hash mismatch")

        sources.add(json.dumps(env["source"], sort_keys=True))
        rows = json.loads((path / "measurements.json").read_text())
        index = {}
        for row in rows:
            rkey = (
                row["case"], row["dtype"], row["split"],
                row["execution_mode"], row["comparison"],
                row["block"], row["position"],
            )
            require(rkey in expected_rows and rkey not in index, "row identity failure")
            seq = (
                ("identity", row["comparison"], row["comparison"], "identity")
                if row["block"] % 2 == 0
                else (row["comparison"], "identity", "identity", row["comparison"])
            )
            require(row["arm"] == seq[row["position"]], "ABBA/BAAB mismatch")
            require(row["rep"] == env["rep"], "repeat mismatch")
            require(row["resource_mode"] == env["resource_mode"], "resource mismatch")
            for field in ("device_us", "wall_us", "setup_us"):
                value = row[field]
                require(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0,
                    "invalid timing",
                )
            index[rkey] = row
        require(set(index) == expected_rows, "timing matrix incomplete")

        quals = json.loads((path / "qualification.json").read_text())
        qindex = {}
        for q in quals:
            qkey = (q["case"], q["dtype"], q["split"])
            require(qkey not in qindex, "duplicate qualification")
            require(q["eager_graph_exact"] is True, "graph parity missing")
            require(q["FP32"]["vectors"] > 0, "FP32 reference missing")
            for action in ("identity", "identity_repeat", "heavy_first"):
                require(q["actions"][action]["exact_output"] is True, "output mismatch")
                require(q["actions"][action]["exact_lse"] is True, "LSE mismatch")
            qindex[qkey] = q
        require(set(qindex) == expected_quals, "qualification matrix incomplete")

        plans = json.loads((path / "plans.json").read_text())
        pindex = {(p["case"], p["dtype"], p["split"]): p for p in plans}
        require(set(pindex) == expected_quals, "plan matrix incomplete")
        runs[key] = {
            "env": env,
            "index": index,
            "quals": qindex,
            "plans": pindex,
        }

    require(len(sources) == 1, "mixed source")
    require(
        set(runs) == set(itertools.product(range(REPS), manifest["resource_modes"])),
        "missing run",
    )
    for rep in range(REPS):
        pristine = runs[(rep, "pristine")]
        cap = runs[(rep, "cap")]
        require(pristine["env"]["job"] == cap["env"]["job"], "different allocation")
        require(
            hardware_identity(pristine["env"]) == hardware_identity(cap["env"]),
            "different physical GPU within repeat",
        )
        for qkey in expected_quals:
            qa = pristine["quals"][qkey]
            qb = cap["quals"][qkey]
            require(qa["input_hashes"] == qb["input_hashes"], "input drift")
            require(qa["output_hash"] == qb["output_hash"], "output drift")
            require(qa["lse_hash"] == qb["lse_hash"], "LSE drift")
            pa = pristine["plans"][qkey]["plan"]
            pb = cap["plans"][qkey]["plan"]
            require(pa["invariant_hash"] == pb["invariant_hash"], "plan invariant drift")
            require(pa["descriptor_hash"] == pb["descriptor_hash"], "descriptor drift")
    return manifest, runs


def log_grid(runs, case, dtype, split, execution, comparison, resource):
    grid = np.empty((REPS, BLOCKS), dtype=np.float64)
    for rep in range(REPS):
        run = runs[(rep, resource)]
        for block in range(BLOCKS):
            native = []
            candidate = []
            for pos in range(4):
                row = run["index"][
                    (case, dtype, split, execution, comparison, block, pos)
                ]
                if row["arm"] == "identity":
                    native.append(row["device_us"])
                else:
                    candidate.append(row["device_us"])
            require(len(native) == len(candidate) == 2, "paired positions missing")
            grid[rep, block] = math.log(float(np.mean(native)) / float(np.mean(candidate)))
    return grid


def bootstrap_draws(grid, confidence, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = np.empty(DRAWS, dtype=np.float64)
    for d in range(DRAWS):
        values = []
        for rep in rng.integers(0, REPS, size=REPS):
            blocks = rng.integers(0, BLOCKS, size=BLOCKS)
            values.extend(grid[rep, blocks].tolist())
        draws[d] = float(np.mean(values))
    tail = (1.0 - confidence) / 2.0
    return draws, [float(x) for x in np.quantile(draws, [tail, 1.0 - tail])]


def ratio_stats(grid, confidence):
    draws, interval = bootstrap_draws(grid, confidence)
    return {
        "ratio": math.exp(float(grid.mean())),
        "CI": [math.exp(x) for x in interval],
        "confidence": confidence,
    }, draws


def interaction_draws(grids, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(DRAWS, dtype=np.float64)
    for d in range(DRAWS):
        sampled_reps = rng.integers(0, REPS, size=REPS)
        means = {}
        for name, grid in grids.items():
            values = []
            for rep in sampled_reps:
                blocks = rng.integers(0, BLOCKS, size=BLOCKS)
                values.extend(grid[rep, blocks].tolist())
            means[name] = float(np.mean(values))
        pristine = means["a_pristine"] - means["b_pristine"]
        cap = means["a_cap"] - means["b_cap"]
        out[d] = abs(cap) - abs(pristine)
    return out


def analyze(root, partition):
    manifest, runs = load_runs(root, partition)
    cases = cases_for("formal")
    by_id = {c["id"]: c for c in cases}
    decisions = {resource: [] for resource in manifest["resource_modes"]}
    cells = {}

    for case, dtype, split, execution, resource in itertools.product(
        cases,
        manifest["dtypes"],
        manifest["split_modes"],
        manifest["execution_modes"],
        manifest["resource_modes"],
    ):
        aa_grid = log_grid(
            runs, case["id"], dtype, split, execution,
            "identity_repeat", resource,
        )
        heavy_grid = log_grid(
            runs, case["id"], dtype, split, execution,
            "heavy_first", resource,
        )
        aa, _ = ratio_stats(aa_grid, 0.90)
        heavy, _ = ratio_stats(heavy_grid, 0.95)
        aa_ok = aa["CI"][0] >= 1 / 1.005 and aa["CI"][1] <= 1.005
        key = (case["id"], dtype, split, execution, resource)
        cells[key] = {
            "AA": {**aa, "equivalent": bool(aa_ok)},
            "heavy": heavy,
            "heavy_grid": heavy_grid,
        }

        geometry = Geometry(tuple(case["q"]), tuple(case["cached"]))
        feature = feature_with_context(
            marginal_moments(geometry),
            dtype=dtype,
            split=split,
            execution=execution,
        )
        decisions[resource].append(
            ResolvedDecision(
                state_id="|".join([case["id"], dtype, split, execution, resource]),
                feature_key=feature,
                action_a="identity",
                action_b="heavy_first",
                ratio_a_over_b=heavy["ratio"],
                ci_low=heavy["CI"][0],
                ci_high=heavy["CI"][1],
                controls_resolve=bool(aa_ok),
                context=tuple(sorted({
                    "case": case["id"],
                    "pair_id": case["pair_id"],
                    "arm": case["arm"],
                    "dtype": dtype,
                    "split": split,
                    "execution": execution,
                    "resource": resource,
                }.items())),
            )
        )

    collision_results = {}
    for resource, rows in decisions.items():
        witnesses = find_opposite_action_collisions(
            rows, relative_margin=manifest["decision_margin"]
        )
        declared = []
        for witness in witnesses:
            left = dict(witness.left.context)
            right = dict(witness.right.context)
            if left["pair_id"] != right["pair_id"]:
                continue
            declared.append({
                "pair_id": left["pair_id"],
                "left": {
                    "state": witness.left.state_id,
                    "preference": witness.left.preference(manifest["decision_margin"]),
                    "ratio": witness.left.ratio_a_over_b,
                    "CI95": [witness.left.ci_low, witness.left.ci_high],
                },
                "right": {
                    "state": witness.right.state_id,
                    "preference": witness.right.preference(manifest["decision_margin"]),
                    "ratio": witness.right.ratio_a_over_b,
                    "CI95": [witness.right.ci_low, witness.right.ci_high],
                },
                "conservative_normalized_minimax_regret": (
                    witness.conservative_normalized_minimax_regret
                ),
                "normalized_minimax_regret_point": (
                    witness.normalized_minimax_regret_point
                ),
                "context": {
                    "dtype": left["dtype"],
                    "split": left["split"],
                    "execution": left["execution"],
                },
            })
        collision_results[resource] = declared

    interactions = []
    for pair_id, dtype, split, execution in itertools.product(
        manifest["primary_pair_ids"],
        manifest["dtypes"],
        manifest["split_modes"],
        manifest["execution_modes"],
    ):
        a = pair_id + "-a"
        b = pair_id + "-b"
        grids = {
            "a_pristine": cells[(a, dtype, split, execution, "pristine")]["heavy_grid"],
            "b_pristine": cells[(b, dtype, split, execution, "pristine")]["heavy_grid"],
            "a_cap": cells[(a, dtype, split, execution, "cap")]["heavy_grid"],
            "b_cap": cells[(b, dtype, split, execution, "cap")]["heavy_grid"],
        }
        pristine = abs(float(grids["a_pristine"].mean() - grids["b_pristine"].mean()))
        cap = abs(float(grids["a_cap"].mean() - grids["b_cap"].mean()))
        seed = (
            SEED
            + 1000 * manifest["primary_pair_ids"].index(pair_id)
            + 100 * manifest["dtypes"].index(dtype)
            + 10 * manifest["split_modes"].index(split)
            + manifest["execution_modes"].index(execution)
        )
        draws = interaction_draws(grids, seed)
        ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        if ci[1] < 0:
            classification = "contracted"
        elif ci[0] > 0:
            classification = "expanded"
        else:
            classification = "unresolved"
        interactions.append({
            "pair_id": pair_id,
            "dtype": dtype,
            "split": split,
            "execution": execution,
            "abs_log_contrast_pristine": pristine,
            "abs_log_contrast_cap": cap,
            "cap_minus_pristine": cap - pristine,
            "CI95": ci,
            "classification": classification,
        })

    primary = [
        w for w in collision_results["pristine"]
        if w["context"]["execution"] == "graph16"
    ]
    graph_interactions = [x for x in interactions if x["execution"] == "graph16"]

    return {
        "kind": "fresh collision-factorial formal analysis",
        "partition": partition,
        "manifest_hash": manifest["case_hash"],
        "bootstrap": {
            "draws": DRAWS,
            "seed": SEED,
            "repetitions": REPS,
            "blocks": BLOCKS,
        },
        "primary": {
            "representation": "marginal_moments",
            "resource": "pristine",
            "execution": "graph16",
            "collision_contexts": len(primary),
            "unique_pair_ids": sorted({w["pair_id"] for w in primary}),
            "max_conservative_normalized_minimax_regret": max(
                (w["conservative_normalized_minimax_regret"] for w in primary),
                default=0.0,
            ),
        },
        "resource_interaction_graph16": {
            "contexts": len(graph_interactions),
            "contracted": sum(x["classification"] == "contracted" for x in graph_interactions),
            "expanded": sum(x["classification"] == "expanded" for x in graph_interactions),
            "unresolved": sum(x["classification"] == "unresolved" for x in graph_interactions),
        },
        "collisions": collision_results,
        "interactions": interactions,
        "default_promotion": False,
        "serving_promotion": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partition", choices=["gpu_4090", "gpu_5090"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior analysis")
    result = analyze(args.root, args.partition)
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + chr(10)
    )
    compact = {
        "partition": result["partition"],
        "primary": result["primary"],
        "resource_interaction_graph16": result["resource_interaction_graph16"],
    }
    (args.out / "RESULTS.md").write_text(
        "# Fresh collision-factorial result" + chr(10) + chr(10)
        + json.dumps(compact, indent=2) + chr(10) + chr(10)
        + "No default or serving promotion follows from this kernel-level assay."
        + chr(10)
    )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
