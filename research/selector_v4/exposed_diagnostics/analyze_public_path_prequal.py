"""Complete-data analysis of actual public-path development measurements."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from public_path_contract import CONTRACT, digest, full_call_choice


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geo(values):
    return math.exp(float(np.log(values).mean()))


def draws(values, seed, hierarchical=False):
    logs = np.log(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    if not hierarchical:
        index = rng.integers(0, len(logs), size=(20000, len(logs)))
        return logs[index].mean(axis=1)
    assert logs.shape == (3, 24)
    processes = rng.integers(0, 3, size=(20000, 3))
    blocks = rng.integers(0, 24, size=(20000, 3, 24))
    return logs[processes[:, :, None], blocks].mean(axis=(1, 2))


def controls_resolve(values, seed):
    ci = np.exp(np.quantile(draws(values, seed), [0.05, 0.95])).tolist()
    return {"ci90": ci, "resolve": ci[0] >= 1 / 1.01 and ci[1] <= 1.01}


def check_rows(rows, role):
    expected = 144 if role == "policy" else 96
    assert len(rows) == expected
    grouped = []
    for block in range(24):
        if role == "policy":
            names = ["native", "oracle", "policy"]
            first = names[block % 3 :] + names[: block % 3]
            order = first + list(reversed(first))
        elif role == "pristine":
            order = ["native"] * 4
        else:
            order = (
                ["native", "candidate", "candidate", "native"]
                if not block % 2
                else ["candidate", "native", "native", "candidate"]
            )
        part = rows[block * len(order) : (block + 1) * len(order)]
        assert [r["arm"] for r in part] == order
        assert [r["position"] for r in part] == list(range(len(order)))
        assert all(r["block"] == block and r["exact"] for r in part)
        for row in part:
            assert row["elapsed_us"] >= 120000 and row["iterations"] > 0
            assert math.isfinite(row["wall_us_per_call"]) and row["wall_us_per_call"] > 0
            assert math.isclose(
                row["elapsed_us"] / row["kernel_calls"], row["wall_us_per_call"], rel_tol=1e-12
            )
        grouped.append(
            {arm: [r["wall_us_per_call"] for r in part if r["arm"] == arm] for arm in set(order)}
        )
    return grouped


def telemetry(path):
    active = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            clock = next((v for k, v in row.items() if k and "clocks.sm" in k), None)
            utilization = next((v for k, v in row.items() if k and "utilization.gpu" in k), None)
            try:
                if float(utilization.split()[0]) >= 90:
                    active.append(float(clock.split()[0]))
            except (ValueError, AttributeError, TypeError):
                continue
    if len(active) < 10:
        return {
            "stable": False,
            "reason": "insufficient active samples",
            "active_samples": len(active),
        }
    lo, hi = np.quantile(active, [0.05, 0.95])
    return {
        "stable": bool(lo > 0 and hi / lo <= 1.05),
        "active_samples": len(active),
        "p05_mhz": float(lo),
        "p95_mhz": float(hi),
    }


def analyze(campaign, gpu, out):
    assert not out.exists()
    binding = json.loads((campaign / "binding.json").read_text())
    assert binding["contract"] == CONTRACT
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())[gpu]
    assert set(jobs) == {"0", "1"}
    files = {
        "binding.json": sha(campaign / "binding.json"),
        "receipts/jobs.json": sha(campaign / "receipts/jobs.json"),
    }
    records = []
    native_cluster = []
    actual_cluster = []
    selected_cluster = []
    regrets = []
    health = []
    for case_index, job in jobs.items():
        assert (campaign / f"receipts/exit-{job}.txt").read_text().strip() == "0"
        root = campaign / f"runs/{gpu}-{job}"
        role_data = {}
        environments = []
        for role in ("pristine", "train", "policy"):
            role_data[role] = {}
            for rep in range(3):
                phase = root / f"{role}-{rep}"
                complete = json.loads((phase / "complete.json").read_text())
                assert complete["complete"] and len(complete["cells"]) == 24
                environment = json.loads((phase / "environment.json").read_text())
                expected_source = (
                    binding["pristine_commit"]
                    if role == "pristine"
                    else binding["candidate_commit"]
                )
                assert environment["source_commit"] == expected_source
                assert environment["role"] == role and environment["rep"] == rep
                assert not environment["profiled_scoring"]
                assert environment["case"] == binding["cases"][int(case_index)]
                environments.append(environment)
                role_data[role][rep] = complete["cells"]
                for file in phase.rglob("*"):
                    if file.is_file():
                        files[str(file.relative_to(campaign))] = sha(file)
        assert len({e["gpu_uuid"] for e in environments}) == 1
        assert len({tuple(e["cpu_affinity"]) for e in environments}) == 1
        assert len({e["pid"] for e in environments}) == 9
        keys = set(role_data["pristine"][0])
        assert all(set(cells) == keys for group in role_data.values() for cells in group.values())
        for key in sorted(keys):
            training = {i: role_data["train"][i][key] for i in range(3)}
            native_ratios, actual_ratios = [], []
            for rep in range(3):
                pristine = role_data["pristine"][rep][key]
                policy = role_data["policy"][rep][key]
                phase = root / f"policy-{rep}" / key
                for role in ("pristine", "train", "policy"):
                    conditioning = json.loads(
                        (root / f"{role}-{rep}" / key / "native-conditioning.json").read_text()
                    )
                    assert (
                        conditioning["seconds_minimum"] == CONTRACT["native_conditioning_seconds"]
                    )
                    assert (
                        conditioning["elapsed_seconds"] >= CONTRACT["native_conditioning_seconds"]
                    )
                    assert not conditioning["includes_scored_windows"]
                decision = json.loads((phase / "frozen-choice-before-scoring.json").read_text())
                assert decision == full_call_choice(training, rep, key + ":" + policy["identity"])
                assert policy["choice"] == decision["choice"]
                assert policy["input_hashes"] == pristine["input_hashes"]
                assert policy["native_output_lse_sha256"] == pristine["native_output_lse_sha256"]
                assert policy["actual_plan"] == pristine["actual_plan"]
                assert policy["every_resource_call_bound"] and policy["binding_failures"] == 0
                for i in range(3):
                    train_rows = json.loads(
                        (root / f"train-{i}" / key / "windows.json").read_text()
                    )
                    blocks = check_rows(train_rows, "train")
                    expected = [
                        [x["native"][0], x["candidate"][0], x["candidate"][1], x["native"][1]]
                        for x in blocks
                    ]
                    assert training[i]["paired_blocks"] == expected
                pb = check_rows(
                    json.loads((root / f"pristine-{rep}" / key / "windows.json").read_text()),
                    "pristine",
                )
                cb = check_rows(json.loads((phase / "windows.json").read_text()), "policy")
                native = [geo(a["native"]) / geo(b["native"]) for a, b in zip(pb, cb, strict=True)]
                actual = [geo(a["native"]) / geo(b["policy"]) for a, b in zip(pb, cb, strict=True)]
                gain = [geo(b["native"]) / geo(b["policy"]) for b in cb]
                native_ratios.append(native)
                actual_ratios.append(actual)
                seed = int(digest([key, rep])[:8], 16)
                control = {
                    arm: controls_resolve([block[arm][0] / block[arm][1] for block in cb], seed ^ i)
                    for i, arm in enumerate(("native", "oracle", "policy"))
                }
                pristine_control = controls_resolve(
                    [geo(block["native"][:2]) / geo(block["native"][2:]) for block in pb],
                    seed ^ 99173,
                )
                selected = decision["choice"] == "resource"
                if selected:
                    selected_cluster.append(draws(gain, seed))
                # The selected path is itself an available tactic. Include its
                # independent measurement in the oracle to keep excess >=0.
                arm_means = {
                    arm: geo([geo(block[arm]) for block in cb])
                    for arm in ("native", "oracle", "policy")
                }
                regrets.append(arm_means["policy"] / min(arm_means.values()) - 1)
                records.append(
                    {
                        "case": int(case_index),
                        "key": key,
                        "rep": rep,
                        "execution": policy["execution"],
                        "dtype": policy["dtype"],
                        "layout": policy["layout"],
                        "selected": selected,
                        "policy_over_native_speedup": geo(gain),
                        "policy_block_worst": min(gain),
                        "native_over_pristine_speedup": geo(native),
                        "actual_policy_over_pristine_speedup": geo(actual),
                        "controls": control,
                        "pristine_control": pristine_control,
                        "absolute_native_us": geo([geo(b["native"]) for b in cb]),
                        "absolute_policy_us": geo([geo(b["policy"]) for b in cb]),
                        "choice_checksum": decision["checksum"],
                    }
                )
            native_cluster.append(draws(native_ratios, int(digest([key, "native"])[:8], 16), True))
            actual_cluster.append(draws(actual_ratios, int(digest([key, "policy"])[:8], 16), True))
        sass_path = campaign / f"receipts/independent-sass-{job}/receipt.json"
        sass = json.loads(sass_path.read_text())
        assert sass["pass"] and sass["binding_sha256"] == files["binding.json"]
        files[str(sass_path.relative_to(campaign))] = sha(sass_path)
        telemetry_path = campaign / f"logs/telemetry-{job}.csv"
        health.append(telemetry(telemetry_path))
        files[str(telemetry_path.relative_to(campaign))] = sha(telemetry_path)
    selected = [r for r in records if r["selected"]]
    joint = lambda array: (
        float(np.exp(np.quantile(np.stack(array).min(axis=0), 0.025))) if array else None
    )
    metrics = {
        "records": len(records),
        "selected": len(selected),
        "coverage": len(selected) / len(records),
        "selected_geomean": geo([r["policy_over_native_speedup"] for r in selected])
        if selected
        else None,
        "selected_worst": min(r["policy_over_native_speedup"] for r in selected)
        if selected
        else None,
        "selected_block_worst": min(r["policy_block_worst"] for r in selected)
        if selected
        else None,
        "selected_joint_min_lcb95": joint(selected_cluster),
        "whole_policy_worst": min(r["policy_over_native_speedup"] for r in records),
        "native_pristine_worst": min(r["native_over_pristine_speedup"] for r in records),
        "native_pristine_joint_min_lcb95": joint(native_cluster),
        "actual_policy_pristine_worst": min(
            r["actual_policy_over_pristine_speedup"] for r in records
        ),
        "actual_policy_pristine_joint_min_lcb95": joint(actual_cluster),
        "regret": {
            "definition": "chosen latency / available held-out oracle latency - 1; oracle never trains",
            "p50": float(np.quantile(regrets, 0.5)),
            "p90": float(np.quantile(regrets, 0.9)),
            "p99": float(np.quantile(regrets, 0.99)),
            "worst": max(regrets),
            "above_one_percent_count": sum(x > 0.01 for x in regrets),
        },
    }
    checks = {
        "complete_independent_process_records": len(records) == 144,
        "selected_nonempty": bool(selected),
        "selected_geomean_at_least_1_05": bool(selected) and metrics["selected_geomean"] >= 1.05,
        "selected_worst_at_least_0_99": bool(selected) and metrics["selected_worst"] >= 0.99,
        "selected_block_worst_at_least_0_99": bool(selected)
        and metrics["selected_block_worst"] >= 0.99,
        "selected_joint_lcb_at_least_0_99": bool(selected)
        and metrics["selected_joint_min_lcb95"] >= 0.99,
        "whole_policy_worst_at_least_0_99": metrics["whole_policy_worst"] >= 0.99,
        "native_pristine_worst_at_least_0_99": metrics["native_pristine_worst"] >= 0.99,
        "native_pristine_joint_lcb_at_least_0_99": metrics["native_pristine_joint_min_lcb95"]
        >= 0.99,
        "actual_policy_pristine_worst_at_least_0_99": metrics["actual_policy_pristine_worst"]
        >= 0.99,
        "actual_policy_pristine_joint_lcb_at_least_0_99": metrics[
            "actual_policy_pristine_joint_min_lcb95"
        ]
        >= 0.99,
        "controls_resolved": all(
            r["pristine_control"]["resolve"] and r["controls"]["native"]["resolve"] for r in records
        )
        and all(all(c["resolve"] for c in r["controls"].values()) for r in selected),
        "all_modes_selected": {r["execution"] for r in selected}
        == {"eager_full_call", "graph1_replay", "graph16_replay"},
        "both_layouts_selected": {r["layout"] for r in selected} == {"ragged", "paged"},
        "both_dtypes_selected": {r["dtype"] for r in selected} == {"float16", "bfloat16"},
        "active_clocks_stable": all(x["stable"] for x in health),
    }
    result = {
        "pass": all(checks.values()),
        "scope": CONTRACT["scope"],
        "gpu": gpu,
        "binding_sha256": files["binding.json"],
        "metrics": metrics,
        "requirements": checks,
        "records": records,
        "telemetry": health,
        "files": files,
        "analyzer_sha256": sha(Path(__file__)),
        "trimmed_windows": 0,
        "fresh_cases_consumed": 0,
        "canary_authority": False,
        "default_promotion": False,
        "serving_promotion": False,
    }
    out.mkdir()
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "metrics": metrics,
                "failed_requirements": [k for k, v in checks.items() if not v],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("gpu", choices=["gpu_4090", "gpu_5090"])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analyze(args.campaign, args.gpu, args.out)
