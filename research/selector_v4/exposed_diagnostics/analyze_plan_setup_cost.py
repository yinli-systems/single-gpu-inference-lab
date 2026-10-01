"""Audit every preregistered public plan/setup call; descriptive costs only."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(campaign, out):
    if out.exists():
        raise FileExistsError("Preserve every previous analysis")
    binding = json.loads((campaign / "binding.json").read_text())
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())
    assert set(jobs) == {"gpu_4090", "gpu_5090"}
    assert binding["observations_per_gpu"] == 32
    assert binding["balanced_abba_blocks"] == 8
    assert binding["independent_processes"] == 1
    assert binding["includes_native_plan_and_public_runner_constructor_and_managed_run"]
    assert binding["flashinfer_source_commit"] == "bab48695bbe72ba23c49bad69818b2975f3de6f7"
    ledger = {"binding.json": sha(campaign / "binding.json")}
    gpus = {}
    for gpu, job in jobs.items():
        assert (campaign / f"receipts/exit-{job}.txt").read_text().strip() == "0"
        roots = list((campaign / "runs").glob(f"{gpu}-*{job}"))
        assert len(roots) == 1, roots
        root = roots[0]
        complete = json.loads((root / "complete.json").read_text())
        assert complete["complete"] and complete["observations"] == 32
        assert complete["all32_observations_retained"]
        assert not complete["strict_window_gates_satisfied"]
        for name, digest in complete["files"].items():
            file = root / name
            assert file.resolve().is_relative_to(root.resolve()), name
            assert sha(file) == digest, name
        env = json.loads((root / "environment.json").read_text())
        assert env["source_commit"] == binding["flashinfer_source_commit"]
        assert env["previously_exposed_geometry"] and env["initial_managed_tactic"] in (-1, 0, 1)
        rows = json.loads((root / "measurements.json").read_text())
        assert len(rows) == 32
        block_ratios = []
        for block in range(8):
            part = rows[4 * block : 4 * block + 4]
            expected = ["native", "new_runner", "new_runner", "native"]
            if block % 2:
                expected = ["new_runner", "native", "native", "new_runner"]
            assert [r["arm"] for r in part] == expected
            assert [r["position"] for r in part] == list(range(4))
            assert all(r["block"] == block and r["exact"] for r in part)
            for row in part:
                components = [
                    row[k]
                    for k in ("plan_call_wall_ms", "constructor_call_wall_ms", "run_and_sync_ms")
                ]
                assert all(math.isfinite(x) and x >= 0 for x in components)
                assert math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0
                assert math.isclose(sum(components), row["wall_ms"], rel_tol=1e-12, abs_tol=1e-9)
                if row["arm"] == "new_runner":
                    assert row["confidence_identity_stable"]
                    assert row["loaded_confidence_checksum"] == (
                        env["confidence_checksum"] if env["confidence_published"] else None
                    )
                else:
                    assert row["confidence_identity_stable"] is None
                    assert row["loaded_confidence_checksum"] is None
            means = {
                arm: statistics.mean(r["wall_ms"] for r in part if r["arm"] == arm)
                for arm in ("native", "new_runner")
            }
            block_ratios.append(means["native"] / means["new_runner"])
        by_arm = {}
        for arm in ("native", "new_runner"):
            part = [r for r in rows if r["arm"] == arm]
            assert len(part) == 16
            by_arm[arm] = {
                key: {
                    "mean": statistics.mean(r[key] for r in part),
                    "median": statistics.median(r[key] for r in part),
                    "minimum": min(r[key] for r in part),
                    "maximum": max(r[key] for r in part),
                }
                for key in (
                    "wall_ms",
                    "plan_call_wall_ms",
                    "constructor_call_wall_ms",
                    "run_and_sync_ms",
                )
            }
        for file in root.rglob("*"):
            if file.is_file():
                ledger[str(file.relative_to(campaign))] = sha(file)
        gpus[gpu] = {
            "job": job,
            "environment": env,
            "observations": 32,
            "all_output_exact": True,
            "by_arm": by_arm,
            "native_over_public_block_ratios": block_ratios,
            "native_over_public_block_geomean": math.exp(
                statistics.mean(map(math.log, block_ratios))
            ),
            "native_over_public_mean_wall": by_arm["native"]["wall_ms"]["mean"]
            / by_arm["new_runner"]["wall_ms"]["mean"],
        }
    result = {
        "binding": binding,
        "gpus": gpus,
        "input_files_sha256": ledger,
        "analyzer_sha256": sha(Path(__file__)),
        "trimmed_observations": 0,
        "all32_observations_per_gpu_retained": True,
        "fresh_cases_consumed": 0,
        "qualification_authority": False,
        "full_http_qualified": False,
        "default_promotion": False,
        "historical_token_divergence_resolved": False,
        "limitations": [
            "One exposed geometry and one process per GPU; short calls do not satisfy release windows or independent-process controls.",
            "Construction includes public source/compiler identity checks and cached confidence validation; initial compilation/calibration/training is excluded.",
            "Component times are wall durations and can include synchronization; they are not pure CPU or kernel cycles.",
            "Ratios are descriptive only; no confidence interval or formal qualification is inferred from these correlated samples.",
        ],
    }
    out.mkdir()
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({gpu: r["native_over_public_mean_wall"] for gpu, r in gpus.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analyze(args.campaign, args.out)
