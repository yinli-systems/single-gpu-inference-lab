"""Summarize every exposed diagnostic block; no qualification authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, rng, level):
    # Resample independent processes first, then all 24 paired blocks within each.
    n_processes, n_blocks = values.shape
    process = rng.integers(0, n_processes, size=(20000, n_processes))
    blocks = rng.integers(0, n_blocks, size=(20000, n_processes, n_blocks))
    samples = np.exp(values[process[:, :, None], blocks].mean(axis=(1, 2)))
    alpha = (1 - level) / 2
    return np.quantile(samples, [alpha, 1 - alpha]).tolist()


def analyze(campaign, out, allow_incomplete_diagnostic=False):
    if out.exists():
        raise FileExistsError("preserve earlier analyses")
    binding = json.loads((campaign / "binding.json").read_text())
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())
    ledger = {"binding.json": digest(campaign / "binding.json")}
    results = {}
    for gpu, job in jobs.items():
        rows, environments, numerics, completeness = [], [], [], []
        for rep in range(3):
            root = campaign / "runs" / f"{gpu}-r{rep}-{job}"
            completed = (root / "complete.json").exists()
            assert completed or allow_incomplete_diagnostic, str(root)
            if completed:
                complete = json.loads((root / "complete.json").read_text())
                assert complete["complete"] and complete["rep"] == rep
                for name, expected in complete["files"].items():
                    assert digest(root / name) == expected, str(root / name)
            part = json.loads((root / "measurements.json").read_text())
            assert len(part) == 2592 if completed else 0 < len(part) < 2592
            assert all(r["rep"] == rep and r["window_wall_ms"] >= 120 for r in part)
            if (root / "raw-windows.jsonl").exists():
                durable = [
                    json.loads(line)
                    for line in (root / "raw-windows.jsonl").read_text().splitlines()
                ]
                assert durable == part
            rows.extend(part)
            environments.append(json.loads((root / "environment.json").read_text()))
            numeric = json.loads((root / "numerics.json").read_text())
            assert len(numeric) == len({r["cell_index"] for r in part})
            assert all(n["exact"] and n["native_module_unchanged"] for n in numeric)
            numerics.append(numeric)
            for f in root.glob("*.json*"):
                ledger[str(f.relative_to(campaign))] = digest(f)
            progress = json.loads((root / "progress.json").read_text())
            completeness.append(
                {
                    "rep": rep,
                    "complete": completed,
                    "persisted_rows": len(part),
                    "last_reported_rows": progress["rows"],
                    "unpersisted_rows_at_least": max(0, progress["rows"] - len(part)),
                    "timed_unpersisted_tail_count_unknown": not completed,
                }
            )
        assert (
            len(
                {
                    (e["gpu_uuid"], tuple(e["affinity"]), e["script_sha256"], e["source"])
                    for e in environments
                }
            )
            == 1
        )
        assert all(
            e["script_sha256"] == digest(campaign / "native_context.py") for e in environments
        )
        # Compare common binary paths across all independent processes and cells.
        libraries = {}
        for numeric in numerics:
            for cell in numeric:
                for name, value in cell["native_libraries"].items():
                    assert name not in libraries or libraries[name] == value
                    libraries[name] = value
        grouped = {}
        for r in rows:
            key = (r["cell_index"], r["execution_mode"], r["comparison"], r["rep"], r["block"])
            grouped.setdefault(key, []).append(r)
        summaries = []
        rng = np.random.default_rng(20261001)
        for cell in range(3):
            for mode in ("eager_prepared", "graph1_pure_replay", "graph16_pure_replay"):
                for comparison, other in (
                    ("native_context", "native_after_resource"),
                    ("fallback_overhead", "uncertified_fallback"),
                    ("resource_gain", "resource"),
                ):
                    reps = sorted(
                        {
                            r["rep"]
                            for r in rows
                            if r["cell_index"] == cell
                            and r["execution_mode"] == mode
                            and r["comparison"] == comparison
                        }
                    )
                    assert len(reps) >= 2, (
                        "Insufficient independent processes even for diagnostic summary"
                    )
                    item = {
                        "cell_index": cell,
                        "execution_mode": mode,
                        "comparison": comparison,
                        "processes": len(reps),
                        "process_reps": reps,
                        "blocks": len(reps) * 24,
                    }
                    for metric in ("wall_us", "device_us"):
                        ratios = np.zeros((len(reps), 24))
                        aa_native = np.zeros((len(reps), 24))
                        aa_other = np.zeros((len(reps), 24))
                        process_latencies = []
                        for process_index, rep in enumerate(reps):
                            latency = {"native": [], other: []}
                            for block in range(24):
                                block_rows = sorted(
                                    grouped[(cell, mode, comparison, rep, block)],
                                    key=lambda r: r["position"],
                                )
                                expected = (
                                    ["native", other, other, "native"]
                                    if block % 2 == 0
                                    else [other, "native", "native", other]
                                )
                                assert [r["arm"] for r in block_rows] == expected
                                assert [r["position"] for r in block_rows] == [0, 1, 2, 3]
                                assert len({r["count"] for r in block_rows}) == 1
                                native = [r[metric] for r in block_rows if r["arm"] == "native"]
                                candidate = [r[metric] for r in block_rows if r["arm"] == other]
                                assert min(native + candidate) > 0
                                ratios[process_index, block] = (
                                    np.log(native).mean() - np.log(candidate).mean()
                                )
                                aa_native[process_index, block] = np.log(native[0] / native[1])
                                aa_other[process_index, block] = np.log(candidate[0] / candidate[1])
                                latency["native"].extend(native)
                                latency[other].extend(candidate)
                            process_latencies.append(
                                {k: float(np.exp(np.log(v).mean())) for k, v in latency.items()}
                            )
                        ci = interval(ratios, rng, 0.95)
                        native_control = interval(aa_native, rng, 0.90)
                        other_control = interval(aa_other, rng, 0.90)
                        item[metric] = {
                            "native_over_other_geomean": float(np.exp(ratios.mean())),
                            "process_ratios": np.exp(ratios.mean(axis=1)).tolist(),
                            "block_min_ratio": float(np.exp(ratios.min())),
                            "process_min_ratio": float(np.exp(ratios.mean(axis=1)).min()),
                            "cluster_bootstrap_95_ci": ci,
                            "native_aa_90_ci": native_control,
                            "other_aa_90_ci": other_control,
                            "controls_resolve_half_percent": all(
                                0.995 <= a <= b <= 1.005 for a, b in (native_control, other_control)
                            ),
                            "descriptive_point_and_lcb_above_099": float(np.exp(ratios.mean()))
                            >= 0.99
                            and ci[0] >= 0.99,
                            "process_latency_geomeans_us": process_latencies,
                        }
                    summaries.append(item)
        assert sum(item["blocks"] * 4 for item in summaries) == len(rows), (
            "Every persisted row must be summarized"
        )
        results[gpu] = {
            "job": str(job),
            "rows": len(rows),
            "environment": environments,
            "process_completeness": completeness,
            "campaign_complete": all(c["complete"] for c in completeness),
            "native_libraries_unchanged": True,
            "all_output_and_lse_exact": True,
            "summaries": summaries,
        }
    result = {
        "binding": binding,
        "input_files_sha256": ledger,
        "gpus": results,
        "bootstrap_resamples": 20000,
        "bootstrap_unit": "independent process, then paired block",
        "all_persisted_rows_retained": True,
        "allow_incomplete_diagnostic": allow_incomplete_diagnostic,
        "campaign_complete": all(g["campaign_complete"] for g in results.values()),
        "analyzer_sha256": digest(Path(__file__)),
        "trimmed_rows": 0,
        "fresh_cases_consumed": 0,
        "qualification_authority": False,
        "full_plan_measured": False,
        "outer_managed_dispatch_measured": False,
        "full_http_qualified": False,
        "historical_token_divergence_resolved": False,
        "limitations": [
            "Three exposed cells; independent process counts reported per cell. Development diagnosis only",
            "An incomplete diagnostic campaign is summarized only with explicit opt-in; unpersisted timing tails cannot be reconstructed",
            "Prepared calls and pure fixed replay; no replanning or Graph input updates",
            "Resource is deliberately uncertified; native fallback uses runner.forward, not AutoTuner.choose_one",
            "Vendor payload is the recorded older CCCL overlay; official pinned-vendor wheel is qualified separately",
            "Observed associations do not attribute the cause of frozen v4.2 native regressions",
        ],
    }
    out.mkdir()
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--allow-incomplete-diagnostic", action="store_true")
    args = parser.parse_args()
    analyze(args.campaign, args.out, args.allow_incomplete_diagnostic)
