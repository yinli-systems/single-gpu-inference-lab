"""Retain all preregistered profiling fits; descriptive exposed diagnosis only."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profiling_times(text):
    result = {}
    for line in text.splitlines():
        match = re.search(
            r"\[Autotuner\]: profiling .*? (-?\d+), shapes=.*?, avg_time ([\d.eE+-]+)$", line
        )
        if match:
            tactic, latency = int(match[1]), float(match[2])
            assert tactic not in result and math.isfinite(latency) and latency > 0
            result[tactic] = latency
    return result


def analyze(campaign, out):
    if out.exists():
        raise FileExistsError("Preserve previous analyses")
    binding = json.loads((campaign / "binding.json").read_text())
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())
    sequence = binding["per_process_fit_repeat_sequence"]
    assert sequence == [10, 256, 256, 10, 256, 10, 10, 256]
    ledger = {"binding.json": sha(campaign / "binding.json")}
    gpus = {}
    for gpu, job in jobs.items():
        assert (campaign / f"receipts/exit-{job}.txt").read_text().strip() == "0"
        rows = []
        for rep in range(3):
            root = campaign / "runs" / f"{gpu}-r{rep}-{job}"
            complete = json.loads((root / "complete.json").read_text())
            assert complete["complete"] and complete["fits"] == 8 and complete["rep"] == rep
            for name, digest in complete["files"].items():
                assert sha(root / name) == digest, str(root / name)
            environment = json.loads((root / "environment.json").read_text())
            assert environment["source_commit"] == binding["flashinfer_source_commit"]
            fits = json.loads((root / "results.json").read_text())
            assert [f["profiling_repeat"] for f in fits] == sequence
            assert [f["fit"] for f in fits] == list(range(8))
            for fit in fits:
                folder = root / f"fit-{fit['fit']:02d}-repeat{fit['profiling_repeat']}"
                times = profiling_times((folder / "profiling.log").read_text())
                expected = {-1, 0, 1} if environment["certificate_accepted"] else {-1, 0}
                assert set(times) == expected and fit["chosen_tactic"] in times
                assert fit["exact"] and fit["same_certificate_for_both_precision_settings"]
                rows.append(
                    dict(
                        rep=rep,
                        **fit,
                        profiling_times_ms=times,
                        best_native_over_resource=min(times[-1], times[0]) / times[1]
                        if 1 in times
                        else None,
                    )
                )
            for f in root.rglob("*"):
                if f.is_file():
                    ledger[str(f.relative_to(campaign))] = sha(f)
        assert len(rows) == 24
        groups = {}
        for count in (10, 256):
            part = [f for f in rows if f["profiling_repeat"] == count]
            groups[str(count)] = dict(
                fits=len(part),
                resource_wins=sum(f["chosen_tactic"] == 1 for f in part),
                resource_offered=sum(f["best_native_over_resource"] is not None for f in part),
                all_output_exact=all(f["exact"] for f in part),
                process_geomean_profile_ratios={
                    str(rep): math.exp(
                        sum(
                            math.log(f["best_native_over_resource"])
                            for f in part
                            if f["rep"] == rep and f["best_native_over_resource"] is not None
                        )
                        / len(
                            [
                                f
                                for f in part
                                if f["rep"] == rep and f["best_native_over_resource"] is not None
                            ]
                        )
                    )
                    for rep in range(3)
                    if any(
                        f["rep"] == rep and f["best_native_over_resource"] is not None for f in part
                    )
                },
            )
        gpus[gpu] = dict(job=job, fits=rows, by_profiling_repeat=groups)
    out.mkdir()
    result = dict(
        binding=binding,
        gpus=gpus,
        input_files_sha256=ledger,
        analyzer_sha256=sha(Path(__file__)),
        trimmed_fits=0,
        all_started_fits_retained=True,
        fresh_cases_consumed=0,
        qualification_authority=False,
        causal_attribution_established=False,
        original_outlier_explained=False,
        default_promotion=False,
        limitations=[
            "One exposed eager geometry, three processes per GPU, twelve correlated fits per precision setting.",
            "Profiler timings diagnose ranking; they are not independent held-out latency or HTTP performance.",
            "A non-reproduced outlier does not establish its cause or prove a larger repeat count prevents it.",
        ],
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({g: r["by_profiling_repeat"] for g, r in gpus.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analyze(args.campaign, args.out)
