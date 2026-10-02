"""Separate dev-only infrastructure correction; never replaces scored work.

Original terminal HOLD and conservative consumption remain unchanged. Reuse
exactly its manifest, source packet, successful smoke and started dev jobs.
Only proven pre-script signal-53 dev allocations may be dispatched once on
another node. All original gates/analyzers remain byte-identical.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from research.selector_v4.continue_graph_http import ACTIVE
from research.selector_v4.public_qualification import pipeline as k
from research.selector_v4.public_qualification.gates import STAGES, need, sha


def read(path):
    return json.loads(path.read_text())


def status(job):
    raw = subprocess.check_output(
        ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed,NodeList", "-n", "-P"],
        text=True,
        timeout=60,
    ).strip()
    fields = raw.split("|")
    need(len(fields) == 5 and fields[0] == job, "Actual known formal allocation")
    return raw, fields


def unstarted(root, gpu, job, fields, node):
    need(
        fields[:3] == [job, "FAILED", "0:53"] and fields[4] == node,
        "Exact unstarted infrastructure failure",
    )
    paths = [
        root / f"runs/{gpu}-{job}",
        root / f"receipts/source-verification-{job}.txt",
        root / f"receipts/validation-verification-{job}.txt",
        root / f"receipts/exit-{job}.txt",
    ]
    paths += list((root / "logs").glob(job + "*"))
    paths += list((root / "logs").glob("telemetry-" + job + "*"))
    need(
        not any(p.exists() for p in paths),
        "No started validation, telemetry or measurement may be replaced",
    )


def copy_frozen_tree(source, target):
    """Private copies: no controller edit can mutate the original files."""
    excluded = {"cache", "__pycache__", ".pytest_cache"}
    for parent, folders, names in os.walk(source):
        current = Path(parent)
        relative = current.relative_to(source)
        destination = target / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in list(folders):
            path = current / name
            if name in excluded:
                folders.remove(name)
            elif path.is_symlink():
                folders.remove(name)
                out = destination / name
                if not out.exists():
                    out.symlink_to(path.resolve(), target_is_directory=True)
        for name in names:
            path, out = current / name, destination / name
            need(not path.is_symlink(), "Actual frozen data")
            if not out.exists():
                shutil.copyfile(path, out)
            else:
                need(sha(path) == sha(out), "Existing copied original byte changed")


def initialize(original, shadow, node):
    need(not shadow.exists(), "New correction branch only")
    old = read(original / "receipts/controller-terminal.json")
    need(
        old["terminal"] and old["state"] == "FORMAL_QUALIFICATION_HOLD",
        "Retained original terminal HOLD",
    )
    need(
        not any((original / "stages" / stage).exists() for stage in STAGES[1:]),
        "Only dev startup failures are eligible",
    )
    binding = read(original / "frozen-binding.json")
    k.verify_helpers(original, binding)
    k.verify_manifest(original, binding)
    k.build_stage_ticket(original, "dev")
    dev = original / "stages/dev"
    jobs = read(dev / "receipts/jobs.json")
    need(
        set(jobs) == set(k.GPUS) and all(set(group) == {"0", "1"} for group in jobs.values()),
        "Original complete two-case dev dispatch",
    )
    failures, retained = {}, {}
    for gpu, group in jobs.items():
        for index, job in group.items():
            raw, fields = status(job)
            key = gpu + ":" + index
            if fields[1] == "FAILED":
                unstarted(dev, gpu, job, fields, node)
                failures[key] = {
                    "original_job": job,
                    "sacct": raw,
                    "scored_requests_started": 0,
                    "gpu_measurements_started": 0,
                }
            else:
                need(
                    fields[1] in ACTIVE or fields[1:3] == ["COMPLETED", "0:0"],
                    "Any started failed allocation blocks correction",
                )
                retained[key] = job
    need(
        0 < len(failures) <= 2 and all(key.startswith("gpu_4090:") for key in failures),
        "At most two original unstarted dev4090 allocations",
    )
    shadow.mkdir()
    for name in ("harness", "sdk-libraries", "smoke"):
        copy_frozen_tree(original / name, shadow / name)
    for name in ("frozen-binding.json", "manifest.json", "harness.tar.gz"):
        shutil.copyfile(original / name, shadow / name)
    (shadow / "receipts").mkdir()
    shutil.copyfile(original / "receipts/dual-smoke.json", shadow / "receipts/dual-smoke.json")
    shutil.copyfile(
        original / "receipts/controller-terminal.json",
        shadow / "receipts/original-controller-hold.json",
    )
    new_dev = shadow / "stages/dev"
    for name in ("runs", "receipts", "logs"):
        (new_dev / name).mkdir(parents=True)
    for name in ("harness", "sdk-libraries"):
        (new_dev / name).symlink_to(shadow / name, target_is_directory=True)
    for name in ("authorization.json", "binding.json"):
        shutil.copyfile(dev / name, new_dev / name)
    shutil.copyfile(dev / "receipts/consumption.json", new_dev / "receipts/consumption.json")
    k.validation_ledger(
        new_dev,
        shadow,
        binding,
        [
            new_dev / "binding.json",
            new_dev / "authorization.json",
            shadow / "frozen-binding.json",
            shadow / "manifest.json",
        ],
    )
    correction = {
        "original": str(original),
        "original_terminal_sha256": sha(original / "receipts/controller-terminal.json"),
        "original_frozen_binding_sha256": sha(original / "frozen-binding.json"),
        "manifest_sha256": sha(original / "manifest.json"),
        "dev_binding_sha256": sha(dev / "binding.json"),
        "original_jobs": jobs,
        "unstarted_failures": failures,
        "retained_started_jobs": retained,
        "scheduler_excluded_node": node,
        "correction_helper_sha256": sha(Path(__file__)),
        "finite_days": 14,
        "fresh_cases_regenerated": 0,
        "started_measurements_retried": 0,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    k.save_new(shadow / "receipts/infrastructure-correction-binding.json", correction)
    shutil.copyfile(Path(__file__), shadow / "receipts/infrastructure-correction-helper.py")
    return correction


def finish_dev(shadow, binding, jobs):
    root = shadow / "stages/dev"
    for group in jobs.values():
        for job in group.values():
            k.execute(
                shadow,
                [
                    shadow
                    / "harness/research/selector_v4/public_qualification/runtime/audit_public_path_sass.py",
                    root,
                    job,
                    root / f"receipts/independent-sass-{job}",
                ],
                root / f"logs/sass-{job}.log",
                5400,
            )
    (root / "analysis").mkdir()
    verdicts = {}
    for gpu in k.GPUS:
        k.execute(
            shadow,
            [
                shadow / "harness/research/selector_v4/public_qualification/analyze.py",
                root,
                gpu,
                root / "analysis" / gpu,
            ],
            root / f"logs/analysis-{gpu}.log",
            7200,
        )
        summary = read(root / "analysis" / gpu / "summary.json")
        verdicts[gpu] = {
            "pass": summary["pass"],
            "failed_requirements": [
                key for key, value in summary["requirements"].items() if not value
            ],
            "summary_sha256": sha(root / "analysis" / gpu / "summary.json"),
        }
    k.save_new(root / "receipts/dual-verdict.json", verdicts)
    need(all(value["pass"] for value in verdicts.values()), "Unchanged dual dev gates HOLD")


def scheduler_command(root, script, arguments, node):
    return [
        "sbatch",
        "--parsable",
        "--exclude=" + node,
        "-p",
        str(arguments[0]),
        "-o",
        str(root / "logs/%j.out"),
        "-e",
        str(root / "logs/%j.err"),
        str(script),
        *map(str, arguments[1:]),
    ]


def submit_excluding(root, script, arguments, node):
    label = str(arguments[0]) + "-" + str(arguments[3])
    intent = root / f"receipts/submit-intent-{label}.json"
    command = scheduler_command(root, script, arguments, node)
    k.save_new(
        intent,
        {
            "arguments": list(map(str, arguments)),
            "command": command,
            "script_sha256": sha(script),
            "scheduler_excluded_node": node,
            "utc": k.now(),
        },
    )
    job = subprocess.check_output(command, text=True, timeout=60).strip().split(";")[0]
    need(job.isdigit(), "Actual unambiguous submission")
    k.save_new(
        root / f"receipts/submit-confirmed-{label}.json",
        {"job": job, "intent_sha256": sha(intent), "utc": k.now()},
    )
    actual = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=60)
    k.save_new(
        root / f"receipts/scheduler-{job}.json",
        {"job": job, "actual_scontrol": actual, "explicit_exclude": node},
    )
    need("ExcNodeList=" + node in actual.split(), "Actual scheduler must acknowledge excluded node")
    return job


def existing_jobs(shadow, correction):
    root = shadow / "stages/dev"
    jobs = read(root / "receipts/jobs.json")
    need(
        set(jobs) == set(k.GPUS) and all(set(group) == {"0", "1"} for group in jobs.values()),
        "Exact four existing dev allocations",
    )
    for gpu, group in jobs.items():
        for index, job in group.items():
            if gpu + ":" + index in correction["unstarted_failures"]:
                expected = read(root / f"receipts/unstarted-correction-{gpu}-{index}.json")[
                    "successor_job"
                ]
            else:
                expected = correction["original_jobs"][gpu][index]
            need(job == expected, "No started job identity may change at scheduler amendment")
    amendment = read(shadow / "receipts/scheduler-amendment.json")
    need(
        sha(root / "receipts/jobs.json") == amendment["existing_jobs_sha256"],
        "Frozen coordinator checkpoint",
    )
    return jobs


def run(shadow, *, continue_existing=False):
    correction = read(shadow / "receipts/infrastructure-correction-binding.json")
    original = Path(correction["original"])
    binding = read(shadow / "frozen-binding.json")
    need(
        not (shadow / "receipts/controller-terminal.json").exists(),
        "Never restart a terminal scientific HOLD",
    )
    if continue_existing:
        amendment = read(shadow / "receipts/scheduler-amendment.json")
        need(
            sha(Path(__file__)) == amendment["new_helper_sha256"],
            "Reviewed scheduler-only amendment",
        )
        k.save_new(
            shadow / f"receipts/controller-restart-{os.getpid()}.json",
            {"pid": os.getpid(), "utc": k.now(), "existing_jobs_only": True},
        )
    else:
        amendment = None
        k.save_new(shadow / "receipts/controller-start.json", {"pid": os.getpid(), "utc": k.now()})
    state = {
        "state": "DEV_UNSTARTED_INFRASTRUCTURE_CORRECTION",
        "terminal": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
        "original_hold_retained": True,
    }
    started = datetime.datetime.fromisoformat(
        read(shadow / "receipts/controller-start.json")["utc"]
    )
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
    deadline = time.monotonic() + max(0, correction["finite_days"] * 86400 - elapsed)
    try:
        need(
            sha(Path(__file__))
            == (
                amendment["new_helper_sha256"]
                if amendment
                else correction["correction_helper_sha256"]
            ),
            "Frozen independent correction helper",
        )
        need(
            sha(original / "receipts/controller-terminal.json")
            == correction["original_terminal_sha256"],
            "Original HOLD unchanged",
        )
        need(
            sha(shadow / "frozen-binding.json") == correction["original_frozen_binding_sha256"]
            and sha(shadow / "manifest.json") == correction["manifest_sha256"],
            "Exactly original source binding and manifest",
        )
        k.verify_helpers(shadow, binding)
        k.build_stage_ticket(shadow, "dev")
        # Only the submission adapter changes; scripts, numerical helpers,
        # manifests, thresholds and all measurement arguments stay frozen.
        k.submit = lambda root, script, args: submit_excluding(
            root, script, args, correction["scheduler_excluded_node"]
        )
        root = shadow / "stages/dev"
        jobs = {gpu: {} for gpu in k.GPUS}
        script = shadow / "harness/research/selector_v4/public_qualification/case.sbatch"
        if continue_existing:
            jobs = existing_jobs(shadow, correction)
        else:
            for gpu, group in correction["original_jobs"].items():
                for index, old_job in group.items():
                    key = gpu + ":" + index
                    if key in correction["unstarted_failures"]:
                        unstarted(
                            original / "stages/dev",
                            gpu,
                            old_job,
                            status(old_job)[1],
                            correction["scheduler_excluded_node"],
                        )
                        job = k.submit(
                            root,
                            script,
                            [
                                gpu,
                                root,
                                binding["candidate_source"],
                                int(index),
                                binding["pristine_source"],
                            ],
                        )
                        k.save_new(
                            root / f"receipts/unstarted-correction-{gpu}-{index}.json",
                            {
                                "original_failed_job": old_job,
                                "successor_job": job,
                                "scheduler_excluded_node": correction["scheduler_excluded_node"],
                                "scored_requests_started_in_original": 0,
                            },
                        )
                    else:
                        job = old_job
                        for prefix in ("submit-intent-", "submit-confirmed-", "dispatch-"):
                            name = f"{prefix}{gpu}-{index}.json"
                            shutil.copyfile(
                                original / "stages/dev/receipts" / name, root / "receipts" / name
                            )
                    jobs[gpu][index] = job
            k.save_new(root / "receipts/jobs.json", jobs)
        # Existing jobs continue in the original root; copy their complete bytes
        # only after successful termination. Their processes are never relaunched.
        active = {(gpu, index, job) for gpu, group in jobs.items() for index, job in group.items()}
        copied = set()
        while active:
            for gpu, index, job in list(active):
                source_root = (
                    original / "stages/dev"
                    if correction["original_jobs"][gpu][index] == job
                    else root
                )
                if k.terminal(source_root, job):
                    if source_root != root:
                        for name in ("runs", "logs", "receipts"):
                            if name == "runs":
                                copy_frozen_tree(
                                    source_root / name / f"{gpu}-{job}",
                                    root / name / f"{gpu}-{job}",
                                )
                            else:
                                for p in (source_root / name).glob("*" + job + "*"):
                                    if p.is_file():
                                        shutil.copyfile(p, root / name / p.name)
                        copied.add(job)
                    active.remove((gpu, index, job))
            need(time.monotonic() < deadline, "Finite correction deadline")
            if active:
                time.sleep(15)
        # Preserve every original intent/confirmation in a separate subtree;
        # it never replaces the successful job map used by the frozen analyzer.
        copy_frozen_tree(original / "stages/dev/receipts", root / "receipts/original-dispatch")
        k.save_new(
            root / "receipts/retained-started-jobs.json",
            {
                "retained_actual_jobs": sorted(copied),
                "original_consumption_sha256": sha(
                    original / "stages/dev/receipts/consumption.json"
                ),
                "started_measurements_retried": 0,
            },
        )
        finish_dev(shadow, binding, jobs)
        for stage in STAGES[1:]:
            k.save_new(shadow / f"receipts/start-{stage}.json", {"utc": k.now()})
            state["state"] = stage.upper()
            k.dispatch_stage(shadow, stage, binding, deadline)
        state.update(state="FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED", terminal=True)
    except BaseException as error:  # noqa: BLE001 - retain first error and stop all successors
        state.update(
            state="FORMAL_QUALIFICATION_HOLD",
            terminal=True,
            error={"type": type(error).__name__, "message": str(error)},
        )
    state["finished_utc"] = k.now()
    k.save_new(shadow / "receipts/controller-terminal.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shadow", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--continue-existing-jobs", action="store_true")
    parser.add_argument("--original", type=Path)
    parser.add_argument("--excluded-node")
    args = parser.parse_args()
    result = (
        initialize(args.original, args.shadow, args.excluded_node)
        if args.initialize
        else run(args.shadow, continue_existing=args.continue_existing_jobs)
    )
    print(json.dumps(result, indent=2))
