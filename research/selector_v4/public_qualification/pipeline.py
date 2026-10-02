"""Finite, bounded formal qualification; initialization requires archived dual PASS.

No operation occurs on import. No serving or upstream publication is authorized
by this controller. A failed/partial stage is retained and stops all successors.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from research.selector_v4.public_qualification.gates import (
    REVISION,
    SCOPE,
    STAGES,
    build_stage_ticket,
    need,
    sha,
    verify_manifest,
)

GPUS = ("gpu_4090", "gpu_5090")
CANDIDATE = "75544a17ce0019ca877f50354d95451ee089f859"
PRISTINE = "85744da1a397c0c4bea1a83b8345281750eb61a9"
BASE = Path("/ssd/scxi253")
E = BASE / "q7b-engines-20260925T1247Z/envs/sglang312/bin/python"


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ledger(root):
    excluded = {"cache", "sdk-libraries", "__pycache__", ".pytest_cache"}
    result = {}
    for current, folders, names in os.walk(root):
        folders[:] = sorted(n for n in folders if n not in excluded and not n.startswith("xdg-"))
        for name in sorted(names):
            path = Path(current) / name
            if path.is_file():
                result[str(path.relative_to(root))] = sha(path)
    return result


def verify_helpers(campaign, binding):
    for name, digest in binding["helper_files"].items():
        need(sha(campaign / "harness" / name) == digest, "Changed frozen helper: " + name)


def initialize(repo, campaign, prerequisite, harness_archive, harness_commit):
    from research.selector_v4.public_qualification.contract import CONTRACT
    from research.selector_v4.public_qualification.fresh_manifest import freeze_manifest

    need(not campaign.exists(), "Never replace an initialized campaign")
    need(len(harness_commit) == 40, "Full committed harness identity required")
    # The producer verifies the complete prerequisite archive before generating
    # any real new geometry. A failed prerequisite leaves no new manifest.
    campaign.mkdir()
    shutil.copyfile(harness_archive, campaign / "harness.tar.gz")
    harness = campaign / "harness"
    harness.mkdir()
    with tarfile.open(harness_archive) as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            need(not path.is_absolute() and ".." not in path.parts, "Unsafe helper archive path")
            need(member.isfile() or member.isdir(), "Helper archive must contain regular files")
        bundle.extractall(harness)
    metadata = json.loads((harness / "harness-commit.json").read_text())
    need(metadata["harness_commit"] == harness_commit, "Committed helper identity")
    for name, digest in metadata["git_blob_files_sha256"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "Helper blob path")
        need(sha(harness / relative) == digest, "Committed helper blob changed: " + name)
    need(
        sha(repo / "research/selector_v4/STATUS.json")
        == sha(harness / "research/selector_v4/STATUS.json"),
        "Consumption state changed after packing",
    )
    manifest = freeze_manifest(harness, prerequisite, campaign / "manifest.json")
    need(
        manifest["families"] == {"dev": 2, "canary": 10, "release": 48, "stress": 12}, "Stage sizes"
    )
    validation = BASE / "sgi-official-wheel-validation-75544a17-20261001"
    candidate_source = validation / "wheel-overlay"
    builds = []
    for commit in (CANDIDATE, PRISTINE):
        path = BASE / f"sgi-flashinfer-official-build-{commit[:8]}-20261001/receipt.json"
        value = json.loads(path.read_text())
        need(value["build_pass"] and value["source_git_commit"] == commit, "Normal build identity")
        need(
            value["normal_upstream_hook"]
            and value["all_non_generated_wheel_package_bytes_match_prebuild_source_ledger"],
            "Normal package provenance",
        )
        need(sha(Path(value["wheel"])) == value["wheel_sha256"], "Wheel changed")
        builds.append((path, value))
    need(builds[0][1]["vendor_git_pins"] == builds[1][1]["vendor_git_pins"], "Vendor pins")
    source_gate = BASE / "sgi-tensor-control-source-prequalification/receipt.json"
    gate = json.loads(source_gate.read_text())
    need(
        gate["terminal"]
        and gate["state"] == "EXACT_TENSOR_CONTROL_DUAL90_AND_SASS_PASS"
        and gate["source_commit"] == CANDIDATE,
        "Exact source prerequisite",
    )
    original = campaign / "pristine-overlay"
    original.mkdir()
    common = {}
    with zipfile.ZipFile(builds[0][1]["wheel"]) as cz, zipfile.ZipFile(builds[1][1]["wheel"]) as pz:
        pz.extractall(original)
        names = {n for n in pz.namelist() if n.startswith("flashinfer/") and not n.endswith("/")}
        need(names <= set(cz.namelist()), "Missing pristine package files")
        import hashlib

        for name in sorted(names - {"flashinfer/prefill.py", "flashinfer/_build_meta.py"}):
            raw = pz.read(name)
            need(cz.read(name) == raw, "Native package difference: " + name)
            common[name] = hashlib.sha256(raw).hexdigest()
        prefix = pz.read("flashinfer/prefill.py")
        need(
            len(prefix) == 344213 and cz.read("flashinfer/prefill.py")[: len(prefix)] == prefix,
            "Native Python prefix",
        )
        common["flashinfer/prefill.py:native-prefix"] = hashlib.sha256(prefix).hexdigest()
    for name in ("include", "csrc"):
        (original / name).symlink_to(original / "flashinfer/data" / name, target_is_directory=True)
    (original / "source.sha256").write_text(
        "".join(
            f"{sha(p)}  {p.relative_to(original)}\n"
            for p in sorted(original.rglob("*"))
            if p.is_file()
        )
    )
    for name in ("logs", "receipts", "stages", "sdk-libraries"):
        (campaign / name).mkdir()
    for path in (validation / "sdk-libraries").iterdir():
        shutil.copyfile(path, campaign / "sdk-libraries" / path.name)
    binding = {
        "scope": SCOPE,
        "qualification_revision": REVISION,
        "manifest_sha256": sha(campaign / "manifest.json"),
        "stage_hashes": manifest["stage_hashes"],
        "harness_archive_sha256": sha(campaign / "harness.tar.gz"),
        "harness_commit": harness_commit,
        "analyzer_sha256": sha(harness / "research/selector_v4/public_qualification/analyze.py"),
        "candidate_commit": CANDIDATE,
        "pristine_commit": PRISTINE,
        "candidate_source": str(candidate_source),
        "pristine_source": str(original),
        "candidate_wheel_sha256": builds[0][1]["wheel_sha256"],
        "pristine_wheel_sha256": builds[1][1]["wheel_sha256"],
        "build_receipt_sha256": {
            commit: sha(path) for commit, (path, _) in zip((CANDIDATE, PRISTINE), builds)
        },
        "source_ledger_sha256": {
            "candidate": sha(candidate_source / "source.sha256"),
            "pristine": sha(original / "source.sha256"),
        },
        "native_common_package_files_sha256": common,
        "contract": CONTRACT,
        "source_gate_sha256": sha(source_gate),
        "prerequisite_receipt_sha256": sha(prerequisite),
        "helper_files": ledger(harness),
        "maximum_inflight_cases": 8,
        "case_allocation_limit_hours": 12,
        "finite_controller_days": 14,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    save_new(campaign / "frozen-binding.json", binding)
    verify_manifest(campaign, binding)
    verify_helpers(campaign, binding)
    return binding


def submit(root, script, arguments):
    label = str(arguments[0]) + "-" + (str(arguments[3]) if len(arguments) > 3 else "smoke")
    intent = root / f"receipts/submit-intent-{label}.json"
    save_new(
        intent, {"arguments": list(map(str, arguments)), "script_sha256": sha(script), "utc": now()}
    )
    job = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "-p",
                arguments[0],
                "-o",
                str(root / "logs/%j.out"),
                "-e",
                str(root / "logs/%j.err"),
                str(script),
                *map(str, arguments[1:]),
            ],
            text=True,
            timeout=60,
        )
        .strip()
        .split(";")[0]
    )
    need(job.isdigit(), "Actual Slurm job ID")
    save_new(
        root / f"receipts/submit-confirmed-{label}.json",
        {"job": job, "intent_sha256": sha(intent), "utc": now()},
    )
    return job


def terminal(root, job):
    """False means still running; any failed allocation stops the controller."""
    states = subprocess.check_output(
        ["sacct", "-j", job, "--format=State,ExitCode", "-n", "-X"], text=True, timeout=60
    ).strip()
    if not states or any(
        s in states for s in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED")
    ):
        return False
    need("COMPLETED" in states and "0:0" in states, "Failed allocation: " + job + " " + states)
    exit_file = root / f"receipts/exit-{job}.txt"
    need(
        exit_file.exists() and exit_file.read_text().strip() == "0",
        "Missing successful exit receipt: " + job,
    )
    return True


def execute(campaign, args, log, timeout):
    with log.open("x") as stream:
        subprocess.run(
            [str(E), *map(str, args)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(campaign / "harness/research/selector_v4/public_qualification")
                + ":"
                + str(campaign / "harness"),
            },
        )


def validation_ledger(root, campaign, binding, extra):
    files = {campaign / "harness" / n: digest for n, digest in binding["helper_files"].items()}
    files.update({p: sha(p) for p in extra})
    files.update({p: sha(p) for p in (campaign / "sdk-libraries").iterdir()})
    (root / "validation.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(files.items()))
    )


def smoke(campaign, binding, deadline):
    root = campaign / "smoke"
    root.mkdir()
    for name in ("logs", "receipts"):
        (root / name).mkdir()
    for name in ("harness", "sdk-libraries"):
        (root / name).symlink_to(campaign / name, target_is_directory=True)
    save_new(root / "binding.json", binding)
    validation_ledger(
        root, campaign, binding, [root / "binding.json", campaign / "frozen-binding.json"]
    )
    script = campaign / "harness/research/selector_v4/public_qualification/smoke.sbatch"
    jobs = {}
    for gpu in GPUS:
        jobs[gpu] = submit(root, script, [gpu, root, binding["candidate_source"]])
        # The individual immutable receipts survive a later submission failure.
        save_new(root / f"receipts/job-{gpu}.json", {"job": jobs[gpu]})
    save_new(root / "receipts/jobs.json", jobs)
    while not all(terminal(root, job) for job in jobs.values()):
        need(time.monotonic() < deadline, "Finite controller deadline")
        time.sleep(15)
    results = {}
    for gpu, job in jobs.items():
        test = ET.parse(root / f"receipts/tests-{job}.xml").getroot()
        suites = [test] if test.tag == "testsuite" else list(test)
        totals = {
            k: sum(int(s.attrib.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
        need(
            totals == {"tests": 90, "failures": 0, "errors": 0, "skipped": 0},
            "Exact ninety source tests",
        )
        target = root / f"receipts/independent-sass-{job}"
        execute(
            campaign,
            [
                campaign
                / "harness/research/selector_v4/public_qualification/runtime/audit_main_sass.py",
                root,
                job,
                target,
            ],
            root / f"logs/sass-{job}.log",
            5400,
        )
        need(json.loads((target / "receipt.json").read_text())["pass"], "Fresh smoke SASS")
        # Retain every raw receipt, all managed mode checks, source checks and SASS.
        results[gpu] = {"pass": True, "job": job, "tests": totals}
    files = {"smoke/" + n: digest for n, digest in ledger(root).items()}
    for result in results.values():
        result["files"] = files
    value = {
        k: binding[k]
        for k in (
            "scope",
            "qualification_revision",
            "manifest_sha256",
            "harness_archive_sha256",
            "candidate_commit",
            "pristine_commit",
        )
    }
    value["results"] = results
    save_new(campaign / "receipts/dual-smoke.json", value)


def dispatch_stage(campaign, stage, binding, deadline):
    verify_manifest(campaign, binding)
    verify_helpers(campaign, binding)
    ticket = build_stage_ticket(campaign, stage)
    manifest = json.loads((campaign / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c["family"] == stage]
    root = campaign / "stages" / stage
    root.mkdir()
    for name in ("runs", "logs", "receipts"):
        (root / name).mkdir()
    for name in ("harness", "sdk-libraries"):
        (root / name).symlink_to(campaign / name, target_is_directory=True)
    save_new(root / "authorization.json", ticket)
    measured = dict(binding, stage=stage, stage_hash=binding["stage_hashes"][stage], cases=cases)
    save_new(root / "binding.json", measured)
    validation_ledger(
        root,
        campaign,
        binding,
        [
            root / "binding.json",
            root / "authorization.json",
            campaign / "frozen-binding.json",
            campaign / "manifest.json",
        ],
    )
    script = campaign / "harness/research/selector_v4/public_qualification/case.sbatch"
    jobs = {gpu: {} for gpu in GPUS}
    consumption = {
        "stage": stage,
        "manifest_sha256": binding["manifest_sha256"],
        "consumed_case_ids": [],
        "events": [],
    }
    save_new(root / "receipts/consumption.json", consumption)
    active = []
    for index, case in enumerate(cases):
        while len(active) > binding["maximum_inflight_cases"] - len(GPUS):
            active = [(g, j) for g, j in active if not terminal(root, j)]
            need(time.monotonic() < deadline, "Finite controller deadline")
            if len(active) > binding["maximum_inflight_cases"] - len(GPUS):
                time.sleep(15)
        # Conservative intent before sbatch: once exposed to dispatch this case
        # remains consumed even if the second GPU submission fails.
        consumption["consumed_case_ids"].append(case["id"])
        consumption["events"].append({"case_id": case["id"], "dispatch_intent_utc": now()})
        temporary = root / "receipts/consumption.tmp"
        temporary.write_text(json.dumps(consumption, indent=2) + "\n")
        temporary.replace(root / "receipts/consumption.json")
        for gpu in GPUS:
            job = submit(
                root,
                script,
                [gpu, root, binding["candidate_source"], index, binding["pristine_source"]],
            )
            jobs[gpu][str(index)] = job
            active.append((gpu, job))
            save_new(
                root / f"receipts/dispatch-{gpu}-{index}.json",
                {"job": job, "case_id": case["id"], "utc": now()},
            )
            temporary = root / "receipts/jobs.tmp"
            temporary.write_text(json.dumps(jobs, indent=2) + "\n")
            temporary.replace(root / "receipts/jobs.json")
    while active:
        active = [(g, j) for g, j in active if not terminal(root, j)]
        need(time.monotonic() < deadline, "Finite controller deadline")
        if active:
            time.sleep(15)
    for group in jobs.values():
        for job in group.values():
            execute(
                campaign,
                [
                    campaign
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
    for gpu in GPUS:
        execute(
            campaign,
            [
                campaign / "harness/research/selector_v4/public_qualification/analyze.py",
                root,
                gpu,
                root / "analysis" / gpu,
            ],
            root / f"logs/analysis-{gpu}.log",
            7200,
        )
        summary = json.loads((root / "analysis" / gpu / "summary.json").read_text())
        verdicts[gpu] = {
            "pass": summary["pass"],
            "failed_requirements": [k for k, v in summary["requirements"].items() if not v],
            "summary_sha256": sha(root / "analysis" / gpu / "summary.json"),
        }
    save_new(root / "receipts/dual-verdict.json", verdicts)
    need(all(v["pass"] for v in verdicts.values()), stage + " dual qualification HOLD")


def run(campaign):
    binding = json.loads((campaign / "frozen-binding.json").read_text())
    # One controller per campaign, exclusive and persistent even after failure.
    save_new(campaign / "receipts/controller-start.json", {"pid": os.getpid(), "utc": now()})
    state = {
        "state": "FORMAL_SMOKE",
        "scope": SCOPE,
        "terminal": False,
        "default_promotion": False,
        "serving_promotion": False,
    }
    deadline = time.monotonic() + binding["finite_controller_days"] * 86400
    try:
        verify_helpers(campaign, binding)
        verify_manifest(campaign, binding)
        need(
            sha(campaign / "harness.tar.gz") == binding["harness_archive_sha256"],
            "Source bundle changed",
        )
        smoke(campaign, binding, deadline)
        for stage in STAGES:
            save_new(campaign / f"receipts/start-{stage}.json", {"utc": now()})
            state["state"] = stage.upper()
            dispatch_stage(campaign, stage, binding, deadline)
        state.update(state="FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED", terminal=True)
    except BaseException:
        state.update(
            state="FORMAL_QUALIFICATION_HOLD", terminal=True, traceback=traceback.format_exc()
        )
        raise
    finally:
        state["finished_utc"] = now()
        save_new(campaign / "receipts/controller-terminal.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--prerequisite", type=Path)
    parser.add_argument("--harness-archive", type=Path)
    parser.add_argument("--harness-commit")
    args = parser.parse_args()
    if args.initialize:
        need(
            all((args.repo, args.prerequisite, args.harness_archive, args.harness_commit)),
            "Initialization inputs",
        )
        initialize(
            args.repo, args.campaign, args.prerequisite, args.harness_archive, args.harness_commit
        )
    else:
        run(args.campaign)
