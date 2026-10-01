"""One complete independent repetition of the two already exposed cases.

The original campaign remains immutable HOLD. No missing-cell continuation,
fresh geometry, retries, threshold changes, or promotion authority.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from research.selector_v4.exposed_diagnostics.public_path_contract import CONTRACT

PHASES = (
    "pristine-0",
    "train-0",
    "train-1",
    "pristine-1",
    "pristine-2",
    "train-2",
    "policy-0",
    "policy-1",
    "policy-2",
)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
MEASUREMENT_SHA = "9052d6be4a3d52b610214b744f50192da5aeaa92c18bab2513fd2cf839d35a0f"
CONTRACT_SHA = "6d873c5a52425e1114328917dc7aee3ef18070d5a14888b9893bb82f4f85bd6b"
CANDIDATE = "75544a17ce0019ca877f50354d95451ee089f859"
PRISTINE = "85744da1a397c0c4bea1a83b8345281750eb61a9"
E = "/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python"


def need(value, message):
    if not value:
        raise ValueError(message)


def hash_stream(stream):
    result = hashlib.sha256()
    while chunk := stream.read(8 << 20):
        result.update(chunk)
    return result.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hash_stream(stream)


def save(path, value):
    value = dict(value, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def status(job, reader=None):
    need(isinstance(job, str) and re.fullmatch(r"[0-9]+", job), "Actual job ID")
    raw = (
        reader(job)
        if reader
        else subprocess.check_output(
            ["sacct", "-X", "-j", job, "--format=JobID,State,ExitCode,Elapsed", "-n", "-P"],
            text=True,
            timeout=60,
        )
    )
    rows = [line.split("|") for line in raw.strip().splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == job]
    need(len(rows) == 1 and len(rows[0]) >= 4, "Unambiguous allocation accounting")
    row = rows[0]
    state = row[1].split()[0].rstrip("+")
    return {
        "job": job,
        "state": state,
        "exit_code": row[2],
        "elapsed": row[3],
        "raw": raw,
        "terminal": state not in ACTIVE,
        "completed": state == "COMPLETED" and row[2] == "0:0",
    }


def verify_archive(root):
    receipt = json.loads((root / "receipt.json").read_text())
    source = root / "raw-public-path-development.tar.gz"
    need(sha(source) == receipt["archive_sha256"], "Prior complete archive SHA")
    verify_payload(source, receipt["files"])
    return receipt


def verify_payload(source, files):
    seen = set()
    with tarfile.open(source, "r|gz") as bundle:
        for member in bundle:
            name = member.name
            need(
                name in files and name not in seen and member.isfile(),
                "Exact original archive member set",
            )
            need(
                hash_stream(bundle.extractfile(member)) == files[name],
                "Prior archive member: " + name,
            )
            seen.add(name)
    need(seen == set(files), "Complete original archive member set")


def verify_sources(campaign):
    subprocess.run(
        ["sha256sum", "-c", "validation.sha256"],
        cwd=campaign,
        stdout=subprocess.DEVNULL,
        check=True,
        timeout=1200,
    )
    binding = json.loads((campaign / "binding.json").read_text())
    for name, expected in binding["frozen_files"].items():
        need(sha(campaign / name) == expected, "Frozen helper changed: " + name)
    for kind in ("candidate", "pristine"):
        source = Path(binding[kind + "_source"])
        need(
            sha(source / "source.sha256") == binding[kind + "_source_ledger_sha256"],
            "Source ledger",
        )
        subprocess.run(
            ["sha256sum", "-c", "source.sha256"],
            cwd=source,
            stdout=subprocess.DEVNULL,
            check=True,
            timeout=1200,
        )
    return binding


def initialize(prior, prior_archive, harness_archive, output):
    need(not output.exists(), "Never replace an exposed campaign")
    original = verify_archive(prior_archive)
    need(original["controller"].get("terminal") is True, "Original controller must terminate")
    need(
        not original["pass_public_path_development"], "This repetition diagnoses the original HOLD"
    )
    binding = json.loads((prior / "binding.json").read_text())
    need(binding == original["binding"], "Original binding matches preserved archive")
    need(
        (binding["candidate_commit"], binding["pristine_commit"]) == (CANDIDATE, PRISTINE),
        "Exact original packages",
    )
    need(binding["contract"] == CONTRACT, "Original complete measurement protocol")
    cases = binding["cases"]
    need(
        len(cases) == 2 and all(c["family"] == "dev" and c["exposed_development"] for c in cases),
        "Only original exposed cases",
    )
    for group in original["jobs"].values():
        for job in group.values():
            need(status(job)["terminal"], "Original allocation still active")
    with tarfile.open(harness_archive) as bundle:
        metadata = json.load(bundle.extractfile("harness-commit.json"))
        expected = metadata["git_blob_files_sha256"]
        need(
            set(bundle.getnames()) == set(expected) | {"harness-commit.json"},
            "Full committed harness",
        )
        for name, value in expected.items():
            need(
                name.startswith("research/") and ".." not in Path(name).parts, "Safe committed path"
            )
            need(hash_stream(bundle.extractfile(name)) == value, "Committed source: " + name)
        prefix = "research/selector_v4/exposed_diagnostics/"
        need(
            expected[prefix + "public_path_prequal.py"] == MEASUREMENT_SHA,
            "Original measurement bytes",
        )
        need(
            expected[prefix + "public_path_contract.py"] == CONTRACT_SHA, "Original contract bytes"
        )
        output.mkdir()
        for directory in ("logs", "runs", "receipts", "sdk-libraries", "harness"):
            (output / directory).mkdir()
        bundle.extractall(output / "harness")
    shutil.copyfile(harness_archive, output / "harness.tar.gz")
    helpers = {
        "public_path_prequal.py": prefix + "public_path_prequal.py",
        "public_path_contract.py": prefix + "public_path_contract.py",
        "analyze_public_path_prequal.py": prefix + "analyze_public_path_prequal.py",
        "public_path_prequal.sbatch": prefix + "complete_public.sbatch",
    }
    for target, source in helpers.items():
        shutil.copyfile(output / "harness" / source, output / target)
    for name in ("audit_public_path_sass.py", "audit_main_sass.py"):
        need(
            sha(prior / name) == original["controller"]["helpers"][name],
            "Original independent SASS helper",
        )
        shutil.copyfile(prior / name, output / name)
    for source in (prior / "sdk-libraries").iterdir():
        need(source.is_file(), "SDK header file")
        shutil.copyfile(source, output / "sdk-libraries" / source.name)
    binding.update(
        harness_commit=metadata["harness_commit"],
        harness_archive_sha256=sha(harness_archive),
        original_campaign=str(prior),
        original_archive=str(prior_archive),
        original_archive_sha256=original["archive_sha256"],
        original_binding_sha256=sha(prior / "binding.json"),
        independent_complete_repetition=True,
        original_campaign_remains_hold=True,
        missing_cell_continuation=False,
        initial_time_limit_hours=12,
        finite_controller_hours=18,
        phases=list(PHASES),
        cells_per_phase=24,
        fresh_cases_consumed=0,
        release_consumed=0,
        stress_consumed=0,
        qualification_authority=False,
        canary_authority=False,
        full_http_qualified=False,
    )
    binding["frozen_files"] = {
        str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    save(output / "binding.json", binding)
    # save adds only a timestamp to the persisted binding. Hash the actual bytes.
    binding = json.loads((output / "binding.json").read_text())
    (output / "validation.sha256").write_text(
        "".join(
            f"{value}  {output / name}\n" for name, value in sorted(binding["frozen_files"].items())
        )
        + f"{sha(output / 'binding.json')}  {output / 'binding.json'}\n"
    )
    verify_sources(output)
    (output / "receipts/jobs.json").write_text("{}\n")
    return binding


def completeness(campaign, gpu, case, job):
    binding = json.loads((campaign / "binding.json").read_text())
    need((campaign / f"receipts/exit-{job}.txt").read_text().strip() == "0", "Actual process exit")
    root = campaign / f"runs/{gpu}-{job}"
    need({d.name for d in root.iterdir() if d.is_dir()} == set(PHASES), "All nine original phases")
    keys = None
    for identity in PHASES:
        role, repetition = identity.rsplit("-", 1)
        phase = root / identity
        complete = json.loads((phase / "complete.json").read_text())
        env = json.loads((phase / "environment.json").read_text())
        need(
            complete.get("complete") is True and len(complete["cells"]) == 24,
            "All 24 cells: " + identity,
        )
        current = set(complete["cells"])
        need(keys is None or keys == current, "Identical complete cell set")
        keys = current
        need(env["case"] == binding["cases"][int(case)], "Original exposed descriptor")
        need(env["role"] == role and env["rep"] == int(repetition), "Phase identity")
        expected = binding["pristine_commit"] if role == "pristine" else binding["candidate_commit"]
        need(env["source_commit"] == expected, "Normal original package identity")
    return True


def execute(campaign, args, name, timeout, deadline):
    env = {**os.environ, "PYTHONPATH": str(campaign / "harness")}
    timeout = min(timeout, deadline - time.monotonic())
    need(timeout > 0, "Finite controller deadline")
    with (campaign / "logs" / name).open("x") as log:
        subprocess.run(
            args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
        )


def run(campaign):
    controller_path = campaign / "receipts/controller.json"
    need(not controller_path.exists(), "No controller retries or resumed dispatch")
    binding = verify_sources(campaign)
    state = {
        "state": "DISPATCH_COMPLETE_EXPOSED_REPETITION",
        "terminal": False,
        "fresh_cases_consumed": 0,
        "qualification_authority": False,
        "default_promotion": False,
        "serving_promotion": False,
        "full_http_qualified": False,
        "original_campaign_remains_hold": True,
        "jobs": {},
        "analyses": {},
    }
    save(controller_path, state)
    deadline = time.monotonic() + 18 * 3600
    try:
        for gpu in ("gpu_4090", "gpu_5090"):
            state["jobs"][gpu] = {}
            for case in ("0", "1"):
                verify_sources(campaign)
                intent = campaign / f"receipts/intent-{gpu}-{case}.json"
                save(
                    intent,
                    {
                        "gpu": gpu,
                        "case": case,
                        "binding_sha256": sha(campaign / "binding.json"),
                        "dispatched": False,
                    },
                )
                raw = subprocess.check_output(
                    [
                        "sbatch",
                        "--parsable",
                        "-p",
                        gpu,
                        "--time=12:00:00",
                        "-o",
                        str(campaign / "logs/%j.out"),
                        "-e",
                        str(campaign / "logs/%j.err"),
                        str(campaign / "public_path_prequal.sbatch"),
                        str(campaign),
                        binding["candidate_source"],
                        case,
                        binding["pristine_source"],
                    ],
                    text=True,
                    timeout=120,
                ).strip()
                job = raw.split(";")[0]
                need(re.fullmatch(r"[0-9]+", job), "Actual submitted job ID")
                state["jobs"][gpu][case] = job
                (campaign / "receipts/jobs.json").write_text(
                    json.dumps(state["jobs"], indent=2) + "\n"
                )
                save(intent, {"gpu": gpu, "case": case, "job": job, "dispatched": True, "raw": raw})
                save(controller_path, state)
        state["state"] = "WAIT_ALL_FOUR_COMPLETE_EXPOSED_JOBS"
        save(controller_path, state)
        ids = [j for group in state["jobs"].values() for j in group.values()]
        while True:
            records = {job: status(job) for job in ids}
            if all(record["terminal"] for record in records.values()):
                break
            need(
                time.monotonic() < deadline,
                "Finite 18h controller deadline; known jobs remain recorded",
            )
            time.sleep(30)
        state["statuses"] = records
        save(controller_path, state)
        need(
            all(record["completed"] for record in records.values()),
            "Every allocation must be COMPLETED 0:0",
        )
        for gpu, group in state["jobs"].items():
            for case, job in group.items():
                completeness(campaign, gpu, case, job)
        verify_sources(campaign)
        state["state"] = "INDEPENDENT_PRISTINE_NATIVE_RESOURCE_SASS"
        save(controller_path, state)
        for job in ids:
            need(time.monotonic() < deadline, "Finite completion deadline")
            target = campaign / f"receipts/independent-sass-{job}"
            execute(
                campaign,
                [E, str(campaign / "audit_public_path_sass.py"), str(campaign), job, str(target)],
                f"sass-{job}.log",
                5400,
                deadline,
            )
            need(
                json.loads((target / "receipt.json").read_text())["pass"], "Independent exact SASS"
            )
        for gpu in state["jobs"]:
            need(time.monotonic() < deadline, "Finite analysis deadline")
            target = campaign / f"analysis/{gpu}"
            target.parent.mkdir(exist_ok=True)
            execute(
                campaign,
                [
                    E,
                    str(campaign / "analyze_public_path_prequal.py"),
                    str(campaign),
                    gpu,
                    str(target),
                ],
                f"analysis-{gpu}.log",
                3600,
                deadline,
            )
            summary = json.loads((target / "summary.json").read_text())
            for name, expected in summary["files"].items():
                need(sha(campaign / name) == expected, "Analysis raw member: " + name)
            state["analyses"][gpu] = dict(
                **{"pass": summary["pass"]},
                summary_sha256=sha(target / "summary.json"),
                metrics=summary["metrics"],
                failed_requirements=[k for k, v in summary["requirements"].items() if not v],
            )
            save(controller_path, state)
        verify_sources(campaign)
        passed = all(value["pass"] for value in state["analyses"].values())
        state.update(
            state="PUBLIC_PATH_DEVELOPMENT_PASS" if passed else "PUBLIC_PATH_DEVELOPMENT_HOLD",
            terminal=True,
        )
    except BaseException as error:  # noqa: BLE001 - persist interrupt/failure; no retry
        state.update(
            state="PUBLIC_PATH_DEVELOPMENT_COMPLETION_HOLD",
            terminal=True,
            error={"type": type(error).__name__, "message": str(error)},
        )
    save(controller_path, state)
    return state


def archive(campaign, output):
    need(not output.exists(), "Never replace a complete evidence archive")
    controller = json.loads((campaign / "receipts/controller.json").read_text())
    need(controller.get("terminal") is True, "Wait for finite controller")
    jobs = json.loads((campaign / "receipts/jobs.json").read_text())
    for intent in (campaign / "receipts").glob("intent-*.json"):
        need(
            json.loads(intent.read_text()).get("dispatched") is True,
            "Unknown submission outcome requires reconciliation",
        )
    statuses = {}
    for group in jobs.values():
        for job in group.values():
            statuses[job] = status(job)
            need(statuses[job]["terminal"], "Known job still active")
    binding = verify_sources(campaign)
    complete = len(statuses) == 4 and all(v["completed"] for v in statuses.values())
    completeness_errors = {}
    if complete:
        for gpu, group in jobs.items():
            for case, job in group.items():
                try:
                    completeness(campaign, gpu, case, job)
                except (ValueError, OSError, KeyError) as error:
                    completeness_errors[job] = {"type": type(error).__name__, "message": str(error)}
        complete = not completeness_errors
    passed = complete and controller["state"] == "PUBLIC_PATH_DEVELOPMENT_PASS"
    need(
        not passed or set(controller["analyses"]) == {"gpu_4090", "gpu_5090"},
        "Both complete dual analyses",
    )
    if passed:
        for gpu, value in controller["analyses"].items():
            summary_path = campaign / f"analysis/{gpu}/summary.json"
            need(sha(summary_path) == value["summary_sha256"], "Original complete summary SHA")
            summary = json.loads(summary_path.read_text())
            need(
                summary["pass"] and all(summary["requirements"].values()), "Every development gate"
            )
            need(
                summary["binding_sha256"] == sha(campaign / "binding.json"),
                "Analysis source binding",
            )
            for name, expected in summary["files"].items():
                need(sha(campaign / name) == expected, "All original raw analysis members")
        for job in statuses:
            root = campaign / f"receipts/independent-sass-{job}"
            sass = json.loads((root / "receipt.json").read_text())
            need(
                sass["pass"] and sass["binding_sha256"] == sha(campaign / "binding.json"),
                "Independent exact SASS binding",
            )
            need(
                sha(root / "raw-sass.tar.gz") == sass["raw_archive_sha256"],
                "Original raw SASS archive",
            )
    paths = {}
    for parent, folders, names in os.walk(campaign):
        folders[:] = sorted(
            d
            for d in folders
            if d not in {"cache", "sdk-libraries", "__pycache__", ".pytest_cache"}
        )
        for name in sorted(names):
            path = Path(parent) / name
            if path.is_file():
                paths[str(path.relative_to(campaign))] = path
    files = {name: sha(path) for name, path in paths.items()}
    output.mkdir()
    target = output / "raw-public-path-development.tar.gz"
    with tarfile.open(target, "w:gz") as bundle:
        for name, path in sorted(paths.items()):
            bundle.add(path, arcname=name, recursive=False)
    verify_payload(target, files)
    receipt = {
        "pass_public_path_development": passed,
        "all_nine_phases_complete_on_both_cases_and_cards": complete,
        "binding": binding,
        "controller": controller,
        "jobs": jobs,
        "statuses": statuses,
        "completeness_errors": completeness_errors,
        "analyses": controller["analyses"],
        "files": files,
        "archive_sha256": sha(target),
        "archive_bytes": target.stat().st_size,
        "raw_references_included": True,
        "trimmed_windows": 0,
        "fresh_cases_consumed": 0,
        "qualification_authority": False,
        "canary_authority": False,
        "full_http_qualified": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    save(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    for name in ("prior", "prior_archive", "harness_archive", "output"):
        init.add_argument(name, type=Path)
    controller = sub.add_parser("run")
    controller.add_argument("campaign", type=Path)
    archival = sub.add_parser("archive")
    archival.add_argument("campaign", type=Path)
    archival.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.prior, args.prior_archive, args.harness_archive, args.output)
    elif args.command == "run":
        result = run(args.campaign)
    else:
        result = archive(args.campaign, args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "state",
                    "terminal",
                    "harness_commit",
                    "jobs",
                    "archive_sha256",
                    "pass_public_path_development",
                )
            }
        )
    )
