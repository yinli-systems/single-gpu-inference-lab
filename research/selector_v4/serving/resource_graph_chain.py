"""One finite, no-retry full-model Graph diagnostic chain after real gates.

Four models, dual GPU, maximum eight allocations of two hours. An exclusive
start receipt prevents duplicate dispatch. This never authorizes promotion.
"""

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.http_measure import save_new
from research.selector_v4.serving.http_pipeline import authorize
from research.selector_v4.serving.resource_graph_contract import resource_trace
from research.selector_v4.serving.resource_graph_http import MODELS, SCOPE, audit_run

GPUS = ("gpu_4090", "gpu_5090")
RUNNING = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def status(job):
    raw = subprocess.check_output(
        ["sacct", "-X", "-j", job, "-n", "-P", "--format=JobID,State,ExitCode,Elapsed"],
        text=True,
        timeout=30,
    )
    rows = [r.split("|") for r in raw.splitlines() if r.split("|")[0] == job]
    if not rows:  # Accounting propagation delay never grants completion.
        return None
    need(len(rows) == 1 and len(rows[0]) == 4, "Unambiguous exact job accounting")
    return dict(zip(("job", "state", "exit", "elapsed"), rows[0]))


def gate_state(kernel, component, *, account=status):
    terminal = kernel / "receipts/controller-terminal.json"
    if terminal.exists():
        value = json.loads(terminal.read_text())
        if (
            value.get("terminal")
            and value.get("state") != "FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED"
        ):
            raise ValueError(
                "Formal chain terminal HOLD; Resource HTTP remains forbidden"
            )
    for gpu, job in component["jobs"].items():
        row = account(job)
        if row is None or row["state"] in RUNNING:
            return "WAIT_COMPONENT_AND_FORMAL_GATES"
        need(
            row["state"] == "COMPLETED" and row["exit"] == "0:0",
            "Component terminal failure",
        )
        path = Path(component["root"]) / "runs" / (gpu + "-" + job) / "complete.json"
        need(path.is_file(), "A successful allocation alone is not a component PASS")
    if not terminal.exists():
        return "WAIT_ALL_EIGHT_FORMAL_KERNEL_VERDICTS"
    value = json.loads(terminal.read_text())
    if not value.get("terminal"):
        return "WAIT_ALL_EIGHT_FORMAL_KERNEL_VERDICTS"
    return "VERIFY_RAW_GATES"


def validate_component(component):
    root = Path(component["root"])
    need(
        sha(root / "harness.tar.gz") == component["archive_sha256"],
        "Original component bundle changed",
    )
    for gpu, job in component["jobs"].items():
        run = root / "runs" / (gpu + "-" + job)
        value = json.loads((run / "complete.json").read_text())
        need(
            value["pass"] is True
            and value["actual_physical_stale_update_rejected"] is True
            and value["resource_kernels_after_stale"] == 0
            and len(value["epochs"]) == 3,
            "Complete real component proof",
        )
        need(
            value["scope"] == "GPU_COMPONENT_ONLY_NOT_SGLANG_HTTP",
            "Actual component scope",
        )
        for epoch in range(3):
            proof = resource_trace(
                json.loads((run / f"epoch-{epoch}.json").read_text())["traceEvents"]
            )
            need(
                proof["actual_resource_graph_kernels"] == 1,
                "Original CUPTI component evidence",
            )
        events = json.loads((run / "stale-rejection.json").read_text())["traceEvents"]
        need(
            not any(
                e.get("cat") == "kernel" and "ResourceKernel" in e.get("name", "")
                for e in events
            ),
            "Stale rejection must execute no Resource kernel",
        )
    return {
        gpu: sha(root / "runs" / (gpu + "-" + job) / "complete.json")
        for gpu, job in component["jobs"].items()
    }


def verify(root):
    proof = json.loads((root / "frozen.json").read_text())
    need(
        sha(root / "validation.sha256") == proof["ledger_sha256"],
        "Frozen ledger changed",
    )
    subprocess.run(
        ["sha256sum", "-c", str(root / "validation.sha256")],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=300,
    )


def run(root):
    binding = json.loads((root / "binding.json").read_text())
    save_new(
        root / "controller-start.json",
        {"pid": os.getpid(), "utc": utc(), "automatic_retries": 0},
    )
    active = []
    result = {
        "terminal": False,
        "scope": SCOPE,
        "state": "STARTING",
        "models": {},
        "full_http_qualified": False,
        "resource_graph_serving_qualified": False,
        "historical_token_divergence_resolved": False,
        "default_promotion": False,
        "serving_promotion": False,
    }
    deadline = binding["deadline_unix"]

    def write():
        result["checked_at_utc"] = utc()
        tmp = root / "controller.tmp"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(root / "controller.json")

    def within_budget():
        need(time.time() < deadline, "Finite preregistered controller deadline")

    try:
        verify(root)
        kernel = Path(binding["kernel"])
        while True:
            within_budget()
            result["state"] = gate_state(kernel, binding["component"])
            write()
            if result["state"] == "VERIFY_RAW_GATES":
                break
            time.sleep(60)
        verify(root)
        authorization = authorize(kernel)
        components = validate_component(binding["component"])
        save_new(
            root / "receipts/authorization.json",
            {
                "kernel": authorization,
                "component_complete_sha256": components,
                "utc": utc(),
            },
        )
        for model in MODELS:
            within_budget()
            verify(root)
            authorize(kernel)
            jobs = {}
            result["state"] = "RUNNING_MODEL_" + model
            for gpu in GPUS:
                save_new(root / f"receipts/intent-{model}-{gpu}.json", {"utc": utc()})
                job = (
                    subprocess.check_output(
                        [
                            "sbatch",
                            "--parsable",
                            "--exclude=wqd10nba06g6",
                            "-p",
                            gpu,
                            "-o",
                            str(root / "logs/%j.out"),
                            "-e",
                            str(root / "logs/%j.err"),
                            str(root / "harness/resource_graph_http.sbatch"),
                            str(root),
                            model,
                        ],
                        text=True,
                        timeout=60,
                    )
                    .strip()
                    .split(";")[0]
                )
                need(
                    job.isdigit(),
                    "Unambiguous job ID; unknown submission never retried",
                )
                active.append(job)
                save_new(
                    root / f"receipts/job-{model}-{gpu}.json",
                    {"job": job, "utc": utc()},
                )
                scontrol = subprocess.check_output(
                    ["scontrol", "show", "job", job], text=True, timeout=30
                )
                (root / f"receipts/scontrol-{job}.txt").write_text(scontrol)
                need(
                    "ExcNodeList=wqd10nba06g6" in scontrol,
                    "Actual explicit excluded node",
                )
                jobs[gpu] = job
            result["models"][model] = {"jobs": jobs, "state": "RUNNING"}
            write()
            while True:
                within_budget()
                rows = {gpu: status(job) for gpu, job in jobs.items()}
                if all(
                    row is not None and row["state"] not in RUNNING
                    for row in rows.values()
                ):
                    break
                time.sleep(30)
            for gpu, row in rows.items():
                save_new(root / f"receipts/terminal-{row['job']}.json", row)
                need(
                    row["state"] == "COMPLETED" and row["exit"] == "0:0",
                    "Original model attempt failed",
                )
                run_root = root / "runs" / (gpu + "-" + row["job"])
                final = json.loads((run_root / "complete.json").read_text())
                need(
                    final["pass"] is True and final["scope"] == SCOPE,
                    "Actual full model functional proof",
                )
                need(
                    len(final["comparisons"]) == 6
                    and all(
                        c["parity_pass"] is True
                        and c["tokens_compared"] == 8
                        and c["full_logprobs_required"] is True
                        for c in final["comparisons"]
                    ),
                    "All complete HTTP cases and full logprobs",
                )
                model_binding = json.loads(
                    (root / "models" / (model + ".json")).read_text()
                )
                raw_audit = audit_run(run_root, model_binding)
                save_new(root / f"receipts/raw-audit-{row['job']}.json", raw_audit)
            verify(root)
            active.clear()
            result["models"][model]["state"] = "DUAL_FUNCTIONAL_PASS"
            write()
        result.update(
            terminal=True, state="FOUR_MODEL_DUAL_FIXED_GRAPH_FUNCTIONAL_PASS"
        )
    except BaseException as error:  # noqa: BLE001 - persist original failure and stop owned jobs
        result.update(
            terminal=True,
            state="HOLD",
            error=str(error),
            traceback=traceback.format_exc(),
        )
        # Only jobs dispatched by this finite owned controller may be stopped.
        for job in active:
            row = status(job)
            if row is None or row["state"] in RUNNING:
                cancellation = subprocess.run(
                    ["scancel", job],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                save_new(
                    root / f"receipts/cancel-{job}.json",
                    {
                        "reason": "finite chain stopped",
                        "returncode": cancellation.returncode,
                        "stderr": cancellation.stderr,
                    },
                )
    write()
    save_new(root / "controller-terminal.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
