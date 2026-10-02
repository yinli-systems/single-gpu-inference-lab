"""Reconstruct actual serving Graph launches from CPU/API/GPU correlations."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from research.release_qualification.serving.workloads import prefix_tokens
from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.serving.metric_gate import WORKLOADS
from research.selector_v4.serving.parity_gate import compare_blocks
from research.selector_v4.serving.slo_report import summarize
from research.selector_v4.serving.workload_modes import PARITY_SCOPE, verify_configuration
from research.selector_v4.serving.workload_modes import specs as mode_specs


def correlated_decode(events, *, allow_eager_resource=False):
    markers = [
        e
        for e in events
        if e.get("name") == "sgi_actual_decode_graph_step"
        and e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
    ]
    launches = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "GraphLaunch" in e.get("name", "")
        and e.get("ph") == "X"
    ]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    need(markers and launches and kernels, "Actual serving CPU/API/GPU events required")
    need(
        all(
            allow_eager_resource and e.get("args", {}).get("graph id") == 0
            for e in kernels
            if "ResourceKernel" in e.get("name", "")
        ),
        "Native diagnostic forbids Resource; qualified serving allows only eager Resource",
    )
    correlated = 0
    attention = 0
    for marker in markers:
        need(marker["dur"] > 0, "Complete Graph step marker")
        apis = [
            e
            for e in launches
            if e["pid"] == marker["pid"]
            and e["tid"] == marker["tid"]
            and marker["ts"] <= e["ts"] <= marker["ts"] + marker["dur"]
        ]
        ids = {e.get("args", {}).get("correlation") for e in apis}
        ids.discard(None)
        linked = [
            e
            for e in kernels
            if e.get("args", {}).get("correlation") in ids
            and "ResourceKernel" not in e.get("name", "")
            and any(
                name in e.get("name", "")
                for name in ("BatchDecodeWithPagedKV", "BatchPrefillWithPagedKV")
            )
        ]
        need(
            linked, "Every traced decode step requires an actual correlated Graph attention kernel"
        )
        need(
            all(e["args"].get("graph id", 0) > 0 for e in linked),
            "Actual kernel graph IDs required",
        )
        correlated += len(apis)
        attention += len(linked)
    return {
        "traced_decode_steps": len(markers),
        "correlated_graph_launch_api_calls": correlated,
        "actual_correlated_native_attention_kernels": attention,
    }


def natural_parity(root, *, require_logprobs=False):
    rows = []
    for block in range(4):
        for name in WORKLOADS:
            key = f"{name}-b{block}.json"
            values = [
                json.loads((root / role / "observations" / key).read_text())
                for role in ("native_baseline", "graph_diagnostic")
            ]
            rows.append(
                {"block": key, **compare_blocks(*values, require_logprobs=require_logprobs)}
            )
    return rows


def observe(root, *, fixed_schedule=False):
    complete = json.loads((root / "complete.json").read_text())
    field = "comparison" if fixed_schedule else "native_instrumentation_token_parity"
    stage = "fixed_parity" if fixed_schedule else "functional"
    need(
        complete["complete"] and len(complete[field]) == 20,
        "Both complete HTTP matrices",
    )
    if fixed_schedule:
        need(
            complete["mode"] == "fixed_schedule_observer"
            and complete["parity_scope"] == PARITY_SCOPE,
            "Declared separately scoped cached parity diagnostic",
        )
    for role in ("native_baseline", "graph_diagnostic"):
        arm = root / role
        done = json.loads((arm / "complete.json").read_text())
        need(done["complete"], "Actual complete HTTP arm")
        for name, digest in done["files"].items():
            relative = Path(name)
            need(
                not relative.is_absolute() and ".." not in relative.parts,
                "Safe diagnostic raw path",
            )
            need(sha(arm / name) == digest, "Original HTTP/Graph artifact changed")
        verify_configuration(json.loads((arm / "server-info.json").read_text()), stage)
        if fixed_schedule:
            command = json.loads((arm / "command.json").read_text())
            need(
                command["fixed_single_request_parity"]
                and command["complete_output_and_top5_logprobs"],
                "Both arms require complete fixed-schedule token/logprob trials",
            )
        reports = json.loads((arm / "descriptive-slo.json").read_text())
        for block in range(4):
            specs = mode_specs(prefix_tokens(), block, stage=stage)
            for name in WORKLOADS:
                key = f"{name}-b{block}.json"
                raw = json.loads((arm / "observations" / key).read_text())
                need(
                    summarize(raw, specs[name]) == reports["blocks"][key],
                    "Independently reconstruct every original HTTP/SLO observation",
                )
    parity = natural_parity(root, require_logprobs=fixed_schedule)
    need(
        parity == complete[field],
        "Parity receipt must equal all original complete token/logprob streams",
    )
    parity_pass = all(p["parity_pass"] for p in parity)
    arm = root / "graph_diagnostic"
    records = []
    for path in sorted((arm / "graph-epochs").glob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    need(records, "Actual scheduler-loaded GPU input epochs required")
    groups = defaultdict(list)
    seen = set()
    last = {}
    for row in records:
        key = (row["pid"], row["graph_key"])
        epoch = (row["pid"], row["load_notifications"])
        need(epoch not in seen, "Distinct original load epochs")
        need(row["load_notifications"] > last.get(row["pid"], 0), "Actual load notification order")
        seen.add(epoch)
        last[row["pid"]] = row["load_notifications"]
        need(
            row["metadata_depth_before_replay"] == row["metadata_depth_after_replay"] == 0,
            "Replay outside metadata writer",
        )
        need(
            row["timing_valid"] is False and row["resource_tactic_selected"] is False,
            "Separate Native-only diagnostic",
        )
        groups[key].append(row)
    witnessed = 0
    for rows in groups.values():
        identities = {
            (
                r["graph_object_id"],
                tuple((k, v["pointer"], tuple(v["shape"])) for k, v in sorted(r["inputs"].items())),
            )
            for r in rows
        }
        need(len(identities) == 1, "Original captured objects/buffers unchanged")
        payloads = {
            tuple((k, v["actual_gpu_payload_sha256"]) for k, v in sorted(r["inputs"].items()))
            for r in rows
        }
        if len(payloads) >= 3:
            witnessed += 1
    need(witnessed > 0, "At least three actual GPU input payload epochs on one unchanged capture")
    proofs = []
    timing = json.loads((arm / "observations/profile-timing-scope.json").read_text())
    need(timing["timing_excluded"], "Profile cannot supply performance measurements")
    for name, digest in timing["trace_files"].items():
        p = arm / name
        need(sha(p) == digest, "Original trace hash")
        if p.name.endswith(".json.gz"):
            with gzip.open(p, "rt") as stream:
                data = json.load(stream)
        elif p.name.endswith(".json"):
            data = json.loads(p.read_text())
        else:
            continue
        if any(
            e.get("name") == "sgi_actual_decode_graph_step" for e in data.get("traceEvents", [])
        ):
            proofs.append(correlated_decode(data["traceEvents"]))
    need(proofs, "Independent correlated serving profile required")
    result = {
        "pass_native_graph_serving_boundary_diagnostic": parity_pass,
        "state": "PASS_NATIVE_DIAGNOSTIC_ONLY" if parity_pass else "HOLD_NATURAL_TOKEN_PARITY",
        "actual_decode_graph_replay_and_input_boundary_verified": True,
        "natural_token_parity_pass": parity_pass,
        "natural_token_parity": parity,
        "actual_loaded_gpu_epochs": len(records),
        "unchanged_capture_groups_with_three_changed_payloads": witnessed,
        "profiles": proofs,
        "full_http_qualified": False,
        "resource_graph_serving_qualified": False,
        "independent_baseline_allocations": 1,
        "serving_gain_or_confidence_claim": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    if fixed_schedule:
        result.pop("natural_token_parity_pass")
        result.pop("natural_token_parity")
        result.update(
            state="PASS_FIXED_SCHEDULE_DIAGNOSTIC_ONLY"
            if parity_pass
            else "HOLD_FIXED_SCHEDULE_PARITY",
            parity_scope=PARITY_SCOPE,
            fixed_schedule_full_token_logprob_parity_pass=parity_pass,
            fixed_schedule_parity=parity,
            original_natural_parity_failure_closed=False,
        )
    return result


def audit(root):
    result = observe(root)
    need(
        result["natural_token_parity_pass"],
        "All natural Native/instrumentation tokens required; no discrepancy removal",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--observe", action="store_true", help="Retain proofs and parity HOLD")
    a = parser.parse_args()
    print(json.dumps((observe if a.observe else audit)(a.root), indent=2))
