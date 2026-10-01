"""Prospective stage tickets; historical diagnostic receipts cannot authorize.

No tickets are created by import. The frozen source bundle and manifest must be
identical for smoke/dev/canary/release/stress; keep every source-bound raw file.
"""

import hashlib
import json
from pathlib import Path

REVISION = "4.3.0"
SCOPE = "PUBLIC_NATIVE_RESOURCE_4_3"
STAGES = ("dev", "canary", "release", "stress")
REQUIREMENTS = (
    "native_source_identity",
    "native_sass_identity",
    "complete_independent_process_records",
    "selected_nonempty",
    "selected_geomean_at_least_1_05",
    "selected_worst_at_least_0_99",
    "selected_block_worst_at_least_0_99",
    "selected_joint_lcb_at_least_0_99",
    "whole_policy_worst_at_least_0_99",
    "native_pristine_worst_at_least_0_99",
    "native_pristine_joint_lcb_at_least_0_99",
    "actual_policy_pristine_worst_at_least_0_99",
    "actual_policy_pristine_joint_lcb_at_least_0_99",
    "controls_resolved",
    "all_modes_selected",
    "both_layouts_selected",
    "both_dtypes_selected",
    "active_clocks_stable",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_summary(campaign, stage, gpu, expected):
    campaign = Path(campaign)
    path = campaign / "stages" / stage / "analysis" / gpu / "summary.json"
    value = json.loads(path.read_text())
    need(value.get("pass") is True, f"{stage} {gpu} HOLD")
    need(
        value.get("scope") == SCOPE and value.get("qualification_revision") == REVISION,
        "Diagnostic/old-revision data cannot authorize a formal stage",
    )
    need(value.get("stage") == stage and value.get("gpu") == gpu, "Stage/GPU identity")
    for field in (
        "manifest_sha256",
        "harness_archive_sha256",
        "candidate_commit",
        "pristine_commit",
    ):
        need(value.get(field) == expected[field], "Source binding: " + field)
    need(value.get("stage_hash") == expected["stage_hashes"][stage], "Stage geometry hash")
    requirements = value.get("requirements", {})
    need(all(requirements.get(k) is True for k in REQUIREMENTS), "Incomplete safety requirements")
    root = campaign / "stages" / stage
    need(bool(value.get("files")), "Missing complete raw evidence ledger")
    for name, digest in value["files"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "Invalid evidence path")
        need(sha(root / relative) == digest, "Raw evidence changed: " + name)
    need(value.get("analyzer_sha256") == expected["analyzer_sha256"], "Analyzer identity")
    return {"summary_sha256": sha(path), "stage": stage, "gpu": gpu}


def predecessor_stages(stage):
    need(stage in STAGES, "Unknown qualification stage")
    return STAGES[: STAGES.index(stage)]


def build_stage_ticket(campaign, stage):
    """Caller persists this before dispatch; never merge/promote through it."""
    campaign = Path(campaign)
    expected = json.loads((campaign / "frozen-binding.json").read_text())
    need(
        expected.get("scope") == SCOPE and expected.get("qualification_revision") == REVISION,
        "A new frozen formal campaign is required",
    )
    need(sha(campaign / "manifest.json") == expected["manifest_sha256"], "Manifest changed")
    need(
        sha(campaign / "harness.tar.gz") == expected["harness_archive_sha256"],
        "Source bundle changed",
    )
    predecessor = predecessor_stages(stage)
    smoke = json.loads((campaign / "receipts" / "dual-smoke.json").read_text())
    need(
        smoke.get("scope") == SCOPE and smoke.get("qualification_revision") == REVISION,
        "Exact formal smoke scope required",
    )
    for field in (
        "manifest_sha256",
        "harness_archive_sha256",
        "candidate_commit",
        "pristine_commit",
    ):
        need(smoke.get(field) == expected[field], "Smoke binding: " + field)
    need(set(smoke.get("results", {})) == {"gpu_4090", "gpu_5090"}, "Both smoke GPUs required")
    for result in smoke["results"].values():
        need(
            result.get("pass") is True and bool(result.get("files")), "Smoke HOLD/missing evidence"
        )
        for name, digest in result["files"].items():
            relative = Path(name)
            need(not relative.is_absolute() and ".." not in relative.parts, "Invalid smoke path")
            need(sha(campaign / relative) == digest, "Smoke evidence changed")
    evidence = [
        verify_summary(campaign, s, gpu, expected)
        for s in predecessor
        for gpu in ("gpu_4090", "gpu_5090")
    ]
    return {
        "scope": SCOPE,
        "qualification_revision": REVISION,
        "authorized_stage": stage,
        "frozen_binding_sha256": sha(campaign / "frozen-binding.json"),
        "stage_hash": expected["stage_hashes"][stage],
        "predecessors": evidence,
        "smoke_receipt_sha256": sha(campaign / "receipts" / "dual-smoke.json"),
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }


def verify_case_ticket(stage_root, index):
    """Measure only a case dispatched under the source-bound prior-stage ticket."""
    stage_root = Path(stage_root)
    binding = json.loads((stage_root / "binding.json").read_text())
    ticket = json.loads((stage_root / "authorization.json").read_text())
    campaign = stage_root.parent.parent
    expected = json.loads((campaign / "frozen-binding.json").read_text())
    # Full raw-ledger validation occurs once before ticket publication. Request
    # processes verify its immutable links, avoiding O(all previous raw data)
    # filesystem reads in every one of nine processes for every fresh case.
    need(
        ticket.get("scope") == SCOPE and ticket.get("qualification_revision") == REVISION,
        "Missing formal stage authorization",
    )
    need(ticket.get("authorized_stage") == binding["stage"], "Wrong authorized stage")
    need(
        ticket.get("frozen_binding_sha256") == sha(campaign / "frozen-binding.json"),
        "Frozen binding changed",
    )
    need(sha(campaign / "manifest.json") == expected["manifest_sha256"], "Manifest changed")
    need(
        sha(campaign / "harness.tar.gz") == expected["harness_archive_sha256"],
        "Source bundle changed",
    )
    need(
        ticket.get("smoke_receipt_sha256") == sha(campaign / "receipts" / "dual-smoke.json"),
        "Smoke receipt changed",
    )
    predecessor = {
        (s, gpu) for s in predecessor_stages(binding["stage"]) for gpu in ("gpu_4090", "gpu_5090")
    }
    links = ticket.get("predecessors", [])
    need(
        len(links) == len(predecessor) and {(r["stage"], r["gpu"]) for r in links} == predecessor,
        "Incomplete predecessor authorization",
    )
    for link in links:
        path = campaign / "stages" / link["stage"] / "analysis" / link["gpu"] / "summary.json"
        need(sha(path) == link["summary_sha256"], "Predecessor verdict changed")
    need(binding["manifest_sha256"] == expected["manifest_sha256"], "Wrong measured manifest")
    need(binding["stage_hash"] == ticket["stage_hash"], "Wrong measured stage")
    need(
        binding["harness_archive_sha256"] == expected["harness_archive_sha256"],
        "Wrong measured source",
    )
    manifest = json.loads((campaign / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c["family"] == binding["stage"]]
    need(binding["cases"] == cases and 0 <= index < len(cases), "Case routing mismatch")
    for field in ("candidate_commit", "pristine_commit"):
        need(binding[field] == expected[field], "Wrong measured build")
    if binding["stage"] != "dev":
        dispatch = json.loads((stage_root / "receipts" / "consumption.json").read_text())
        need(cases[index]["id"] in dispatch["consumed_case_ids"], "Fresh consumption not recorded")
