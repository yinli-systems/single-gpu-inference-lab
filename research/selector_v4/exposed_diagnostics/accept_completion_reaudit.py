"""Accept corrected complete development data in a separate immutable branch.

This never rewrites the original controller, archive, measurement, decision or
gate. Only the separately reviewed reference-directory bookkeeping correction
and unchanged complete-data analyses can supply a new development prerequisite.
Historical canary HOLD, stronger unmet regret targets and promotion remain open.
"""

import argparse
import copy
import importlib.util
import json
import os
import shutil
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha


def accepted_analyses(reaudit):
    result = json.loads((reaudit / "result.json").read_text())
    need(
        result["pass_complete_data_development_reaudit"] is True
        and result["measurement_or_threshold_changes"] is False
        and result["gpu_jobs_dispatched"] == result["fresh_cases_consumed"] == 0,
        "Complete unchanged corrective analysis required",
    )
    need(set(result["summaries"]) == {"gpu_4090", "gpu_5090"}, "Both original GPU analyses")
    analyses = {}
    for gpu, proof in result["summaries"].items():
        root = reaudit / "verified-inputs"
        path = root / f"analysis/{gpu}/summary.json"
        need(sha(path) == proof["sha256"], "Exact unchanged analysis summary")
        summary = json.loads(path.read_text())
        need(
            summary["pass"] is True and all(summary["requirements"].values()),
            "Every original frozen development requirement",
        )
        need(summary["binding_sha256"] == sha(root / "binding.json"), "Original analysis binding")
        need(summary["metrics"] == proof["metrics"], "Every original metric retained")
        for name, digest in summary["files"].items():
            need(sha(root / name) == digest, "Original complete analysis member")
        analyses[gpu] = {
            "pass": True,
            "summary_sha256": sha(path),
            "metrics": summary["metrics"],
            "failed_requirements": [],
        }
    return analyses


def accept(reaudit, campaign, original_archive, corrected_helper, output, verifier_commit):
    need(not output.exists() and len(verifier_commit) == 40, "New independently committed acceptance")
    spec = importlib.util.spec_from_file_location("reviewed_completeness", corrected_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    integrity = json.loads((reaudit / "input-integrity.json").read_text())
    need(sha(corrected_helper) == integrity["corrected_helper_sha256"], "Only reviewed checker fix")
    original = helper.verify_archive(original_archive)
    need(
        original["archive_sha256"] == integrity["original_archive_sha256"]
        and not original["pass_public_path_development"],
        "Exact preserved original HOLD archive",
    )
    need(
        sha(campaign / "receipts/controller.json") == integrity["old_controller_sha256"],
        "Original terminal controller unchanged",
    )
    for name, digest in original["files"].items():
        need(sha(campaign / name) == digest, "Original HOLD input changed")
    helper.verify_sources(campaign)
    analyses = accepted_analyses(reaudit)
    shadow = reaudit / "verified-inputs"
    for gpu, cases in original["jobs"].items():
        for case, job in cases.items():
            helper.completeness(shadow, gpu, case, job)
    output.mkdir()
    inputs = output / "accepted-inputs"

    def redundant_sass_view(directory, names):
        # All original .sass bytes remain in Q and inside the SHA-bound original
        # raw-sass.tar.gz. Omit only their redundant unpacked view from this
        # branch; no disassembly or measurement is removed from the evidence.
        if Path(directory).name.startswith("independent-sass-"):
            return {"raw"} & set(names)
        return {"cache", "sdk-libraries", "__pycache__", ".pytest_cache"} & set(names)

    shutil.copytree(
        shadow,
        inputs,
        copy_function=os.link,
        ignore=redundant_sass_view,
    )
    (inputs / "cache").symlink_to(campaign / "cache", target_is_directory=True)
    (inputs / "sdk-libraries").symlink_to(campaign / "sdk-libraries", target_is_directory=True)
    controller_path = inputs / "receipts/controller.json"
    old = json.loads(controller_path.read_text())
    shutil.copyfile(controller_path, inputs / "receipts/original-controller-hold.json")
    # Critical: unlink the new branch's hardlink before writing. The original
    # terminal HOLD and the independent reanalysis snapshot must stay untouched.
    controller_path.unlink()
    controller = copy.deepcopy(old)
    controller.pop("error", None)
    controller.update(
        state="PUBLIC_PATH_DEVELOPMENT_PASS",
        terminal=True,
        analyses=analyses,
        correction_scope="Reference directory bookkeeping; complete original data and gates unchanged",
        original_terminal_hold_retained=True,
        original_controller_sha256=integrity["old_controller_sha256"],
        original_archive_sha256=original["archive_sha256"],
        independent_reaudit_result_sha256=sha(reaudit / "result.json"),
        corrected_completeness_helper_sha256=sha(corrected_helper),
        acceptance_verifier_commit=verifier_commit,
        stronger_p99_regret_target_satisfied=False,
        original_sass_archives_preserved=True,
        redundant_unpacked_sass_view_retained_at=str(shadow),
        default_promotion=False,
        serving_promotion=False,
        historical_token_divergence_resolved=False,
    )
    helper.save(controller_path, controller)
    shutil.copyfile(reaudit / "result.json", inputs / "receipts/independent-complete-reaudit.json")
    shutil.copyfile(reaudit / "input-integrity.json", inputs / "receipts/independent-input-integrity.json")
    shutil.copyfile(corrected_helper, inputs / "receipts/reviewed-completeness-helper.py")
    shutil.copyfile(Path(__file__), inputs / "receipts/acceptance-verifier.py")
    receipt = helper.archive(inputs, output / "archive")
    need(receipt["pass_public_path_development"], "Independent corrected development prerequisite")
    need(
        sha(campaign / "receipts/controller.json") == integrity["old_controller_sha256"]
        and sha(shadow / "receipts/controller.json") == integrity["old_controller_sha256"],
        "Original and reanalysis controller HOLD still immutable",
    )
    helper.verify_archive(original_archive)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reaudit", "campaign", "original-archive", "corrected-helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verifier-commit", required=True)
    a = parser.parse_args()
    r = accept(a.reaudit, a.campaign, a.original_archive, a.corrected_helper, a.output, a.verifier_commit)
    print(json.dumps({"development_prerequisite_pass": r["pass_public_path_development"], "archive_sha256": r["archive_sha256"]}))
