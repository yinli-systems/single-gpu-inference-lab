import json

import pytest

from research.selector_v4.exposed_diagnostics.accept_completion_reaudit import accepted_analyses
from research.selector_v4.public_qualification.gates import sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    inputs = root / "verified-inputs"
    save(inputs / "binding.json", {"original": True})
    raw = inputs / "original-cell.json"
    save(raw, {"all_windows": [1, 2, 3]})
    result = {
        "pass_complete_data_development_reaudit": True,
        "measurement_or_threshold_changes": False,
        "gpu_jobs_dispatched": 0,
        "fresh_cases_consumed": 0,
        "summaries": {},
    }
    for gpu in ("gpu_4090", "gpu_5090"):
        summary = {
            "pass": True,
            "requirements": {"unchanged_test_gate": True},
            "binding_sha256": sha(inputs / "binding.json"),
            "metrics": {"original": 1.0},
            "files": {"original-cell.json": sha(raw)},
        }
        path = inputs / f"analysis/{gpu}/summary.json"
        save(path, summary)
        result["summaries"][gpu] = {"sha256": sha(path), "metrics": summary["metrics"]}
    save(root / "result.json", result)
    return result


def test_both_complete_analyses_keep_every_original_metric(tmp_path):
    fixture(tmp_path)
    values = accepted_analyses(tmp_path)
    assert set(values) == {"gpu_4090", "gpu_5090"}
    assert all(v["pass"] and v["metrics"] == {"original": 1.0} for v in values.values())


@pytest.mark.parametrize("fault", ["hold", "changed-gates", "missing-gpu", "raw-edit", "metric-edit"])
def test_failed_or_edited_reanalysis_cannot_authorize_new_development(tmp_path, fault):
    result = fixture(tmp_path)
    if fault == "hold":
        result["pass_complete_data_development_reaudit"] = False
    elif fault == "changed-gates":
        result["measurement_or_threshold_changes"] = True
    elif fault == "missing-gpu":
        result["summaries"].pop("gpu_5090")
    elif fault == "raw-edit":
        save(tmp_path / "verified-inputs/original-cell.json", {"all_windows": [1]})
    else:
        result["summaries"]["gpu_4090"]["metrics"]["original"] = 2.0
    save(tmp_path / "result.json", result)
    with pytest.raises(ValueError):
        accepted_analyses(tmp_path)
