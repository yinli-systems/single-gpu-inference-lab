"""No partial, token-only or wrong GPU control may start Resource HTTP."""

import copy

import pytest

from research.selector_v4.continue_graph_http import GPUS, check_native_verdicts


def passed():
    return {
        gpu: {
            "state": "PASS_FIXED_SCHEDULE_DIAGNOSTIC_ONLY",
            "fixed_schedule_full_token_logprob_parity_pass": True,
            "actual_decode_graph_replay_and_input_boundary_verified": True,
            "historical_token_divergence_resolved": False,
        }
        for gpu in GPUS
    }


def test_two_complete_independent_protocol_controls_are_required():
    check_native_verdicts(passed())


@pytest.mark.parametrize(
    "fault", ["missing-card", "parity", "no-actual-graph", "historical-closure", "natural-only"]
)
def test_partial_or_relabelled_preflight_cannot_unlock_http(fault):
    value = copy.deepcopy(passed())
    if fault == "missing-card":
        value.pop("gpu_5090")
    elif fault == "parity":
        value["gpu_5090"]["fixed_schedule_full_token_logprob_parity_pass"] = False
    elif fault == "no-actual-graph":
        value["gpu_4090"]["actual_decode_graph_replay_and_input_boundary_verified"] = False
    elif fault == "historical-closure":
        value["gpu_4090"]["historical_token_divergence_resolved"] = True
    else:
        value["gpu_5090"]["state"] = "PASS_NATIVE_DIAGNOSTIC_ONLY"
    with pytest.raises(ValueError):
        check_native_verdicts(value)


@pytest.mark.parametrize("failure", ["native-parity", "kernel-hold"])
def test_failed_stage_stops_every_resource_http_successor(tmp_path, monkeypatch, failure):
    import json

    from research.selector_v4 import continue_graph_http as c

    root, native, kernel = (tmp_path / name for name in ("continuation", "native", "kernel"))
    for directory in (root, native / "receipts", kernel / "receipts"):
        directory.mkdir(parents=True)
    (kernel / "receipts/controller-terminal.json").write_text(json.dumps({"terminal": True}))
    (root / "binding.json").write_text(
        json.dumps(
            {
                "finite_days": 1,
                "native": str(native),
                "kernel": str(kernel),
                "native_jobs": {"gpu_4090": "11", "gpu_5090": "12"},
                "original_native_jobs": {"gpu_4090": "11", "gpu_5090": "12"},
            }
        )
    )
    for job in ("11", "12"):
        (native / f"receipts/exit-{job}.txt").write_text("0\n")
    monkeypatch.setattr(c, "verify", lambda binding: None)
    monkeypatch.setattr(
        c,
        "native_status",
        lambda job: (f"{job}|COMPLETED|0:0|00:01", [job, "COMPLETED", "0:0", "00:01"]),
    )
    monkeypatch.setattr(c, "archive_native", lambda *args: {"archive_sha256": "native-proof"})
    verdict = passed()["gpu_4090"]
    if failure == "native-parity":
        verdict["fixed_schedule_full_token_logprob_parity_pass"] = False
    monkeypatch.setattr(c, "observe", lambda *args, **kwargs: copy.deepcopy(verdict))
    monkeypatch.setattr(c.kernel_archive, "job_records", lambda formal: {})
    monkeypatch.setattr(c, "wait_terminal", lambda *args: None)
    monkeypatch.setattr(
        c.kernel_archive,
        "archive",
        lambda *args: {"archive_sha256": "kernel-proof", "pass_formal_kernel_qualification": False},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A failed predecessor must never activate or initialize Resource HTTP")

    monkeypatch.setattr(c, "authorize_http_training", forbidden)
    monkeypatch.setattr(c.http_pipeline, "initialize", forbidden)
    state = c.run(root)
    assert state["terminal"] and state["state"] == "GRAPH_HTTP_CONTINUATION_HOLD"
    assert (
        not state["full_http_qualified"]
        and not state["default_promotion"]
        and not state["serving_promotion"]
    )
    assert not state["historical_token_divergence_resolved"]
    assert state["error"]["type"] == "ValueError"
    assert (root / "controller-start.json").is_file()


@pytest.mark.parametrize("fault", [None, "started", "changed-protocol", "scored"])
def test_only_proven_unstarted_submission_may_have_one_successor(tmp_path, monkeypatch, fault):
    import json

    from research.selector_v4 import continue_graph_http as c

    native = tmp_path / "native"
    receipts = native / "receipts"
    receipts.mkdir(parents=True)

    def write(name, obj):
        (receipts / name).write_text(json.dumps(obj))

    write("jobs.json", {"gpu_4090": "11", "gpu_5090": "12"})
    old = [
        "sbatch",
        "--partition=gpu_4090",
        "script",
        "native",
        "model",
        "0.76",
        "module",
        "fixed_schedule_observer",
    ]
    for gpu, job in (("gpu_4090", "11"), ("gpu_5090", "12")):
        write(f"submit-confirmed-{gpu}.json", {"job": job})
        write(f"submit-intent-{gpu}.json", {"command": old})
    correction = {
        "original_job": "11",
        "successor_job": "13",
        "failure_before_batch_script": True,
        "scored_requests_started": 0,
        "gpu_measurements_started": 0,
        "failed_node_excluded": "badnode",
    }
    if fault == "scored":
        correction["scored_requests_started"] = 1
    write("unscored-startup-successor-gpu_4090.json", correction)
    write("submit-confirmed-unscored-gpu_4090.json", {"job": "13"})
    new = [old[0], "--exclude=badnode", *old[1:]]
    if fault == "changed-protocol":
        new[-3] = "0.50"
    write("submit-intent-unscored-gpu_4090.json", {"command": new})
    if fault == "started":
        (receipts / "source-pre-11.txt").write_text("even an empty precheck is a started script")
    monkeypatch.setattr(
        c, "native_status", lambda job: ("failure", [job, "FAILED", "0:53", "00:00"])
    )
    if fault is None:
        assert c.native_jobs(native) == {"gpu_4090": "13", "gpu_5090": "12"}
        assert json.loads((receipts / "jobs.json").read_text())["gpu_4090"] == "11"
    else:
        with pytest.raises(ValueError):
            c.native_jobs(native)
