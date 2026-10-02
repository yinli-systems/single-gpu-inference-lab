"""Only actual pre-script infrastructure failures admit a separate dev branch."""

import pytest

from research.selector_v4.repair_unstarted_formal import copy_frozen_tree, unstarted


def test_signal_failure_without_any_started_work_is_eligible(tmp_path):
    (tmp_path / "logs").mkdir()
    unstarted(tmp_path, "gpu_4090", "11", ["11", "FAILED", "0:53", "00:00", "badnode"], "badnode")


@pytest.mark.parametrize(
    "path",
    [
        "runs/gpu_4090-11",
        "receipts/source-verification-11.txt",
        "receipts/validation-verification-11.txt",
        "receipts/exit-11.txt",
        "logs/11-pristine-0.log",
        "logs/telemetry-11.csv",
    ],
)
def test_any_started_script_or_measurement_blocks_replacement(tmp_path, path):
    (tmp_path / "logs").mkdir()
    p = tmp_path / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    with pytest.raises(ValueError, match="started"):
        unstarted(
            tmp_path, "gpu_4090", "11", ["11", "FAILED", "0:53", "00:00", "badnode"], "badnode"
        )


@pytest.mark.parametrize(
    "fields",
    [
        ["11", "FAILED", "1:0", "00:01", "badnode"],
        ["11", "COMPLETED", "0:0", "00:01", "badnode"],
        ["11", "FAILED", "0:53", "00:00", "different"],
    ],
)
def test_scientific_or_other_node_failure_cannot_be_relabelled(tmp_path, fields):
    (tmp_path / "logs").mkdir()
    with pytest.raises(ValueError):
        unstarted(tmp_path, "gpu_4090", "11", fields, "badnode")


def test_shadow_edits_cannot_mutate_original_and_existing_bytes_cannot_change(tmp_path):
    original, shadow = tmp_path / "old", tmp_path / "new"
    original.mkdir()
    (original / "source.json").write_text("original")
    copy_frozen_tree(original, shadow)
    (shadow / "source.json").write_text("new branch")
    assert (original / "source.json").read_text() == "original"
    with pytest.raises(ValueError, match="changed"):
        copy_frozen_tree(original, shadow)


def test_exclude_is_an_explicit_scheduler_option_and_every_measurement_argument_is_preserved(
    tmp_path,
):
    from research.selector_v4.repair_unstarted_formal import scheduler_command

    args = ["gpu_4090", tmp_path, "/normal/source", 17, "/pristine/source"]
    command = scheduler_command(tmp_path, tmp_path / "frozen.sbatch", args, "badnode")
    assert command[:3] == ["sbatch", "--parsable", "--exclude=badnode"]
    assert command[-4:] == list(map(str, args[1:]))
    assert command[command.index("-p") + 1] == args[0]


def test_existing_job_checkpoint_cannot_change_a_started_measurement(tmp_path):
    import hashlib
    import json

    from research.selector_v4.repair_unstarted_formal import existing_jobs

    receipts = tmp_path / "stages/dev/receipts"
    receipts.mkdir(parents=True)
    (tmp_path / "receipts").mkdir()
    jobs = {"gpu_4090": {"0": "31", "1": "32"}, "gpu_5090": {"0": "21", "1": "22"}}
    correction = {
        "original_jobs": {"gpu_4090": {"0": "11", "1": "12"}, "gpu_5090": {"0": "21", "1": "22"}},
        "unstarted_failures": {"gpu_4090:0": {}, "gpu_4090:1": {}},
    }
    for index, job in jobs["gpu_4090"].items():
        (receipts / f"unstarted-correction-gpu_4090-{index}.json").write_text(
            json.dumps({"successor_job": job})
        )
    file = receipts / "jobs.json"
    file.write_text(json.dumps(jobs))
    (tmp_path / "receipts/scheduler-amendment.json").write_text(
        json.dumps({"existing_jobs_sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    )
    assert existing_jobs(tmp_path, correction) == jobs
    jobs["gpu_5090"]["0"] = "99"
    file.write_text(json.dumps(jobs))
    with pytest.raises(ValueError, match="identity"):
        existing_jobs(tmp_path, correction)
