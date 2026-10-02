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
