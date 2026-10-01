"""Real NVIDIA alias, invalid active samples, GPU identity and unchanged floors."""

import pytest

from research.selector_v4.public_qualification.telemetry import read_clock_csv


def fixture(tmp_path, column="clocks.current.sm [MHz]", *, count=10, fault=None):
    path = tmp_path / "telemetry.csv"
    rows = ["timestamp, uuid, utilization.gpu [%], " + column]
    for i in range(count):
        clock = "2500"
        if fault == "unstable" and i % 2:
            clock = "2200"
        if fault == "active-invalid" and i == 0:
            clock = "N/A"
        rows.append(f"timestamp, GPU-actual, 100 %, {clock} MHz")
    rows.append("timestamp, GPU-actual, 0 %, N/A")
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.mark.parametrize("column", ["clocks.current.sm [MHz]", "clocks.sm [MHz]"])
def test_actual_nvidia_alias_and_short_alias_read_every_active_sample(tmp_path, column):
    result = read_clock_csv(fixture(tmp_path, column), "actual")
    assert result == {"stable": True, "active_samples": 10, "p05_mhz": 2500, "p95_mhz": 2500}


@pytest.mark.parametrize(
    "fault", ["too-few", "unstable", "active-invalid", "wrong-uuid", "wrong-column"]
)
def test_unknown_or_invalid_active_telemetry_cannot_hide_behind_valid_samples(tmp_path, fault):
    path = fixture(
        tmp_path,
        column="unsupported" if fault == "wrong-column" else "clocks.current.sm [MHz]",
        count=5 if fault == "too-few" else 10,
        fault=fault,
    )
    assert not read_clock_csv(path, "other" if fault == "wrong-uuid" else "actual")["stable"]
