import py_compile
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
DIR = (
    ROOT
    / "research"
    / "decision_sufficiency"
    / "collision_factorial"
)


def test_collision_factorial_python_sources_compile(tmp_path):
    files = sorted(DIR.glob("*.py"))
    assert {p.name for p in files} >= {
        "manifest.py",
        "measure.py",
        "analyze.py",
        "validate_canary.py",
        "prepare_campaign.py",
    }
    for path in files:
        py_compile.compile(
            str(path),
            cfile=str(tmp_path / (path.stem + ".pyc")),
            doraise=True,
        )


def test_collision_factorial_launcher_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(DIR / "run.sbatch")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_snapshot_file_list_resolves():
    text = (DIR / "prepare_campaign.py").read_text()
    for relative in [
        "collision_factorial/manifest.py",
        "collision_factorial/measure.py",
        "collision_factorial/analyze.py",
        "collision_factorial/validate_canary.py",
        "collision_factorial/PROTOCOL.md",
        "collision_factorial/run.sbatch",
        "collision_factorial/prepare_campaign.py",
        "section6/metadata_adapter.py",
        "plan-order-mechanism/plan_contract.py",
        "resource_generalization/prepare.py",
    ]:
        assert relative in text
