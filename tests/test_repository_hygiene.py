"""Keep unrelated local operator state out of the published Git tree."""

from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FILES = {
    "AUTONOMY_PROMPT.md",
    "AUTONOMY_STATE.md",
    "run_claude_watchdog.sh",
}
FORBIDDEN_DIRS = {"autonomy_logs", ".claude", ".codex"}


def forbidden_paths(paths):
    """Match operator state exactly, never legitimate workload names."""
    return sorted(
        path for path in paths
        if path in FORBIDDEN_FILES or path.split("/", 1)[0] in FORBIDDEN_DIRS
    )


class RepositoryHygieneTests(unittest.TestCase):
    def test_published_tree_has_no_local_operator_state(self):
        if (ROOT / ".git").exists():
            paths = subprocess.check_output(
                ["git", "ls-files", "-z"], cwd=ROOT
            ).decode("utf-8").split("\0")
        else:
            paths = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()]
        self.assertEqual(forbidden_paths(paths), [])

    def test_actual_inference_workloads_remain_allowed(self):
        self.assertEqual(forbidden_paths([
            "benchmarks/results/live-trace-replay/raw/replay-mooncake-toolagent/run.json.gz",
            "scripts/dp_ep/pinner.py",
        ]), [])

    def test_operator_state_is_rejected(self):
        paths = ["AUTONOMY_STATE.md", "autonomy_logs/run.jsonl", ".claude/settings.json"]
        self.assertEqual(forbidden_paths(paths), sorted(paths))


if __name__ == "__main__":
    unittest.main()
