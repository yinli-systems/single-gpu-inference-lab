"""Committed synthetic file packaging only; no GPU geometry is generated."""

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from research.selector_v4.public_qualification.pack_harness import pack


@pytest.fixture
def repository(tmp_path):
    git = Path("/Library/Developer/CommandLineTools/usr/bin/git")
    if not git.is_file():
        git = Path(shutil.which("git"))
    repo = tmp_path / "synthetic-repository"
    repo.mkdir()
    subprocess.run([str(git), "init", "-q", str(repo)], check=True)
    (repo / "research").mkdir()
    (repo / "research/control.txt").write_bytes(b"synthetic committed source\n")
    subprocess.run([str(git), "add", "research"], cwd=repo, check=True)
    subprocess.run(
        [
            str(git),
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-qm",
            "synthetic",
        ],
        cwd=repo,
        check=True,
    )
    return repo, str(git)


def test_archive_contains_exact_committed_blobs_and_identity(repository, tmp_path):
    repo, git = repository
    output = tmp_path / "source.tar.gz"
    result = pack(repo, output, git)
    expected = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert result["harness_commit"] == expected
    with tarfile.open(output) as bundle:
        assert set(bundle.getnames()) == {"research/control.txt", "harness-commit.json"}
        raw = bundle.extractfile("research/control.txt").read()
        metadata = json.loads(bundle.extractfile("harness-commit.json").read())
    assert raw == (repo / "research/control.txt").read_bytes()
    assert (
        metadata["git_blob_files_sha256"]["research/control.txt"] == hashlib.sha256(raw).hexdigest()
    )
    assert not metadata["qualification_authority"] and metadata["fresh_cases_generated"] == 0
    with pytest.raises(FileExistsError):
        pack(repo, output, git)


@pytest.mark.parametrize("untracked", [False, True])
def test_uncommitted_or_untracked_source_cannot_be_packed(repository, tmp_path, untracked):
    repo, git = repository
    (repo / "research" / ("new.txt" if untracked else "control.txt")).write_text("uncommitted")
    output = tmp_path / "must-not-exist.tar.gz"
    with pytest.raises(ValueError, match="Commit"):
        pack(repo, output, git)
    assert not output.exists()
