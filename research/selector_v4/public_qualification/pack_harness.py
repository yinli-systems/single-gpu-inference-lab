"""Create one immutable helper archive from committed research blobs only."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def pack(repo, output, git="git"):
    if output.exists():
        raise FileExistsError("Never replace a frozen helper archive")
    run = lambda *args: subprocess.check_output([git, *args], cwd=repo)
    if run("status", "--porcelain", "--untracked-files=all", "--", "research"):
        raise ValueError("Commit every research change before packing")
    commit = run("rev-parse", "HEAD").decode().strip()
    names = run("ls-tree", "-r", "--name-only", "HEAD", "research").decode().splitlines()
    files = {name: run("show", "HEAD:" + name) for name in names}
    if not files:
        raise ValueError("Missing committed research source")
    metadata = {
        "harness_commit": commit,
        "git_blob_files_sha256": {n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()},
        "fresh_cases_generated": 0,
        "qualification_authority": False,
    }
    files["harness-commit.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(raw))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--git", default="git")
    args = parser.parse_args()
    metadata = pack(args.repo, args.output, args.git)
    print(
        json.dumps(
            {
                "harness_commit": metadata["harness_commit"],
                "files": len(metadata["git_blob_files_sha256"]),
                "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )
