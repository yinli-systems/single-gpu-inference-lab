#!/usr/bin/env python3
"""Apply or revert the compact sampling-mask patch on an installed vLLM.

The patch is a plain ``git diff`` against vLLM v0.29.0, kept upstream-shaped
so it can be submitted as-is. This installer only locates the installed
package, checks the version, and shells out to ``patch``.

    python integrations/vllm/install_compact_sampling_mask.py            # apply
    python integrations/vllm/install_compact_sampling_mask.py --revert   # revert
    python integrations/vllm/install_compact_sampling_mask.py --check    # status
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parent / "vllm-v0.29.0-compact-sampling-mask.patch"
SUPPORTED_VERSIONS = {"0.29.0"}
MARKER = "_pack_compact_support_kernel"


def site_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm is not importable in this interpreter")
    return Path(spec.origin).resolve().parent.parent


def installed_version() -> str:
    import vllm  # noqa: PLC0415

    return vllm.__version__


def is_applied(root: Path) -> bool:
    return MARKER in (root / "vllm/v1/worker/gpu/sample/output.py").read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--revert", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--allow-unsupported-version", action="store_true")
    args = parser.parse_args()

    root = site_root()
    version = installed_version()
    applied = is_applied(root)
    print(f"vllm {version} at {root}; compact sampling mask patch applied={applied}")
    if args.check:
        return 0
    if version not in SUPPORTED_VERSIONS and not args.allow_unsupported_version:
        print(f"patch is validated for {sorted(SUPPORTED_VERSIONS)} only", file=sys.stderr)
        return 2
    if args.revert and not applied:
        print("nothing to revert")
        return 0
    if not args.revert and applied:
        print("already applied")
        return 0
    cmd = ["patch", "-p1", "--forward", "--no-backup-if-mismatch"]
    if args.revert:
        cmd.append("--reverse")
    with PATCH.open("rb") as fh:
        proc = subprocess.run(cmd, cwd=root, stdin=fh)
    if proc.returncode != 0:
        print("patch failed; the installed sources may be partially modified", file=sys.stderr)
        return proc.returncode
    print("reverted" if args.revert else "applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
