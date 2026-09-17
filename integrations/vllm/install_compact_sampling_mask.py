#!/usr/bin/env python3
"""Apply or revert the compact sampling-mask patch on an installed vLLM v0.29.0.

Superseded upstream: vllm-project/vllm#54901 (merged 2026-09-04, shipped in
0.29.1) landed the same top_k-bounded compact layout. This patch exists as an
independent implementation for 0.29.0 and must not be applied to 0.29.1+,
where the engine already carries the fix; the version guard enforces that.

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


def package_only(patch_text: str, prefix: str = "vllm/") -> str:
    """Keep only the per-file sections of a git diff that touch ``prefix``."""

    sections = patch_text.split("diff --git ")
    kept = [sec for sec in sections[1:] if sec.startswith(f"a/{prefix}")]
    return "".join("diff --git " + sec for sec in kept)


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
    # The checked-in patch is the full upstream change, tests included; an
    # installed wheel has no tests/ tree, so apply only the package files.
    proc = subprocess.run(cmd, cwd=root, input=package_only(PATCH.read_text()), text=True)
    if proc.returncode != 0:
        print("patch failed; the installed sources may be partially modified", file=sys.stderr)
        return proc.returncode
    print("reverted" if args.revert else "applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
