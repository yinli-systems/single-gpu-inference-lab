"""Owned CUDA13 CCCL prefix for DeepGEMM; Native FlashInfer includes unchanged."""

import json
import shutil
from pathlib import Path

from research.selector_v4.public_qualification.gates import need, sha

SOURCE = Path(
    "/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/lib/python3.12/site-packages/nvidia/cu13/include/cccl"
)


def prepare(root, *, source=SOURCE):
    need(not root.exists(), "New owned CCCL namespace")
    need((source / "cuda/std/utility").is_file(), "Actual CUDA13 CCCL standard utility required")
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    need(
        files and all(not p.is_symlink() for p in source.rglob("*")), "Regular complete CCCL source"
    )
    root.mkdir()
    shutil.copytree(source, root / "cccl")
    for name, digest in files.items():
        need(sha(root / "cccl" / name) == digest, "Exact owned CCCL header copy")
    # Expose only cccl/cuda/... . An include of cuda/std/... still resolves to
    # the original FlashInfer-pinned libcudacxx rather than this newer copy.
    need(not (root / "cuda").exists(), "Native pinned CUDA standard includes must not be shadowed")
    binding = {
        "source": str(source),
        "headers": {"cccl/" + n: v for n, v in files.items()},
        "native_cuda_std_includes_shadowed": False,
    }
    (root / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    return binding
