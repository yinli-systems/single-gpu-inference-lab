"""Shared CUDA architecture handling for PyTorch extension builds."""

from __future__ import annotations

import os
import re

DEFAULT_TORCH_CUDA_ARCH_LIST = "8.9"
_CUDA_ARCH_PATTERN = re.compile(
    r"^(?:sm_|compute_)?"
    r"(?:(?P<major>[1-9][0-9]*)\.(?P<minor>[0-9])|"
    r"(?P<compact>[1-9][0-9]{1,2}))$"
)


def normalize_cuda_arch(value: str) -> str:
    """Return an nvcc compute capability such as ``89`` or ``80``."""

    match = _CUDA_ARCH_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError(
            "CUDA_ARCH must be a compute capability such as '89', '80', or '8.0'"
        )
    compact = match.group("compact")
    if compact is not None:
        return compact
    return f"{int(match.group('major'))}{match.group('minor')}"


def torch_cuda_arch(value: str) -> str:
    """Convert one compact ``CUDA_ARCH`` alias to PyTorch's dotted form."""

    compact = normalize_cuda_arch(value)
    return f"{int(compact[:-1])}.{compact[-1]}"


def configure_torch_cuda_arch_list(
    *,
    default: str = DEFAULT_TORCH_CUDA_ARCH_LIST,
    environ: dict[str, str] | None = None,
) -> str:
    """Set a deterministic PyTorch CUDA target when the caller did not.

    ``TORCH_CUDA_ARCH_LIST`` is PyTorch's native interface and takes priority.
    ``CUDA_ARCH`` remains a single-architecture compatibility alias for the
    standalone Makefile runner and older reproduction commands.
    """

    target_environ = os.environ if environ is None else environ
    native = target_environ.get("TORCH_CUDA_ARCH_LIST")
    if native:
        return native
    alias = target_environ.get("CUDA_ARCH")
    selected = torch_cuda_arch(alias) if alias else default
    target_environ["TORCH_CUDA_ARCH_LIST"] = selected
    return selected
