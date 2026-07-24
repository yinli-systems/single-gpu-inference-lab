from pathlib import Path

import pytest

from l20_stack.cuda_build import (
    configure_torch_cuda_arch_list,
    normalize_cuda_arch,
    torch_cuda_arch,
)

CUDA_EXTENSION_BUILD_ENTRY_POINTS = (
    Path("integrations/vllm/install_l20_paged_decode.py"),
    Path("scripts/benchmark_cuda_paged_decode.py"),
    Path("scripts/benchmark_cuda_paged_fp8_decode.py"),
    Path("scripts/run_vllm_l20_sparse_penalty_triangle.sh"),
    Path("scripts/run_vllm_l20_sparse_repetition_penalty_serving_ab.sh"),
    Path("scripts/smoke_cuda_paged_decode_op.py"),
    Path("scripts/smoke_cuda_paged_fp8_decode_op.py"),
    Path("scripts/smoke_cuda_sparse_repetition_penalty_op.py"),
    Path("scripts/stress_cuda_paged_decode.py"),
)


def test_cuda_arch_defaults_to_l20_sm89():
    environ = {}

    assert configure_torch_cuda_arch_list(environ=environ) == "8.9"
    assert environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("80", "80"),
        ("8.0", "80"),
        ("sm_80", "80"),
        ("compute_89", "89"),
        ("100", "100"),
        ("10.0", "100"),
    ),
)
def test_cuda_arch_normalizes_common_compute_capability_forms(value, expected):
    assert normalize_cuda_arch(value) == expected


@pytest.mark.parametrize(
    "value",
    ("", "8", "sm_", "sm_8x", "80;touch /tmp/unsafe", "8.0+PTX", "8000"),
)
def test_cuda_arch_rejects_invalid_or_compound_values(value):
    with pytest.raises(ValueError, match="CUDA_ARCH must be a compute capability"):
        normalize_cuda_arch(value)


def test_cuda_arch_environment_can_target_a100():
    environ = {"CUDA_ARCH": "80"}

    assert configure_torch_cuda_arch_list(environ=environ) == "8.0"
    assert environ["TORCH_CUDA_ARCH_LIST"] == "8.0"


def test_native_torch_cuda_arch_list_takes_priority_over_alias():
    environ = {
        "CUDA_ARCH": "invalid-alias-is-not-read",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9+PTX",
    }

    assert configure_torch_cuda_arch_list(environ=environ) == "8.0;8.9+PTX"


def test_compact_arch_converts_to_pytorch_dotted_form():
    assert torch_cuda_arch("89") == "8.9"
    assert torch_cuda_arch("100") == "10.0"


def test_cuda_extension_build_entry_points_share_validated_architecture_helper():
    for path in CUDA_EXTENSION_BUILD_ENTRY_POINTS:
        source = path.read_text(encoding="utf-8")
        assert "configure_torch_cuda_arch_list()" in source, path
        assert "-gencode=" not in source, path


def test_kernel_dependencies_include_ninja_for_torch_cpp_extension_load():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    kernels = pyproject.split("[project.optional-dependencies]", 1)[1].split(
        "production-kernels", 1
    )[0]

    assert '"ninja>=1.11"' in kernels
