import pytest

from research.selector_v4.public_qualification.gates import sha
from research.selector_v4.serving.cccl_namespace import prepare


def test_complete_cccl_prefix_copied_without_shadowing_native_cuda_std(tmp_path):
    source = tmp_path / "installed-cuda13-cccl"
    header = source / "cuda/std/utility"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"synthetic exact standard utility\n")
    (source / "cuda/std/type_traits").write_bytes(b"synthetic traits\n")
    root = tmp_path / "owned"
    binding = prepare(root, source=source)
    assert len(binding["headers"]) == 2
    assert not (root / "cuda").exists()
    assert binding["headers"]["cccl/cuda/std/utility"] == sha(root / "cccl/cuda/std/utility")
    with pytest.raises(ValueError):
        prepare(root, source=source)


def test_missing_real_cccl_namespace_rejected_before_creating_output(tmp_path):
    root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        prepare(root, source=tmp_path / "missing")
    assert not root.exists()
