"""Contract checks for the vLLM v0.29.0 compact sampling-mask patch.

CPU-safe checks parse the checked-in patch; the GPU check (skipped without a
vLLM install) builds both layouts from the same logits and requires identical
CSR output.
"""

from pathlib import Path

import pytest

PATCH = Path("integrations/vllm/vllm-v0.29.0-compact-sampling-mask.patch")


def test_patch_is_scoped_and_documents_its_contract():
    text = PATCH.read_text()
    files = [line.split()[-1][2:] for line in text.splitlines() if line.startswith("+++ b/")]
    assert files == [
        "tests/v1/test_outputs.py",
        "vllm/envs.py",
        "vllm/v1/worker/gpu/sample/output.py",
        "vllm/v1/worker/gpu/sample/sampler.py",
    ]
    # the installer must strip the tests section for wheel installs
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "installer", Path("integrations/vllm/install_compact_sampling_mask.py")
    )
    installer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(installer)
    pkg = installer.package_only(text)
    assert "tests/v1/test_outputs.py" not in pkg
    assert pkg.count("diff --git ") == 3
    for marker in (
        "_pack_compact_support_kernel",
        "VLLM_SAMPLING_MASK_COMPACT",
        "VLLM_SAMPLING_MASK_COMPACT_MAX_K",
        "_sampling_mask_max_support",
        "the batch's top_k bound was violated",
        "overflow",
    ):
        assert marker in text, marker
    # the bit-packed layout must remain available as the fallback
    assert "np.unpackbits(" in text
    assert "-        unpacked = np.unpackbits(" in text


def test_patched_layouts_agree_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    output = pytest.importorskip("vllm.v1.worker.gpu.sample.output")
    if not hasattr(output.SamplingMaskTensors, "_from_logits_compact"):
        pytest.skip("installed vLLM does not carry the compact sampling-mask patch")
    import numpy as np

    gen = torch.Generator(device="cuda").manual_seed(9)
    batch, vocab, top_k = 64, 151_936, 50
    logits = torch.randn((batch, vocab), device="cuda", generator=gen)
    thresh = torch.topk(logits, top_k, dim=1).values[:, -1:]
    logits = torch.where(logits >= thresh, logits, torch.full_like(logits, -float("inf")))
    num_sampled = torch.ones(batch, device="cuda", dtype=torch.int32)
    num_sampled[3] = 0
    ns_np = num_sampled.cpu().numpy()
    bitmap = output.SamplingMaskTensors.from_logits(logits, num_sampled).to_cpu_nonblocking()
    compact = output.SamplingMaskTensors.from_logits(logits, num_sampled, max_support=64).to_cpu_nonblocking()
    torch.cuda.synchronize()
    a = bitmap.tolists(ns_np)
    b = compact.tolists(ns_np)
    assert np.array_equal(a.token_ids, b.token_ids)
    assert np.array_equal(a.offsets, b.offsets)
    assert a.cu_num_generated_tokens == b.cu_num_generated_tokens
    # overflow must raise, never truncate silently
    tight = output.SamplingMaskTensors.from_logits(logits, num_sampled, max_support=32).to_cpu_nonblocking()
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="exceeded max_support"):
        tight.tolists(ns_np)
