"""Explicit Native-only full-model witness of the exact-source prefix hook.

Empty frozen bindings leave every actual attention invocation native. Geometry,
metadata and router observations invalidate timings. This diagnostic proves no
resource-enabled HTTP behavior, performance or historical divergence closure.
"""

import functools
import inspect
import json
import os
import runpy
from pathlib import Path


def install():
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

    from research.selector_v4.serving import metadata_boundary_server  # noqa: F401
    from research.selector_v4.serving.prefix_backend_hook import install as install_hook
    from research.selector_v4.serving.prefix_router import PagedPrefixRouter

    original_init = FlashInferAttnBackend.__init__
    if getattr(original_init, "_sgi_native_prefix_router", False):
        raise RuntimeError("Native prefix-router diagnostic already installed")

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        router = PagedPrefixRouter(
            self._sgi_metadata_adapter,
            graph_or_tracing=lambda: (
                torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
            ),
        )
        install_hook(self, router, source_path=inspect.getfile(FlashInferAttnBackend))
        self._sgi_native_prefix_router = router

    initialize._sgi_native_prefix_router = True
    FlashInferAttnBackend.__init__ = initialize
    original_sample = ModelRunner.sample
    root = Path(os.environ["SGI_METADATA_BOUNDARY_DIR"])

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        router = self.attn_backend._sgi_native_prefix_router
        if router.bindings or router.counters["prepared_attempts"]:
            raise RuntimeError("Native diagnostic must never route a Resource attempt")
        with (root / f"prefix-router-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(router.evidence(), sort_keys=True) + "\n")
        return result

    ModelRunner.sample = sample


if os.environ.get("SGI_METADATA_BOUNDARY_DIR"):
    install()

if __name__ == "__main__":
    if not os.environ.get("SGI_METADATA_BOUNDARY_DIR"):
        raise RuntimeError("Explicit Native-only prefix-router diagnostic required")
    runpy.run_module("sglang.launch_server", run_name="__main__")
