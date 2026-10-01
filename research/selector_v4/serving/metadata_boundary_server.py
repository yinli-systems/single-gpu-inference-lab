"""Native-only full-model witness of real SGLang metadata notification order.

This diagnostic attaches empty registries and leaves attention selection native.
The independent geometry observer is active; timings are invalid. The actual
prefill updater must be entered inside an announced backend metadata boundary.
"""

import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path


def install():
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

    from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter

    root = Path(os.environ["SGI_METADATA_BOUNDARY_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    source = Path(inspect.getfile(FlashInferAttnBackend))
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != os.environ["SGI_METADATA_BACKEND_SHA256"]:
        raise RuntimeError("Unreviewed SGLang metadata backend source")
    original_init = FlashInferAttnBackend.__init__
    if getattr(original_init, "_sgi_metadata_boundary", False):
        raise RuntimeError("Metadata diagnostic already installed")

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        adapter = MetadataEpochAdapter(self)
        adapter.install()
        self._sgi_metadata_adapter = adapter
        self._sgi_prefill_update_witnesses = 0
        original_update = self.indices_updater_prefill.update

        @functools.wraps(original_update)
        def checked_update(*arguments, **options):
            if adapter.depth <= 0:
                raise RuntimeError("Actual prefill/Triton update escaped metadata notification")
            self._sgi_prefill_update_witnesses += 1
            return original_update(*arguments, **options)

        self.indices_updater_prefill.update = checked_update

    initialize._sgi_metadata_boundary = True
    FlashInferAttnBackend.__init__ = initialize
    original_sample = ModelRunner.sample

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        backend = self.attn_backend
        adapter = getattr(backend, "_sgi_metadata_adapter", None)
        if adapter is None:
            raise RuntimeError("Diagnostic requires the reviewed actual FlashInfer backend")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sampler evidence cannot execute inside capture")
        row = {
            "backend_source_sha256": actual_sha,
            "adapter_counters": dict(adapter.counters),
            "actual_prefill_updater_calls_inside_notified_boundary": backend._sgi_prefill_update_witnesses,
            "registered_resource_geometries": len(adapter.bindings),
            "adapter_depth_after_model": adapter.depth,
            "resource_tactic_selected": False,
            "timing_valid": False,
            "full_http_resource_qualified": False,
        }
        with (root / f"boundaries-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return result

    ModelRunner.sample = sample


if os.environ.get("SGI_METADATA_BOUNDARY_DIR"):
    install()
    # Both observers are explicitly diagnostic and inherited by spawn workers.
    from research.selector_v4.serving import geometry_server  # noqa: F401

if __name__ == "__main__":
    if not os.environ.get("SGI_METADATA_BOUNDARY_DIR"):
        raise RuntimeError("Explicit isolated metadata diagnostic directory required")
    runpy.run_module("sglang.launch_server", run_name="__main__")
