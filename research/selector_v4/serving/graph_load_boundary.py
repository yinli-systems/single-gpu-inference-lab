"""Notify leases before real decode input/glue-graph writes, not only Python plan.

Explicit installation only. This does not authorise Resource inside a Graph.
The native load and replay continue unchanged; future serving entry points must
install this before accepting a request. Existing frozen campaigns are separate.
"""

import functools
import hashlib
import inspect
from pathlib import Path

REVIEWED_DECODE_SHA256 = "ba44a02251df0ddeee6802b9f305c0257c860aeae4069dc43eda0a7fde31c5a2"


def wrap_load(original):
    if getattr(original, "_sgi_graph_load_boundary", False):
        raise RuntimeError("Decode load boundary already installed")

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        backend = self.model_runner.attn_backend
        adapter = getattr(backend, "_sgi_metadata_adapter", None)
        if adapter is None:
            adapter = getattr(getattr(backend, "_sgi_training_router", None), "adapter", None)
        if adapter is not None:
            if adapter.backend is not backend or adapter.depth != 0:
                raise RuntimeError("Decode load requires its idle original backend adapter")
            # This occurs before native input copies AND GPU-only glue replay.
            # A nested Python plan may notify again; repeated release is safe.
            adapter.before_update(True)
            backend._sgi_decode_load_notifications = (
                getattr(backend, "_sgi_decode_load_notifications", 0) + 1
            )
        return original(self, *args, **kwargs)

    load._sgi_graph_load_boundary = True
    return load


def install():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner

    source = Path(inspect.getfile(DecodeCudaGraphRunner))
    if hashlib.sha256(source.read_bytes()).hexdigest() != REVIEWED_DECODE_SHA256:
        raise RuntimeError("Unreviewed actual decode Graph source")
    DecodeCudaGraphRunner.load_batch = wrap_load(DecodeCudaGraphRunner.load_batch)
