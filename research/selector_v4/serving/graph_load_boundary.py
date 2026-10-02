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
REVIEWED_PREFILL_SHA256 = "48de8c6add0616f8d4824d170264b2f3d9dd9ccb7950a1c1cbdbea4c3d10e620"


def wrap_load(original, *, kind="decode"):
    if kind not in ("decode", "prefill"):
        raise ValueError("Reviewed decode or prefill load boundary required")
    if getattr(original, "_sgi_graph_load_boundary", False):
        raise RuntimeError("Decode load boundary already installed")

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        backend = self.model_runner.attn_backend
        router = getattr(backend, "_sgi_training_router", None)
        # seal() replaces the training adapter. Resolve its current object at
        # every load, even when a diagnostic attribute retains an old adapter.
        adapter = (
            router.adapter if router is not None else getattr(backend, "_sgi_metadata_adapter", None)
        )
        if adapter is not None:
            if adapter.backend is not backend or adapter.depth != 0:
                raise RuntimeError("Graph load requires its idle original backend adapter")
            # This occurs before native input copies AND GPU-only glue replay.
            # A nested Python plan may notify again; repeated release is safe.
            adapter.before_update(True)
            name = "_sgi_" + kind + "_load_notifications"
            setattr(backend, name, getattr(backend, name, 0) + 1)
        return original(self, *args, **kwargs)

    load._sgi_graph_load_boundary = True
    return load


def install(*, include_prefill=False):
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner

    classes = [(DecodeCudaGraphRunner, "decode", REVIEWED_DECODE_SHA256)]
    if include_prefill:
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        classes.append((PrefillCudaGraphRunner, "prefill", REVIEWED_PREFILL_SHA256))
    replacements = []
    for cls, kind, digest in classes:
        source = Path(inspect.getfile(cls))
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Unreviewed actual " + kind + " Graph source")
        replacements.append((cls, wrap_load(cls.load_batch, kind=kind)))
    # Validate every requested source and installation before changing classes.
    for cls, wrapped in replacements:
        cls.load_batch = wrapped
