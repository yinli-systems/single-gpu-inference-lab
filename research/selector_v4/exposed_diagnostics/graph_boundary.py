"""Uncertified diagnostic of fixed Graph bindings and conservative invalidation.

This deliberately bypasses deployment selection. It is never a serving runner.
Only payload updates in the same Q/K/V buffers may reuse a captured graph.
The exclusive native owner must invalidate the lease before untracked metadata
writes. Replanning, snapshot changes, replacement buffers, options or streams
fall back to the current owner's eager forward, and never replay a stale graph.
"""

from __future__ import annotations


class GraphBoundaryProbe:
    def __init__(self, lease, inputs, *, replays=1, allow_uncertified_diagnostic=False):
        if not allow_uncertified_diagnostic:
            raise RuntimeError("Explicit uncertified diagnostic opt-in is required")
        if replays not in (1, 16):
            raise ValueError("Only the exposed fixed Graph1/16 diagnostic modes are supported")
        import torch

        self.lease = lease
        self.binding = self.input_binding(inputs)
        self.invalidated = False
        self.last_execution = None
        self.replays = replays
        if not lease.current(inputs) or not lease.runner._current(inputs):
            raise RuntimeError("Native owner or owned metadata snapshot is stale")
        if not lease.runner._eligible or not lease.runner._prepare(inputs):
            raise RuntimeError("Exposed resource kernel is unsupported")
        assert lease.runner.receipt is None, "No eager certificate may be reused for Graph"
        # Prepare and warm outside capture. The raw resource call is intentional
        # and provides no authorization for managed selection or full serving.
        for _ in range(3):
            lease.runner._resource(inputs, **lease.run_options)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for _ in range(replays):
                self._result = lease.runner._resource(inputs, **lease.run_options)
        if not lease.current(inputs) or not lease.runner._current(inputs):
            raise RuntimeError("Owner changed while capturing the diagnostic graph")

    @staticmethod
    def input_binding(inputs):
        return tuple(
            (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
            for t in inputs
        )

    def invalidate(self):
        self.invalidated = True
        self.lease.invalidate()

    def replay(self, inputs, *, forward_options):
        if (
            self.invalidated
            or dict(forward_options) != self.lease.options
            or self.input_binding(inputs) != self.binding
            or not self.lease.current(inputs)
            or not self.lease.runner._current(inputs)
        ):
            self.invalidate()
            self.last_execution = "native_eager"
            return self.lease.native(inputs, forward_options)
        self._graph.replay()
        self.last_execution = "uncertified_resource_graph"
        return self._result

    def evidence(self):
        return {
            "diagnostic_only": True,
            "graph_replays": self.replays,
            "same_buffer_payload_updates_supported": True,
            "resource_graph_metadata_updates_supported": False,
            "exclusive_native_owner_required": True,
            "untracked_metadata_requires_explicit_invalidation": True,
            "managed_certificate_reused": False,
            "serving_qualified": False,
            "fresh_cases_consumed": 0,
            "default_promotion": False,
            "historical_token_divergence_resolved": False,
        }
