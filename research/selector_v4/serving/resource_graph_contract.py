"""One-use replay tickets for a separate fixed-geometry diagnostic graph.

Full SGLang Native graphs keep their original split-KV plan. A separate unsplit
graph is eligible only for one exact positive request geometry; padding and
variable lengths must use the original Native graph. No certificate authority.
"""

from dataclasses import dataclass


def supported_batch(batch, tokens):
    from research.selector_v4.serving.metadata_adapter import host_lengths

    mode = getattr(batch, "forward_mode", None)
    return bool(
        type(tokens) is int
        and tokens > 0
        and getattr(batch, "batch_size", None) == 1
        and mode is not None
        and mode.is_extend_without_speculative()
        and host_lengths(getattr(batch, "extend_seq_lens_cpu", None)) == (tokens,)
        and host_lengths(getattr(batch, "extend_prefix_lens_cpu", None)) == (0,)
        and host_lengths(getattr(batch, "seq_lens_cpu", None)) == (tokens,)
        and getattr(batch, "extend_num_tokens", None) == tokens
        and getattr(batch, "spec_info", None) is None
        and not getattr(batch, "mm_inputs", None)
        and not getattr(batch, "lora_ids", None)
    )


@dataclass
class ReplayTicket:
    state: str = "IDLE"
    epoch: int = 0
    frame: object = None

    def begin(self):
        if self.state not in ("IDLE", "CONSUMED", "NATIVE"):
            self.state = "RETIRED"
            raise RuntimeError("Previous epoch incomplete or adapter retired")
        self.epoch += 1
        self.frame = None
        self.state = "LOADING"

    def finish(self, frame, *, eligible):
        if self.state != "LOADING":
            self.state = "RETIRED"
            raise RuntimeError("Metadata update was not announced")
        self.frame = frame if eligible else None
        self.state = "READY" if eligible else "NATIVE"

    def consume(self, frame):
        if self.state == "NATIVE":
            return False
        if self.state != "READY" or frame != self.frame:
            self.state = "RETIRED"
            raise RuntimeError("Stale or already consumed Resource Graph epoch")
        self.state = "CONSUMED"
        return True

    def fail(self):
        self.frame = None
        self.state = "RETIRED"


def resource_trace(events):
    """Require CPU marker -> real launch -> graph Resource kernel with 64KiB."""
    markers = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and e.get("name") == "sgi_resource_full_model_graph"
    ]
    if not markers:
        raise ValueError("No actual Resource serving graph markers")
    consumed, launches, kernels = set(), 0, 0
    for marker in markers:
        if marker.get("dur", 0) <= 0:
            raise ValueError("Incomplete graph marker")
        apis = [
            e
            for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and e.get("ph") == "X"
            and "GraphLaunch" in e.get("name", "")
            and e.get("pid") == marker.get("pid")
            and e.get("tid") == marker.get("tid")
            and marker["ts"] <= e["ts"] <= marker["ts"] + marker["dur"]
        ]
        ids = {e.get("args", {}).get("correlation") for e in apis} - {None}
        if not ids or ids & consumed:
            raise ValueError("Missing or duplicate actual Graph launch correlation")
        consumed |= ids
        linked = [
            e
            for e in events
            if e.get("cat") == "kernel"
            and "ResourceKernel" in e.get("name", "")
            and e.get("args", {}).get("correlation") in ids
        ]
        if not linked or any(
            e["args"].get("graph id", 0) <= 0
            or e["args"].get("shared memory", -1) != 65536
            for e in linked
        ):
            raise ValueError("Actual captured 64KiB Resource kernels required")
        launches += len(apis)
        kernels += len(linked)
    return {
        "markers": len(markers),
        "graph_launches": launches,
        "actual_resource_graph_kernels": kernels,
        "timing_valid": False,
        "serving_promotion": False,
    }
