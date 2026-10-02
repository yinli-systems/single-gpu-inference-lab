"""Full-model Native Graph boundary diagnostic; readbacks invalidate timing.

Actual fixed graph objects/buffers are checked across changing request inputs.
No Resource tactic, registry, training or serving authority is installed.
"""

import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path

SOURCES = {
    "ModelRunner": "f4b9965f3db0667e9c72d6db88e5a18727ebb38a317bff2a623144816fb4218c",
    "DecodeCudaGraphRunner": "ba44a02251df0ddeee6802b9f305c0257c860aeae4069dc43eda0a7fde31c5a2",
    "FullCudaGraphBackend": "737f6d589845a10a934e32bc06bfb4836fe028dcd73850ecf6f2241c57df8c9d",
    "FlashInferAttnBackend": "5c8baba0d14eeca4c9d4bef8674318be1697b0160f105224282cd4aa27575810",
}


def install():
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    from research.selector_v4.serving.graph_load_boundary import install as notify
    from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter

    root = Path(os.environ["SGI_GRAPH_SERVING_DIAGNOSTIC"])
    root.mkdir(parents=True, exist_ok=True)
    for cls in (ModelRunner, DecodeCudaGraphRunner, FullCudaGraphBackend, FlashInferAttnBackend):
        if (
            hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            != SOURCES[cls.__name__]
        ):
            raise RuntimeError("Unreviewed GraphServing source: " + cls.__name__)
    notify()
    original_init = FlashInferAttnBackend.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        adapter = MetadataEpochAdapter(self)
        adapter.install()
        self._sgi_metadata_adapter = adapter

    FlashInferAttnBackend.__init__ = initialize
    original_load = DecodeCudaGraphRunner.load_batch
    original_execute = DecodeCudaGraphRunner.execute
    graphs = {}

    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    @functools.wraps(original_load)
    def load(self, batch, *args, **kwargs):
        result = original_load(self, batch, *args, **kwargs)
        backend = self.model_runner.attn_backend
        fields = {}
        for name in ("input_ids", "positions", "seq_lens", "out_cache_loc", "req_pool_indices"):
            tensor = getattr(self.buffers, name)
            n = (
                self.raw_num_token
                if name in ("input_ids", "positions", "out_cache_loc")
                else self.raw_bs
            )
            fields[name] = {
                "pointer": tensor.data_ptr(),
                "shape": list(tensor.shape),
                "active_rows": n,
                "actual_gpu_payload_sha256": digest(tensor[:n]),
            }
        graph = self.backend._graphs[self._replay_graph_key]
        key = repr(self._replay_graph_key)
        identity = {
            "graph_object_id": id(graph),
            "buffers": {
                k: {"pointer": v["pointer"], "shape": v["shape"]} for k, v in fields.items()
            },
        }
        if key in graphs and graphs[key] != identity:
            raise RuntimeError("Native captured Graph object/buffer identity changed")
        graphs[key] = identity
        self._sgi_graph_epoch = {
            "kind": "actual_decode_graph_step",
            "pid": os.getpid(),
            "graph_key": key,
            "batch_size": batch.batch_size,
            "inputs": fields,
            "graph_object_id": id(graph),
            "load_notifications": backend._sgi_decode_load_notifications,
            "adapter_counters": dict(backend._sgi_metadata_adapter.counters),
            "metadata_depth_before_replay": backend._sgi_metadata_adapter.depth,
            "resource_tactic_selected": False,
            "timing_valid": False,
            "full_http_qualified": False,
            "qualification_authority": False,
        }
        return result

    @functools.wraps(original_execute)
    def execute(self, batch, *args, **kwargs):
        with torch.profiler.record_function("sgi_actual_decode_graph_step"):
            output = original_execute(self, batch, *args, **kwargs)
        row = self._sgi_graph_epoch
        logits = output.next_token_logits
        row["logits_sha256"] = digest(logits.float())
        row["metadata_depth_after_replay"] = (
            self.model_runner.attn_backend._sgi_metadata_adapter.depth
        )
        with (root / f"graph-steps-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return output

    DecodeCudaGraphRunner.load_batch = load
    DecodeCudaGraphRunner.execute = execute


if os.environ.get("SGI_GRAPH_SERVING_DIAGNOSTIC"):
    install()

if __name__ == "__main__":
    if not os.environ.get("SGI_GRAPH_SERVING_DIAGNOSTIC"):
        raise RuntimeError("Explicit owned diagnostic directory required")
    runpy.run_module("sglang.launch_server", run_name="__main__")
