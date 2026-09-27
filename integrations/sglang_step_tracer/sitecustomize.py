"""Experiment-only SGLang step tracer, loaded through sitecustomize (put this directory on PYTHONPATH).

When SGL_EXP_STEP_TRACE is set, every ModelRunner.forward call in every process (the scheduler
runs in a child process) is bracketed by a CUDA event pair on the current stream. One JSON line
per forward is written once its end event has completed (checked lazily on later calls and at
exit), so tracing adds no synchronization to the step: mode, batch size, per-request extend
(chunk) and prefix (KV depth) lengths, and cuda_ms. Never upstreamed.
"""

import importlib.abc
import importlib.util
import os
import sys

_PATH = os.environ.get("SGL_EXP_STEP_TRACE")
_TARGET = "sglang.srt.model_executor.model_runner"


def _install(mod):
    import atexit
    import json
    import torch

    runner = mod.ModelRunner
    orig = runner.forward
    fh = open(f"{_PATH}.{os.getpid()}", "a", buffering=1)
    pending = []

    def drain(block=False):
        while pending and (block or pending[0][1].query()):
            e0, e1, meta = pending.pop(0)
            e1.synchronize()
            meta["cuda_ms"] = e0.elapsed_time(e1)
            fh.write(json.dumps(meta) + "\n")

    def forward(self, forward_batch, *args, **kwargs):
        drain()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig(self, forward_batch, *args, **kwargs)
        e1.record()
        mode = forward_batch.forward_mode
        pending.append((e0, e1, {
            "id": self.forward_pass_id, "mode": getattr(mode, "name", str(mode)), "bs": int(forward_batch.batch_size),
            "extend": list(forward_batch.extend_seq_lens_cpu or []), "prefix": list(forward_batch.extend_prefix_lens_cpu or [])}))
        return out

    runner.forward = forward
    atexit.register(lambda: drain(block=True))


class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        sys.meta_path.insert(0, self)
        if spec is None:
            return None
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        sys.meta_path.remove(self)
        _install(module)


if _PATH:
    sys.meta_path.insert(0, _Hook())
