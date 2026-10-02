"""Source-bound CPU markers only while an independent profiler is active.

Both formal HTTP arms use the same marker wrapper. It performs no input/logit
readbacks, changes no Native result, and cannot authorize a Resource tactic.
The scored processes retain only the symmetric profiler-enabled check.
"""

import functools
import hashlib
import inspect
import runpy
from pathlib import Path

from research.selector_v4.serving.graph_load_boundary import REVIEWED_DECODE_SHA256


def wrap_execute(original, *, profiling, marker):
    if getattr(original, "_sgi_graph_profile_boundary", False):
        raise RuntimeError("Graph profile boundary already installed")

    @functools.wraps(original)
    def execute(self, *args, **kwargs):
        if not profiling():
            return original(self, *args, **kwargs)
        with marker("sgi_actual_decode_graph_step"):
            return original(self, *args, **kwargs)

    execute._sgi_graph_profile_boundary = True
    return execute


def install():
    import torch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner

    source = Path(inspect.getfile(DecodeCudaGraphRunner))
    if hashlib.sha256(source.read_bytes()).hexdigest() != REVIEWED_DECODE_SHA256:
        raise RuntimeError("Unreviewed actual decode Graph profile source")
    DecodeCudaGraphRunner.execute = wrap_execute(
        DecodeCudaGraphRunner.execute,
        profiling=torch.autograd._profiler_enabled,
        marker=torch.profiler.record_function,
    )


# The explicit launch entry point must repeat installation in spawned workers.
if __name__ == "__main__" or __name__ == "__mp_main__":
    install()
    if __name__ == "__main__":
        runpy.run_module("sglang.launch_server", run_name="__main__")
