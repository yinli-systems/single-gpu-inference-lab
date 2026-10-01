"""Observe actual SGLang prefill geometry before serving integration.

Diagnostic only: CPU metadata readbacks and durable writes invalidate timings.
The official forward/plan/run methods execute normally; no resource tactic is
selected. Run through this entry point only in an isolated qualification server.
CUDA capture is never inspected or synchronized by this observer.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path


def install():
    destination = os.environ.get("SGI_PREFILL_GEOMETRY_DIR")
    if not destination:
        raise RuntimeError("Explicit diagnostic output directory is required")
    import flashinfer
    import torch
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    path = root / f"geometry-{os.getpid()}.jsonl"

    def signature(t):
        return {
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "dtype": str(t.dtype),
            "device": str(t.device),
        }

    def observe(wrapper, method, bound, paged):
        q = bound["q"]
        if not q.is_cuda or torch.cuda.is_current_stream_capturing():
            return
        batch = getattr(wrapper, "_batch_size", None)
        qi = wrapper._qo_indptr_buf.detach().cpu().tolist()
        if batch is None:
            batch = len(qi) - 1
        qi = qi[: batch + 1]
        qs = [b - a for a, b in zip(qi, qi[1:])]
        if paged:
            kv = bound["paged_kv_cache"]
            k, v = (kv[0], kv[1]) if isinstance(kv, tuple) else kv.unbind(0)
            page_size = k.shape[1 if wrapper._kv_layout == "NHD" else 2]
            ki = wrapper._paged_kv_indptr_buf.detach().cpu().tolist()[: batch + 1]
            last = wrapper._paged_kv_last_page_len_buf.detach().cpu().tolist()[:batch]
            lengths = [(b - a - 1) * page_size + n for a, b, n in zip(ki, ki[1:], last)]
        else:
            k, v = bound["k"], bound["v"]
            page_size = 1
            ki = wrapper._kv_indptr_buf.detach().cpu().tolist()[: batch + 1]
            lengths = [b - a for a, b in zip(ki, ki[1:])]
        if len(qs) != batch or len(lengths) != batch or sum(qs) != q.shape[0]:
            raise RuntimeError("Diagnostic metadata does not describe the actual Q batch")
        plan = list(wrapper._plan_info or [])
        causal = bool(bound["causal"])
        config = {}
        for name in (
            "pos_encoding_mode",
            "use_fp16_qk_reduction",
            "sm_scale",
            "window_left",
            "logits_soft_cap",
            "k_scale",
            "v_scale",
            "rope_scale",
            "rope_theta",
        ):
            value = bound.get(name)
            config[name] = signature(value) if isinstance(value, torch.Tensor) else value
        row = {
            "method": method,
            "layout": "paged" if paged else "ragged",
            "backend": wrapper._backend,
            "causal": causal,
            "qo_lengths": qs,
            "kv_lengths": lengths,
            "page_size": page_size,
            "plan_info": plan,
            "actual_split": bool(plan[14]) if len(plan) == 15 else None,
            "inputs": [signature(t) for t in (q, k, v)],
            "run_options": config,
            "original_cached_module": type(wrapper._cached_module).__name__,
            "metadata_version_tracking": {
                "q_indptr": version(wrapper._qo_indptr_buf),
                "kv_indptr": version(
                    wrapper._paged_kv_indptr_buf if paged else wrapper._kv_indptr_buf
                ),
            },
            "diagnostic_only": True,
            "timing_valid": False,
            "resource_tactic_selected": False,
            "fresh_kernel_cases_consumed": 0,
        }
        reasons = []
        if not causal:
            reasons.append("frozen_v42_operator_identity_requires_causal")
        if paged and page_size != 16:
            reasons.append("frozen_v42_paged_qualification_requires_page16")
        if len(plan) != 15 or row["actual_split"] is not False:
            reasons.append("frozen_v42_requires_official15_field_unsplit_plan")
        if any(x is None for x in row["metadata_version_tracking"].values()):
            reasons.append("current_main_prepared_runner_rejects_untracked_metadata")
        row["integration_gaps_to_validate"] = reasons
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen:
            return
        if len(seen) >= 256:
            raise RuntimeError("Diagnostic geometry limit reached; preserve bounded evidence")
        seen.add(key)
        props = torch.cuda.get_device_properties(q.device)
        row.update(
            geometry_sha256=key,
            gpu_name=props.name,
            gpu_uuid=str(props.uuid),
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            flashinfer_version=flashinfer.__version__,
            wrapper_source_sha256=hashlib.sha256(
                Path(inspect.getfile(type(wrapper))).read_bytes()
            ).hexdigest(),
            observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        with path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def version(t):
        try:
            return t._version
        except RuntimeError:
            return None

    def instrument(cls, method, paged):
        original = getattr(cls, method)
        if getattr(original, "_sgi_geometry_observer", False):
            raise RuntimeError("Observer already installed")
        params = inspect.signature(original)

        @functools.wraps(original)
        def forward(self, *args, **kwargs):
            bound = params.bind(self, *args, **kwargs)
            bound.apply_defaults()
            observe(self, method, bound.arguments, paged)
            return original(self, *args, **kwargs)

        forward._sgi_geometry_observer = True
        setattr(cls, method, forward)

    for cls, paged in (
        (BatchPrefillWithPagedKVCacheWrapper, True),
        (BatchPrefillWithRaggedKVCacheWrapper, False),
    ):
        for method in ("forward", "forward_return_lse"):
            instrument(cls, method, paged)


if os.environ.get("SGI_PREFILL_GEOMETRY_DIR"):
    install()  # Also runs when multiprocessing spawn imports __mp_main__.

if __name__ == "__main__":
    if not os.environ.get("SGI_PREFILL_GEOMETRY_DIR"):
        raise RuntimeError("Explicit diagnostic output directory is required")
    runpy.run_module("sglang.launch_server", run_name="__main__")
