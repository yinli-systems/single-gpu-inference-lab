"""Experimental eager serving bridge for one exclusively owned native plan.

This is not a promotion of the frozen v4.2 campaign. The current-main resource
runner requires immutable, versioned metadata. A lease copies the small metadata
buffers outside inference mode and binds them to the original planning epoch and
CUDA stream. Native scratch workspaces remain shared for this one epoch only.
Changing the owner plan invalidates the lease before any resource invocation.
Dynamic CUDA Graph replay is deliberately outside this bridge's contract.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import torch


class ServingPlanLease:
    def __init__(
        self, wrapper, inputs, *, qo_lengths, kv_lengths, forward_options, return_lse=False
    ):
        if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
            raise RuntimeError("Prepare the eager plan lease outside capture/tracing")
        from flashinfer.prefill import (
            BatchPrefillWithPagedKVCacheWrapper,
            make_prefill_resource_runner,
        )

        self.owner = wrapper
        self.paged = isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper)
        self.epoch = wrapper._plan_info
        self.native_module = wrapper._cached_module
        self.owner_metadata = self.metadata_binding(wrapper)
        self.stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
        self.options = dict(forward_options)
        self.return_lse = bool(return_lse)
        self.invalidated = False
        # Workspace buffers contain native planner schedules and execution
        # scratch; they are shared only while the owner's plan epoch is current.
        # Copy all metadata, including physical page indices, with version
        # counters even when the model itself uses torch.inference_mode().
        with torch.inference_mode(False):
            prepared = copy.copy(wrapper)
            prepared._plan_info = list(self.epoch)
            for name, value in vars(wrapper).items():
                if (
                    isinstance(value, torch.Tensor)
                    and name.endswith("_buf")
                    and "workspace_buffer" not in name
                ):
                    setattr(prepared, name, value.detach().clone())
        fields = {
            "causal": "_causal",
            "pos_encoding_mode": "_pos_encoding_mode",
            "use_fp16_qk_reduction": "_use_fp16_qk_reduction",
            "window_left": "_window_left",
            "logits_soft_cap": "_logits_soft_cap",
            "sm_scale": "_sm_scale",
            "rope_scale": "_rope_scale",
            "rope_theta": "_rope_theta",
        }
        method = wrapper.forward_return_lse if self.return_lse else wrapper.forward
        q, k, v = inputs
        args = (q, (k, v)) if self.paged else (q, k, v)
        bound = inspect.signature(method).bind(*args, **self.options)
        bound.apply_defaults()
        for name, field in fields.items():
            setattr(prepared, field, bound.arguments[name])
        self.run_options = {"return_lse": self.return_lse}
        for name in ("k_scale", "v_scale"):
            if name in bound.arguments:
                self.run_options[name] = bound.arguments[name]
        self.prepared = prepared
        self.runner = make_prefill_resource_runner(
            prepared,
            inputs,
            qo_lengths=qo_lengths,
            kv_lengths=kv_lengths,
            execution_mode="eager_run",
            run_kwargs=self.run_options,
        )
        if not self.current(inputs):
            raise RuntimeError("Owner replanned while preparing metadata snapshot")

    @staticmethod
    def metadata_binding(wrapper):
        def version(tensor):
            try:
                return tensor._version
            except RuntimeError:
                return None

        return tuple(
            (
                name,
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                version(value),
            )
            for name, value in sorted(vars(wrapper).items())
            if isinstance(value, torch.Tensor)
            and name.endswith("_buf")
            and "workspace_buffer" not in name
        )

    def invalidate(self):
        """The exclusive owner calls this before any untracked metadata write."""
        self.invalidated = True

    def current(self, inputs):
        valid = (
            not self.invalidated
            and self.owner._plan_info is self.epoch
            and self.owner._cached_module is self.native_module
            and self.metadata_binding(self.owner) == self.owner_metadata
            and torch.cuda.current_stream(inputs[0].device).cuda_stream == self.stream
            and not torch.cuda.is_current_stream_capturing()
            and not torch.compiler.is_compiling()
        )
        if not valid:
            self.invalidated = True
        return valid

    def native(self, inputs, options):
        q, k, v = inputs
        method = self.owner.forward_return_lse if self.return_lse else self.owner.forward
        args = (q, (k, v)) if self.paged else (q, k, v)
        return method(*args, **options)

    def calibrate(self, inputs):
        if not self.current(inputs):
            raise RuntimeError("Cannot calibrate an expired native planning epoch")
        receipt = self.runner.calibrate(inputs)
        if not self.current(inputs):
            raise RuntimeError("Native planning epoch changed during calibration")
        return receipt

    def run(self, inputs, *, forward_options):
        if dict(forward_options) != self.options or not self.current(inputs):
            return self.native(inputs, forward_options)
        # The runner independently revalidates the snapshot, certificate and
        # run options. Missing/inconclusive certificates remain native.
        return self.runner.run(inputs, **self.run_options)

    def evidence(self):
        return {
            "bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runner_identity": self.runner.identity,
            "native_plan_info": list(self.epoch),
            "metadata_owned": True,
            "dynamic_graph_supported": False,
            "exclusive_planning_owner_required": True,
            "source_epoch_invalidated": self.invalidated,
            "default_promotion": False,
            "serving_qualified": False,
            "historical_token_divergence_resolved": False,
        }
