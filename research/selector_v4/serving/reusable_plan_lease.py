"""Explicit eager ownership transitions for an already prepared serving runner.

The scheduler must release this lease before replanning or any foreign metadata
write. Only a subsequent explicit bind can authorize another resource call.
This draft does not install hooks, train on requests, or authorize serving.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import torch

from research.selector_v4.serving.plan_lease import ServingPlanLease


class ReusableServingPlanLease(ServingPlanLease):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = "BOUND"
        self.owner_generation = 0
        self.bound_generation = 0
        self.owner_plan_values = tuple(self.epoch)
        self.last_reason = None

    def release_before_update(self):
        """Call before the first scheduler/planner/Triton metadata write."""
        if self.state == "RETIRED":
            raise RuntimeError("A retired lease cannot resume resource execution")
        self.owner_generation += 1
        self.invalidated = True
        self.state = "RELEASED"

    def invalidate(self):
        # Unannounced mutation or boundary rejection is terminal. The explicit
        # release operation alone permits a later same-geometry eager bind.
        self.invalidated = True
        self.state = "RETIRED"

    def current(self, inputs):
        # ServingPlanLease.__init__ invokes this before subclass fields exist.
        if hasattr(self, "state"):
            if self.state != "BOUND":
                return False
            if (
                self.bound_generation != self.owner_generation
                or tuple(self.owner._plan_info or ()) != self.owner_plan_values
            ):
                self.last_reason = "unannounced_owner_plan_mutation"
                self.invalidate()
                return False
        valid = super().current(inputs)
        if not valid and hasattr(self, "state"):
            self.invalidate()
        return valid

    def bind_current_plan(self, inputs, *, qo_lengths, kv_lengths, forward_options):
        """Reuse source audit/module/certificate; include this cost in serving.

        Metadata is recopied into owned versioned tensors after the native plan.
        Public rebind verifies the actual GPU metadata, complete plan values,
        compiler envelope, stream, input geometry and options. A failed bind
        retires this lease; it never retries calibration or compilation.
        """
        if self.state != "RELEASED":
            self.last_reason = "update_not_announced"
            self.invalidate()
            return False
        if (
            torch.cuda.is_current_stream_capturing()
            or torch.compiler.is_compiling()
            or dict(forward_options) != self.options
            or tuple(qo_lengths) != self.runner.qo_lengths
            or tuple(kv_lengths) != self.runner.kv_lengths
            or not self.runner._receipt_valid
            or self.runner._proxy is None
            or self.owner._cached_module is not self.native_module
        ):
            self.last_reason = "unsupported_envelope_or_unprepared_runner"
            self.invalidate()
            return False
        owner_epoch = self.owner._plan_info
        owner_metadata = self.metadata_binding(self.owner)
        q, k, v = inputs
        method = self.owner.forward_return_lse if self.return_lse else self.owner.forward
        arguments = (q, (k, v)) if self.paged else (q, k, v)
        bound = inspect.signature(method).bind(*arguments, **forward_options)
        bound.apply_defaults()
        # Keep the wrapper object that the public runner owns. Replacing its
        # metadata is permitted only while the lease cannot execute a resource.
        with torch.inference_mode(False):
            fresh = copy.copy(self.owner)
            fresh._plan_info = list(owner_epoch or ())
            for name, value in vars(self.owner).items():
                if (
                    isinstance(value, torch.Tensor)
                    and name.endswith("_buf")
                    and "workspace_buffer" not in name
                ):
                    setattr(fresh, name, value.detach().clone())
        for name, field in (
            ("causal", "_causal"),
            ("pos_encoding_mode", "_pos_encoding_mode"),
            ("use_fp16_qk_reduction", "_use_fp16_qk_reduction"),
            ("window_left", "_window_left"),
            ("logits_soft_cap", "_logits_soft_cap"),
            ("sm_scale", "_sm_scale"),
            ("rope_scale", "_rope_scale"),
            ("rope_theta", "_rope_theta"),
        ):
            setattr(fresh, field, bound.arguments[name])
        vars(self.prepared).clear()
        vars(self.prepared).update(vars(fresh))
        rebound = self.runner.rebind_same_geometry(inputs)
        unchanged_owner = (
            self.owner._plan_info is owner_epoch
            and self.owner._cached_module is self.native_module
            and self.metadata_binding(self.owner) == owner_metadata
            and torch.cuda.current_stream(inputs[0].device).cuda_stream == self.stream
        )
        if not rebound or not unchanged_owner:
            self.last_reason = "public_rebind_or_owner_epoch_rejected"
            self.invalidate()
            return False
        self.epoch = owner_epoch
        self.owner_plan_values = tuple(owner_epoch)
        self.owner_metadata = owner_metadata
        self.bound_generation = self.owner_generation
        self.invalidated = False
        self.state = "BOUND"
        self.last_reason = None
        return self.current(inputs)

    def run(self, inputs, *, forward_options):
        if dict(forward_options) != self.options:
            self.last_reason = "forward_options_changed"
            self.invalidate()
        if self.state != "BOUND":
            return self.native(inputs, forward_options)
        return super().run(inputs, forward_options=forward_options)

    def evidence(self):
        return {
            **super().evidence(),
            "reuse_bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state": self.state,
            "owner_generation": self.owner_generation,
            "bound_generation": self.bound_generation,
            "last_reason": self.last_reason,
            "explicit_release_before_foreign_write_required": True,
            "public_eager_rebind_used": True,
            "resource_graph_metadata_updates_supported": False,
        }
