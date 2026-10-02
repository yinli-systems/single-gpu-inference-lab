"""Uncertified paged Resource Graph with explicit, fixed-geometry epochs.

This probe owns every captured metadata/workspace buffer. It stages a new
Native plan into those SAME buffers only after an announced, synchronized
update. It never rebinds a public Graph runner or reuses an eager certificate.
Device metadata readback is deliberately included in this diagnostic boundary.
It is not a serving adapter, performance qualification or promotion authority.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import dataclass


@dataclass
class EpochGuard:
    frame: object
    state: str = "READY"
    generation: int = 0
    reason: str | None = None

    def retire(self, reason):
        self.state = "RETIRED"
        self.reason = reason

    def begin(self, frame):
        if self.state != "READY" or frame != self.frame:
            self.retire("unannounced_change_or_invalid_transition")
            return False
        self.state = "UPDATING"
        self.generation += 1
        return True

    def finish(self, frame, *, compatible):
        if self.state != "UPDATING" or not compatible:
            self.retire("new_plan_incompatible_or_update_not_announced")
            return False
        self.frame = frame
        self.state = "READY"
        self.reason = None
        return True

    def permits(self, frame):
        if self.state == "UPDATING":
            raise RuntimeError("No attention execution during metadata update")
        if self.state != "READY":
            return False
        if frame != self.frame:
            self.retire("unannounced_metadata_or_capture_binding_change")
            return False
        return True


class GraphEpochProbe:
    def __init__(self, owner, inputs, *, qo_lengths, kv_lengths, forward_options,
                 replays=1, allow_uncertified_diagnostic=False):
        if not allow_uncertified_diagnostic:
            raise RuntimeError("Explicit uncertified diagnostic opt-in required")
        import torch
        from flashinfer.prefill import (
            BatchPrefillWithPagedKVCacheWrapper,
            make_prefill_resource_runner,
        )

        if replays not in (1, 16) or not isinstance(owner, BatchPrefillWithPagedKVCacheWrapper):
            raise ValueError("Only fixed paged Graph1/16 diagnostics are supported")
        if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
            raise RuntimeError("Construct and validate outside capture/tracing")
        self.owner = owner
        self.qo_lengths, self.kv_lengths = tuple(qo_lengths), tuple(kv_lengths)
        self.options = dict(forward_options)
        self.input_binding = self.bind_inputs(inputs)
        self.stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
        self.plan_values = tuple(owner._plan_info or ())
        self.native_module = owner._cached_module
        self.owner_envelope = self.envelope(owner)
        self.replays = replays
        self.last_execution = None
        self.replay_count = 0
        self.update_count = 0
        self.copy_bytes = []
        q, k, v = inputs
        bound = inspect.signature(owner.forward_return_lse).bind(q, (k, v), **self.options)
        bound.apply_defaults()
        with torch.inference_mode(False):
            self.owned = copy.copy(owner)
            self.owned._plan_info = list(self.plan_values)
            for name, value in vars(owner).items():
                if isinstance(value, torch.Tensor) and (name.endswith("_buf") or "workspace_buffer" in name):
                    setattr(self.owned, name, value.detach().clone())
        for name, field in (
            ("causal", "_causal"), ("pos_encoding_mode", "_pos_encoding_mode"),
            ("use_fp16_qk_reduction", "_use_fp16_qk_reduction"), ("window_left", "_window_left"),
            ("logits_soft_cap", "_logits_soft_cap"), ("sm_scale", "_sm_scale"),
            ("rope_scale", "_rope_scale"), ("rope_theta", "_rope_theta"),
        ):
            setattr(self.owned, field, bound.arguments[name])
        self.runner = make_prefill_resource_runner(
            self.owned, inputs, qo_lengths=self.qo_lengths, kv_lengths=self.kv_lengths,
            execution_mode="graph_replay", graph_replays=replays, run_kwargs={"return_lse": True},
        )
        if not self.runner._eligible or not self.runner._prepare(inputs):
            raise RuntimeError("Unsupported or unprepared Resource Graph envelope")
        assert self.runner.receipt is None and not self.runner._receipt_valid
        self.owned_binding = self.buffer_binding(self.owned)
        if not self.compatible(inputs, self.options):
            raise RuntimeError("Invalid initial physical metadata or buffer binding")
        frame = self.frame(inputs)
        for _ in range(3):
            self.runner._resource(inputs, return_lse=True)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for _ in range(replays):
                self._result = self.runner._resource(inputs, return_lse=True)
        torch.cuda.synchronize()
        if self.frame(inputs) != frame:
            raise RuntimeError("Owner changed while capturing")
        self.guard = EpochGuard(frame)
        self.accepted_owned_frame = self.metadata_frame(self.owned)

    @staticmethod
    def envelope(wrapper):
        return tuple((name, repr(getattr(wrapper, name, None))) for name in (
            "_backend", "_kv_layout", "_causal", "_pos_encoding_mode", "_use_fp16_qk_reduction",
            "_window_left", "_logits_soft_cap", "_sm_scale", "_rope_scale", "_rope_theta",
            "_prefix_len_ptr", "_token_pos_in_items_ptr", "_max_item_len_ptr", "_sinks",
        ))

    @staticmethod
    def bind_inputs(inputs):
        return tuple((x.data_ptr(), tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device)) for x in inputs)

    @staticmethod
    def buffer_binding(wrapper):
        import torch
        return tuple((name, value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                      str(value.dtype), str(value.device)) for name, value in sorted(vars(wrapper).items())
                     if isinstance(value, torch.Tensor) and (name.endswith("_buf") or "workspace_buffer" in name))

    @staticmethod
    def metadata_frame(wrapper):
        import torch
        # Read values too: foreign writes to inference tensors do not increment
        # PyTorch versions. Readback is expensive and is not hidden as replay-only.
        return tuple((name, value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                      str(value.dtype), str(value.device),
                      hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                     for name, value in sorted(vars(wrapper).items())
                     if isinstance(value, torch.Tensor) and name.endswith("_buf") and "workspace_buffer" not in name)

    def frame(self, inputs):
        import torch
        return (id(self.owner._plan_info), tuple(self.owner._plan_info or ()),
                id(self.owner._cached_module), self.metadata_frame(self.owner),
                self.buffer_binding(self.owner), self.bind_inputs(inputs),
                torch.cuda.current_stream(inputs[0].device).cuda_stream, self.envelope(self.owner))

    def compatible(self, inputs, options):
        import torch
        if (dict(options) != self.options or self.bind_inputs(inputs) != self.input_binding
                or tuple(self.owner._plan_info or ()) != self.plan_values
                or self.owner._cached_module is not self.native_module
                or self.envelope(self.owner) != self.owner_envelope
                or self.buffer_binding(self.owned) != self.owned_binding
                or torch.cuda.current_stream(inputs[0].device).cuda_stream != self.stream
                or torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
            return False
        if {x[0] for x in self.buffer_binding(self.owner)} != {x[0] for x in self.owned_binding}:
            return False
        # Every buffer copied into the capture must preserve size/type/stride.
        for name, _, shape, stride, dtype, device in self.owned_binding:
            actual = getattr(self.owner, name, None)
            if not isinstance(actual, torch.Tensor) or (tuple(actual.shape), tuple(actual.stride()), str(actual.dtype), str(actual.device)) != (shape, stride, dtype, device):
                return False
        w = self.owner
        qo = w._qo_indptr_buf.cpu().tolist()
        ptr = w._paged_kv_indptr_buf.cpu().tolist()
        last = w._paged_kv_last_page_len_buf.cpu().tolist()
        indices = w._paged_kv_indices_buf.cpu().tolist()
        page_size = inputs[1].shape[1 if w._kv_layout == "NHD" else 2]
        actual_q = tuple(b-a for a, b in zip(qo, qo[1:]))
        if len(ptr) != len(self.kv_lengths)+1 or len(last) != len(self.kv_lengths):
            return False
        actual_kv = tuple((ptr[i+1]-ptr[i]-1)*page_size+last[i] for i in range(len(last)))
        return (actual_q == self.qo_lengths and actual_kv == self.kv_lengths
                and qo[0] == ptr[0] == 0 and ptr[-1] == len(indices)
                and all(b > a for a, b in zip(ptr, ptr[1:]))
                and all(0 < x <= page_size for x in last)
                and all(0 <= x < inputs[1].shape[0] for x in indices))

    def before_metadata_update(self, inputs):
        import torch
        if not self.guard.begin(self.frame(inputs)):
            return False
        try:
            torch.cuda.synchronize(inputs[0].device)
            return True
        except Exception:
            self.guard.retire("synchronization_failed")
            raise

    def bind_after_metadata_update(self, inputs, *, forward_options):
        if self.guard.state != "UPDATING":
            self.guard.retire("update_not_announced")
            return False
        try:
            if not self.compatible(inputs, forward_options):
                self.guard.retire("unsupported_new_plan_geometry_or_binding")
                return False
            copied = 0
            # Full buffer copy includes the Native plan's integer scheduling
            # state; private workspaces isolate Native-after-Resource execution.
            for name, *_ in self.owned_binding:
                source, target = getattr(self.owner, name), getattr(self.owned, name)
                target.copy_(source)
                copied += source.numel()*source.element_size()
            self.copy_bytes.append(copied)
            self.accepted_owned_frame = self.metadata_frame(self.owned)
            ok = self.guard.finish(self.frame(inputs), compatible=True)
            self.update_count += int(ok)
            return ok
        except Exception:
            self.guard.retire("metadata_copy_or_validation_failed")
            raise

    def native(self, inputs, options):
        return self.owner.forward_return_lse(inputs[0], (inputs[1], inputs[2]), **options)

    def replay(self, inputs, *, forward_options):
        if self.guard.state == "UPDATING":
            raise RuntimeError("No Native or Resource execution during update")
        valid = (dict(forward_options) == self.options and self.guard.permits(self.frame(inputs))
                 and self.metadata_frame(self.owned) == self.accepted_owned_frame
                 and self.buffer_binding(self.owned) == self.owned_binding)
        if not valid:
            self.guard.retire("stale_or_unsupported_replay_boundary")
            self.last_execution = "native_eager"
            return self.native(inputs, forward_options)
        self._graph.replay()
        self.replay_count += 1
        self.last_execution = "uncertified_resource_graph"
        return self._result

    def evidence(self):
        return {"diagnostic_only": True, "fixed_geometry_metadata_epochs": self.update_count,
                "graph_replays_per_call": self.replays, "resource_graph_replay_calls": self.replay_count,
                "captured_buffers_owned": True, "copied_bytes_per_epoch": self.copy_bytes,
                "metadata_readback_included": True, "guard_state": self.guard.state,
                "guard_reason": self.guard.reason, "certificate_reused": False,
                "public_graph_runner_rebound": False, "resource_graph_serving_qualified": False,
                "qualification_authority": False, "fresh_cases_consumed": 0,
                "default_promotion": False, "serving_promotion": False,
                "historical_token_divergence_resolved": False}
