"""Explicit benchmark adapter for fixed, owned geometry and a private resource JIT.

This is not a serving runner. It reuses native buffers while repeating the
unaltered native planner, then hands the new native plan to a private proxy.
Ownership, metadata and configuration changes reject the trial. Deployment
selection, CUDA Graph input updates, and independent holdouts remain unqualified.
"""

from __future__ import annotations

import functools


def metadata(wrapper):
    import torch

    return tuple(
        (
            name,
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            value._version,
        )
        for name, value in sorted(vars(wrapper).items())
        if name.endswith("_buf") and isinstance(value, torch.Tensor)
    )


def configuration(wrapper):
    return tuple(
        (name, repr(getattr(wrapper, name, None)))
        for name in (
            "_backend",
            "_causal",
            "_kv_layout",
            "_pos_encoding_mode",
            "_window_left",
            "_logits_soft_cap",
            "_use_fp16_qk_reduction",
            "_sm_scale",
            "_rope_scale",
            "_rope_theta",
        )
    )


class SharedResourceRuntime:
    def __init__(self, native, inputs, *, qo_lengths, kv_lengths):
        from flashinfer.prefill import make_prefill_resource_runner

        native.plan()
        self.native = native
        self.torch = native.torch
        self.layout = native.layout
        self.ws = native.ws
        self.out = native.out
        self.lse = native.lse
        self.wrapper = native.wrapper
        self.eager_calls = native.eager_calls
        self.graphs = {}
        self.inputs = tuple(inputs)
        self.plan_signature = tuple(self.wrapper._plan_info)
        self.module = self.wrapper._cached_module
        self.metadata = metadata(self.wrapper)
        self.configuration = configuration(self.wrapper)
        self.runner = make_prefill_resource_runner(
            self.wrapper,
            inputs,
            qo_lengths=qo_lengths,
            kv_lengths=kv_lengths,
            run_kwargs={"out": self.out, "lse": self.lse, "return_lse": True},
        )
        if not self.runner._eligible or not self.runner._prepare(inputs):
            raise RuntimeError("fixed target does not support the private resource tactic")
        self.proxy = self.runner._proxy
        q, k, v = inputs
        kwargs = {"out": self.out, "lse": self.lse, "return_lse": True}
        self._call = (
            functools.partial(self.proxy.run, q, (k, v), **kwargs)
            if self.layout == "paged"
            else functools.partial(self.proxy.run, q, k, v, **kwargs)
        )
        assert self.proxy._float_workspace_buffer is self.wrapper._float_workspace_buffer
        assert self.proxy._int_workspace_buffer is self.wrapper._int_workspace_buffer
        self._check()

    def _check(self):
        if (
            self.wrapper._cached_module is not self.module
            or metadata(self.wrapper) != self.metadata
            or configuration(self.wrapper) != self.configuration
        ):
            raise RuntimeError("fixed native owner metadata/module/configuration changed")

    def attach(self, q):
        if q is not self.inputs[0]:
            raise RuntimeError("new Q binding requires a new diagnostic adapter")

    def plan(self, inspect=True):
        self._check()
        self.native.plan(inspect=False)
        self._check()
        if tuple(self.wrapper._plan_info) != self.plan_signature:
            raise RuntimeError("native planner changed the fixed geometry plan")
        self.proxy._plan_info = self.wrapper._plan_info
        if inspect:
            return list(self.plan_signature), False
        return None

    def call(self, q):
        if q is not self.inputs[0]:
            raise RuntimeError("new Q binding requires a new diagnostic adapter")
        return self._call()

    def capture(self, q, count):
        self.plan()
        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph):
            for _ in range(count):
                self.call(q)
        self.graphs[count] = graph

    def pointers(self, q, k, v):
        return self.native.pointers(q, k, v)
