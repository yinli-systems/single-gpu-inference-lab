"""Gated, uncertified fixed-geometry full-model Resource Graph diagnostic.

This creates a second graph and dedicated unsplit workspaces. Native SGLang
graphs and split-KV metadata are never replaced. It performs expensive readbacks
and duplicate Native references; all timings are invalid. No production mode.
"""

import __future__

import ast
import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path

from research.selector_v4.serving.resource_graph_contract import (
    ReplayTicket,
    supported_batch,
)

SOURCES = {
    "FlashInferAttnBackend": "5c8baba0d14eeca4c9d4bef8674318be1697b0160f105224282cd4aa27575810",
    "PrefillCudaGraphRunner": "48de8c6add0616f8d4824d170264b2f3d9dd9ccb7950a1c1cbdbea4c3d10e620",
    "FullCudaGraphBackend": "737f6d589845a10a934e32bc06bfb4836fe028dcd73850ecf6f2241c57df8c9d",
}


def transform(source, expected=SOURCES["FlashInferAttnBackend"]):
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("Unreviewed SGLang forward source")
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"
    )
    calls = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "prefill_wrapper_paged"
        and n.func.attr == "forward"
    ]
    if len(calls) != 1:
        raise ValueError("Exactly one reviewed full paged forward required")
    call = calls[0]
    call.func = ast.Name(id="_sgi_full_graph_paged", ctx=ast.Load())
    call.args = [
        ast.Name(id="self", ctx=ast.Load()),
        ast.Name(id="prefill_wrapper_paged", ctx=ast.Load()),
        ast.Name(id="layer", ctx=ast.Load()),
        *call.args,
    ]
    return ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))


def tensor_digest(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def binding(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


class LayerEntry:
    """Owned buffers for a single real layer; private functional execution only."""

    def __init__(self, owner, q, cache, options, tokens):
        import torch
        from flashinfer.prefill import (
            BatchPrefillWithPagedKVCacheWrapper,
            make_prefill_resource_runner,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Resource preparation must finish during explicit warmup"
            )
        if q.ndim != 3 or q.shape[0] != tokens or q.shape[-1] != 128:
            raise ValueError("Exact fixed positive query geometry required")
        if not isinstance(cache, (tuple, list)) or len(cache) != 2:
            raise ValueError("Explicit real layer K/V tuple required")
        k, v = cache
        if k.ndim == 3:
            k, v = k.unsqueeze(1), v.unsqueeze(1)
        if k.ndim != 4 or k.shape[1] != 1 or k.shape != v.shape:
            raise ValueError("Actual native NHD page1 cache required")
        if (
            options.get("causal") is not True
            or options.get("window_left", -1) != -1
            or options.get("logits_soft_cap", 0) not in (None, 0)
        ):
            raise ValueError("Only unscaled standard causal attention is reviewed")
        if any(options.get(name) not in (None, 1.0) for name in ("k_scale", "v_scale")):
            raise ValueError("Quantized KV scaling is unsupported")
        self.owner, self.options, self.tokens = owner, dict(options), tokens
        self.cache_binding = (binding(k), binding(v))
        # Resource's metadata must have version counters, even when the server
        # constructs model/cache tensors under inference_mode.
        with torch.inference_mode(False):
            self.q = torch.empty_like(q)
            self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
            self.qptr = torch.tensor([0, tokens], dtype=torch.int32, device=q.device)
            self.kptr = self.qptr.clone()
            self.indices = owner._paged_kv_indices_buf[:tokens].detach().clone()
            self.last = torch.ones(1, dtype=torch.int32, device=q.device)
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace, "NHD", backend="fa2"
            )
            self.wrapper.plan(
                self.qptr,
                self.kptr,
                self.indices,
                self.last,
                q.shape[1],
                k.shape[2],
                128,
                1,
                causal=True,
                sm_scale=options["sm_scale"],
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                disable_split_kv=True,
            )
        self.inputs = [self.q, k, v]
        self.q.copy_(q)
        self.runner = make_prefill_resource_runner(
            self.wrapper,
            self.inputs,
            qo_lengths=[tokens],
            kv_lengths=[tokens],
            execution_mode="graph_replay",
            graph_replays=1,
            run_kwargs={"return_lse": True},
        )
        if not self.runner._eligible or not self.runner._prepare(self.inputs):
            raise RuntimeError("Dedicated unsplit Resource plan not eligible")
        if self.runner.receipt is not None or self.runner._receipt_valid:
            raise RuntimeError(
                "Functional diagnostic must not fabricate/reuse a certificate"
            )
        self.owned_binding = self.frame(include_payload=False)
        self.resource_result = self.native_result = None

    def frame(self, *, include_payload=True):
        import torch

        tensors = (self.qptr, self.kptr, self.indices, self.last)
        return (
            tuple(binding(t) for t in tensors),
            tuple(self.wrapper._plan_info),
            binding(self.q),
            self.cache_binding,
            tuple(
                (name, binding(value))
                for name, value in sorted(vars(self.wrapper).items())
                if isinstance(value, torch.Tensor)
                and (name.endswith("_buf") or "workspace_buffer" in name)
            ),
            tuple(tensor_digest(t) for t in tensors) if include_payload else None,
        )

    def live_owner_frame(self):
        owner = self.owner
        metadata = (
            owner._qo_indptr_buf,
            owner._paged_kv_indptr_buf,
            owner._paged_kv_indices_buf[: self.tokens],
            owner._paged_kv_last_page_len_buf,
        )
        return (
            id(owner._plan_info),
            tuple(owner._plan_info),
            tuple((binding(t), tensor_digest(t)) for t in metadata),
        )

    def refresh(self):
        owner = self.owner
        if owner._qo_indptr_buf.cpu().tolist() != [
            0,
            self.tokens,
        ] or owner._paged_kv_indptr_buf.cpu().tolist() != [0, self.tokens]:
            raise RuntimeError("Live native metadata is outside the fixed geometry")
        if owner._paged_kv_last_page_len_buf.cpu().tolist() != [1]:
            raise RuntimeError("Native page1 boundary changed")
        indices = owner._paged_kv_indices_buf[: self.tokens]
        if (
            indices.numel() != self.tokens
            or bool((indices < 0).any())
            or bool((indices >= self.inputs[1].shape[0]).any())
        ):
            raise RuntimeError("Physical page ownership outside real KV cache")
        if self.frame(include_payload=False) != self.owned_binding:
            raise RuntimeError("Captured Resource buffers changed")
        self.indices.copy_(indices)

    def run(self, q, cache, options):
        import torch

        k, v = cache
        if k.ndim == 3:
            k, v = k.unsqueeze(1), v.unsqueeze(1)
        if (
            dict(options) != self.options
            or (binding(k), binding(v)) != self.cache_binding
        ):
            raise RuntimeError("Real layer cache/options differ from prepared capture")
        self.q.copy_(q)
        self.native_result = self.wrapper.run(self.q, (k, v), return_lse=True)
        self.resource_result = self.runner._resource(self.inputs, return_lse=True)
        if not torch.cuda.is_current_stream_capturing() and not all(
            torch.equal(a, b) for a, b in zip(self.native_result, self.resource_result)
        ):
            raise RuntimeError("Warmup O/LSE exact comparison failed")
        return self.resource_result[0]

    def verify_output(self):
        import torch

        if not all(
            torch.equal(a, b) for a, b in zip(self.native_result, self.resource_result)
        ):
            raise RuntimeError("Actual Resource graph O/LSE mismatch")
        pages = self.indices.long()
        return {
            "q_sha256": tensor_digest(self.q),
            "k_sha256": tensor_digest(self.inputs[1].index_select(0, pages)),
            "v_sha256": tensor_digest(self.inputs[2].index_select(0, pages)),
            "page_indices_sha256": tensor_digest(self.indices),
            "output_sha256": tensor_digest(self.resource_result[0]),
            "out_lse_exact": True,
        }


def install(root, kernel_campaign, *, tokens=128):
    # Fail before SGLang/Torch import or server allocation on any missing gate.
    from research.selector_v4.serving.http_pipeline import authorize

    authorize(kernel_campaign)
    import torch
    from flashinfer._build_meta import __git_commit__
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    if __git_commit__ != "75544a17ce0019ca877f50354d95451ee089f859" or tokens != 128:
        raise ValueError("Only pinned candidate and preregistered 128-token geometry")
    classes = (FlashInferAttnBackend, PrefillCudaGraphRunner, FullCudaGraphBackend)
    for cls in classes:
        if (
            hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            != SOURCES[cls.__name__]
        ):
            raise RuntimeError("Unreviewed Graph source: " + cls.__name__)
    if getattr(FlashInferAttnBackend.forward_extend, "_sgi_resource_graph", False):
        raise RuntimeError("Resource Graph diagnostic already installed")
    root.mkdir(parents=True, exist_ok=True)

    def record(value):
        value.update(
            pid=os.getpid(),
            timing_valid=False,
            certificate_authority=False,
            default_promotion=False,
            serving_promotion=False,
        )
        with (root / f"resource-graph-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def route(backend, owner, layer, q, cache, **options):
        context = getattr(backend, "_sgi_resource_capture", None)
        if context is None:
            return owner.forward(q, cache, **options)
        key = layer.layer_id
        if key not in context["entries"]:
            if len(context["entries"]) >= 40:
                raise RuntimeError("Finite per-layer diagnostic workspace limit")
            context["entries"][key] = LayerEntry(owner, q, cache, options, tokens)
        return context["entries"][key].run(q, cache, options)

    original_forward = FlashInferAttnBackend.forward_extend
    method = inspect.unwrap(original_forward)
    namespace = dict(method.__globals__, _sgi_full_graph_paged=route)
    flags = sum(
        getattr(__future__, n).compiler_flag for n in __future__.all_feature_names
    )
    tree = transform(Path(inspect.getfile(FlashInferAttnBackend)).read_text())
    exec(  # noqa: S102 - exact-source single-call AST rewrite
        compile(
            tree,
            inspect.getfile(FlashInferAttnBackend),
            "exec",
            flags=method.__code__.co_flags & flags,
            dont_inherit=True,
        ),
        namespace,
    )
    changed = namespace["forward_extend"]
    changed._sgi_resource_graph = True

    original_capture = FullCudaGraphBackend.capture_one
    original_load = PrefillCudaGraphRunner.load_batch
    original_replay = FullCudaGraphBackend.replay

    @functools.wraps(original_capture)
    def capture(self, shape_key, forward_fn, *args, **kwargs):
        original_capture(self, shape_key, forward_fn, *args, **kwargs)
        runner = self._cuda_graph_runner
        if (
            not isinstance(runner, PrefillCudaGraphRunner)
            or shape_key.size != tokens
            or shape_key.variant_label is not None
        ):
            return
        if runner._capture_req_slots != 1:
            raise RuntimeError("Exactly one full-prefill request slot required")
        backend = runner.model_runner.attn_backend
        native_graph, native_output = self._graphs[shape_key], self._outputs[shape_key]
        context = {"entries": {}}
        backend._sgi_resource_capture = context
        try:
            original_capture(self, shape_key, forward_fn, *args, **kwargs)
            if not context["entries"]:
                raise RuntimeError(
                    "Resource capture reached no real paged attention layers"
                )
            context.update(
                graph=self._graphs[shape_key],
                output=self._outputs[shape_key],
                key=shape_key,
                ticket=ReplayTicket(),
                native_graph=native_graph,
            )
            self._sgi_resource_context = context
            record(
                {
                    "event": "captured",
                    "layers": len(context["entries"]),
                    "native_graph_object": id(native_graph),
                    "resource_graph_object": id(context["graph"]),
                }
            )
        finally:
            backend._sgi_resource_capture = None
            self._graphs[shape_key], self._outputs[shape_key] = (
                native_graph,
                native_output,
            )

    def frame(context):
        return (
            id(context["graph"]),
            id(context["native_graph"]),
            torch.cuda.current_stream().cuda_stream,
            tuple(
                (key, entry.frame(), entry.live_owner_frame())
                for key, entry in sorted(context["entries"].items())
            ),
        )

    @functools.wraps(original_load)
    def load(self, batch, *args, **kwargs):
        context = getattr(self.backend, "_sgi_resource_context", None)
        if context is None:
            return original_load(self, batch, *args, **kwargs)
        ticket = context["ticket"]
        if self.backend._graphs[context["key"]] is not context["native_graph"]:
            ticket.fail()
            raise RuntimeError("Native capture identity changed")
        ticket.begin()  # Before any real input/metadata/glue writes.
        try:
            result = original_load(self, batch, *args, **kwargs)
            eligible = supported_batch(batch, tokens)
            if eligible:
                for entry in context["entries"].values():
                    entry.refresh()
            ticket.finish(frame(context) if eligible else None, eligible=eligible)
            return result
        except BaseException:
            ticket.fail()
            raise

    @functools.wraps(original_replay)
    def replay(self, shape_key, batch, **kwargs):
        context = getattr(self, "_sgi_resource_context", None)
        if context is None or shape_key != context["key"]:
            return original_replay(self, shape_key, batch, **kwargs)
        ticket = context["ticket"]
        if self._graphs[shape_key] is not context["native_graph"]:
            ticket.fail()
            raise RuntimeError("Native capture identity changed before replay")
        if not ticket.consume(frame(context) if ticket.state == "READY" else None):
            record({"event": "native_fallback", "epoch": ticket.epoch})
            return original_replay(self, shape_key, batch, **kwargs)
        try:
            # The complete Native model graph executes before Resource. Its
            # output is copied before any shared graph-pool buffers are reused.
            control = original_replay(self, shape_key, batch, **kwargs)
            if not torch.is_tensor(control):
                raise RuntimeError("Reviewed full-prefill tensor output required")
            expected = control.clone()
            if frame(context) != ticket.frame:
                raise RuntimeError("Metadata changed after Native reference replay")
            with torch.profiler.record_function("sgi_resource_full_model_graph"):
                context["graph"].replay()
            actual = context["output"]
            if not torch.equal(actual, expected):
                raise RuntimeError("Full-model graph output differs from Native")
            layers = {
                str(key): entry.verify_output()
                for key, entry in context["entries"].items()
            }
            record(
                {
                    "event": "resource_replay",
                    "epoch": ticket.epoch,
                    "graph_object": id(context["graph"]),
                    "layers": layers,
                    "full_model_output_exact": True,
                    "metadata_frame_sha256": hashlib.sha256(
                        repr(ticket.frame).encode()
                    ).hexdigest(),
                }
            )
            return actual
        except BaseException as error:
            ticket.fail()
            record(
                {
                    "event": "replay_failure",
                    "error": type(error).__name__,
                    "message": str(error),
                }
            )
            raise

    # Only install after all exact-source/AST checks have passed.
    FlashInferAttnBackend.forward_extend = changed
    FullCudaGraphBackend.capture_one = capture
    PrefillCudaGraphRunner.load_batch = load
    FullCudaGraphBackend.replay = replay


if __name__ in ("__main__", "__mp_main__"):
    root = os.environ.get("SGI_RESOURCE_GRAPH_DIAGNOSTIC")
    kernel = os.environ.get("SGI_FORMAL_KERNEL_CAMPAIGN")
    if not root or not kernel:
        raise RuntimeError(
            "Explicit diagnostic output and fully qualified kernel campaign required"
        )
    install(Path(root), Path(kernel))
    if __name__ == "__main__":
        runpy.run_module("sglang.launch_server", run_name="__main__")
