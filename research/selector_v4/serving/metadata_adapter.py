"""Explicit SGLang metadata boundaries for externally prepared eager registries.

No training or operator interception is installed here. The caller supplies real
pretrained leases/signature tensors and explicitly chooses this research hook.
Only ordinary page1 cached-prefix metadata can bind; graph boundaries release
leases and remain native. Hooking the backend precedes Triton/fast-plan writes.
"""

import functools
from dataclasses import dataclass


def host_lengths(value):
    """Never copy device metadata into the CPU on a request path."""
    if value is None or getattr(value, "is_cuda", False):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value):
        return None
    return tuple(value)


@dataclass
class Binding:
    registry: object
    key: object
    signature_inputs: object
    qo_lengths: tuple
    kv_lengths: tuple
    forward_options: dict


class MetadataEpochAdapter:
    def __init__(self, backend):
        self.backend = backend
        self.bindings = []
        self.frozen = False
        self.depth = 0
        self.originals = {}
        self.counters = {"updates": 0, "graph_updates": 0, "early_binds": 0}

    def attach(self, registry, key, signature_inputs, *, qo_lengths, kv_lengths, forward_options):
        if self.frozen:
            raise RuntimeError("Attach prepared registries before installation")
        if registry.owner not in self.backend.prefill_wrappers_paged:
            raise ValueError("Registry must belong to this backend's original paged wrapper")
        if not registry.frozen:
            raise ValueError("Freeze real trained decisions before serving attachment")
        if any(b.registry is registry and b.key == key for b in self.bindings):
            raise ValueError("Duplicate prepared binding")
        self.bindings.append(
            Binding(
                registry,
                key,
                signature_inputs,
                tuple(qo_lengths),
                tuple(kv_lengths),
                dict(forward_options),
            )
        )

    def before_update(self, graph):
        self.counters["updates"] += 1
        self.counters["graph_updates"] += int(graph)
        seen = set()
        for binding in self.bindings:
            registry = binding.registry
            if id(registry) not in seen:
                registry.before_metadata_update()
                seen.add(id(registry))

    def after_update(self, forward_batch, graph):
        if graph or self.backend.page_size != 1:
            return
        mode = forward_batch.forward_mode
        if not mode.is_extend_without_speculative():
            return
        metadata = self.backend.forward_metadata
        if not getattr(metadata, "use_ragged", False) or getattr(
            metadata, "extend_no_prefix", True
        ):
            return
        qo = host_lengths(forward_batch.extend_seq_lens_cpu)
        kv = host_lengths(forward_batch.extend_prefix_lens_cpu)
        if qo is None or kv is None or len(qo) != len(kv) or not all(qo) or not all(kv):
            return
        owners = metadata.prefill_wrappers
        for binding in self.bindings:
            if (
                binding.registry.owner not in owners
                or binding.qo_lengths != qo
                or binding.kv_lengths != kv
            ):
                continue
            rebound = binding.registry.bind_after_metadata_update(
                binding.key,
                binding.signature_inputs,
                qo_lengths=qo,
                kv_lengths=kv,
                forward_options=binding.forward_options,
            )
            self.counters["early_binds"] += int(rebound)

    def install(self):
        if self.frozen or self.originals:
            raise RuntimeError("Metadata adapter already installed")
        methods = (
            "init_forward_metadata",
            "init_forward_metadata_out_graph",
            "init_cuda_graph_state",
        )
        originals = {name: getattr(self.backend, name, None) for name in methods}
        if not all(callable(original) for original in originals.values()):
            raise ValueError("All reviewed metadata entry points are required")
        if any(getattr(original, "_sgi_epoch_adapter", False) for original in originals.values()):
            raise RuntimeError("Backend already has an epoch adapter")
        self.frozen = True
        for name, original in originals.items():
            self.originals[name] = original

            @functools.wraps(original)
            def boundary(
                *args, _original=original, _graph=name != "init_forward_metadata", **kwargs
            ):
                outer = self.depth == 0
                if outer:
                    self.before_update(_graph)
                self.depth += 1
                try:
                    result = _original(*args, **kwargs)
                    if outer and not _graph:
                        batch = kwargs.get("forward_batch", args[0] if args else None)
                        if batch is not None:
                            self.after_update(batch, False)
                    return result
                finally:
                    self.depth -= 1

            boundary._sgi_epoch_adapter = True
            setattr(self.backend, name, boundary)

    def restore(self):
        # Release before removing notifications; any retained lease stays unable
        # to execute until a new explicit owner performs its own binding.
        self.before_update(True)
        for name, original in self.originals.items():
            setattr(self.backend, name, original)
        self.originals.clear()
