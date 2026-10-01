"""Explicit frozen eager cached-prefix routing through real public leases.

There is no registration, constructor, training or metadata readback here.
Unknown geometry, options, graphs and leases not bound before the model stay
native. A routed attempt is not evidence of an actual resource kernel launch.
"""

import functools


def signature(inputs):
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in inputs)


class PagedPrefixRouter:
    def __init__(self, adapter, *, graph_or_tracing):
        if not adapter.frozen:
            raise ValueError("Install the frozen metadata adapter before routing")
        self.adapter = adapter
        self.graph_or_tracing = graph_or_tracing
        self.bindings = {}
        self.counters = {"native_calls": 0, "prepared_attempts": 0}
        for binding in adapter.bindings:
            entry = binding.registry.entries.get(binding.key)
            if entry is None:
                continue  # Native/inconclusive training is retained, never overridden.
            if not entry.lease.return_lse:
                raise ValueError("Actual cached-prefix merge requires output and LSE")
            key = (
                id(binding.registry.owner),
                binding.qo_lengths,
                binding.kv_lengths,
                signature(binding.signature_inputs),
            )
            if key in self.bindings:
                raise ValueError("Keep one frozen decision for each actual input geometry")
            self.bindings[key] = binding

    def run(self, owner, q, kv_cache, **forward_options):
        if not self.bindings:
            self.counters["native_calls"] += 1
            return owner.forward_return_lse(q, kv_cache, **forward_options)
        native = functools.partial(owner.forward_return_lse, q, kv_cache, **forward_options)
        geometry = self.adapter.current_prefix_lengths
        if geometry is None or self.adapter.depth or self.graph_or_tracing():
            self.counters["native_calls"] += 1
            return native()
        inputs = [q, *kv_cache]
        key = (id(owner), *geometry, signature(inputs))
        binding = self.bindings.get(key)
        if binding is None or dict(forward_options) != binding.forward_options:
            self.counters["native_calls"] += 1
            return native()
        registry = binding.registry
        entry = registry.entries.get(binding.key)
        if (
            entry is None
            or entry.bound_epoch != registry.epoch
            or binding.key not in registry.bound_keys
            or entry.lease.state != "BOUND"
        ):
            # Never move a missed metadata binding into first-layer computation.
            self.counters["native_calls"] += 1
            return native()
        self.counters["prepared_attempts"] += 1
        return registry.run(
            binding.key,
            inputs,
            qo_lengths=binding.qo_lengths,
            kv_lengths=binding.kv_lengths,
            forward_options=forward_options,
            native_call=native,
        )

    def evidence(self):
        return {
            "frozen_input_geometries": len(self.bindings),
            "counters": dict(self.counters),
            "request_time_training": False,
            "request_time_metadata_readback": False,
            "actual_resource_launch_count": None,
            "full_http_resource_qualified": False,
        }
