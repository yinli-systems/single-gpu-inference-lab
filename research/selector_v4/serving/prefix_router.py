"""Explicit frozen eager cached-prefix routing through real public leases.

There is no registration, constructor, training or metadata readback here.
Unknown geometry, options, graphs and leases not bound before the model stay
native. A routed attempt is not evidence of an actual resource kernel launch.
"""

import functools


def signature(inputs):
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in inputs)


class PagedPrefixRouter:
    def __init__(self, adapter, *, graph_or_tracing, unpack_paged_cache=None):
        if not adapter.frozen:
            raise ValueError("Install the frozen metadata adapter before routing")
        self.adapter = adapter
        self.graph_or_tracing = graph_or_tracing
        self.bindings = {}
        self.counters = {"native_calls": 0, "prepared_attempts": 0}
        self.owners = set()
        self.host_geometries = set()
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
            self.owners.add(id(binding.registry.owner))
            self.host_geometries.add(
                (id(binding.registry.owner), binding.qo_lengths, binding.kv_lengths)
            )
        self.unpack_paged_cache = unpack_paged_cache
        if self.bindings and self.unpack_paged_cache is None:
            from flashinfer.utils import _unpack_paged_kv_cache

            self.unpack_paged_cache = _unpack_paged_kv_cache

    def run(self, owner, q, kv_cache, **forward_options):
        if not self.bindings or id(owner) not in self.owners:
            self.counters["native_calls"] += 1
            return owner.forward_return_lse(q, kv_cache, **forward_options)
        native = functools.partial(owner.forward_return_lse, q, kv_cache, **forward_options)
        geometry = self.adapter.current_prefix_lengths
        if geometry is None or self.adapter.depth or self.graph_or_tracing():
            self.counters["native_calls"] += 1
            return native()
        if (id(owner), *geometry) not in self.host_geometries:
            self.counters["native_calls"] += 1
            return native()
        # Native supports tuple and packed Tensor payloads, including an implicit
        # page-size-one dimension. Use its real view transformation, never
        # iterate a packed token-major Tensor as if it contained only K and V.
        k, v = self.unpack_paged_cache(kv_cache, owner._kv_layout)
        inputs = [q, k, v]
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
