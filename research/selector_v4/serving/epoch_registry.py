"""Frozen, eager-only serving lookup with one rebind per metadata epoch/key.

Training and real managed-cache validation happen before registration. Resource
execution still goes through the public runner's independent certificate/cache
checks. The scheduler exclusively owns the wrapper and announces every update.
This registry is a research integration component, not serving qualification.
"""

from dataclasses import dataclass


@dataclass
class PreparedEntry:
    lease: object
    bound_epoch: int


class ServingEpochRegistry:
    def __init__(self, owner, *, maximum_keys=256):
        if not isinstance(maximum_keys, int) or maximum_keys <= 0:
            raise ValueError("A positive, bounded registry size is required")
        self.owner = owner
        self.maximum_keys = maximum_keys
        self.epoch = 0
        self.frozen = False
        self.entries = {}
        self.bound_keys = set()
        self.counters = {
            "metadata_epochs": 0,
            "public_rebinds": 0,
            "failed_rebinds": 0,
            "prepared_eager_attempts": 0,
            "native_calls": 0,
            "graph_native_calls": 0,
        }

    def register(self, key, lease, *, actual_managed_tactic):
        """Register after explicit, unmeasured real calibration/v2 training.

        A native/inconclusive result is retained as a native entry. This method
        cannot manufacture a receipt or force the public runner's tactic.
        """
        if self.frozen:
            raise RuntimeError("Serving choices are frozen; no request-time registration")
        if key in self.entries:
            raise ValueError("Keep the first registered result; no fastest-result replacement")
        if len(self.entries) >= self.maximum_keys:
            raise ValueError("Registry capacity reached; retain native for additional geometries")
        if actual_managed_tactic != 1:
            self.entries[key] = None
            return
        if (
            lease.owner is not self.owner
            or lease.runner.execution_mode != "eager_run"
            or not lease.runner._receipt_valid
            or lease.runner._proxy is None
            or lease.state != "BOUND"
        ):
            raise ValueError("A current owned, certified, prepared eager lease is required")
        self.entries[key] = PreparedEntry(lease, self.epoch)
        self.bound_keys.add(key)

    def freeze(self):
        self.frozen = True

    def before_metadata_update(self):
        """Called before the first native planner or foreign metadata write."""
        self.epoch += 1
        self.counters["metadata_epochs"] += 1
        for key in self.bound_keys:
            entry = self.entries[key]
            if entry.lease.state == "BOUND":
                entry.lease.release_before_update()
        self.bound_keys.clear()

    def run(
        self,
        key,
        inputs,
        *,
        qo_lengths,
        kv_lengths,
        forward_options,
        native_call,
        graph_or_tracing=False,
    ):
        """No constructor, training, source audit or compiler on cache misses.

        The caller supplies the host geometry already used by the scheduler.
        Real public rebind still verifies the device metadata outside capture.
        Cloning/validation costs are included once per key/epoch and amortized
        over the model's layers, never excluded from measured serving steps.
        """
        entry = self.entries.get(key)
        if graph_or_tracing:
            self.counters["graph_native_calls"] += 1
            self.counters["native_calls"] += 1
            return native_call()
        if not self.frozen or entry is None or entry.lease.state == "RETIRED":
            self.counters["native_calls"] += 1
            return native_call()
        lease = entry.lease
        if (
            tuple(qo_lengths) != lease.runner.qo_lengths
            or tuple(kv_lengths) != lease.runner.kv_lengths
            or dict(forward_options) != lease.options
        ):
            lease.invalidate()
            self.counters["native_calls"] += 1
            return native_call()
        if entry.bound_epoch != self.epoch:
            rebound = lease.bind_current_plan(
                inputs,
                qo_lengths=qo_lengths,
                kv_lengths=kv_lengths,
                forward_options=forward_options,
            )
            self.counters["public_rebinds"] += 1
            if not rebound:
                self.counters["failed_rebinds"] += 1
                self.counters["native_calls"] += 1
                return native_call()
            entry.bound_epoch = self.epoch
            self.bound_keys.add(key)
        if not lease.current(inputs):
            self.counters["native_calls"] += 1
            return native_call()
        # This counts attempts, not kernel launches. Independent profiling must
        # witness actual v2 dispatch/resource launches for HTTP qualification.
        self.counters["prepared_eager_attempts"] += 1
        return lease.run(inputs, forward_options=forward_options)

    def evidence(self):
        return {
            "frozen": self.frozen,
            "registered_keys": len(self.entries),
            "native_keys": sum(entry is None for entry in self.entries.values()),
            "counters": dict(self.counters),
            "request_time_training": False,
            "resource_graph_metadata_updates_supported": False,
            "actual_resource_launch_count": None,
            "full_http_qualified": False,
            "qualification_authority": False,
        }
