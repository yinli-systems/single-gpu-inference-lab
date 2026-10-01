"""Explicit idle-boundary transition from Native calibration to frozen routing.

No background polling or file reads occur on model steps. The actual reviewed
scheduler invokes begin/seal only after its own idle check and stream barrier.
This component is a draft integration, not actual Resource HTTP qualification.
"""

from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter
from research.selector_v4.serving.prefix_router import PagedPrefixRouter


class TrainingPrefixRouter:
    def __init__(
        self, adapter, *, graph_or_tracing, unpack_paged_cache, adapter_factory=MetadataEpochAdapter
    ):
        if not adapter.frozen or adapter.bindings or adapter.depth:
            raise ValueError("Installed empty Native adapter required")
        self.adapter = adapter
        self.graph_or_tracing = graph_or_tracing
        self.unpack_paged_cache = unpack_paged_cache
        self.adapter_factory = adapter_factory
        self.phase = "NATIVE_STARTUP"
        self.session = None
        self.frozen_router = None
        self.control_sequence = 0
        original_before = adapter.before_update

        def before_update(graph):
            if self.phase == "EXPLICIT_CALIBRATION":
                self.session.before_metadata_update()
            return original_before(graph)

        adapter.before_update = before_update

    def begin(self, session):
        if self.phase != "NATIVE_STARTUP" or self.adapter.depth:
            raise RuntimeError("One explicit idle-boundary calibration phase required")
        if session.sealed or session.failed or session.bindings:
            raise ValueError("New authorized training session required")
        self.session = session
        self.phase = "EXPLICIT_CALIBRATION"

    def run(self, owner, q, kv_cache, **options):
        if self.phase == "FROZEN_SERVING":
            return self.frozen_router.run(owner, q, kv_cache, **options)
        geometry = self.adapter.current_prefix_lengths
        if (
            self.phase == "EXPLICIT_CALIBRATION"
            and geometry is not None
            and not self.adapter.depth
            and not self.graph_or_tracing()
        ):
            k, v = self.unpack_paged_cache(kv_cache, owner._kv_layout)
            self.session.record_current_plan(
                owner,
                [q, k, v],
                qo_lengths=geometry[0],
                kv_lengths=geometry[1],
                forward_options=options,
            )
        # Calibration output never participates in the model's result.
        return owner.forward_return_lse(q, kv_cache, **options)

    def seal(self):
        if self.phase != "EXPLICIT_CALIBRATION" or self.adapter.depth:
            raise RuntimeError("Seal once at an explicit idle boundary outside metadata updates")
        try:
            bindings = self.session.seal()
            # Restore invokes before_update. The training session is now sealed
            # and has already released every lease; never notify it again.
            self.phase = "SEALING"
            backend = self.adapter.backend
            self.adapter.restore()
            replacement = self.adapter_factory(backend)
            for binding in bindings:
                replacement.attach(
                    binding.registry,
                    binding.key,
                    binding.signature_inputs,
                    qo_lengths=binding.qo_lengths,
                    kv_lengths=binding.kv_lengths,
                    forward_options=binding.forward_options,
                )
            replacement.install()
            self.adapter = replacement
            self.frozen_router = PagedPrefixRouter(
                replacement,
                graph_or_tracing=self.graph_or_tracing,
                unpack_paged_cache=self.unpack_paged_cache,
            )
            self.phase = "FROZEN_SERVING"
        except BaseException:
            self.phase = "FAILED"
            raise

    def evidence(self):
        return {
            "phase": self.phase,
            "request_time_training": False,
            "full_http_qualified": False,
            "router": self.frozen_router.evidence() if self.frozen_router is not None else None,
            "metadata_boundaries": dict(self.adapter.counters),
            "metadata_depth": self.adapter.depth,
        }
