"""Opt-in source-bound Resource HTTP bootstrap after full kernel qualification.

The existing local HTTP set_internal_state channel carries only an explicit
research phase command. Scheduler.is_fully_idle and the actual forward-stream
barrier precede begin/seal. No periodic polling or request-time tuning after seal.
No actual model/GPU HTTP qualification has run through this draft entry point.
"""

import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path

REVIEWED_SCHEDULER_SHA256 = "13f93b9f21f0ef09a5a951e837db45b6790b6b7f054193ac806d9d8322dcecac"
CONTROL_KEY = "sgi_prefix_phase"


def phase_transition(scheduler, values, *, make_session):
    if (
        set(values) != {CONTROL_KEY}
        or type(values[CONTROL_KEY]) is not int
        or values[CONTROL_KEY] not in (1, 2, 3)
        or not scheduler.is_fully_idle()
    ):
        return False
    runner = scheduler.tp_worker.model_runner
    router = runner.attn_backend._sgi_training_router
    expected = {1: "NATIVE_STARTUP", 2: "EXPLICIT_CALIBRATION", 3: "FROZEN_SERVING"}[
        values[CONTROL_KEY]
    ]
    if router.adapter.depth or router.phase != expected:
        return False
    runner.forward_stream.synchronize()
    if values[CONTROL_KEY] == 1:
        router.begin(make_session())
    elif values[CONTROL_KEY] == 2:
        router.seal()
    router.control_sequence += 1
    return True


def install():
    import torch
    from flashinfer._build_meta import __git_commit__
    from flashinfer.utils import _unpack_paged_kv_cache
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.managers.scheduler import Scheduler

    from research.selector_v4.serving.metadata_adapter import MetadataEpochAdapter
    from research.selector_v4.serving.prefix_backend_hook import (
        REVIEWED_BACKEND_SHA256,
    )
    from research.selector_v4.serving.prefix_backend_hook import (
        install as install_hook,
    )
    from research.selector_v4.serving.training_router import TrainingPrefixRouter
    from research.selector_v4.serving.training_session import PrefixTrainingSession

    backend_source = Path(inspect.getfile(FlashInferAttnBackend))
    scheduler_source = Path(inspect.getfile(Scheduler))
    for source, digest in (
        (backend_source, REVIEWED_BACKEND_SHA256),
        (scheduler_source, REVIEWED_SCHEDULER_SHA256),
    ):
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Unreviewed actual SGLang bootstrap source")
    original_init = FlashInferAttnBackend.__init__
    original_control = Scheduler.set_internal_state
    if getattr(original_init, "_sgi_prefix_training", False):
        raise RuntimeError("Training bootstrap already installed")

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        adapter = MetadataEpochAdapter(self)
        adapter.install()
        router = TrainingPrefixRouter(
            adapter,
            graph_or_tracing=lambda: (
                torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
            ),
            unpack_paged_cache=_unpack_paged_kv_cache,
        )
        install_hook(self, router, source_path=backend_source)
        self._sgi_training_router = router

    initialize._sgi_prefix_training = True
    FlashInferAttnBackend.__init__ = initialize

    @functools.wraps(original_control)
    def control(self, request):
        values = request.server_args
        if CONTROL_KEY not in values:
            return original_control(self, request)
        root = Path(os.environ["SGI_HTTP_TRAINING_OUTPUT"])

        def make_session():
            root.mkdir(parents=True, exist_ok=True)
            return PrefixTrainingSession(
                os.environ["SGI_FORMAL_KERNEL_CAMPAIGN"],
                root / f"worker-{os.getpid()}",
                candidate_commit=__git_commit__,
                maximum_keys=16,
            )

        try:
            if not phase_transition(self, values, make_session=make_session):
                return SetInternalStateReqOutput(updated=False)
            router = self.tp_worker.model_runner.attn_backend._sgi_training_router
            record = (
                root
                / f"phase-{os.getpid()}-{values[CONTROL_KEY]}-{router.control_sequence:04d}.json"
            )
            with record.open("x") as stream:
                stream.write(json.dumps(router.evidence(), indent=2) + "\n")
            return SetInternalStateReqOutput(updated=True)
        except Exception as error:  # noqa: BLE001 - retain failed control, never promote it
            root.mkdir(parents=True, exist_ok=True)
            (root / f"control-failure-{os.getpid()}.json").write_text(
                json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n"
            )
            return SetInternalStateReqOutput(updated=False)

    Scheduler.set_internal_state = control


if os.environ.get("SGI_FORMAL_KERNEL_CAMPAIGN"):
    install()

if __name__ == "__main__":
    if not os.environ.get("SGI_FORMAL_KERNEL_CAMPAIGN"):
        raise RuntimeError("Explicit fully qualified kernel campaign is required")
    runpy.run_module("sglang.launch_server", run_name="__main__")
