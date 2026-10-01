"""Explicit premeasurement training; only complete kernel qualification authorizes.

No server hooks or request-time tuning are installed. The caller announces
metadata updates before all writes, supplies real current-plan tensors during a
separate calibration phase, seals once, then attaches frozen entries for serving.
Every first training result is retained; native winners are never replaced.
"""

import hashlib
import json
import traceback
from contextlib import contextmanager
from pathlib import Path

from research.selector_v4.public_qualification.gates import (
    STAGES,
    build_stage_ticket,
    need,
    sha,
    verify_manifest,
    verify_summary,
)
from research.selector_v4.serving.prefix_router import signature


def authorize_http_training(campaign, candidate_commit):
    campaign = Path(campaign)
    expected = json.loads((campaign / "frozen-binding.json").read_text())
    need(expected["candidate_commit"] == candidate_commit, "Different serving kernel source")
    terminal = json.loads((campaign / "receipts/controller-terminal.json").read_text())
    need(
        terminal.get("terminal") is True
        and terminal.get("state") == "FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED",
        "Complete formal kernel qualification is required before Resource HTTP work",
    )
    verify_manifest(campaign, expected)
    build_stage_ticket(campaign, "stress")  # Verify dual smoke and all prior raw links.
    need(
        sha(campaign / "harness.tar.gz") == expected["harness_archive_sha256"],
        "Kernel source bundle changed",
    )
    # Includes stress itself, rather than accepting its authorization ticket
    # (which only certifies the preceding release stage).
    summaries = [
        verify_summary(campaign, stage, gpu, expected)
        for stage in STAGES
        for gpu in ("gpu_4090", "gpu_5090")
    ]
    return {
        "scope": "FORMAL_KERNEL_PASS_EXPLICIT_HTTP_PREMEASUREMENT_TRAINING_ONLY",
        "candidate_commit": candidate_commit,
        "frozen_binding_sha256": sha(campaign / "frozen-binding.json"),
        "kernel_terminal_sha256": sha(campaign / "receipts/controller-terminal.json"),
        "summaries": summaries,
        "request_time_training": False,
        "full_http_resource_qualified": False,
        "default_promotion": False,
    }


def geometry_key(inputs, qo_lengths, kv_lengths, options):
    for lengths in (qo_lengths, kv_lengths):
        need(
            bool(lengths) and all(type(v) is int and v > 0 for v in lengths),
            "Positive host lengths",
        )
    need(len(qo_lengths) == len(kv_lengths), "Host batch length mismatch")
    need(
        all(value is None or type(value) in (int, float, bool, str) for value in options.values()),
        "Only reviewed scalar forward options are eligible",
    )
    raw = json.dumps(
        [list(qo_lengths), list(kv_lengths), signature(inputs), options],
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class PrefixTrainingSession:
    def __init__(self, campaign, output, *, candidate_commit, maximum_keys=256):
        need(type(maximum_keys) is int and 0 < maximum_keys <= 256, "Bounded geometry capacity")
        self.authorization = authorize_http_training(campaign, candidate_commit)
        self.output = Path(output)
        self.output.mkdir()  # Never replace an earlier calibration campaign.
        (self.output / "authorization.json").write_text(
            json.dumps(self.authorization, indent=2) + "\n"
        )
        self.maximum_keys = maximum_keys
        self.sealed = False
        self.failed = False
        self.registries = {}
        self.bindings = []
        self.attempted = set()

    @contextmanager
    def _attempt(self, root):
        try:
            yield
        except BaseException as error:
            self.failed = True
            (root / "failure.json").write_text(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "first_result_retained": True,
                        "full_http_resource_qualified": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            self.before_metadata_update()
            raise

    def before_metadata_update(self):
        need(not self.sealed, "The serving adapter owns updates after sealing")
        for registry in self.registries.values():
            registry.before_metadata_update()

    def record_current_plan(self, owner, inputs, *, qo_lengths, kv_lengths, forward_options):
        """Explicit calibration-phase call after an announced real Native plan.

        Callers supply the actual serving stream and native-unpacked page1
        geometry. No plan is reconstructed here and no production cache miss
        invokes this function. Store real template tensors for later early bind.
        """
        need(not self.sealed, "No training after the calibration phase is sealed")
        need(not self.failed, "An incomplete calibration session cannot resume")
        import torch
        from flashinfer import MeasurementPolicy, autotune_v2
        from flashinfer._build_meta import __git_commit__, __version__
        from flashinfer.autotuner import AutoTuner

        from research.selector_v4.serving.epoch_registry import ServingEpochRegistry
        from research.selector_v4.serving.metadata_adapter import Binding
        from research.selector_v4.serving.reusable_plan_lease import ReusableServingPlanLease

        need(
            (__git_commit__, __version__) == (self.authorization["candidate_commit"], "0.7.1"),
            "Actual loaded package identity",
        )
        need(
            not torch.cuda.is_current_stream_capturing() and not torch.compiler.is_compiling(),
            "Train outside capture/tracing",
        )
        need(
            len(inputs) == 3 and all(isinstance(t, torch.Tensor) and t.is_cuda for t in inputs),
            "Real CUDA tensors required",
        )
        need(
            getattr(owner, "_kv_layout", None) == "NHD"
            and inputs[1].ndim == 4
            and inputs[1].shape[1] == 1,
            "Actual native-unpacked page1 NHD inputs",
        )
        key = geometry_key(inputs, qo_lengths, kv_lengths, forward_options)
        identity = (id(owner), key)
        if identity in self.attempted or len(self.attempted) >= self.maximum_keys:
            return False  # Keep first result; overflow stays unknown/Native.
        self.attempted.add(identity)
        root = self.output / f"entry-{len(self.attempted) - 1:04d}"
        root.mkdir()
        (root / "started.json").write_text(
            json.dumps(
                {
                    "key": key,
                    "qo_lengths": list(qo_lengths),
                    "kv_lengths": list(kv_lengths),
                    "input_signatures": signature(inputs),
                    "forward_options": forward_options,
                    "phase": "EXPLICIT_PREMEASUREMENT_CALIBRATION",
                    "kernel_source": __git_commit__,
                },
                indent=2,
            )
            + "\n"
        )
        with self._attempt(root):
            registry = self.registries.setdefault(
                id(owner), ServingEpochRegistry(owner, maximum_keys=self.maximum_keys)
            )
            lease = ReusableServingPlanLease(
                owner,
                inputs,
                qo_lengths=qo_lengths,
                kv_lengths=kv_lengths,
                forward_options=forward_options,
                return_lse=True,
            )
            receipt = lease.calibrate(inputs)
            lease.runner.tuning_config.profiling_repeat = 256
            with autotune_v2(
                mode="tune",
                measurement_policy=MeasurementPolicy(execution_mode="eager"),
                cache_root=root / "managed-cache",
            ):
                lease.run(inputs, forward_options=forward_options)
                _, tactic = AutoTuner.get().choose_one(
                    "experimental_prefill_resource",
                    [lease.runner],
                    lease.runner.tuning_config,
                    inputs,
                )
            (root / "training.json").write_text(
                json.dumps(
                    {
                        "key": key,
                        "certificate": receipt,
                        "actual_managed_tactic": tactic,
                        "first_result_retained": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            registry.register(key, lease, actual_managed_tactic=tactic)
            self.bindings.append(
                Binding(
                    registry,
                    key,
                    inputs,
                    tuple(qo_lengths),
                    tuple(kv_lengths),
                    dict(forward_options),
                )
            )
            (root / "complete.json").write_text(
                json.dumps(
                    {
                        "key": key,
                        "certificate": receipt,
                        "actual_managed_tactic": tactic,
                        "registry": registry.evidence(),
                        "first_result_retained": True,
                        "full_http_resource_qualified": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        return True

    def seal(self):
        need(not self.sealed, "The calibration phase can be sealed only once")
        need(not self.failed, "An incomplete calibration session cannot authorize serving")
        # Release all last calibration leases before model metadata can change.
        self.before_metadata_update()
        for registry in self.registries.values():
            registry.freeze()
        self.sealed = True
        (self.output / "sealed.json").write_text(
            json.dumps(
                {
                    "attempted_geometries": len(self.attempted),
                    "attached_geometries": len(self.bindings),
                    "request_time_training": False,
                    "full_http_resource_qualified": False,
                },
                indent=2,
            )
            + "\n"
        )
        return tuple(self.bindings)
