"""Supplemental actual-hit/dispatch witnesses for the immutable epoch smoke.

Instrument only post-training functional calls. Calibration and managed tactic
profiling use the original uninstrumented runner. Retain the real cache files.
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from flashinfer import prefill
from flashinfer.autotuner import AutoTuner

from research.selector_v4.serving.test_reusable_plan_lease_gpu import (
    test_real_managed_decision_and_owned_metadata_rebind as epoch_smoke,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("paged", [False, True])
def test_every_rebound_call_has_actual_cache_hit_and_tactic(paged, tmp_path, monkeypatch):
    root = Path(os.environ.get("SGI_REUSABLE_LEASE_EVIDENCE", str(tmp_path)))
    name = "paged" if paged else "ragged"
    persistent = root / (name + "-artifacts")
    persistent.mkdir(parents=True, exist_ok=False)
    witnesses = []
    factory = prefill.make_prefill_resource_runner

    def checked_factory(*args, **kwargs):
        runner = factory(*args, **kwargs)
        original_run = runner.run
        original_resource = runner._resource
        counting_installed = False
        count = 0

        def resource(*arguments, **options):
            nonlocal count
            count += 1
            return original_resource(*arguments, **options)

        def run(inputs, **options):
            nonlocal counting_installed, count
            tuner = AutoTuner.get()
            if tuner.is_tuning_mode:
                return original_run(inputs, **options)
            if not counting_installed:
                runner._resource = resource
                counting_installed = True
            config = runner.tuning_config
            policy = tuner._effective_measure_policy
            if policy is not None:
                config = tuner._apply_measure_policy(config, policy)
            hit, index, tactic, _ = tuner.search_cache(
                "experimental_prefill_resource",
                [runner],
                tuple(tuner._get_input_sizes(inputs)),
                config,
                inputs=inputs,
            )
            assert hit and index == 0, "Actual lookup miss cannot prove a selected epoch"
            count = 0
            result = original_run(inputs, **options)
            witness = {
                "actual_cache_hit": hit,
                "cached_tactic": tactic,
                "resource_invocations": count,
                "certificate_accepted": runner._receipt_valid,
                "identity": runner.identity,
                "certificate_checksum": runner.receipt["checksum"],
                "native_fallback": tactic != 1,
            }
            witnesses.append(witness)
            # Persist a disagreement before raising; this is diagnostic only.
            (root / (name + "-dispatch-witnesses.json")).write_text(
                json.dumps(witnesses, indent=2) + "\n"
            )
            assert count == (1 if tactic == 1 else 0)
            assert tactic != 1 or runner._receipt_valid
            return result

        monkeypatch.setattr(runner, "run", run)
        return runner

    monkeypatch.setattr(prefill, "make_prefill_resource_runner", checked_factory)
    epoch_smoke(paged, persistent, monkeypatch)
    summary = json.loads((root / (name + ".json")).read_text())
    expected = 3 if summary["certificate"]["tactic"] == 1 else 0
    assert len(witnesses) == expected
    assert all(w["cached_tactic"] == summary["actual_managed_tactic"] for w in witnesses)
    entries = list((persistent / "managed-cache").glob("v2/*/entries/*.json"))
    assert entries, "Real managed cache files must survive in the campaign"
