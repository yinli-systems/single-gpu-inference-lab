"""Synthetic manifest component freshness and untouched-stage preservation."""

import copy
import json

import pytest

from research.selector_v4.manifest_v4 import digest
from research.selector_v4.public_qualification.fresh_manifest import (
    collect_history,
    freeze_manifest,
    replace_canary,
)
from research.selector_v4.public_qualification.gates import sha


def synthetic_base():
    families = {"dev": 2, "canary": 10, "release": 48, "stress": 12}
    cases = []
    for family, count in families.items():
        for index in range(count):
            number = len(cases) + 1
            cases.append(
                {
                    "id": family + str(index),
                    "family": family,
                    "q": [number],
                    "cached": [number + 100],
                }
            )
    return {
        "families": families,
        "cases": cases,
        "case_hash": digest(cases),
        "stage_hashes": {f: digest([c for c in cases if c["family"] == f]) for f in families},
    }


def test_new_ten_canaries_preserve_original_dev_release_stress_verbatim():
    base = synthetic_base()
    original = copy.deepcopy(base)
    value = replace_canary(base, [], generator_seed=37)
    assert base == original and value["fresh_cases_consumed"] == 0
    assert (
        len(value["cases"]) == 72
        and value["stage_hashes"]["canary"] != base["stage_hashes"]["canary"]
    )
    for family in ("dev", "release", "stress"):
        assert value["stage_hashes"][family] == base["stage_hashes"][family]
        assert [c for c in value["cases"] if c["family"] == family] == [
            c for c in base["cases"] if c["family"] == family
        ]


def test_q_cached_and_pair_components_are_deduplicated_against_consumed_history():
    base = synthetic_base()
    first = replace_canary(base, [], generator_seed=37)
    history = [c for c in first["cases"] if c["family"] == "canary"]
    second = replace_canary(base, history, generator_seed=37)
    old_q = {tuple(c["q"]) for c in base["cases"] + history}
    old_kv = {tuple(c["cached"]) for c in base["cases"] + history}
    for c in (x for x in second["cases"] if x["family"] == "canary"):
        assert tuple(c["q"]) not in old_q and tuple(c["cached"]) not in old_kv


def test_hold_prerequisite_cannot_generate_or_write_a_real_manifest(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"pass_public_path_development": False}))
    output = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="prerequisite must pass"):
        freeze_manifest(tmp_path, receipt, output)
    assert not output.exists()


def test_existing_frozen_manifest_is_never_replaced(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_text("preserve")
    with pytest.raises(ValueError, match="replace"):
        freeze_manifest(tmp_path, tmp_path / "missing.json", output)
    assert output.read_text() == "preserve"


def test_renamed_sibling_and_exposed_manifests_enter_component_history(tmp_path):
    root = tmp_path / "research/historical"
    root.mkdir(parents=True)
    files = [root / "autotune-branch-manifest.json", root / "exposed-old-manifest.json"]
    for i, p in enumerate(files):
        p.write_text(json.dumps({"cases": [{"q": [i + 1], "cached": [i + 10]}]}))
    base = {"historical_manifest_hashes": {str(files[0].relative_to(tmp_path)): sha(files[0])}}
    history, hashes = collect_history(tmp_path, base)
    assert len(history) == len(hashes) == 2
    files[0].write_text("changed")
    with pytest.raises(ValueError, match="Historical manifest changed"):
        collect_history(tmp_path, base)
