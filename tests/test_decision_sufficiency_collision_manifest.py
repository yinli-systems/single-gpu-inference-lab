import importlib.util
import json
from pathlib import Path

from l20_stack.decision_sufficiency import (
    Geometry,
    aggregate_sums,
    analytical_work,
    marginal_moments,
    paired_multiset,
    vidur_prefill_lookup_key,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = (
    ROOT
    / "research"
    / "decision_sufficiency"
    / "collision_factorial"
    / "manifest.py"
)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "decision_collision_manifest", MANIFEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collision_factorial_manifest_is_exact_and_fresh_by_geometry():
    module = _load_manifest_module()
    manifest = module.build()

    assert manifest["decision_margin"] == 0.01
    assert manifest["blocks"] == 12
    assert manifest["resource_modes"] == ["pristine", "cap"]
    assert manifest["order_actions"] == ["identity", "heavy_first"]
    assert len(manifest["primary_pair_ids"]) == 5

    index = {case["id"]: case for case in manifest["cases"]}
    assert len(index) == 12
    assert {len(index[p + "-a"]["q"]) for p in manifest["primary_pair_ids"]} == {
        6,
        10,
        14,
    }

    for pair_id in manifest["primary_pair_ids"] + manifest["canary_pair_ids"]:
        a = index[pair_id + "-a"]
        b = index[pair_id + "-b"]
        ga = Geometry(tuple(a["q"]), tuple(a["cached"]))
        gb = Geometry(tuple(b["q"]), tuple(b["cached"]))

        assert sorted(ga.query) == sorted(gb.query)
        assert sorted(ga.cached) == sorted(gb.cached)
        assert aggregate_sums(ga) == aggregate_sums(gb)
        assert analytical_work(ga) == analytical_work(gb)
        assert vidur_prefill_lookup_key(ga, 64) == vidur_prefill_lookup_key(gb, 64)
        assert marginal_moments(ga) == marginal_moments(gb)
        assert paired_multiset(ga) != paired_multiset(gb)

    old_plan_cases = json.loads(
        (
            ROOT
            / "benchmarks"
            / "results"
            / "plan-order-mechanism"
            / "cases.json"
        ).read_text()
    )
    old_plan_geometries = {
        (tuple(row["query"]), tuple(row["cached"])) for row in old_plan_cases
    }

    old_resource_module_path = (
        ROOT / "research" / "resource_generalization" / "manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "old_resource_manifest", old_resource_module_path
    )
    old_resource = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(old_resource)
    old_resource_geometries = {
        (tuple(row["q"]), tuple(row["cached"]))
        for row in old_resource.build()["cases"]
    }

    for case in manifest["cases"]:
        geometry = (tuple(case["q"]), tuple(case["cached"]))
        assert geometry not in old_plan_geometries
        assert geometry not in old_resource_geometries
