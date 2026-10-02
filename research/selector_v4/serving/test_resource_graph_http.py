import ast
import copy
import json
from pathlib import Path

import pytest

from research.selector_v4.serving.resource_graph_chain import gate_state
from research.selector_v4.serving.resource_graph_http import (
    command,
    protocol,
    validate_case,
)
from research.selector_v4.serving.test_resource_graph_contract import trace


def test_protocol_exact_resource_and_negative_geometry_are_preregistered():
    rows = protocol()
    assert len(rows) == 6 and sum(r["expect_resource"] for r in rows) == 3
    assert [len(r["cells"][0]["input_ids"]) for r in rows] == [
        128,
        128,
        128,
        127,
        129,
        256,
    ]
    assert rows[-1]["cells"][0]["input_ids"][:128] == rows[-1]["seed_ids"]
    assert rows[-1]["cells"][0]["expect_cached"] == 128
    assert [r["prime_tokens"] for r in rows[:3]] == [0, 256, 512]
    assert len({r["cells"][0]["input_ids"][0] for r in rows}) == 6


def test_both_arms_explicit_full_graph_page1_and_same_schedule():
    native, resource = [
        command("/weights", 12000, 0.70, r) for r in ("native", "resource")
    ]
    assert native[:2] + native[3:] == resource[:2] + resource[3:]
    assert native[native.index("--page-size") + 1] == "1"
    config = json.loads(native[native.index("--cuda-graph-config") + 1])
    assert config["prefill"] == {
        "backend": "full",
        "bs": [128, 256],
        "full_prefill_max_req": 1,
    }
    assert "--disable-radix-cache" not in native


def epoch(layers=2):
    return {
        "event": "resource_replay",
        "full_model_output_exact": True,
        "layers": {str(i): {"out_lse_exact": True} for i in range(layers)},
    }


def test_full_model_requires_every_real_layer_in_correlated_graph():
    events = trace()
    events.append(copy.deepcopy(events[-1]))
    assert validate_case(protocol()[0], events, [epoch()], layers=2, resource=True)[
        "pass"
    ]
    with pytest.raises(ValueError):
        validate_case(protocol()[0], trace(), [epoch()], layers=2, resource=True)
    with pytest.raises(ValueError):
        validate_case(protocol()[0], events, [], layers=2, resource=True)
    eager = copy.deepcopy(events)
    eager[-1]["args"]["graph id"] = 0
    with pytest.raises(ValueError):
        validate_case(protocol()[0], eager, [epoch()], layers=2, resource=True)


def test_native_fallback_cannot_hide_any_resource_execution():
    assert validate_case(
        protocol()[3],
        [{"cat": "kernel", "name": "NativeAttention"}],
        [],
        layers=2,
        resource=True,
    )["pass"]
    for resource in (False, True):
        with pytest.raises(ValueError):
            validate_case(protocol()[3], trace(), [], layers=2, resource=resource)


def test_model_equality_and_layer_equality_are_mandatory():
    for bad in (
        {**epoch(), "full_model_output_exact": False},
        {**epoch(), "layers": {"0": {"out_lse_exact": False}}},
    ):
        with pytest.raises(ValueError):
            validate_case(protocol()[0], trace(), [bad], layers=2, resource=True)


def components(tmp_path):
    root = tmp_path / "component"
    for gpu, job in (("gpu_4090", "1"), ("gpu_5090", "2")):
        path = root / "runs" / (gpu + "-" + job)
        path.mkdir(parents=True)
        (path / "complete.json").write_text("{}")
    return {"root": str(root), "jobs": {"gpu_4090": "1", "gpu_5090": "2"}}


def done(job):
    return {"job": job, "state": "COMPLETED", "exit": "0:0"}


def test_no_kernel_terminal_never_authorizes_submission(tmp_path):
    assert gate_state(
        tmp_path / "kernel", components(tmp_path), account=done
    ).startswith("WAIT_ALL")


def test_component_pending_or_failure_never_authorizes(tmp_path):
    component = components(tmp_path)
    assert gate_state(
        tmp_path / "kernel", component, account=lambda j: {"state": "PENDING"}
    ).startswith("WAIT_COMPONENT")
    with pytest.raises(ValueError, match="Component terminal"):
        gate_state(
            tmp_path / "kernel",
            component,
            account=lambda j: {"state": "FAILED", "exit": "1:0"},
        )


def test_kernel_terminal_hold_is_not_a_retriable_wait(tmp_path):
    receipts = tmp_path / "kernel/receipts"
    receipts.mkdir(parents=True)
    (receipts / "controller-terminal.json").write_text(
        json.dumps({"terminal": True, "state": "HOLD"})
    )
    with pytest.raises(ValueError, match="terminal HOLD"):
        gate_state(tmp_path / "kernel", components(tmp_path), account=done)


def test_advertised_pass_only_requests_raw_verification(tmp_path):
    receipts = tmp_path / "kernel/receipts"
    receipts.mkdir(parents=True)
    (receipts / "controller-terminal.json").write_text(
        json.dumps(
            {
                "terminal": True,
                "state": "FORMAL_KERNEL_QUALIFICATION_PASS_HTTP_REQUIRED",
            }
        )
    )
    assert (
        gate_state(tmp_path / "kernel", components(tmp_path), account=done)
        == "VERIFY_RAW_GATES"
    )


def test_spawned_worker_installs_once_without_launching_recursive_server(monkeypatch):
    from research.selector_v4.serving import resource_graph_server

    source = ast.parse(Path(resource_graph_server.__file__).read_text())
    entry = ast.Module(body=[source.body[-1]], type_ignores=[])
    monkeypatch.setenv("SGI_RESOURCE_GRAPH_DIAGNOSTIC", "/owned/epochs")
    monkeypatch.setenv("SGI_FORMAL_KERNEL_CAMPAIGN", "/kernel")
    import os
    from types import SimpleNamespace

    for name, expected_launches in (
        ("__main__", 1),
        ("__mp_main__", 0),
        ("imported", 0),
    ):
        installs, launches = [], []
        namespace = {
            "__name__": name,
            "os": os,
            "Path": Path,
            "install": lambda *args, result=installs: result.append(args),
            "runpy": SimpleNamespace(
                run_module=lambda *args, result=launches, **kwargs: result.append(args)
            ),
        }
        exec(compile(entry, "real_entry", "exec"), namespace)  # noqa: S102 - actual entry under mocks
        assert len(installs) == (name != "imported")
        assert len(launches) == expected_launches


def test_missing_profiler_is_not_a_negative_control_pass():
    with pytest.raises(ValueError, match="Actual GPU trace"):
        validate_case(protocol()[3], [], [], layers=2, resource=True)
