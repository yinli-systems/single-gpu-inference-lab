import copy
import json

import pytest

from research.selector_v4.serving.graph_serving_audit import correlated_decode, natural_parity
from research.selector_v4.serving.metric_gate import WORKLOADS
from research.selector_v4.serving.test_http_analysis import measured


def events():
    return [
        {
            "name": "sgi_actual_decode_graph_step",
            "ph": "X",
            "cat": "user_annotation",
            "pid": 123,
            "tid": 456,
            "ts": 100,
            "dur": 30,
        },
        {
            "name": "cudaGraphLaunch",
            "ph": "X",
            "cat": "cuda_runtime",
            "pid": 123,
            "tid": 456,
            "ts": 110,
            "dur": 2,
            "args": {"correlation": 77},
        },
        {
            "name": "BatchDecodeWithPagedKVCacheKernel",
            "ph": "X",
            "cat": "kernel",
            "pid": 0,
            "tid": 7,
            "ts": 140,
            "dur": 2,
            "args": {"correlation": 77, "graph id": 3},
        },
    ]


def test_async_gpu_attention_can_finish_after_the_cpu_marker():
    r = correlated_decode(events())
    assert r["traced_decode_steps"] == 1 and r["actual_correlated_native_attention_kernels"] == 1


def test_profiler_gpu_annotation_copy_is_not_a_second_cpu_api_launch():
    e = events()
    gpu = copy.deepcopy(e[0])
    gpu.update(cat="gpu_user_annotation", pid=0, tid=7, ts=140)
    e.append(gpu)
    assert correlated_decode(e)["traced_decode_steps"] == 1
    with pytest.raises(ValueError):
        correlated_decode(e[1:])


@pytest.mark.parametrize(
    "fault", ["wrong-thread", "wrong-correlation", "no-graph-id", "resource", "no-launch"]
)
def test_markers_or_counters_without_actual_correlated_native_graph_rejected(fault):
    e = copy.deepcopy(events())
    if fault == "wrong-thread":
        e[1]["tid"] = 999
    elif fault == "wrong-correlation":
        e[2]["args"]["correlation"] = 99
    elif fault == "no-graph-id":
        del e[2]["args"]["graph id"]
    elif fault == "resource":
        e[2]["name"] = "BatchPrefillWithPagedKVCacheResourceKernel"
    else:
        e.pop(1)
    with pytest.raises(ValueError):
        correlated_decode(e)


def test_all_twenty_natural_token_blocks_are_retained_even_when_one_differs(tmp_path):
    _, original = measured()
    for role in ("native_baseline", "graph_diagnostic"):
        out = tmp_path / role / "observations"
        out.mkdir(parents=True)
        for block in range(4):
            for name in WORKLOADS:
                value = copy.deepcopy(original)
                if role == "graph_diagnostic" and block == 3 and name == "decode":
                    value["requests"][0]["tokens"][5] = 9999
                (out / f"{name}-b{block}.json").write_text(json.dumps(value))
    result = natural_parity(tmp_path)
    assert len(result) == 20
    failures = [r for r in result if not r["parity_pass"]]
    assert len(failures) == 1 and failures[0]["block"] == "decode-b3.json"
    assert failures[0]["mismatches"][0]["positions"] == [5]
    (tmp_path / "graph_diagnostic/observations/mixed-b3.json").unlink()
    with pytest.raises(FileNotFoundError):
        natural_parity(tmp_path)
