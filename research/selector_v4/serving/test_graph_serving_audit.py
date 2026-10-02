import copy

import pytest

from research.selector_v4.serving.graph_serving_audit import correlated_decode


def events():
    return [
        {
            "name": "sgi_actual_decode_graph_step",
            "ph": "X",
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
