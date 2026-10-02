from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from research.selector_v4.serving.graph_profile_server import wrap_execute
from research.selector_v4.serving.graph_serving_audit import correlated_decode
from research.selector_v4.serving.test_graph_serving_audit import events


def test_scored_native_output_unchanged_and_marker_only_when_profile_active():
    recorded = []
    state = SimpleNamespace(active=False)

    @contextmanager
    def marker(name):
        recorded.append(name)
        yield
        recorded.append("marker-end")

    def original(self, value):
        recorded.append("native")
        return value

    wrapped = wrap_execute(original, profiling=lambda: state.active, marker=marker)
    result = object()
    assert wrapped(None, result) is result and recorded == ["native"]
    recorded.clear()
    state.active = True
    assert wrapped(None, result) is result
    assert recorded == ["sgi_actual_decode_graph_step", "native", "marker-end"]
    with pytest.raises(RuntimeError):
        wrap_execute(wrapped, profiling=lambda: False, marker=marker)


def test_candidate_eager_resource_does_not_authorize_resource_inside_a_graph():
    e = events()
    resource = {
        "name": "BatchPrefillWithPagedKVCacheResourceKernel",
        "cat": "kernel", "ph": "X", "args": {"graph id": 0, "correlation": 900},
    }
    e.append(resource)
    assert correlated_decode(e, allow_eager_resource=True)["actual_correlated_native_attention_kernels"] == 1
    with pytest.raises(ValueError):
        correlated_decode(e)
    resource["args"]["graph id"] = 3
    with pytest.raises(ValueError):
        correlated_decode(e, allow_eager_resource=True)
