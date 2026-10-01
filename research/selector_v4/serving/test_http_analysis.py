"""Independent raw metric reconstruction rejects edited metrics and lost rows."""

import copy
import hashlib
import json

import numpy as np
import pytest

from research.release_qualification.serving.workloads import prefix_tokens, work_specs
from research.selector_v4.serving.http_analysis import derived_metrics


def measured():
    work = work_specs(prefix_tokens(), 0)["guarded_prefix"]
    rows = []
    for index, cell in enumerate(work["cells"]):
        times = [0.2 + i * 0.01 for i in range(cell["output_tokens"])]
        rows.append(
            {
                "id": cell["id"],
                "rid": f"unique-{index}",
                "complete": True,
                "http_status": 200,
                "payload": {
                    "input_ids": cell["input_ids"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": cell["output_tokens"],
                        "ignore_eos": True,
                    },
                },
                "tokens": list(range(cell["output_tokens"])),
                "token_times": times,
                "ttft": times[0],
                "tpot": (times[-1] - times[0]) / (len(times) - 1),
                "cached_tokens": 8192,
                "finish_reason": {"type": "length"},
                "request_start": 0.0,
                "completion": 0.6,
            }
        )
    elapsed = 0.7
    for row in rows:
        row["raw_events"] = []
        for index, (token, stamp) in enumerate(zip(row["tokens"], row["token_times"])):
            meta = {"completion_tokens": index + 1, "cached_tokens": 8192}
            if index == len(row["tokens"]) - 1:
                meta["finish_reason"] = {"type": "length"}
            row["raw_events"].append(
                {
                    "client_arrival_seconds": stamp,
                    "data": json.dumps({"output_ids": [token], "meta_info": meta}),
                }
            )
        row["raw_events"].append({"client_arrival_seconds": 0.52, "data": "[DONE]"})
    block = {
        "complete": True,
        "errors": [],
        "requests": rows,
        "elapsed": elapsed,
        "workload_sha256": hashlib.sha256(
            json.dumps(work, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "output_tokens": sum(len(r["tokens"]) for r in rows),
    }
    block["output_tokens_per_second"] = block["output_tokens"] / elapsed
    block["strict_slo_goodput"] = len(rows) / elapsed
    for kind in ("TTFT", "TPOT"):
        for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            block[f"{kind}-{label}"] = float(np.quantile([r[kind.lower()] for r in rows], q))
    return work, block


def test_all_eight_metrics_are_reconstructed_from_every_actual_request():
    work, block = measured()
    result = derived_metrics(block, work)
    assert (
        len(result) == 8 and result["output_tokens_per_second"] == block["output_tokens_per_second"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "edited-tail",
        "deleted-request",
        "duplicate-rid",
        "wrong-token-time",
        "cache-miss",
        "wrong-input",
        "failed-request",
        "completion-outside-block",
    ],
)
def test_changed_or_incomplete_raw_requests_cannot_get_a_metric_pass(fault):
    work, block = measured()
    block = copy.deepcopy(block)
    if fault == "edited-tail":
        block["TPOT-p99"] *= 0.9
    elif fault == "deleted-request":
        block["requests"].pop()
    elif fault == "duplicate-rid":
        block["requests"][1]["rid"] = block["requests"][0]["rid"]
    elif fault == "wrong-token-time":
        block["requests"][0]["token_times"][5] = 0.19
    elif fault == "cache-miss":
        block["requests"][0]["cached_tokens"] = 4096
    elif fault == "wrong-input":
        block["requests"][0]["payload"]["input_ids"][0] = 99
    elif fault == "failed-request":
        block["requests"][0]["complete"] = False
    else:
        block["requests"][0]["completion"] = 1.0
    with pytest.raises(ValueError):
        derived_metrics(block, work)
