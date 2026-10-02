"""Additional descriptive HTTP/SLO metrics from all raw requests, no gate edits."""

import math

import numpy as np

from research.selector_v4.serving.http_analysis import derived_metrics

TTFT_LIMITS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
TPOT_LIMITS = (0.005, 0.01, 0.02, 0.05, 0.1)


def summarize(block, expected):
    legacy = derived_metrics(block, expected)  # Independently reparse every SSE.
    rows = block["requests"]
    elapsed = block["elapsed"]
    lengths = [len(r["payload"]["input_ids"]) for r in rows]
    cached = [r["cached_tokens"] for r in rows]
    if any(not 0 <= c <= n for c, n in zip(cached, lengths)):
        raise ValueError("Cache count outside actual logical input")
    itl = [b - a for r in rows for a, b in zip(r["token_times"], r["token_times"][1:])]
    latency = [r["completion"] - r["request_start"] for r in rows]
    if any(not math.isfinite(v) or v < 0 for v in itl) or any(v <= 0 for v in latency):
        raise ValueError("Invalid complete HTTP interval")
    grid = []
    for ttft in TTFT_LIMITS:
        for tpot in TPOT_LIMITS:
            passed = sum(r["ttft"] <= ttft and r["tpot"] <= tpot for r in rows)
            grid.append(
                {
                    "ttft_limit_seconds": ttft,
                    "tpot_limit_seconds": tpot,
                    "requests_meeting_both": passed,
                    "requests_total": len(rows),
                    "strict_goodput_requests_per_second": passed / elapsed,
                    "failure_fraction": 1 - passed / len(rows),
                }
            )
    return {
        "legacy_frozen_metrics": legacy,
        "request_per_second": len(rows) / elapsed,
        "logical_input_tokens_per_second": sum(lengths) / elapsed,
        "uncached_input_tokens_per_second": sum(n - c for n, c in zip(lengths, cached)) / elapsed,
        "output_tokens_per_second": block["output_tokens"] / elapsed,
        "client_arrival_itl_seconds": {
            str(q): float(np.quantile(itl, q)) for q in (0.5, 0.95, 0.99)
        },
        "co_delivered_zero_intervals": sum(v == 0 for v in itl),
        "client_admitted_request_latency_seconds": {
            str(q): float(np.quantile(latency, q)) for q in (0.5, 0.95, 0.99)
        },
        "slo_threshold_grid": grid,
        "concurrency": expected["concurrency"],
        "scope": "Descriptive complete fixed-workload block; no confidence/gain/qualification verdict",
        "itl_scope": "Client SSE arrival intervals; co-delivered tokens retained, not device token emission intervals",
        "latency_scope": "Request latency after client concurrency admission; full block throughput also includes client queue",
        "failed_requests_omitted": 0,
        "qualification_authority": False,
        "full_http_qualified": False,
    }
