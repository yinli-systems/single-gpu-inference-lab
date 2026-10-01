"""Collect every actual HTTP outcome; durable writing is outside block timing.

The caller owns an aiohttp session and full server qualification. This collector
does not launch a server, train a tactic, authorize serving or omit failures.
"""

import asyncio
import hashlib
import json
import time

import numpy as np

from research.selector_v4.serving.http_stream import TokenStream


async def collect_request(session, url, cell, rid, origin, *, logprobs=False):
    started = time.perf_counter()
    payload = {
        "rid": rid,
        "input_ids": cell["input_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": cell["output_tokens"],
            "ignore_eos": True,
        },
        "stream": True,
    }
    if logprobs:
        payload.update(return_logprob=True, top_logprobs_num=5, logprob_start_len=-1)
    stream = TokenStream(cell["output_tokens"])
    record = {
        "id": cell["id"],
        "rid": rid,
        "payload": payload,
        "request_start": started - origin,
        "complete": False,
    }
    try:
        async with session.post(url + "/generate", json=payload) as response:
            record["http_status"] = response.status
            if response.status != 200:
                record["error_body"] = await response.text()
                raise RuntimeError(f"HTTP {response.status}")
            async for chunk in response.content.iter_any():
                stream.feed(chunk, time.perf_counter() - started)
        record.update(stream.complete(expected_cached_tokens=cell["expect_cached"]))
    except Exception as error:  # noqa: BLE001 - retain every transport/protocol failure
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["tokens"] = list(stream.tokens)
        record["token_times"] = list(stream.token_times)
    finally:
        ended = time.perf_counter()
        record.update(
            latency=ended - started,
            completion=ended - origin,
            raw_events=stream.events,
            trailing_received_bytes_hex=bytes(stream.buffer).hex(),
            trailing_sse_data_fields_hex=[bytes(v).hex() for v in stream.fields],
        )
    return record


async def collect_block(session, url, work, tag, *, logprobs=False):
    """One fixed workload block; return all responses even when one fails."""
    cells = work["cells"]
    concurrency = work["concurrency"]
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("Complete unique logical request set required")
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError("Bounded HTTP concurrency required")
    gate = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    wall_started = time.time()

    async def one(index, cell):
        async with gate:
            return await collect_request(
                session, url, cell, f"{tag}-{work['name']}-{index:03d}", started, logprobs=logprobs
            )

    rows = await asyncio.gather(*(one(i, cell) for i, cell in enumerate(cells)))
    elapsed = time.perf_counter() - started
    errors = [{"rid": r["rid"], **r["error"]} for r in rows if not r["complete"]]
    record = {
        "workload": work["name"],
        "concurrency": concurrency,
        "elapsed": elapsed,
        "monotonic_started": started,
        "unix_started": wall_started,
        "requests": rows,
        "errors": errors,
        "complete": not errors,
        "workload_sha256": hashlib.sha256(
            json.dumps(work, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "timing_scope": "Complete HTTP block including client queue and every failed request",
        "raw_writes_during_measured_block": False,
        "full_http_qualified": False,
    }
    if errors:
        return record  # Preserve partial outcomes; no success-only metric.
    total = sum(len(r["tokens"]) for r in rows)
    record.update(
        output_tokens=total,
        output_tokens_per_second=total / elapsed,
        strict_slo_goodput=sum(r["ttft"] <= 2 and r["tpot"] <= 0.05 for r in rows) / elapsed,
        cached_min=min(r["cached_tokens"] for r in rows),
        cached_max=max(r["cached_tokens"] for r in rows),
    )
    for kind in ("TTFT", "TPOT"):
        values = [r[kind.lower()] for r in rows]
        for label, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            record[f"{kind}-{label}"] = float(np.quantile(values, quantile))
    return record
