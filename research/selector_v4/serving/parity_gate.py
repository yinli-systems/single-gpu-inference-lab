"""Complete request/token/logprob parity; historical missing state stays open."""

import json
import math


def full_logprobs(request):
    count = len(request["tokens"])
    latest = None
    for event in request["raw_events"]:
        if event["data"] == "[DONE]":
            continue
        obj = json.loads(event["data"])
        meta = obj["meta_info"]
        values = meta.get("output_token_logprobs")
        tops = meta.get("output_top_logprobs")
        if values is None or tops is None:
            continue
        if not isinstance(values, list) or not isinstance(tops, list):
            raise TypeError("Full cumulative logprob arrays required")
        if latest is not None and (
            values[: len(latest[0])] != latest[0] or tops[: len(latest[1])] != latest[1]
        ):
            raise ValueError("Cumulative logprob prefix changed")
        latest = (values, tops)
    if latest is None or any(len(v) != count for v in latest):
        raise ValueError("Complete cumulative output/top logprobs required")
    values, tops = latest
    for token, value, top in zip(request["tokens"], values, tops):
        if (
            not isinstance(value, list)
            or len(value) != 3
            or type(value[1]) is not int
            or value[1] != token
            or type(value[0]) not in (float, int)
            or not math.isfinite(value[0])
            or not isinstance(top, list)
            or len(top) != 5
            or any(
                not isinstance(v, list)
                or len(v) != 3
                or type(v[0]) not in (float, int)
                or not math.isfinite(v[0])
                or type(v[1]) is not int
                for v in top
            )
        ):
            raise ValueError("Invalid full token/top-logprob record")
    return latest


def compare_blocks(pristine, candidate, *, require_logprobs):
    if (
        not pristine["complete"]
        or not candidate["complete"]
        or pristine["errors"]
        or candidate["errors"]
        or pristine["workload_sha256"] != candidate["workload_sha256"]
    ):
        raise ValueError("Complete same-workload blocks required for parity")
    left = {r["id"]: r for r in pristine["requests"]}
    right = {r["id"]: r for r in candidate["requests"]}
    if (
        set(left) != set(right)
        or len(left) != len(pristine["requests"])
        or len(right) != len(candidate["requests"])
    ):
        raise ValueError("Complete unique logical request set required")
    mismatches = []
    tokens = 0
    for key in sorted(left):
        base, changed = left[key], right[key]
        for field in ("input_ids", "sampling_params"):
            if base["payload"][field] != changed["payload"][field]:
                raise ValueError("Parity request payload changed")
        expected = base["payload"]["sampling_params"]["max_new_tokens"]
        if (
            not base["complete"]
            or not changed["complete"]
            or len(base["tokens"]) != expected
            or len(changed["tokens"]) != expected
        ):
            raise ValueError("Truncated parity output")
        tokens += expected
        differences = [
            i for i, (a, b) in enumerate(zip(base["tokens"], changed["tokens"])) if a != b
        ]
        if differences:
            mismatches.append(
                {
                    "logical_id": key,
                    "kind": "token",
                    "first_difference": differences[0],
                    "positions": differences,
                }
            )
        if require_logprobs and full_logprobs(base) != full_logprobs(changed):
            mismatches.append({"logical_id": key, "kind": "full_token_or_top_logprobs"})
    return {
        "parity_pass": not mismatches,
        "requests_compared": len(left),
        "tokens_compared": tokens,
        "full_logprobs_required": require_logprobs,
        "mismatches": mismatches,
        "full_http_qualified": False,
        "historical_token_divergence_resolved": False,
    }
