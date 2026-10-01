"""Bounded HTTP SSE observation with complete tokens and explicit failure.

Receive timestamps measure client transport arrival, including server queues.
Tokens in one SSE event share its arrival timestamp. Persist raw events only
outside the measured block; synchronous file writes would perturb the client.
This parser alone grants no source, metric, parity or serving qualification.
"""

import json
import math

from research.release_qualification.serving.workloads import update


class TokenStream:
    def __init__(self, expected_tokens, *, maximum_event_bytes=8 << 20):
        if type(expected_tokens) is not int or expected_tokens < 2:
            raise ValueError("At least two tokens required for TTFT/TPOT")
        self.expected_tokens = expected_tokens
        self.maximum_event_bytes = maximum_event_bytes
        self.buffer = bytearray()
        self.fields = []
        self.events = []
        self.tokens = []
        self.token_times = []
        self.count = 0
        self.done = False
        self.finish = None
        self.cached_tokens = 0
        self.last_arrival = 0.0

    def feed(self, chunk, elapsed):
        if not isinstance(chunk, bytes):
            raise TypeError("Actual received bytes required")
        if not math.isfinite(elapsed) or elapsed < self.last_arrival:
            raise ValueError("Finite monotonic client arrival time required")
        self.last_arrival = elapsed
        self.buffer.extend(chunk)
        if len(self.buffer) > self.maximum_event_bytes:
            raise ValueError("Bounded SSE buffer exceeded")
        while b"\n" in self.buffer:
            line, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            line = line.rstrip(b"\r")
            if not line:
                if self.fields:
                    body = b"\n".join(self.fields)
                    self.fields.clear()
                    self._event(body, elapsed)
            elif line.startswith(b"data:"):
                self.fields.append(bytes(line[5:]).removeprefix(b" "))
                if sum(len(v) for v in self.fields) > self.maximum_event_bytes:
                    raise ValueError("Bounded SSE event exceeded")
            elif line.startswith((b":", b"id:", b"event:", b"retry:")):
                continue
            else:
                raise ValueError("Malformed SSE field")

    def _event(self, body, elapsed):
        # Retain even the malformed/failing event before validation. An explicit
        # caller failure record accompanies this in the raw block archive.
        self.events.append({"client_arrival_seconds": elapsed, "data": body.decode("utf-8")})
        if self.done:
            raise ValueError("Data arrived after terminal DONE")
        if body == b"[DONE]":
            self.done = True
            return
        obj = json.loads(body)
        if not isinstance(obj, dict) or "error" in obj:
            raise ValueError("HTTP stream error or nonobject response")
        meta = obj.get("meta_info")
        if not isinstance(meta, dict):
            raise TypeError("Token metadata missing")
        count = meta.get("completion_tokens")
        tokens, count, added = update(self.tokens, self.count, obj.get("output_ids"), count)
        if count > self.expected_tokens:
            raise ValueError("Excess output tokens")
        cached = meta.get("cached_tokens", 0)
        if type(cached) is not int or cached < 0:
            raise ValueError("Invalid cached-token count")
        self.cached_tokens = max(self.cached_tokens, cached)
        self.tokens, self.count = tokens, count
        self.token_times.extend([elapsed] * added)
        if meta.get("finish_reason") is not None:
            self.finish = meta["finish_reason"]

    def complete(self, *, expected_cached_tokens=0):
        if self.buffer or self.fields:
            raise ValueError("Truncated SSE framing")
        if (
            not self.done
            or self.count != self.expected_tokens
            or len(self.token_times) != self.expected_tokens
            or not isinstance(self.finish, dict)
            or self.finish.get("type") != "length"
        ):
            raise ValueError("Incomplete output or missing length/DONE termination")
        if self.cached_tokens < expected_cached_tokens:
            raise ValueError("Expected radix-prefix cache hit missing")
        ttft = self.token_times[0]
        tpot = (self.token_times[-1] - ttft) / (self.expected_tokens - 1)
        if ttft <= 0 or tpot <= 0:
            raise ValueError("Nonpositive HTTP metric; no epsilon replacement")
        return {
            "tokens": list(self.tokens),
            "token_times": list(self.token_times),
            "ttft": ttft,
            "tpot": tpot,
            "cached_tokens": self.cached_tokens,
            "finish_reason": self.finish,
            "timing_scope": "Client SSE arrival; co-delivered tokens share arrival time",
            "complete": True,
        }
