"""Differential stress test: candidate vs upstream stop-string preprocessing.

Runs on the real Qwen2.5 vocabulary (151k byte-level tokens) plus adversarial
synthetic vocabularies (repeated units, stop-string-containing tokens, tokens
longer than the stop string, empty stop string, multi-byte UTF-8 boundaries).
"""

from __future__ import annotations

import random
import sys

from bench_stopstring import (
    load_clean_vocab,
    stop_positions_candidate,
    stop_positions_upstream,
)


def check(tokens, indices, stops, label):
    up = stop_positions_upstream(tokens, indices, stops)
    ca = stop_positions_candidate(tokens, indices, stops)
    if up != ca:
        for s in stops:
            for kind, a, b in (("valid", up[0][s], ca[0][s]), ("end", up[1][s], ca[1][s])):
                if a != b:
                    bad = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
                    k = bad[0]
                    print(f"FAIL [{label}] stop={s!r} {kind} token_idx={k} "
                          f"token={tokens[indices.index(k)]!r}\n  upstream={a.get(k)}\n  candidate={b.get(k)}")
                    return False
    return True


def main():
    rng = random.Random(20260923)
    real_tokens, real_indices, _ = load_clean_vocab("tokenizer.json")
    failures = 0
    checks = 0

    # 1. Real vocabulary, many stop strings (one at a time to keep it quick).
    real_stops = [
        "<|im_end|>", "</tool_call>", "STOP", "\n\n", "###", "=" * 40,
        "。", "😀", "Ω≈ç√", "  ", "\t\n", "a", "<|endoftext|>", "Final Answer:",
        "</s></s>", "éé", "]]>", "```", "assistant\n", "0",
    ]
    subset = real_tokens[:40000], real_indices[:40000]
    for s in real_stops:
        checks += 1
        if not check(subset[0], subset[1], (s.encode("utf-8"),), f"real40k/{s!r}"):
            failures += 1

    # 2. Adversarial synthetic vocabularies over a tiny alphabet: every token of
    #    length 1..4 over "ab", plus tokens that embed the stop string.
    alpha = b"ab"
    exhaustive = []
    for n in range(1, 5):
        stack = [b""]
        for _ in range(n):
            stack = [p + bytes([c]) for p in stack for c in alpha]
        exhaustive.extend(stack)
    for stop in (b"a", b"ab", b"aba", b"abab", b"aa", b"aaa", b"b" * 6):
        toks = tuple(exhaustive + [stop, stop + stop, b"x" + stop + b"x", stop[:-1], b""])
        idx = tuple(range(len(toks)))
        checks += 1
        if not check(toks, idx, (stop,), f"exhaustive/{stop!r}"):
            failures += 1

    # 3. Random tokens and random stop strings, including long tokens.
    for trial in range(300):
        alphabet = rng.choice([b"ab", b"abc", bytes(range(0x20, 0x30)), b"\xe4\xb8\xad\xe6\x96\x87"])
        toks = tuple(
            bytes(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            for _ in range(200)
        )
        idx = tuple(range(len(toks)))
        slen = rng.randint(1, 10)
        stop = bytes(rng.choice(alphabet) for _ in range(slen))
        checks += 1
        if not check(toks, idx, (stop,), f"random/{trial}"):
            failures += 1
            if failures > 3:
                break

    # 4. Degenerate inputs.
    toks = (b"", b"a", b"ab", b"aaa", b"\xff\xfe", b"x" * 64)
    idx = tuple(range(len(toks)))
    for stop in (b"", b"a", b"x" * 3, b"\xff", b"x" * 100):
        checks += 1
        if not check(toks, idx, (stop,), f"degenerate/{stop!r}"):
            failures += 1

    # 5. str (non-byte) mode, which is what a non-byte-level tokenizer gives.
    str_toks = ("st", "op", "sto", "pper", "las", "topper", "stop", "s", "", "stopstop")
    str_idx = tuple(range(len(str_toks)))
    for stop in ("stop", "s", "stopstop", "pp"):
        checks += 1
        if not check(str_toks, str_idx, (stop,), f"str/{stop!r}"):
            failures += 1

    print(f"{checks - failures}/{checks} differential checks identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
