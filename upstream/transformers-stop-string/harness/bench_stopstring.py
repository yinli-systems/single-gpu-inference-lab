"""Does the transformers stop-string preprocessing cost matter on a REAL vocabulary?

Upstream helper is copied verbatim from
huggingface/transformers main:src/transformers/generation/stopping_criteria.py
(StopStringCriteria._stop_string_get_matching_positions), fetched 2026-09-23.

The cleaned token list is reproduced the way upstream builds it for a byte-level
tokenizer (Qwen2.5): every vocab token is mapped through the inverse of
bytes_to_unicode(), so tokens are `bytes` and stop strings are UTF-8 bytes.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict


# ---------------------------------------------------------------- upstream ---
def stop_positions_upstream(token_list, token_indices, stop_strings):
    """Verbatim copy of StopStringCriteria._stop_string_get_matching_positions."""
    token_valid_positions = {}
    token_end_overlaps = {}
    for stop_string in stop_strings:
        reversed_stop_string = stop_string[::-1]
        token_valid_positions[stop_string] = {}
        token_end_overlaps[stop_string] = {}
        for token, tok_idx in zip(token_list, token_indices):
            reversed_token = token[::-1]
            matching_positions = []
            possible_end_lengths = []
            for i in range(1 - len(token), len(stop_string)):
                if i < 0:
                    tok = reversed_token[-i:]
                    i = 0
                else:
                    tok = reversed_token
                stop = reversed_stop_string[i : i + len(tok)]
                if tok.startswith(stop):
                    if i == 0:
                        possible_end_lengths.append(min(len(tok), len(stop)))
                    else:
                        matching_positions.append(i)

            if matching_positions:
                token_valid_positions[stop_string][tok_idx] = matching_positions
            if possible_end_lengths:
                token_end_overlaps[stop_string][tok_idx] = possible_end_lengths
    return token_valid_positions, token_end_overlaps


# --------------------------------------------------------------- candidate ---
def stop_positions_candidate(token_list, token_indices, stop_strings):
    """Same output, but candidates are found with native find/slice comparisons.

    Upstream iterates i from 1-L to S-1. Note that for i < 0 it slices
    `reversed_token[-i:]`, i.e. it DROPS the first (-i) units of the reversed
    token, so the compared prefix has length k = L + i, ascending with i, and i
    is then rebound to 0 (so every i < 0 contributes an end overlap, never a
    position). Translated to forward coordinates, for stop string s (len S) and
    token t (len L):

      end overlaps, in this order:
        * k = 1..L-1 ascending:
            k <= S : matches iff t[:k] == s[S-k:]      -> end length k
            k >  S : matches iff t[k-S:k] == s         -> end length S
        * then i == 0:
            L <= S : matches iff t == s[S-L:]          -> end length L
            L >  S : matches iff t[L-S:] == s          -> end length S

      positions (i > 0), in this order:
        * internal, i = S-L-start > 0 for each occurrence of t in s
          (start descending == i ascending)
        * left overhang, i = S-m for m = min(L,S)-1..1 (m descending),
          matches iff t[L-m:] == s[:m]
    """
    token_valid_positions = {}
    token_end_overlaps = {}

    for stop_string in stop_strings:
        S = len(stop_string)
        if S == 0:
            # Degenerate: upstream's loop never reaches i == 0, so the fast
            # paths below do not apply. Defer to the reference implementation.
            ref_valid, ref_overlaps = stop_positions_upstream(
                token_list, token_indices, (stop_string,)
            )
            token_valid_positions[stop_string] = ref_valid[stop_string]
            token_end_overlaps[stop_string] = ref_overlaps[stop_string]
            continue
        valid = {}
        overlaps = {}

        # For the two "hang off the edge" families the first/last unit of the
        # token pins the candidate length, so index them once per stop string.
        # head_by_unit[c] = [k : s[S-k] == c]  (candidate end overlaps)
        # tail_by_unit[c] = [m : s[m-1] == c]  (candidate start overlaps)
        head_by_unit = defaultdict(list)
        tail_by_unit = defaultdict(list)
        for k in range(1, S + 1):
            head_by_unit[stop_string[S - k : S - k + 1]].append(k)
            tail_by_unit[stop_string[k - 1 : k]].append(k)

        for token, tok_idx in zip(token_list, token_indices):
            L = len(token)
            if L == 0:
                # Upstream: tok and stop are both empty for every i in 1..S-1,
                # and startswith("") is true, so an empty token "matches"
                # everywhere except i == 0 (which is not reached).
                if S > 1:
                    valid[tok_idx] = list(range(1, S))
                continue
            first = token[:1]
            last = token[-1:]

            # --- end overlaps: k = 1..L-1 ascending, then the i == 0 case ---
            possible_end_lengths = []
            for k in head_by_unit.get(first, ()):  # ascending k, all k <= S
                if k < L and token[:k] == stop_string[S - k :]:
                    possible_end_lengths.append(k)
            if L - 1 > S:
                # k = start + S > S: the whole stop string sits inside the
                # token's first k units. start == 0 would be k == S, which the
                # loop above already covered.
                start = token.find(stop_string, 1)
                while start != -1 and start + S <= L - 1:
                    possible_end_lengths.append(S)
                    start = token.find(stop_string, start + 1)
            if L <= S:
                if token == stop_string[S - L :]:
                    possible_end_lengths.append(L)
            elif token.endswith(stop_string):
                possible_end_lengths.append(S)

            # --- internal matches: token is a substring of the stop string ---
            matching_positions = []
            if L < S:
                # i = S - L - start, so start descending == i ascending
                starts = []
                start = stop_string.find(token)
                while start != -1:
                    starts.append(start)
                    start = stop_string.find(token, start + 1)
                for start in reversed(starts):
                    i = S - L - start
                    if i > 0:
                        matching_positions.append(i)

            # --- left overhang: token tail == stop head, m descending ---
            for m in reversed(tail_by_unit.get(last, ())):  # descending m
                if m < L and m < S and token[L - m :] == stop_string[:m]:
                    matching_positions.append(S - m)

            if matching_positions:
                valid[tok_idx] = matching_positions
            if possible_end_lengths:
                overlaps[tok_idx] = possible_end_lengths

        token_valid_positions[stop_string] = valid
        token_end_overlaps[stop_string] = overlaps

    return token_valid_positions, token_end_overlaps


# ------------------------------------------------------------------ vocab ----
def bytes_to_unicode():
    """GPT-2 byte<->unicode table (same as transformers.convert_slow_tokenizer)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def load_clean_vocab(path):
    """Reproduce clean_tokenizer_vocab(..., stop_string_matching_mode='byte_level')."""
    data = json.load(open(path))
    vocab = dict(data["model"]["vocab"])
    for added in data.get("added_tokens", []):
        vocab.setdefault(added["content"], added["id"])
    byte_decoder = {u: b for b, u in bytes_to_unicode().items()}
    tokens, indices, undecodable = [], [], 0
    for token, idx in vocab.items():
        if all(ch in byte_decoder for ch in token):
            tokens.append(bytes(byte_decoder[ch] for ch in token))
        else:
            # upstream falls back to convert_tokens_to_string(); those tokens are
            # the special ones, kept as their utf-8 form for this measurement.
            tokens.append(token.encode("utf-8"))
            undecodable += 1
        indices.append(idx)
    return tuple(tokens), tuple(indices), undecodable


# ------------------------------------------------------------------- main ----
def timeit(fn, *args, repeats=3):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3, samples, out


def main():
    tokens, indices, undecodable = load_clean_vocab("tokenizer.json")
    print(f"real vocab: {len(tokens)} tokens ({undecodable} not byte-level), "
          f"mean token length {sum(map(len, tokens)) / len(tokens):.2f} bytes")

    suites = {
        "single short stop ('<|im_end|>')": ["<|im_end|>"],
        "4 stops (agent-style)": ["</tool_call>", "\nObservation:", "<|im_end|>", "STOP"],
        "1 long stop (40 chars)": ["=" * 40],
        "8 stops (max_model 'stop' list)": [
            "</s>", "\nUser:", "\nAssistant:", "###", "<|im_end|>",
            "</tool_call>", "\n\n\n", "Final Answer:",
        ],
    }

    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"\n{'suite':36} {'upstream ms':>12} {'candidate ms':>13} {'ratio':>7}  identical")
    for name, stops in suites.items():
        stop_bytes = tuple(s.encode("utf-8") for s in stops)
        up_ms, up_s, up_out = timeit(stop_positions_upstream, tokens, indices, stop_bytes, repeats=repeats)
        ca_ms, ca_s, ca_out = timeit(stop_positions_candidate, tokens, indices, stop_bytes, repeats=repeats)
        same = up_out == ca_out
        print(f"{name:36} {up_ms:12.1f} {ca_ms:13.2f} {up_ms / ca_ms:6.1f}x  {same}")
        if not same:
            up_v, up_e = up_out
            ca_v, ca_e = ca_out
            for s in stop_bytes:
                for label, a, b in (("valid", up_v[s], ca_v[s]), ("end", up_e[s], ca_e[s])):
                    if a != b:
                        diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5]
                        print(f"    MISMATCH {label} {s!r}: {[(k, a.get(k), b.get(k)) for k in diff]}")
        print(f"{'':36} upstream samples {[round(x * 1e3, 1) for x in up_s]}  "
              f"candidate {[round(x * 1e3, 2) for x in ca_s]}")


if __name__ == "__main__":
    main()
