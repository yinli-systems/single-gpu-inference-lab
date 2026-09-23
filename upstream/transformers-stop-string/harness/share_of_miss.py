"""How much of a StopStringCriteria cache miss is the matching helper?

The embedding-vec builder below is the upstream body with the torch.tensor()
call dropped (torch is not installed here); everything else — the numpy
gather_vec fill — is verbatim, so the share it reports is a lower bound on the
non-helper cost.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from bench_stopstring import (
    load_clean_vocab,
    stop_positions_candidate,
    stop_positions_upstream,
)


def build_embedding_vec(token_list, token_indices, stop_strings, positions_fn):
    token_valid_positions, token_end_overlaps = positions_fn(
        token_list, token_indices, stop_strings
    )
    t0 = time.perf_counter()
    all_valid_positions = [
        len(val) for positions in token_valid_positions.values() for val in positions.values()
    ]
    max_valid_positions = max(all_valid_positions) if all_valid_positions else 1
    valid_end_lens = [
        len(val) for positions in token_end_overlaps.values() for val in positions.values()
    ]
    if not valid_end_lens:
        raise ValueError("no end lens")
    max_valid_end_lens = max(valid_end_lens)
    vec_size = len(stop_strings) * (max_valid_positions + max_valid_end_lens) + 1
    gather_vec = np.full((max(token_indices) + 2, vec_size), dtype=np.int32, fill_value=-1)

    for i, stop_string in enumerate(stop_strings):
        positions = token_valid_positions[stop_string]
        end_lens = token_end_overlaps[stop_string]
        for token_idx, valid_positions in positions.items():
            gather_vec[
                token_idx, max_valid_positions * i : max_valid_positions * i + len(valid_positions)
            ] = valid_positions
        for token_idx, possible_end_lens in end_lens.items():
            base = max_valid_positions * len(stop_strings) + max_valid_end_lens * i
            gather_vec[token_idx, base : base + len(possible_end_lens)] = possible_end_lens

    for token, token_idx in zip(token_list, token_indices):
        gather_vec[token_idx, -1] = len(token)
    return (time.perf_counter() - t0) * 1e3


def median_ms(fn, repeats=3):
    return statistics.median(fn() for _ in range(repeats))


def main():
    tokens, indices, _ = load_clean_vocab("tokenizer.json")

    def clean_vocab_cost():
        t0 = time.perf_counter()
        load_clean_vocab("tokenizer.json")
        return (time.perf_counter() - t0) * 1e3

    vocab_ms = median_ms(clean_vocab_cost)

    print(f"{'stop strings':30} {'helper up':>10} {'helper new':>11} {'vec fill':>9} "
          f"{'miss up':>9} {'miss new':>9} {'miss ratio':>10}")
    for label, stops in (
        ("1  ('<|im_end|>')", ["<|im_end|>"]),
        ("4  (agent-style)", ["</tool_call>", "\nObservation:", "<|im_end|>", "STOP"]),
        ("8  (long stop list)", ["</s>", "\nUser:", "\nAssistant:", "###", "<|im_end|>",
                                 "</tool_call>", "\n\n\n", "Final Answer:"]),
    ):
        sb = tuple(s.encode("utf-8") for s in stops)

        def t(fn):
            t0 = time.perf_counter()
            fn(tokens, indices, sb)
            return (time.perf_counter() - t0) * 1e3

        up_ms = median_ms(lambda: t(stop_positions_upstream))
        new_ms = median_ms(lambda: t(stop_positions_candidate))
        vec_ms = median_ms(lambda: build_embedding_vec(tokens, indices, sb, stop_positions_candidate))
        miss_up = up_ms + vec_ms + vocab_ms
        miss_new = new_ms + vec_ms + vocab_ms
        print(f"{label:30} {up_ms:9.1f}  {new_ms:10.1f} {vec_ms:8.1f}  {miss_up:8.1f}  "
              f"{miss_new:8.1f}  {miss_up / miss_new:9.2f}x")
    print(f"\nvocab prep (this script's byte-level reproduction of "
          f"clean_tokenizer_vocab): {vocab_ms:.1f} ms")
    print("Cache: STOP_STRING_EMBEDDING_CACHE is an OrderedDict capped at 8 entries (LRU),")
    print("keyed on (token_list, token_indices, stop_strings, matching_mode).")


if __name__ == "__main__":
    main()
