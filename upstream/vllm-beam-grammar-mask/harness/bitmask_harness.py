"""Grammar-bitmask unpacking in offline beam search, on real grammar masks.

PYTHONPATH must point at the tree with the change. The reference below is
main's _bitmask_to_token_ids copied verbatim (feb87b19).

Part A: differential check, reference vs change, on synthetic edge cases and
        on every mask a real xgrammar matcher produces along a JSON trajectory.
Part B: per-beam cost in _build_beam_sampling_params with vLLM's real
        XgrammarBackend - compile_grammar, accept_tokens (replay), fill_bitmask,
        unpack - at several depths of that trajectory.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from vllm.entrypoints.generate.beam_search.offline import _bitmask_to_token_ids
from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

MODEL = "/data/run01/scxi253/inference/models/Qwen1.5-MoE-A2.7B-Chat"

# --- main's implementation, verbatim ---------------------------------------
_bitmask_cache: dict = {}


def reference(bitmask_row: torch.Tensor, vocab_size: int) -> list[int]:
    if vocab_size not in _bitmask_cache:
        indices = torch.arange(vocab_size)
        _bitmask_cache[vocab_size] = (
            indices,
            indices >> 5,  # i // 32
            indices & 31,  # i % 32
        )
    indices, word_indices, bit_indices = _bitmask_cache[vocab_size]
    mask = ((bitmask_row[word_indices] >> bit_indices) & 1).bool()
    return indices[mask].tolist()


# ---------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"enum": ["admin", "editor", "viewer"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "role", "tags", "active"],
}
INSTANCE = {
    "name": "Ada Lovelace, Countess of Lovelace, analyst of the Analytical Engine",
    "age": 36,
    "role": "editor",
    "tags": ["mathematics", "computing", "poetical science", "notes on Menabrea"],
    "active": True,
}


def median_ms(fn, reps=21):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab_size = 151936
    backend = XgrammarBackend(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
            speculative_config=None,
        ),
        tokenizer=tok,
        vocab_size=vocab_size,
    )
    spec = json.dumps(SCHEMA)
    trajectory = tok.encode(
        json.dumps(INSTANCE, separators=(",", ":")), add_special_tokens=False
    )
    bitmask = backend.allocate_token_bitmask(1)
    print(f"vocab {vocab_size}, bitmask row {tuple(bitmask[0].shape)} "
          f"{bitmask.dtype}, trajectory {len(trajectory)} tokens", flush=True)

    # ---------------- Part A: differential ----------------
    checks = 0
    rng = random.Random(0)

    def check(row, vs):
        nonlocal checks
        checks += 1
        a, b = reference(row, vs), _bitmask_to_token_ids(row, vs)
        assert a == b, (vs, len(a), len(b))
        assert all(type(x) is int for x in b[:5])

    # real masks along the trajectory
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, spec)
    real_sizes = []
    for t in trajectory:
        grammar.fill_bitmask(bitmask, 0)
        check(bitmask[0], vocab_size)
        real_sizes.append(len(reference(bitmask[0], vocab_size)))
        assert grammar.accept_tokens("x", [t]), "trajectory rejected by grammar"
    # synthetic edge cases
    for vs in (1, 31, 32, 33, 63, 64, 65, 1000, 32000, 151936, 151937):
        words = (vs + 31) // 32
        for fill in ("zero", "ones", "rand", "sparse", "high"):
            for extra_words in (0, 3):
                row = torch.zeros(words + extra_words, dtype=torch.int32)
                if fill == "ones":
                    row.fill_(-1)  # every bit, including padding past vocab
                elif fill == "rand":
                    row = torch.randint(-(2**31), 2**31 - 1, row.shape, dtype=torch.int32)
                elif fill == "sparse":
                    for _ in range(5):
                        i = rng.randrange(vs)
                        row[i // 32] |= torch.tensor(1 << (i % 32)).to(torch.int32)
                elif fill == "high":
                    row.fill_(torch.tensor(-(2**31), dtype=torch.int32).item())
                check(row, vs)
                # the same data as a non-contiguous view
                wide = torch.zeros(row.numel(), 2, dtype=torch.int32)
                wide[:, 0] = row
                check(wide[:, 0], vs)
    # the caller's tensor must stay resizable (no storage shared with NumPy)
    probe = torch.zeros(8, dtype=torch.int32)
    _bitmask_to_token_ids(probe, 200)
    probe.resize_(16)
    print(f"Part A: {checks} masks identical (real trajectory masks: allowed-set "
          f"size min {min(real_sizes)} / median {int(statistics.median(real_sizes))} "
          f"/ max {max(real_sizes)})", flush=True)

    # ---------------- Part B: per-beam cost ----------------
    print("\ndepth  allowed   compile   replay    fill   unpack(main)  unpack(new)  "
          "per-beam main -> new", flush=True)
    rows = []
    for depth in (0, 8, 24, 48, len(trajectory) - 1):
        prefix = trajectory[:depth]

        def compile_only():
            return backend.compile_grammar(StructuredOutputOptions.JSON, spec)

        def compile_and_replay():
            g = backend.compile_grammar(StructuredOutputOptions.JSON, spec)
            if prefix:
                g.accept_tokens("beam", prefix)
            return g

        t_compile = median_ms(compile_only)
        t_replay = median_ms(compile_and_replay) - t_compile
        g = compile_and_replay()
        t_fill = median_ms(lambda: g.fill_bitmask(bitmask, 0))
        g.fill_bitmask(bitmask, 0)
        allowed = len(reference(bitmask[0], vocab_size))
        t_old = median_ms(lambda: reference(bitmask[0], vocab_size))
        t_new = median_ms(lambda: _bitmask_to_token_ids(bitmask[0], vocab_size))
        base = t_compile + t_replay + t_fill
        rows.append(dict(depth=depth, allowed=allowed, compile=t_compile,
                         replay=t_replay, fill=t_fill, unpack_main=t_old,
                         unpack_new=t_new))
        print(f"{depth:5d} {allowed:8d}  {t_compile:7.3f}  {t_replay:7.3f}  "
              f"{t_fill:6.3f}  {t_old:10.3f}   {t_new:10.3f}    "
              f"{base + t_old:7.3f} -> {base + t_new:7.3f} ms", flush=True)
    json.dump(dict(real_sizes=real_sizes, rows=rows), open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
