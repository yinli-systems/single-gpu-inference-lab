# Transformers `StopStringCriteria` preprocessing on a real vocabulary

`harness/bench_stopstring.py` holds `StopStringCriteria._stop_string_get_matching_positions`
copied verbatim from Transformers `main` (2026-09-23) next to a candidate that finds the same
positions with native `find` plus first/last-unit indices. The cleaned token list is rebuilt the
way upstream does for a byte-level tokenizer. Input: Qwen2.5's `tokenizer.json`
(151,665 tokens), not committed — fetch it with
`curl -L -o tokenizer.json https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json`.

| Run | File |
| --- | --- |
| 336/336 differential checks (real vocab × 20 stop strings incl. CJK/emoji/repeats, exhaustive `{a,b}` vocabularies, 300 random trials, degenerate inputs, str mode) | `results/stress.txt` |
| Helper, upstream vs candidate: 1 stop 419 → 83 ms (5.0×), 4 stops 1.60 → 0.32 s (5.0×), 1 long stop 981 → 81 ms (12.2×), 8 stops 2.85 → 0.61 s (4.6×); outputs identical | `results/bench-real-vocab.txt` |
| Whole cache miss (helper + vocab prep + embedding fill): 1 stop 636 → 298 ms (2.13×), 4 stops 1.87 → 0.53 s (3.50×), 8 stops 3.18 → 0.84 s (3.78×) | `results/share-of-cache-miss.txt` |

Absolute times in these files are from a loaded laptop and are ~1.5× the first run's; the ratios
match it. Why it matters: `STOP_STRING_EMBEDDING_CACHE` keeps 8 entries keyed on the exact
stop-string set, so a server whose requests bring varied stop strings pays the whole miss
repeatedly. Semantics that the naive reading gets wrong (and the first draft did): for negative
`i` the upstream loop slices `reversed_token[-i:]` — a positive start index — and rebinds `i` to
0, so every negative offset is an end overlap, emitted in ascending length. Not submitted; the
end-to-end `generate()` share is unmeasured.
