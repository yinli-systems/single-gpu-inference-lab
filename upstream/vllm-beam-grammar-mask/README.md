# vLLM offline beam search — no dense allowed-token sets under grammars

`0001-no-dense-allowed-sets.patch`. Measured in context first: with vLLM's real
`XgrammarBackend`, the Qwen1.5 tokenizer (151,936) and a JSON-schema grammar, per beam per step
compile is ~0.008 ms (cached), replay ≤ 0.11 ms, fill ≤ 0.07 ms, and turning the bitmask into a
token list 0.8–4.4 ms (`results/per-beam-breakdown-real-xgrammar.json`). Inside JSON strings the
grammar allows ~147k tokens; the list was then turned into a `set()` just to test ~2B candidates
(9.05 ms per dense beam vs 0.09 ms for a direct bit test, `harness/dense_probe.py`). A faster
unpack alone gives only 1.7–2.7× on these masks; not building the list and set is the fix.

Grammar work per step at B=32 over a 53-token conforming trajectory
(`harness/grammar_step_harness.py bench`): main 206.0 / 216.9 ms, change 6.9 / 6.3 ms
(`results/grammar-step-w32-*.json`). Equivalence on 56 real grammar states plus a terminated and
an off-grammar beam: identical drop decisions, engine `allowed_token_ids`, and membership for
every token ID (`harness/grammar_step_harness.py diff`, digests in
`../vllm-beam-defer-materialization/results/differential-dumps.sha256`). 68 unit tests; each of
four injected bugs (bit order, missing bounds check, off-by-one at the 1024 cap, keeping dead-end
beams) fails them.
