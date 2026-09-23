# vLLM #58181 — integer token IDs for generate output logprobs

`0001-generate-logprobs-shape.patch` replaces `ChatCompletionLogProbs` with `"token_id:N"`
placeholder strings on `/inference/v1/generate` with `GenerateLogProbs`
(`token_id`, `logprob`, `rank`, `top_logprobs` as a rank-ordered list), in the Python and Rust
frontends, streaming and non-streaming; derender converts back to the OpenAI shapes with the
existing U+FFFD byte-fallback correction. The shape was settled with the maintainers on
[#57574](https://github.com/vllm-project/vllm/issues/57574).

Verification: `cargo test -p vllm-server raw_generate` 9/9 (wire shape asserted end to end
through the mock engine); `pytest test_generate_logprobs_conversion.py test_tokens_logprobs.py`
7/7. The first draft of the byte-fallback test asserted the wrong expectation for a lone
first half of a two-byte character; a simulation of the real conversion code caught it before
submission.
