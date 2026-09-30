# Separate upstream scopes — checked 2026-09-30

## Performance candidate

Selector-v3.2.1 is an isolated FlashInfer0.7 FA2 resource-policy prototype. Its selector expression and thirty unexecuted v3.2 release geometries are unchanged; its measurement/native-policy implementation is revised. The old v3.2 canary HOLD remains evidence, not a passing result for this implementation.

Required before a performance submission: CUDA compilation and dual-GPU canary qualification for this exact source; independent release worst-case, controls, and paired net-cost gates; evidence of actual tactic activation; current upstream API/caller review; paged and graph lifecycle checks; and full-serving validation with the original token-divergence investigation still independently tracked. No serving speedup, default promotion, current-main compatibility, or maintainer acceptance is claimed.

The legacy native plan call defaults OFF. The added explicit policy entry and sixteen-field plan representation are a research ABI change and require deliberate upstream design review. Source hashes and a private JIT cache are not evidence that every compiled binary is ABI-compatible with unrelated callers.

## HTTP cleanup ownership fix (separate PR27)

Our research PR27 remains at `672822f29953c3f466212bdd00de48b1d32f847d`; no selector changes belong in that PR.

SGLang issue38788 explicitly describes a scripted-runtime test-harness bug, not a serving bug. Duplicate-ID rejection is intended; test synchronization and error propagation are the stated targets.

SGLang PR38850, head `fd553e47cf9fa3687f4929b5e063110635884ec7` when checked, changes only test-harness implementation. Importantly, its description already discusses delayed cleanup aborting a replacement and includes a real tokenizer-generator diagnostic. Therefore our production-path ownership guard is not a first discovery of that underlying hazard. Its possible contribution is a separately validated production mitigation and HTTP regression coverage, to be coordinated with the existing work.

Before any lifecycle submission: compare current production cleanup code, confirm all relevant return/error/abort paths, bound retained request state, cover multi-request/cancellation behavior, and state which parts overlap upstream evidence. The historical full-length token divergence is not explained by the independent request-truncation fix.

Sources: https://github.com/sgl-project/sglang/issues/38788 ; https://github.com/sgl-project/sglang/pull/38850 . Metadata was read through the connected GitHub API. No official upstream PR was created or merged during this recovery.
