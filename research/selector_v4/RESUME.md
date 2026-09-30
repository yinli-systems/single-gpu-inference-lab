# Resume selector v4

Worktree: `/Users/kevin/Projects/sgi-selector-v4-autotune-20260930`.
Branch: `kevin/selector-v4-autotune-20260930`.

The official-0.7 source overlay was generated under `/ssd/scxi253/single-gpu-inference-selector-v4-dev-20260930`. The first smoke campaign `/ssd/scxi253/single-gpu-inference-selector-v4-canary-20260930T185501Z` is invalid because `FLASHINFER_WORKSPACE_BASE` was omitted and a shared-home JIT module was reused. Preserve jobs1643723/1643724 and `evidence/superseded-cache-alias/`; never count them as v4 GPU validation.

Before release, submit corrected private-cache smoke on both GPUs. Require exact ragged/paged output/LSE, policy bits, C++/Python candidate-pool agreement and `audit_binary.py` success. Then run one-shard full canary pipeline per GPU: calibration3×8, frozen decisions, evaluation3×8. Only dual canary PASS may authorize the 48-case untouched release manifest (`release_hash=ee80e2cce74be6c17c94e2ebc98e3911d7ce634ade54a79e303b5f6972b4ca16`).

Default and serving remain OFF. Full HTTP and the historical token-divergence investigation remain separate later gates.
