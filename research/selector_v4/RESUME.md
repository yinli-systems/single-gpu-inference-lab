# Resume selector v4 safely

Current worktree: `/Users/kevin/Projects/sgi-selector-v4-safe-autotune-20260930`; branch `kevin/selector-v4-safe-autotune-20260930`.

Read `STATUS.json`, `PROTOCOL.md`, and any campaign receipt before acting. V3.2.2 is permanently HOLD and its 30 release geometries are development evidence. Never tune v4 using canary/release/stress outcomes and continue calling the same data fresh.

Next admissible step is a dual-GPU `dev` smoke using only the two explicitly exposed historical geometries. It validates JIT, exact output/LSE, frozen windows, candidate identity and evidence format. It cannot authorize canary or change thresholds. After a clean smoke, freeze source archive and submit v4 canary on both GPU families. Both canaries must PASS before release authorization.

Do not start full serving. The historical 2/432 token divergence remains unresolved and independently blocks default promotion.
