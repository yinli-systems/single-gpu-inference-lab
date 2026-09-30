# Resume selector v4.1 safely

## Branch and state

- Worktree: `/Users/kevin/Projects/sgi-selector-v4-safe-autotune-20260930`
- Branch: `kevin/selector-v4-safe-autotune-20260930`
- Last committed provenance checkpoint before v4.1 integration: `5849c16f5ac6ad78643b8876cab9145491d8f05e`.
- V4.1 integration is not valid GPU evidence until committed source, immutable campaign receipt and job IDs are recorded below.
- No active v4 jobs were present when integration resumed. Do not repeat v3.2.2 96/96 release.

## Persisted evidence to reuse

- Old-overlay safe dev jobs1643638/1643639 completed. Reanalysis from campaign manifest/source is HOLD only because duplicate controls do not resolve; use only to validate cross-fit/cache machinery.
- Sibling isolated-symbol private-cache smoke jobs1643750/1643751 produced exact dual-GPU outputs and compiled co-resident symbols, but Slurm ended nonzero due an obsolete audit path. Copied under `evidence/kernel-isolation-smoke-r2-legacy/`; supporting development evidence only.
- First canary jobs1643723/1643724 used an aliased shared JIT cache and are invalid/retained.
- Sibling jobs1643750/1643751 and all old failures must not be rerun as if fresh.

## V4.1 implementation delta

- separate native/resource ragged+paged kernel symbols;
- legacy plan defaults native; explicit policy supports only native0/cap1;
- runtime requires an isolation marker before applying cached cap;
- unsupported/ineligible cap becomes an explicit second-native null arm;
- native-after-cap exact check per paired cell;
- identity includes binding hash and shared-memory limits;
- 96ms windows, 16 blocks/process, >=32 training blocks, 20k bootstrap draws;
- per-shard compiled binary audit and active-clock telemetry gate;
- all sibling-v4 shapes added to the historical freshness ledger.

## Next admissible action

1. Finish full remote CPU/source validation and commit the exact v4.1 source.
2. Build a new immutable dev-smoke campaign from the official0.7 pristine overlay using `prepare_campaign.py`.
3. Submit exactly one smoke job per GPU using `smoke_v4.sbatch`; use private pristine/candidate JIT roots.
4. Require exact ragged/paged outputs for native-before/cap/native-after and a passing compiled binary audit.
5. Only then run the 2 exposed dev cases. Do not consume the 10 canary shapes until dev controls resolve and all source/binary contracts pass.
6. Canary/release/full HTTP remain OFF/HOLD until their own gates pass. Historical2/432 divergence remains unresolved.

## Reasoning-setting limitation

The UI showed Pro power5/5, but an option explicitly named Max could not be verified. Do not claim Max was enabled.
