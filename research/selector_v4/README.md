# Selector v4 safe autotuning

V4 converts the resource-cap mechanism from a static selector into a deployment-matched, confidence-gated tactic autotuner with native fallback.

Current state:
- v3.2.2 final dual-GPU HOLD archived and immutable;
- v4 core identity, eligibility, confidence selector and atomic cache implemented;
- 10 canary + 48 release + 12 stress geometries frozen and exact-deduplicated from all historical manifests;
- GPU harness measures native/cap under eager, Graph1 and Graph16 with three-process cross-fitting;
- default and serving promotion remain disabled.

Entry points: `PROTOCOL.md`, `measure_v4.py`, `analyze_v4.py`, `run_v4.sbatch`, and `authorize_v4.py`.
