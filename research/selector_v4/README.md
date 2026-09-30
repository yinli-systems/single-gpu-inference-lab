# Selector v4 — isolated tactics and deployment-mode autotuning

Selector v4 is a fail-closed continuation of the completed v3.2.2 HOLD result. It does not reinterpret that release as a pass. The exposed v3.2.2 cells are development evidence only.

The implementation has two safety layers:

1. **Kernel-state isolation.** The official native kernel definition remains source-identical. A renamed clone carries the 64-KiB dynamic-shared-memory tactic, so setting the capped symbol's function attribute cannot mutate the native symbol used by fallback calls.
2. **Measurement-based final choice.** A conservative 2.4 block-wave geometry rule only admits candidates to calibration. Native versus cap is then selected separately for eager, Graph1 and Graph16 from exact-output, clock-stable, hierarchical paired evidence. Cache miss, corruption, insufficient evidence or unresolved controls all select native.

The v3.2.2 development evidence motivates the 2.4-wave floor: it keeps the strong 4090 region and excludes the exposed 5090 47-descriptor counterexample. That retrospective check is not v4 validation.

The frozen v4 manifest contains 8 development canaries and 48 untouched release cases across six regimes and batch sizes 5–10. Release cases cannot be used to change v4 and remain fresh.

Current status: source contracts and official-source overlay generation are implemented. Default and serving promotion remain disabled. The historical 2/432 full-model token divergence remains an independent blocker.
