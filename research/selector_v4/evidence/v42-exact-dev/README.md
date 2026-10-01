# V4.2 exact-source dual-GPU dev PASS

Both exposed geometries passed every frozen requirement on RTX4090 and RTX5090. Four jobs COMPLETED0:0; immutable measurement source b769e7c, archive60ffb32e. All225 raw measurement/analysis/source/binary/telemetry/dispatch files are individually hash-bound in receipt.json and preserved in raw-dev-evidence.tar.gz. Raw summaries are retained unchanged.

Actual policy relative to official pristine:4090 geometric mean1.383822, worst1.311622;5090 geometric mean1.285451, worst0.992863. Selected cap means1.385991 and1.398568 with coverage100% and75%; joint-min95%LCB1.316414 and1.304803. Native/pristine joint-min95%LCB0.993662 and0.992892 passes the unchanged0.99 floor. These are two exposed dev geometries, not release or HTTP performance claims. Raw native/pristine individual block minima can fall below0.99; the predeclared native gate uses per-cell means and joint confidence, while selected-cap block minima separately remain above0.99.

Frozen analyzer description mistakenly inverts regret wording; the implemented numeric formula is chosen latency / oracle latency -1. Receipt annotates this without modifying the frozen summaries.

Dual dev PASS authorized ten fresh canary shapes. Canary jobs1644188–1644207 consumed those ten shapes on2026-10-01T01:43UTC. Release48 and stress12 remain untouched at this snapshot. Defaults and serving promotion remainOFF; historical2/432 divergence remains unresolved.
