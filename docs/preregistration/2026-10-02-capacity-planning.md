# Pre-registration: does the step-cost clock change a capacity decision? (H100, 2026-10-02)

Committed before any live run of this campaign. The simulator predictions it tests were computed and
committed with it, from data that already existed (the H100 Qwen3-4B shape campaign) and the public
traces.

## Why

The report's representation claim (per-request attention work, not an aggregate state) has so far been
tested on step-time prediction. A serving simulator uses the same step-cost model as its clock, and a
capacity planner reads its p99 TTFT to decide how much load one replica can take. Offline, the three
clock forms available in `simulate_geometry_prevalence.py` agree on which configuration is best but
disagree on absolute p99 TTFT on Azure code traffic by up to 4x, so they would provision differently.
This campaign measures which one is right, live.

## Setup (fixed)

- RunPod H100 80GB HBM3 (driver 580.126.09), vLLM 0.29.0 from PyPI with tracer v2 (`apply_tracer_v2.py`;
  `core.py` md5 0b151b41..., `model_runner.py` md5 7d281bfa..., identical to the L20 campaign), FA3,
  Qwen3-4B bf16 (published weights), prefix caching on, `gpu_memory_utilization` 0.9, `max_num_seqs`
  256, `max_model_len` 40960. KV capacity measured on this GPU before the campaign: 445,408 tokens.
- Traces: Mooncake tool-agent (`toolagent_trace.jsonl`, sha256 48a2db1a..., 3 s arrival jitter as in
  `live-trace-replay`) and Azure 2023 code (`AzureLLMInferenceTrace_code.csv`, sha256 54e9a6d2...),
  first 240 s (scaled) of arrivals, then drain; synthetic token ids with the traces' lengths; Mooncake
  block hashes map to real 512-token prefix blocks.
- Configurations: `b2048` (MBT 2048, default), `b8192` (MBT 8192), `b4096t1024` (MBT 4096,
  `long_prefill_token_threshold` 1024).
- Grid: Azure code x {0.75, 1, 1.5, 2, 3}, Mooncake x {0.5, 0.6, 0.7, 0.8}: 27 cells, one server each,
  spread over four GPUs of one pod. Harness `replay_trace_serving.py`, campaign
  `benchmarks/results/h100-capacity-planning/campaign/campaign_h100_capacity.sh`.
- Predictions: `simulate_geometry_prevalence.py` with the H100 Qwen3-4B shape-campaign steps as its
  clock (`h100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv`), the same KV capacity, configurations and
  window, for `--clock aggregate` (M0's (Σq)(Σk) interaction), `meanfield` ((Σq)(Σk)/n, the best
  marginal-only linear form) and `geometry` (Σ qᵢ(kᵢ + (qᵢ+1)/2)); files in
  `benchmarks/results/h100-capacity-planning/predictions/`.

## Measures

- p99 TTFT per cell, recomputed from per-request TTFTs with `np.quantile` (linear), the simulator's
  definition, over requests without errors.
- Capacity of a source (live or a clock) for a (trace, configuration) and an SLO (p99 TTFT ≤ 1 s,
  ≤ 2 s): the largest grid rate whose p99 meets the SLO with every smaller grid rate meeting it too; 0 if
  the smallest misses. Analysis `scripts/analyze_capacity_planning.py`, written before the runs.

## Predictions

- **K1 (accuracy, Azure code).** The geometry clock's median |relative error| of p99 TTFT over the 15
  Azure cells is smaller than the aggregate clock's, and the aggregate clock over-predicts p99 TTFT in
  at least 9 of the 15.
- **K2 (capacity decisions, primary).** Over the 12 decisions (2 traces × 3 configurations × 2 SLOs),
  the geometry clock's capacity equals live capacity in at least 8, and in more decisions than the
  aggregate clock's.
- **K3 (over-provisioning, Azure code, SLO 1 s).** For at least 2 of the 3 configurations, live
  capacity is at least 1.5x the aggregate clock's capacity (or the aggregate clock says no grid rate
  meets the SLO while live says one does).
- **K4 (control, Mooncake).** For every Mooncake decision, the three clocks and live lie within one grid
  step of each other: the knee is sharp, so the clock form matters little there.
- **Reported, not predicted:** meanfield against geometry (the pairing part of the clock), p99 TPOT,
  goodput at the harness's SLO pairs, and the share of multi-prefill steps from the tracer.

## Exclusions

None beyond requests with errors (reported). A cell whose server fails to start is rerun once and
recorded; a cell that fails twice is reported missing and its decisions are dropped.
