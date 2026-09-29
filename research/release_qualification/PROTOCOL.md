# Release qualification protocol

This campaign freezes a new 48-case family before GPU execution. It compares official FlashInfer 0.7 (`pristine`), a source-overhead control (`off`), unconditional 64-KiB resource cap (`cap`), and a plan-time conservative guard (`guarded`).

The guard is deliberately simple and integer-only: enable the resource cap only when at least one cached prefix is at least 8192 tokens and `max(query_length) / mean(query_length) >= 1.2`. Unknown or ineligible paths remain native. Descriptor order and device mathematical code are unchanged.

Primary performance endpoint: CUDA-graph 16-call `run_device_us`, aggregated across every frozen release cell. Primary safety endpoints: worst point ratio, count below 0.99, output/LSE bit parity, FP32-reference checks, and declared plan-decision conformance. Plan-plus-run cycle cost is reported separately.

Six independent shards are intended on RTX 4090, with three fresh process repeats and balanced mode order. No release case may be edited after the first canary submission. Existing development and confirmatory families are exposed and are not reused as fresh evidence.

Promotion requires: complete matrix, exact resource-only outputs, all guard decisions matching the manifest, guarded worst ratio at least 0.99 on resolved cells, positive aggregate net benefit, and no hidden failed jobs. Results remain architecture- and workload-conditional.
