# Frozen v4.2 4090 canary: complete, qualification HOLD

All 60 original processes are complete. Five processes which never started because of Slurm RPC errors were resumed on the same original GPU UUID and CPU cores, with the same source and immutable protocol. No completed GPU measurement was repeated; no new fresh case was consumed. All retained original files were hash reverified.

The frozen analyzer reports selected gain 1.537934x, coverage 476/720, actual policy/pristine mean 1.328097x, and native/pristine joint-min 95% LCB 0.992702. Qualification remains HOLD because `held_out_controls_resolve` is false. The independent 5090 HOLD is unchanged. Release 48 and stress 12 remain untouched.

Original RPC errors, UUID/CPU preflight rejections, CPU cache-path discovery failure, and the first analysis failure caused by four late original audit links are preserved. The final CPU audit used explicit read-only views of actual per-repeat binaries; it never relocated or compiled GPU libraries. Telemetry union is lossless and SHA-bound. The frozen source, scoring rules, thresholds, measurements and original terminal receipts are unchanged.

`receipt.json` binds every regular-file archive member. The 33,474,664-byte archive and every member were independently hash checked after download. It includes all 60 raw process directories, raw SASS, recovery preregistration, original/resume/derived receipts, logs, telemetry and frozen full analysis. The numeric regret formula is chosen latency/oracle latency - 1; the inverted frozen descriptive label is retained and annotated.

No default/serving promotion, release authority, historical 2/432 closure, or full HTTP qualification follows from this archive.
