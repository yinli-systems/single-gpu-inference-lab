# Frozen fallback/host-tail diagnosis (not release qualification)

Basis: source a849cf4, campaign selector-v321-native-20260930T100521Z. Preserve both failed canaries and all their observations. The original 30 release geometries remain unexecuted. No selector, threshold, or existing result changes here.

The 5090 cited Graph1 failure has one guarded wall sample1289.334125us while its device event is1196.938038us. Its peer device intervals are comparable. That is an observed host/device-span separation, not proof of a stable fallback-code regression or a proven OS-descheduling cause. Graph replay excludes native plan-time work.

Freeze three exposed BF16 ragged coordinates: below/auto (cited failure, split plan), below/unsplit (fallback with unsplit plan), both/unsplit (positive selected control). Per GPU: three independent processes; same physical GPU and driver within the job. Historical source/overlay identities verified. No profiling, clock locking, cache flushing, changed shapes, or relaxed gates.

For each coordinate capture independent off/guarded runtimes, then off/guarded graphs against the SAME runtime/storage (native flag can differ only on selected control). Add a same-graph null comparison: both labels replay literally the same off graph. Null results are measurement controls, never candidate improvements. Record original pointer identities, native plan vectors, graph topology/grid/block/shared-memory metadata, full output/LSE equality versus saved pristine reference, and post-case memory.

Measure Graph1x16 and Graph16x1 separately, eight ABBA/BAAB blocks. Compare three explicitly diagnostic timers: old per-window events; pre-materialized persistent events; full wall with stream completion and no event instrumentation. Record wall stages, thread CPU time, voluntary/involuntary switches, and device spans when available. Keep every observation. Positive-control host sleeps after GPU completion validate detector sensitivity but are excluded from all performance aggregates.

No diagnostic dataset can overturn the old canary verdict, qualify release, or resolve historical2/432 full-model divergence. Any subsequent measured-path change requires a new version and new canaries. Both correctness and whole-policy release gates stay mandatory.

## Pre-execution amendment

Before any diagnostic GPU submission, the completed4090 canary reports two additional failed FP16/ragged near-opposite coordinates, requested auto andunsplit, both Graph16. Include these two exposed coordinates, for five total; this is not a release-set addition. Record their device and host behavior separately from the cited5090 host-span spike. All other controls/counts stay unchanged.
