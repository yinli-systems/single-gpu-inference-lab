# Selector-v3.1 post-hoc measurement-contract audit

This is an exposed, post-hoc audit of the immutable selector-v3 campaign. It does not change the v3 selector, raw rows, preregistered gate, or HOLD verdict, and it cannot promote v3.2.

## Contract defect

The v3 release gate treated `calls=0 / run_device_us` as eager deployment latency. That field is one device-event interval around the kernel call. The same raw row also contains `cycle_us`, measured as recurring `plan + run + synchronize` wall time; this is the relevant eager full-call boundary. Graph1 and Graph16 correctly use captured replay timing. The v3 `main/repeat` labels are within-arm position controls, not cross-treatment ABBA; pristine/off/cap/guarded were separate processes and binaries.

The corrected deployment identities are therefore:

- eager: `calls=0 / cycle_us`;
- Graph1: `calls=1 / run_device_us`;
- Graph16: `calls=16 / run_device_us`.

This follows FlashInfer Autotuner v2's distinction between recurring eager full-call latency and CUDA Graph replay. No Nsight Compute/CUPTI timing is used.

## Exposed 5090 diagnostic

Across selected deployment cells, guarded geomean is 1.267074x and point worst is 0.988477x. Eager full-call geomean is 1.249633x with the 0.988477x point worst; Graph1 is 1.274778x with worst 1.011840x; Graph16 is 1.276995x with worst 1.011504x. Forced cap has geomean 1.267317x and worst 1.005983x.

The guarded eager point worst is `holdout-v3-tied-00 / BF16 / ragged / auto`. Two of 24 guarded observations are approximately 2262 us and 1757 us while its guarded median is approximately 1493 us, near forced cap. Its controls are unresolved. The result is not evidence of a stable resource-cap regression and must not be repaired by deleting observations; the cell remains unqualified.

## Exposed 4090 diagnostic

Across selected deployment cells, guarded geomean is 1.354862x and point worst is 1.237787x. Eager full-call is 1.328071x with worst 1.237787x; Graph1 is 1.367956x with worst 1.268139x; Graph16 is 1.368962x with worst 1.269265x. Forced cap is similar.

## Decision

V3 remains HOLD. Its numerical and worst-case evidence is useful, but its independent-process design does not establish the requested treatment-paired resolution. All 30 release geometries are exposed development evidence. V3.2 preserves the selector rule while replacing only the measurement contract and uses an independent zero-overlap geometry set.
