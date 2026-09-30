# Selector v4 safe-autotune qualification

**gpu_5090: HOLD**

- Cross-fit records: 144; selected 108 (75.0%).
- Held-out selected geomean 1.400026x; point worst 1.307094x; block worst 1.295103x; joint-min LCB 1.305203x.
- Whole-policy geomean 1.287069x; worst 1.000000x.
- Candidate-native / pristine geomean 0.999170x; worst 0.994911x; joint-min LCB 0.988980x.
- Requirements: `{"all_execution_modes_selected": true, "both_dtypes_selected": true, "both_layouts_selected": true, "cache_roundtrip": true, "compiled_kernel_symbol_isolation": true, "held_out_controls_resolve": true, "native_after_cap_exact": true, "native_overlay_controls_resolve_one_percent": true, "native_overlay_joint_min_lcb_at_least_0_99": false, "native_overlay_worst_at_least_0_99": true, "numerical_exact": true, "policy_worst_at_least_0_99": true, "selected_block_worst_at_least_0_99": true, "selected_geomean_at_least_1_05": true, "selected_joint_min_lcb_at_least_0_99": true, "selected_nonempty": true, "selected_worst_at_least_0_99": true, "telemetry_stable": true}`
