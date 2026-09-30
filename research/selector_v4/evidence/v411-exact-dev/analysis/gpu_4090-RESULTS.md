# Selector v4 safe-autotune qualification

**gpu_4090: HOLD**

- Cross-fit records: 144; selected 131 (91.0%).
- Held-out selected geomean 1.387585x; point worst 1.320191x; block worst 1.286007x; joint-min LCB 1.315340x.
- Whole-policy geomean 1.347152x; worst 1.000000x.
- Candidate-native / pristine geomean 0.998721x; worst 0.995879x; joint-min LCB 0.992703x.
- Requirements: `{"all_execution_modes_selected": true, "both_dtypes_selected": true, "both_layouts_selected": true, "cache_roundtrip": true, "compiled_kernel_symbol_isolation": true, "held_out_controls_resolve": false, "native_after_cap_exact": true, "native_overlay_controls_resolve_one_percent": true, "native_overlay_joint_min_lcb_at_least_0_99": true, "native_overlay_worst_at_least_0_99": true, "numerical_exact": true, "policy_worst_at_least_0_99": true, "selected_block_worst_at_least_0_99": true, "selected_geomean_at_least_1_05": true, "selected_joint_min_lcb_at_least_0_99": true, "selected_nonempty": true, "selected_worst_at_least_0_99": true, "telemetry_stable": true}`
