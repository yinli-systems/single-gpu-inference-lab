# Selector v4 safe-autotune qualification

**gpu_4090: HOLD**

- Cross-fit records: 144; selected 113 (78.5%).
- Held-out selected geomean 1.391954x; point worst 1.323446x; block worst 1.273680x; joint-min LCB 1.301486x.
- Whole-policy geomean 1.296300x; worst 1.000000x.
- Requirements: `{"all_execution_modes_selected": true, "both_dtypes_selected": true, "both_layouts_selected": true, "cache_roundtrip": true, "held_out_controls_resolve": false, "numerical_exact": true, "policy_worst_at_least_0_99": true, "selected_block_worst_at_least_0_99": true, "selected_geomean_at_least_1_05": true, "selected_joint_min_lcb_at_least_0_99": true, "selected_nonempty": true, "selected_worst_at_least_0_99": true}`
