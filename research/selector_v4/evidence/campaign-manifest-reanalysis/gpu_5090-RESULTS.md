# Selector v4 safe-autotune qualification

**gpu_5090: HOLD**

- Cross-fit records: 144; selected 106 (73.6%).
- Held-out selected geomean 1.396594x; point worst 1.303738x; block worst 1.291951x; joint-min LCB 1.301493x.
- Whole-policy geomean 1.278756x; worst 1.000000x.
- Requirements: `{"all_execution_modes_selected": true, "both_dtypes_selected": true, "both_layouts_selected": true, "cache_roundtrip": true, "held_out_controls_resolve": false, "numerical_exact": true, "policy_worst_at_least_0_99": true, "selected_block_worst_at_least_0_99": true, "selected_geomean_at_least_1_05": true, "selected_joint_min_lcb_at_least_0_99": true, "selected_nonempty": true, "selected_worst_at_least_0_99": true}`
