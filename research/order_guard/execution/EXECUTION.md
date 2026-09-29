# GPU execution of frozen order-guard protocol

Algorithm and case source remain at 8aa93527c44dbe67862bf32746b9242e0beebb0f.
No new algorithm parameters or cases are introduced by this execution wrapper.
Canary: one RTX 4090 and one RTX 5090 allocation, separately validated before formal submission.
Formal: four disjoint manifest-index shards per family, three process repeats, 24 one-GPU tasks total.
Parallelism cap: three 4090 tasks, one 5090 task. Partition counts are not summed.
The predeclared primary is locality_packet8 vs identity; causal_heavy remains secondary.
All modes, dtypes, cache conditions and exposed sentinel are retained.
No frozen probe code is changed. Unsupported paths and correctness failures remain HOLD.
Launcher reservations reject duplicate run keys; failed launches are retained rather than reused.
No shared package, driver, GPU clocks or unrelated job is modified.
Telemetry and Slurm allocations are captured. Node sharing remains a potential confound.
Formal jobs require explicit CANARY_PASS and refuse a preexisting campaign HOLD.
The historical 0.6.18 study is not a current-version or production qualification.
