# Frozen-policy native implementation validation

Registered before the native GPU validation. The first held-out analysis has already been opened and is retained unchanged. This is a fresh execution measurement of identical frozen policies, NOT a new task-generalization test and NOT test-set retraining.

The initial Python selector made decisions too slowly to justify deployment. Lower the exact selected causal model into C++/CPython, preserving its feature values, predictions and policy decisions. All weights, hyperparameters and the selected fixed policies remain those of model_selection SHA256 d41c87f9e2f1c8eeac8ab411f63925e0eb2781f0229f8d7974ae6b6463a81f1c, committed before test evaluation at 2a5f3c6aa6081f5e344d6b5d6764a13690b2b1b9.

Before GPU validation, verify feature and score equivalence on all 152 registered geometries plus 100 timing-free random input contracts. Compare all four causal models. Require zero changed policy decisions. Generated model tables and shared objects are local derived artifacts with hashes; never import untrusted pickle files.

Measure four arms: stock auto, the best fixed policy selected separately for hardware/head on TRAIN cycle cost, native GPU-cost selector (secondary), native cycle-cost selector (PRIMARY). A native choice is recomputed inside every timed interval; there is no selector cache. Each interval includes selection, the public FA2 planner, actual GPU execution and synchronization. Do not retrospectively swap the primary selector to whichever is faster.

Use all 72 held-out geometries and both head configurations, three independent process/seed repetitions on each GPU, eight paired randomized forward/reverse blocks per cell. Use new tensor seeds 20261001+replicate. Canary covers only two discovery shapes and one training shape. Numerical checks use the existing independent selected FP32 vectors and full-output tolerance comparison to stock. Workspace is the same maximum reserved allocation for every arm. Report its cost and do not infer unchanged full-serving KV capacity.

Summarize median process repetitions per geometry/head and paired family-cluster intervals over the six held-out geometry families. Report all >5% regressions. Primary promotion requires lower 95% interval bound >1 against the head-conditioned fixed policy after real selection overhead. No universal cross-hardware gain without both cards passing. Report each card separately and retain all failed gates.

These are full *operator planning cycles*, not full LLM requests: they omit projections, MLPs, layer orchestration, queues, model outputs and application networking. If no candidate passes the registered deployment gate, do not expend large serving campaigns on it. If a scoped candidate passes, qualify the existing vLLM runtime/backend before any separate matched live-serving claim. TTFT/TPOT/goodput require independently measured end-to-end outputs.
