# Prospective SM-boundary crossover diagnostic

Registered after the repaired discovery canary and before these new measurements. It does NOT modify the original 72/72 learning split or select any learned policy from test outcomes.

## Narrow hypothesis
At GQA group size four and CTA query tile 128, a request occupies ceil(q/32) query CTAs per KV head. For unsplit attention, a one-tile packing difference can cross an SM work-wave boundary. This is a hypothesis about the measured FA2 cliff, not a claim that wave quantization is new (FlashAttention-3 already discusses it).

Choose Q in {960,1024,1088,1280,1344,1408,2656,2720,2784}. Let b=32*floor(Q/64), a=b/2-1. A has queries (a,Q-a), B has (b,Q-b), and both have cached depths (8192,8192-(Q-a-b)). W_A-W_B=(a-b)*(k1-k2+a+b-Q)=0 exactly. Total Q, total cached K and the multiset of total KV lengths also agree. Query marginals differ. These are equal-work controls, NOT optimization speedups.

Intervene on auto, split disabled, fixed512 and fixed1024 with identical Q/K/V within each shape. Same FA2 backend, dtype, correctness thresholds and measurement protocol as measure.py. Two heads configurations, three process/seed repetitions on each GPU. Each run has 18 shapes*2 head configurations*4 policies=144 records; six runs have 864 records.

## Predictions locked before data
Primary uses split disabled to exclude automatic split selection as the explanation. On RTX4090 (128 SMs), Q1024 has 132 versus 128 CTAs at Hkv4 and 264 versus 256 at Hkv8; Q1024 is a previously observed positive control, not a new finding. On RTX5090 (170 SMs), Q1344 has 172 versus168 CTAs at Hkv4 and344 versus336 at Hkv8. Q2720 has344 versus340 and688 versus680. Those cross a work-wave boundary on5090 but not4090. Therefore the A/B excess should migrate from4090 at Q1024 to5090 at Q1344/Q2720. This distinguishes a hardware-wave account from simply labeling one shape inherently slow.

Report every Q, both heads, all policies and all repetitions. Quantitative directional screen: for each previously unmeasured target Q/head, require median A/B>1.10 on5090 and a larger ratio than4090. Report neighbor controls rather than excluding discrepancies. Confidence is limited by three process repeats; do not use inner calls as independent observations. Exact latency factors are NOT predicted, because intra-SM overlap, L2 reuse and masking also affect execution.

Fixed small split settings are secondary interventions, testing whether exposing shorter work mitigates cliffs. Their planning and workspace costs remain counted. A positive crossover supports this specific mechanism but does not prove all residual latency or produce a new serving optimization. No full-vLLM claim follows from it.
