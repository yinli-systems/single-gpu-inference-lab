# Multi-GPU opportunity ledger

Standard: a line is worth starting only if, as of the date noted, no implementation of the same
mechanism was found in vLLM (issues + PRs), SGLang, TensorRT-LLM, Dynamo, or arXiv/MLSys/OSDI/
SOSP/EuroSys/ASPLOS/NSDI 2024–2026. "Nobody has done this" is never claimable.

| id | mechanism | novelty check (date, sources, what was found) | status | gate |
| --- | --- | --- | --- | --- |
| G1–G4 | Cross-rank DP padding amplification under synchronized CUDA-graph mode; padding-aware graph admission (group-wide graph+padded vs eager+unpadded) | 2026-09-19: vLLM `dp_utils.py` pads all ranks to max when synced cudagraph_mode≠NONE (verified); GLM-5.2 speculative-padding post makes mixed decode uniform to keep graphs — does not choose eager when padding exceeds graph savings; #44806 (6× mixed-step slowdown) open without fix. No graph-vs-padding controller found. | flagship, not started | oracle <5% kill / >10% alive / >20% strong |
| P1 | Generic low-precision (FP8 activation) communication for `allgather_reducescatter` EP on PCIe | 2026-09-19: DeepEP/NIXL/FlashInfer have specialized FP8 transports; no generic path found for the allgather backend. Confidence medium. | upper bound first | bound >15% to implement |
| C1 | Same-node DP metadata coordination via shared memory instead of a Gloo/NCCL all-reduce | 2026-09-19: no PR found. Engineering-leaning. | measure first | ms/step material at small batch |
| R1 | Phase/shape-aware DP routing | 2026-09-19: BalanceRoute (arXiv:2605.06113) studies online routing for barrier-synchronized DP. Downgraded. | comparison only | — |
| — | Adaptive TP (Nitsum), adaptive CP (Vertumnus), DBO unanimity (needs DeepEP) | occupied / not feasible on 4090 | not pursued | — |
