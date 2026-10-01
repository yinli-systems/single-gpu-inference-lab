# Actual serving-layout kernel profile

Source75544a17 normalwheel; isolated jobs1647801/1647802 completed0:0. Previously exposed BF16 query lengths only. No deployment confidence/selection/HTTP or release authority.

| GPU | Operator | Native CUPTI median us | Resource CUPTI median us | Native/resource |
|---|---|---:|---:|---:|
| gpu_4090 | ragged_projection_suffix | 50.7045 | 53.6800 | 0.944570 |
| gpu_4090 | paged1_cached_prefix | 1990.4305 | 1424.0750 | 1.397701 |
| gpu_5090 | ragged_projection_suffix | 38.1600 | 40.0795 | 0.952108 |
| gpu_5090 | paged1_cached_prefix | 1780.4580 | 1305.1365 | 1.364193 |

Each profile has16 actual launches/arm. Dynamic shared-memory launch arguments are49152 native and65536 resource, distinct symbols, and matching registers/thread peroperator/card. Every numerical comparison includes fulloutput/LSE and native-after-resource. Earlier normal75544 dual176-pairs/card disassembly evidence remains a separate source-bound proof; these profile jobs do not claim a new complete binary audit.

All64 balanced event windows/operator/card remain in the archive: warm/cold L2,8ABBA/BAAB blocks/state,64 observations/window,4096 event observations/operator/card. FlashInfer preferred cupti-python was unavailable; its documented CUDA-event fallback includes host launch gaps. Torch Kineto/CUPTI kernel durations are a separate profiled16-launch diagnostic. Profiled times never enter qualification orHTTP scores. Ragged suffix shows no kernel gain and about2x warm-event overhead; retain native. The long cached-prefix witness has about1.37–1.40x event gain. These are exposed single-process observations, not confidence bounds or end-to-end gains.

Both traces report102400bytes shared memory/SM,101376opt-in bytes/block and49152default bytes/block. Kineto static occupancy estimates report0active blocks/SM for the resource function despite its successful16 actual launches. Preserve the raw estimator result and do not treat it as measured residency. The memory-capacity-only upper bounds are2native and1resource CTA/SM; actual achieved residency/cache/issue behavior requires hardware counters. No Nsight Compute or hardware residency counter data was collected.

Rawarchive48members and archiveSHA256 independently verified locally. Source/helper hashes, compiler envelope, normalwheel binding, all rawtrace events and complete event samples remain in raw-kernel-profile.tar.gz.

Official interpretation references: [NVIDIA CUDA Graph execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html). These explain tool/API semantics, not empirical performance claims.
