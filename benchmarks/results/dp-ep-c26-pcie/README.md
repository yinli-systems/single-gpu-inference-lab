# Task 26 — PCIe arbitration: bulk KV movement on rank 0's GPU vs rank 1's EP communication

Status: **job 1602508 running (2026-09-19 09:16 cluster clock)**; analysis pending (round 57).

## Question

On PCIe-only GPUs (2×RTX 4090, driver 580.82, no NVLink, no P2P — NCCL moves the EP
`allgather_reducescatter` tensors through host shared memory, i.e. across the same PCIe links), can
a bulk CPU↔GPU KV transfer on **rank 0's** GPU degrade the decode latency of **rank 1**, which moves
no KV, because every step rendezvous at the per-layer EP collectives?

Pre-registered gate (standing prompt): rank-0 transfer raising rank-1 p95 TPOT by >10 % = strong;
<5 % = kill. Policy work (defer / rate-limit KV movement in an EP-critical window) only if the
oracle exceeds 10 %.

## Design (job 1602508)

- Server: vLLM 0.29.0, Qwen1.5-MoE-A2.7B-Chat, DP2/EP2 (`--data-parallel-size 2
  --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend allgather_reducescatter`),
  graph mode, multi-port external LB (rank r on port 8300+r), `--max-num-batched-tokens 512`,
  `--gpu-memory-utilization 0.85` (0.03 lower than the c21/G1 runs to leave room for the hog's
  context + 512 MiB device window on GPU 0). Step tracer on both ranks.
- Workload: B ∈ {8, 32} plain decode streams (temperature 0, 128-token prompts, `ignore_eos`) on
  **both** ranks; 4 s settle, 8 s measurement window, 3 repeats, shuffled cell order per repeat.
- Injector `pcie_hog.py` (separate process on `cuda:0`, pinned host buffer of 1.5 GiB = 8k tokens
  of Qwen1.5-MoE KV at 192 KiB/token; 64 MiB chunk copies on its own stream; the burst starts
  exactly at the window start via a start file; per-burst `time.monotonic()` log on the node
  clock shared with the step trace):
  | spec | meaning |
  | --- | --- |
  | `off` | no hog (baseline) |
  | `idle` | hog process alive on GPU 0 (context + buffers) but not copying — context-sharing control |
  | `h2d:1.0`, `d2h:1.0`, `both:1.0` | continuous host→GPU0 / GPU0→host / alternating bursts |
  | `h2d:0.5`, `h2d:0.25` | 50 % / 25 % duty (1.5 GiB burst, then a pause) |
  | `h2d:cap4` | bandwidth-capped to ~4 GB/s (16 MiB chunks + sleeps) |
  | `d2d:1.0` | GPU0-internal copies (copy engine busy, **no PCIe traffic**) — separates "GPU 0 is busy" from PCIe arbitration |
- Standalone link bandwidth (h2d/d2h/d2d, 3 s each) is measured before the server starts
  (`hog-standalone.jsonl`); `nvidia-smi dmon -s ut -d 1` records per-GPU PCIe rx/tx MB/s for the
  whole job (`dmon.log`).
- Metrics: both ranks' ITL p50/p95 inside the window (rank 1 is the unaffected client), per-rank
  step period and step CUDA time from `trace/step.jsonl.dp{0,1}`, split into hog-active vs pause
  steps for duty < 1, hog achieved GB/s and active fraction.

Exact commands (git SHA `5694920` for the scripts; run from `/data/run01/scxi253/inference`):

```
sbatch lab-scripts/sbatch_c26.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512      # -> job 1602508
LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_pcie.py \
  --dir results/c26-1602508-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512 --output results/c26-analysis/c26-1602508.json
```

## Prior art (2026-09-19)

GitHub (vLLM issues+PRs: "PCIe contention KV offload", "KV transfer interference NCCL", "offloading
connector latency PCIe", "cpu offload bandwidth contention collective", "PCIe bandwidth interference
expert parallel"; SGLang "PCIe contention KV offload") — no report of KV-offload traffic degrading
collectives or other DP ranks. Adjacent: vLLM #57555 (load-path DMA/Triton thresholds per device),
#44294 (offload loads serialised), #43470 (PD RFC, transfer backlog stalls between pools), #42212
(Triton H2D fast path). arXiv API ("PCIe AND contention AND collective AND inference", "KV AND
transfer AND interference AND serving", "expert AND parallel AND PCIe AND offload"): SwiftCache
(2608.14333), DualPath (2603.13358, storage-bandwidth path), semi-PD (2511.04791) — none study
PCIe arbitration between KV movement and EP collectives. No implementation of a PCIe-window-aware
KV mover found in these sources as of 2026-09-19.

## Results

_pending — see `raw/` after round 57._

## Caveats (pre-registered)

- The hog is a separate process: it shares the link and root complex like a connector's copies
  but not the worker's copy-engine scheduling; a slowdown of rank 0 itself may come from context
  sharing on GPU 0 (the `idle` and `d2d` controls isolate that) — only the *rank-1* effect with
  `h2d/d2h` > `d2d` counts as PCIe arbitration.
- Graph mode: no EP collective trace (hooks do not run in captured graphs); the collective cost is
  inferred from step CUDA time and ITL.
- ITL p95 in temperature-0 streams is bimodal (≈2×p50, detokenizer chunks) — p50 and the step
  trace are the robust signals; p95 is reported because the gate names it.
