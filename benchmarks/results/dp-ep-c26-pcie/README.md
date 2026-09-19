# Task 26 — PCIe arbitration: bulk KV movement on rank 0's GPU vs rank 1's EP communication

Status: **round 1 (job 1602508, unlogged GPU pair): d2h ×1.20/×1.47 p50 — NOT reproduced on three same-socket pairs (rounds 2–3 + topology job, 3 repeats: copy-engine d2h ×1.03–1.16, caps/duty ≤ ×1.07); SM-issued copies ×1.9–2.1 (SM contention, both directions); cross-socket pair result below; prior art narrowed by SGLang #34805.**

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

## Results (pooled medians over 3 repeats; ratio vs `off` at the same B; both ranks show identical numbers in every cell = lockstep)

Standalone GPU-0 link: h2d 23.1 GB/s, d2h 24.6 GB/s (PCIe 4.0 x16); d2d 454 GB/s.

| hog on GPU 0 | achieved GB/s | B | rank-1 ITL p50 (ratio) | rank-1 ITL p95 (ratio) | step period / CUDA ms both ranks |
| --- | ---: | ---: | --- | --- | --- |
| idle (context only) | — | 8 / 32 | 14.2 → 14.3 (1.00) / 21.9 → 21.9 (1.00) | 1.11¹ / 1.01 | 14.1 → 14.0 / 21.6 → 21.7 |
| h2d continuous | 20.7 / 21.8 | 8 / 32 | 15.6 (1.09) / 23.5 (1.07) | 1.20 / 1.08 | → 15.0 / 23.1 |
| **d2h continuous** | 17.5 / 14.8 | 8 / 32 | **17.0 (1.20)** / **32.3 (1.47)** | **1.28 / 1.48** | → 17.0 / 32.1 |
| both (alternating) | 20.5 / 20.2 | 8 / 32 | 16.1 (1.13) / 30.0 (1.37) | 1.26 / 1.42 | → 15.8 / 27.5 |
| h2d 50 % duty | 20.7 / 21.8 | 8 / 32 | 14.9 (1.05) / 22.8 (1.04) | 1.15 / 1.06 | in-burst CUDA 14.9 / 22.7 vs pause 14.7 / 22.6 |
| h2d 25 % duty | 20.6 / 20.5 | 8 / 32 | 14.2 (1.00) / 22.5 (1.02) | 1.11 / 1.03 | in-burst 14.3 / 22.5 |
| h2d capped 4 GB/s | 4.0 / 4.0 | 8 / 32 | 14.2 (1.00) / 21.9 (1.00) | 1.01 / 1.01 | → 14.2 / 21.6 |
| d2d control (GPU-0 internal, 206 GB/s) | 206 | 8 / 32 | 27.8 (1.96) / 41.2 (1.88) | 2.04 / 1.93 | → 25.5 / 41.4 |

¹ `off` B=8 p95 is the bimodal 25.9 ms baseline; idle's 1.11 is within that noise (p50 1.00, step period 14.0 vs 14.1).

Reading (bounded):

1. **A bulk GPU→host stream on rank 0's GPU degrades rank 1's decode by 20 % (B=8) to 47 % (B=32),
   p95 likewise (×1.28 / ×1.48)** — above the pre-registered 10 % gate, strong band. The step
   period rises identically on both ranks (lockstep), so rank 1 pays even though it moves no KV.
2. **Direction asymmetry**: host→GPU0 traffic at a *higher* rate (21 GB/s) costs only 7–9 %, while
   GPU0→host at 15–17 GB/s costs 20–47 %, and the d2h hog itself is slowed from 24.6 to 15–17 GB/s
   by the server — the upstream (GPU→host) direction of GPU 0's link is the contended resource.
   Hypothesis to verify: NCCL's SHM transport (no P2P on 4090) has the sender's SMs write the EP
   allgather/reduce-scatter payloads into host memory = upstream on the sender's link; the
   effect grows with B (payload 32 → 128 KB per collective), consistent with a bandwidth-shared path.
3. **Context sharing is not the cause** (`idle` = 1.00) and **GPU memory bandwidth cannot be** (the
   PCIe hogs move ≤ 22 GB/s of GDDR traffic, 2 % of the card); the `d2d` control (206 GB/s
   GPU-internal copies, ×1.9) is a memory-bandwidth-contention arm, not a copy-engine-only arm.
4. **Rate limiting works**: 4 GB/s continuous = 1.00; 50 %/25 % duty at full rate = 1.05 / 1.00
   on the window average, with only +0.2 ms in-burst CUDA time for h2d — i.e. an 8k-token KV
   burst (1.5 GiB) at line rate costs ~5 steps × +1–2 ms for h2d; the d2h duty-cycled cells were
   not run (next round).

Gate outcome: **ALIVE, strong (>20 %) for GPU→host movement; 7–9 % (engineering band) for
host→GPU.** Next: d2h duty/cap sweep (0.5, 0.25, cap 2/4/8/12 GB/s) to find the harmless rate,
NCCL transport confirmation (`NCCL_DEBUG=INFO`), and a real-connector check (vLLM
OffloadingConnector CPU offload on rank 0) before any policy claim.

## Round 2 groundwork (round 57): PCIe counters, the real mover, prior art

**Per-cell PCIe counters of job 1602508** (`nvidia-smi dmon -s ut`, 1 s samples, aligned to the
windows through the `hog/start-NNN` mtimes; `scripts/dp_ep/analyze_dmon.py`, `raw/dmon-1602508.json`;
medians of MB/s over the window, GPU0 rx/tx | GPU1 rx/tx):

| cell | B=8 GPU0 | B=8 GPU1 | B=32 GPU0 | B=32 GPU1 |
| --- | --- | --- | --- | --- |
| off | 6399 / 889 | 6038 / 846 | 5853 / 1064 | 5524 / 1019 |
| idle | 6236 / 882 | 5808 / 825 | 5954 / 1034 | 5431 / 1014 |
| h2d:1.0 | **26513** / 3228 | 6594 / 852 | **27628** / 3546 | 5782 / 1015 |
| d2h:1.0 | 3713 / **20353** | 7565 / 907 | 3431 / **17552** | 8042 / 1072 |
| both:1.0 | 4028 / 15721 | 6784 / 860 | 13890 / 3475 | 7763 / 1081 |
| h2d:cap4 | 10474 / 1344 | 5996 / 832 | 9972 / 1522 | 5568 / 1018 |
| d2d:1.0 | 2039 / 358 | **12344** / 1258 | 1753 / 459 | 6488 / 864 |

Reading: (a) the EP payload is tiny — at B=32 each rank sends 32 × 2048 × 2 B = 128 KiB per
allgather and receives the same per reduce-scatter, × 24 layers ≈ 6 MiB per 21 ms step ≈ 0.3 GB/s —
yet the *plain* baseline shows **~6 GB/s of PCIe rx on both GPUs** with ~0.9 GB/s tx. That is
the receiver-side polling of NCCL's host-memory (SHM) transport: a GPU waiting for its peer spins
on flags in host memory over PCIe, so rx scales with waiting time — it doubles on GPU 1 (12.3 GB/s)
in the `d2d` cell, where GPU 0 is slowed by GDDR contention and GPU 1 waits longer. (b) The
hog saturates one direction of GPU 0's link (rx 26.5–27.6 GB/s for h2d, tx 17.6–20.4 GB/s for
d2h); rank 1's own traffic barely changes, so the harm is not bandwidth starvation of rank 1 but
delayed completion of rank 0's *sends*, which sit on the saturated GPU0→host direction. (c) The
tx in the baseline (~0.9–1.1 GB/s) is the sender side (payload + flags) — the direction the d2h
hog contends with.

**NCCL transport confirmed (job 1602541, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET,ENV`,
`raw/nccl-1602541-rank{0,1}.log`)**: `Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0`;
`Channel 00/01 : 0[0] -> 1[1] via SHM/direct/direct` and `1[1] -> 0[0] via SHM/direct/direct` for
both communicators (busId 41000 and 61000, i.e. the two GPUs hang off different PCIe segments;
`NET/IB` initialised but unused intra-node). "SHM/direct" = the sending GPU's kernel writes the
payload straight into host shared memory over its own PCIe link (GPU0→host, the direction the
d2h hog saturates) and the receiver's kernel reads it from host memory over its link — no
copy-engine proxy in the path, which is why the requester probe (round 3) is meaningful.

**The real mover in vLLM 0.29.0** (`vllm/v1/kv_offload/cpu/gpu_worker.py`, read on the cluster):
`SingleDirectionOffloadingHandler` runs each transfer on its own CUDA stream, serialised after the
previous transfer, unpaced; **GPU→CPU always uses the copy engine** (`ops.swap_blocks_batch` =
`cudaMemcpyBatchAsync`, comment "GPU->CPU is bandwidth-bound; the dedicated copy engine beats
Triton"), CPU→GPU uses the Triton kernel `swap_blocks_triton._swap_blocks_kernel` (SM-issued
loads over the UVA host pointer, 12 programs) for pages < 28 KiB and the copy engine otherwise.
So the harmful direction measured here (d2h, copy engine, line rate, no pacing) is exactly what
a real offload store does; the hog is a faithful mechanism proxy differing only in duty cycle
(a real 4k-token store is one 0.75 GiB burst ≈ 30 ms ≈ 2 steps).

**Requester probe (round 3, job 1602557)**: `pcie_hog.py --engine sm` moves the same bytes with a
Triton kernel over the pinned pointer (validated standalone on an idle 4090, job 1602548: ce and
sm both 26.2–26.3 GB/s in each direction, even with 4 programs; d2d ce 454 vs sm 221 GB/s;
`raw/hogtest-1602548.txt`). If SM-issued D2H writes interfere less than copy-engine writes at the
same rate, the fix is a one-line policy in `_select_swap_blocks_fn` (use the Triton path for
GPU→CPU when the DP/EP transport is host-memory NCCL); if they interfere equally, only pacing
(rate cap) remains.

**Prior art update (2026-09-19, round 57)**: SGLang PR **#34805** "[Diffusion] Avoid H2D/A2A
contention during layerwise offload" (open, opt-in, unmerged, 2026-08-14) — "layerwise offload can
enqueue a full layer's pinned-memory H2D copy while Ulysses `all_to_all_single` is using the same
PCIe fabric. On PCIe-only multi-GPU systems, this overlap slows both the collective and the
denoise step"; remedy = bounded H2D submission + a hard no-H2D window around the NCCL collectives.
That is the same mechanism class (bulk DMA vs a collective on a PCIe-only box) with a windowing
remedy, for weight prefetch in diffusion serving. **Task 26's novelty is therefore narrowed** to:
KV movement in the D2H direction under DP/EP LLM decode, where the collectives are continuous
(48 per 14–21 ms step) so a no-DMA window is not applicable and the remedy must be rate- or
requester-based; and the cross-rank victim (rank 1 moves nothing). vLLM: no issue/PR on offload
traffic degrading collectives or peer ranks ("offload PCIe NCCL interference", "KV offload
collective contention", "swap_blocks_batch copy engine", "cudaMemcpyBatchAsync NCCL", "copy engine
all_reduce latency PCIe", "SHM transport PCIe contention", "offloading connector rate limit";
adjacent #42212, #39306, #52838). SGLang HiCache PRs (#38358 load-back-aware prefill reorder,
#37635/#37701 transfer-kernel block quota) tune transfer throughput, not collective interference.
TensorRT-LLM / Dynamo: no hits. arXiv API ("copy engine" AND collective AND interference; PCIe AND
offloading AND NCCL AND contention; "KV cache" AND offload AND "PCIe bandwidth" AND interference):
none.

## Pre-registered predictions for rounds 2–3 (written 2026-09-19 06:03 Mac, before any cell of jobs 1602541/1602557 had run)

Round 2 (`sbatch_c26r2.sh … --specs off,d2h:1.0,d2h:0.5,d2h:0.25,d2h:cap2,d2h:cap4,d2h:cap8,d2h:cap12,h2d:cap12,d2h:1.0:0.75,d2h:0.5:0.75,both:cap8 --batch-sizes 8,32 --repeats 3 --seed 65`, job 1602541):
1. Duty scales the *window-average* p50 linearly: d2h 0.5 / 0.25 ≈ ×1.24 / ×1.12 at B=32
   (×1.10 / ×1.05 at B=8), but the p95 stays near the continuous value (×1.4) for duty ≥ 0.25
   because in-burst steps exceed 5 % of tokens — i.e. the p95 gate can be passed by realistic
   bursty movers even when the average looks fine.
2. Caps are monotone in rate; d2h cap4 ≈ ×1.00–1.05, cap8 / cap12 intermediate; the harmless
   rate for d2h is lower than for h2d. `h2d:cap12` < `d2h:cap12` (direction asymmetry survives
   at equal rate).
3. 0.75 GiB bursts behave like 1.5 GiB bursts at the same duty (rate, not burst size, matters).

Round 3 (`sbatch_c26r3.sh … --specs off,d2h:1.0,d2h-sm:1.0,d2h:cap8,d2h-sm:cap8,h2d:1.0,h2d-sm:1.0 --batch-sizes 8,32 --repeats 3 --seed 67`, job 1602557): two competing hypotheses, no
strong prior — H_a "requester matters": SM-issued writes share the link fairly with NCCL's
SM-issued writes, so `d2h-sm` ≪ `d2h` (then the one-line Triton-for-D2H policy is the fix);
H_b "bytes and direction matter": `d2h-sm` ≈ `d2h` (then only pacing remains). Decision rule:
H_a if `d2h-sm:1.0` p50 ratio ≤ half of `d2h:1.0`'s excess at both B; H_b if within ±20 % of it.

Real-mover projection (for `sbatch_c26kv.sh`, not yet run): the native connector's store traffic
is bounded by the prefill rate (≈ 12k tok/s × 192 KiB ≈ 2.4 GB/s average, ~4 ms bursts per
512-token chunk) and its loads go the mild h2d direction, so the *realistic* effect on rank 1 is
predicted in the ≤ 10 % band — the ×1.47 headline needs sustained line-rate D2H (PD KV export
staged through host memory, KV snapshots, host-side replication), not this connector on this model.

## Rounds 2–3 results (jobs 1602541 / 1602557, 3 repeats each, same-socket GPU pairs; `raw/c26-1602541.json`, `raw/c26-1602557.json`, `raw/waves-*.log`)

Exact commands (scripts at git SHA `c535f14`, analysis `74bd87d`):

```
sbatch lab-scripts/sbatch_c26r2.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512    # -> 1602541 (node wqd10nba07g3, GPUs bus 41/61 = NUMA 1/0, socket 0)
sbatch lab-scripts/sbatch_c26r3.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512    # -> 1602557 (node wqd10nba07g5, NUMA 2/0, socket 0)
LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_pcie.py --dir results/c26-<job>-… --output results/c26-analysis/c26-<job>.json
```

**Topology facts (established this round).** The 4090 nodes are 8 × RTX 4090, one GPU per PCIe root port, on 2-socket
EPYC 7542/7402 with 8 NUMA nodes (NPS4; socket = NUMA // 4), no PCIe switches, `nvidia-smi topo -m` = `SYS` for every
pair. Slurm does **not** confine the job's CPUs (cpuset = all cores), so every pinned host buffer (NCCL SHM segments, the
hog's buffer, vLLM's offload tier) is first-touched on whatever NUMA node the allocating process happened to run on —
an uncontrolled variable in *all* runs so far (round 1 included). Rounds 2 and 3 got same-socket GPU pairs.

**Round 3 — requester probe** (node 07g5, pooled medians over 3 repeats; rank-1 ITL p50 / p95 with ratio vs `off` at the same B; step period = rank-1 step period p50):

| hog on GPU 0 | achieved GB/s | B=8 p50 (ratio) | B=8 p95 (ratio) | B=32 p50 (ratio) | B=32 p95 (ratio) | r1 step period B=8 / 32 |
| --- | ---: | --- | --- | --- | --- | --- |
| off | — | 14.5 | 28.5 | 21.7 | 44.0 | 13.8 / 21.4 |
| d2h:1.0 copy engine | 19.8 / 19.0 | 14.9 (1.03) | 30.0 (1.05) | 24.5 (**1.13**) | 49.8 (**1.13**) | 14.8 / 24.3 |
| d2h:cap8 copy engine | 8.0 | 14.6 (1.01) | 29.4 (1.03) | 23.0 (1.06) | 46.3 (1.05) | 14.3 / 22.5 |
| h2d:1.0 copy engine | 21.1 / 21.0 | 15.1 (1.04) | 30.4 (1.07) | 23.8 (1.10) | 48.2 (1.09) | 15.0 / 23.3 |
| d2h-sm:1.0 Triton kernel | 9.7 | 27.2 (**1.87**) | 55.9 (**1.96**) | 45.1 (**2.08**) | 92.6 (**2.10**) | 27.2 / 44.4 |
| d2h-sm:cap8 Triton kernel | 5.6 | 20.3 (1.40) | 40.1 (1.41) | 31.6 (1.46) | 64.1 (1.46) | 20.2 / 31.4 |
| h2d-sm:1.0 Triton kernel | 12.4 | 27.2 (1.87) | 55.8 (1.96) | 44.3 (2.04) | 92.0 (2.09) | 27.2 / 44.5 |

**Round 2 — d2h duty / rate-cap sweep** (node 07g3, pooled medians over 3 repeats; rank-1 ITL p50 (ratio) / p95 (ratio)):

| hog on GPU 0 | GB/s | B=8 p50 / p95 | B=32 p50 / p95 |
| --- | ---: | --- | --- |
| off | — | 13.9 / 26.9 | 22.1 / 44.7 |
| d2h:1.0 | 22.1 | 14.9 (1.07) / 29.9 (1.11) | 22.9 (1.04) / 46.2 (1.03) |
| d2h:1.0, 0.75 GiB bursts | 21.4 | 14.5 (1.04) / 28.8 (1.07) | 22.8 (1.03) / 46.0 (1.03) |
| d2h:0.5 | 22.3 | 14.4 (1.03) / 29.0 (1.08) | 22.0 (1.00) / 44.2 (0.99) |
| d2h:0.5, 0.75 GiB bursts | 22.2 | 14.5 (1.04) / 29.3 (1.09) | 23.8 (1.07) / 48.0 (1.07) |
| d2h:0.25 | 21.6 | 14.3 (1.03) / 28.8 (1.07) | 22.4 (1.01) / 45.2 (1.01) |
| d2h cap 2 / 4 / 8 / 12 GB/s | 2 / 4 / 8 / 12 | 1.02 / 1.03 / 1.01 / 1.05 (p95 1.06 / 1.07 / 1.05 / 1.09) | 1.01 / 1.00 / 1.02 / 1.07 (p95 1.02 / 1.00 / 1.02 / 1.06) |
| h2d:cap12 | 12 | 14.5 (1.04) / 29.2 (1.09) | 22.5 (1.02) / 45.4 (1.02) |
| both:cap8 | 8 | 14.1 (1.01) / 28.4 (1.06) | 22.4 (1.01) / 45.7 (1.02) |

Reading (bounded):

1. **On same-socket GPU pairs the copy-engine mechanism is in the engineering band**: line-rate GPU0→host costs the peer
   ×1.03–1.07 (B=8) and ×1.04–1.13 (B=32) over two pairs (plus ×1.07 / ×1.12–1.16 on the third same-socket pair of the
   topology job, below); every duty cycle and rate cap ≤ ×1.07; host→GPU0 ×1.04–1.10. The hog is **not** slowed by the
   server here (19–22 GB/s), whereas in round 1 it fell to 14.8–17.5 GB/s while costing the peer ×1.47 — the two
   symptoms of a shared saturated host path go together, and neither appears on these pairs.
2. **Predictions 1–2 (linear duty scaling, monotone caps) are not supported**: with a ≤ ×1.07 total effect there is
   no rate dependence to resolve (cap 2 ≈ cap 12; duty 0.25 ≈ 1.0 within repeat noise). Prediction 3 (burst size
   irrelevant) holds trivially.
3. **H_a rejected, decisively**: SM-issued copies of the same bytes cost the peer ×1.87–2.10 in *both* directions and
   ×1.40–1.46 even when capped to 5.6 GB/s — the resident Triton copy kernel (12 programs) steals SM/memory bandwidth on
   GPU 0 and lockstep exports the slowdown to rank 1 (the same signature as the `d2d` control). The copy engine is the
   benign requester. Practical reading for vLLM's `swap_blocks_triton` path (SM copies for CPU→GPU pages < 28 KiB): on a
   DP/EP node such a kernel costs the *whole group* ~×2 while resident; the real connector (`sbatch_c26kv.sh`) is the
   required next evidence before any claim, and our 128 KiB pages take the copy engine anyway.
4. The round-1 ×1.47 therefore needs a topology explanation (cross-socket GPU pair and/or cross-socket pinned buffers);
   the topology-controlled job 1602580 is reported next.

## Topology round (job 1602580, `sbatch_c26topo.sh`, node wqd10nba07g4, 6 × 4090: same-socket pair vs cross-socket pair, 3 repeats each; `raw/c26topo-1602580-{same,cross}.json`, `raw/waves-1602580-{same,cross}.log`, `raw/topo-1602580.txt`)

Pairs chosen from the allocation by sysfs NUMA node (`pairs.txt`: `same:0,1:3-2 cross:0,4:3-6`, physical GPUs 0/1/5 = bus
01/25/A1; socket = NUMA // 4): **same** = GPU bus 01 (NUMA 3) + bus 25 (NUMA 2), both socket 0; **cross** = bus 01 (NUMA 3) +
bus A1 (NUMA 6), sockets 0 + 1. `nvidia-smi topo -m` = SYS for both. Same server config as rounds 1–3, specs `off, d2h:1.0,
h2d:1.0`, B ∈ {8, 32}, hog = copy engine, 1.5 GiB bursts, continuous.

| pair (job 1602580, node 07g4) | hog on GPU 0 | achieved GB/s B=8 / 32 | B=8 rank-1 p50 (ratio) / p95 (ratio) | B=8 step period | B=32 rank-1 p50 (ratio) / p95 (ratio) | B=32 step period |
| --- | --- | ---: | --- | --- | --- | --- |
| same socket (NUMA 3 + 2) | off | — | 14.7 / 29.4 | 14.3 | 22.0 / 44.6 | 21.8 |
| same socket | d2h:1.0 | 20.4 / 18.9 | 15.6 (1.06) / 31.0 (1.05) | 15.4 (×1.08) | 25.0 (**1.13**) / 50.4 (**1.13**) | 24.8 (×1.14) |
| same socket | h2d:1.0 | 21.6 / 21.5 | 15.2 (1.03) / 29.4 (1.00) | 15.1 (×1.06) | 23.6 (1.07) / 47.7 (1.07) | 23.3 (×1.07) |
| **cross socket (NUMA 3 + 6)** | off | — | 15.1 / 29.0 | 14.3 | 22.2 / 45.1 | 21.9 |
| cross socket | d2h:1.0 | 17.5 / **14.5** | 17.8 (**1.18**) / 34.1 (**1.18**) | 17.9 (**×1.25**) | 32.2 (**1.45**) / 64.9 (**1.44**) | 32.1 (**×1.47**) |
| cross socket | h2d:1.0 | 21.5 / 20.2 | 15.6 (1.03) / 30.9 (1.07) | 15.6 (×1.09) | 23.9 (1.08) / 47.8 (1.06) | 23.6 (×1.08) |

Pooled medians over 3 repeats (one `off` B=8 repeat of the cross pair shows the known bimodal delivery artifact, p50 28.2; the
3-repeat median absorbs it). Both ranks show identical step periods in every cell (lockstep).


Reading (bounded):

1. **The cross-rank effect is set by the GPU pair's socket topology.** On the *same node*, the same hog on the same GPU
   (bus 01) costs the peer ×1.13 when the peer sits on the same socket and ×1.45 (p95 ×1.44, step period ×1.47) when it sits on the other
   socket — the round-1 numbers (×1.47 p50 / ×1.48 p95 at B=32, ×1.20 at B=8, hog throttled to 14.8 / 17.5 GB/s) are
   reproduced by the cross-socket pair including the hog throttling (14.5 / 17.5 GB/s), so round 1's unlogged pair was
   cross-socket with high confidence.
2. Mechanism reading: with no P2P, NCCL's SHM/direct transport moves every EP payload GPU0 → host memory → GPU1. For a
   cross-socket pair that path crosses the inter-socket fabric and the memory controller of whichever NUMA node holds the
   SHM segment; the D2H stream adds ~15–20 GB/s of writes into host memory on GPU 0's side. That both the hog *and* the
   collectives slow down (the hog loses 25–30 % of its rate only on cross-socket pairs) says the two streams share a
   saturated host-side resource, not merely GPU 0's PCIe link (which is equally loaded on same-socket pairs, where the
   hog keeps 19–22 GB/s and the peer pays ≤ ×1.16). Which resource (fabric vs. memory controller, and where the pinned
   buffers actually live — uncontrolled so far because Slurm leaves the cpuset unrestricted) is the NUMA round's job
   (`sbatch_c26numa.sh`, jobs 1602619 on the round-1 node / 1602620: hog buffer bound to GPU 0's node, another node of
   the same socket, a node of the other socket; placement verified with `move_pages(2)` in the hog's start record).
3. Policy implication (still to be tested with the real connector, jobs 1602608/1602609): a DP/EP pair should not span
   sockets on these PCIe-only boxes when either rank moves bulk KV, or the mover's pinned buffers should be bound to the
   GPU's socket — a placement rule, not a runtime controller.

## Caveats (pre-registered)

- The hog is a separate process: it shares the link and root complex like a connector's copies
  but not the worker's copy-engine scheduling; a slowdown of rank 0 itself may come from context
  sharing on GPU 0 (the `idle` and `d2d` controls isolate that) — only the *rank-1* effect with
  `h2d/d2h` > `d2d` counts as PCIe arbitration.
- Graph mode: no EP collective trace (hooks do not run in captured graphs); the collective cost is
  inferred from step CUDA time and ITL.
- ITL p95 in temperature-0 streams is bimodal (≈2×p50, detokenizer chunks) — p50 and the step
  trace are the robust signals; p95 is reported because the gate names it.
