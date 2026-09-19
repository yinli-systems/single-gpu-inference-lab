# Task 26 — PCIe arbitration: bulk KV movement on rank 0's GPU vs rank 1's EP communication

Status (2026-09-19, round 59): **hog (synthetic line-rate D2H): topology- and placement-gated — same-socket pair with a local buffer ×1.13 (B=32), buffer on the far socket +×1.14–1.16, pair spanning sockets +×1.15, worst corner ×1.48 = round 1 (NUMA round, `move_pages`-verified); SM-issued copies ×1.9–2.1 regardless. Real connector (vLLM CPU offload): H2D loads benign for the peer (×0.97–0.98, and ×0.5 vs recompute); D2H stores leave the peer's decode steps at ×1.01 but rank 0's own 512-token chunk step goes 48 → 86 ms (bimodal) with the connector on → peer p95 ×1.80 at B=32 — cause (connector store path vs cross-socket collective slowdown) pending the DP=1 control (jobs 1603014/1603015). Prior art for the mechanism class: SGLang #34805.**

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

## NUMA round (jobs 1602620 / 1602619, `sbatch_c26numa.sh`, hog buffer bound with `set_mempolicy(MPOL_BIND)` and read back with `move_pages(2)`; `raw/c26numa-16026{20,19}.json`, `raw/waves-16026{20,19}.log`, `raw/gpus-numa-kv.txt`)

Same server config as rounds 1–3; specs `off, d2h:1.0` (unbound, first-touch), `d2h:1.0@L` (buffer on GPU 0's own NUMA node),
`d2h:1.0@S` (another node of GPU 0's socket), `d2h:1.0@X` (a node of the other socket), `h2d:1.0@L`, `h2d:1.0@X`; B ∈ {8, 32},
3 repeats, copy engine, 1.5 GiB bursts, continuous. Every hog start record reads back 256/256 sampled pages on the requested
node (`pages_by_node`), so placement is controlled for the first time. Both nodes confine the cpuset to 12 CPUs.

| pair | node | spec (buffer node) | hog GB/s B=8 / 32 | B=8 rank-1 p50 (ratio) / p95 (ratio) | B=32 rank-1 p50 (ratio) / p95 (ratio) | B=32 step period |
| --- | --- | --- | ---: | --- | --- | --- |
| **same socket** (GPU0 NUMA 2, GPU1 NUMA 1) | naf13g7 (1602620) | off | — | 14.8 / 28.8 | 22.0 / 44.9 | 21.7 |
| same socket | | d2h unbound (landed on N2 = local) | 22.2 / 20.9 | 14.6 (0.98) / 29.1 (1.01) | 24.5 (1.11) / 49.6 (1.10) | 24.4 |
| same socket | | d2h @2 (local) | 22.2 / 20.9 | 14.9 (1.01) / 30.0 (1.04) | 24.8 (1.13) / 50.4 (1.12) | 24.4 |
| same socket | | d2h @3 (same socket, other node) | 21.5 / 20.2 | 14.7 (1.00) / 29.5 (1.02) | 24.9 (1.13) / 50.2 (1.12) | 24.3 |
| same socket | | **d2h @6 (other socket)** | 19.3 / **17.1** | 16.0 (**1.08**) / 32.2 (**1.11**) | 28.5 (**1.29**) / 57.4 (**1.28**) | 27.9 (×1.29) |
| same socket | | h2d @2 / h2d @6 | 22.6 / 22.5 · 21.1 / 21.0 | 15.0 (1.02) / 30.3 (1.05) · 15.0 (1.01) / 30.2 (1.05) | 23.4 (1.06) / 47.4 (1.06) · 23.2 (1.05) / 47.4 (1.06) | 23.0 |
| **cross socket** (GPU0 NUMA 1, GPU1 NUMA 7) | nba06g6 = round-1 node (1602619) | off | — | 14.1 / 28.4 | 22.1 / 44.7 | 21.7 |
| cross socket | | d2h unbound (landed on N5 = GPU 1's socket) | 17.3 / **14.5** | 17.4 (1.24) / 34.7 (1.22) | 32.7 (**1.48**) / 65.9 (**1.47**) | 32.3 (×1.49) |
| cross socket | | d2h @1 (local to GPU 0) | 18.9 / 16.7 | 16.2 (1.15) / 32.7 (1.15) | 28.9 (**1.31**) / 58.3 (**1.30**) | 28.7 (×1.32) |
| cross socket | | d2h @2 (GPU 0's socket, other node) | 19.0 / 16.6 | 16.4 (1.16) / 32.9 (1.16) | 28.8 (1.30) / 58.2 (1.30) | 28.6 |
| cross socket | | **d2h @5 (GPU 1's socket)** | 17.3 / **14.5** | 17.3 (1.23) / 34.8 (1.23) | 32.8 (**1.48**) / 66.2 (**1.48**) | 32.3 (×1.49) |
| cross socket | | h2d @1 / h2d @5 | 21.9 / 21.7 · 20.7 / 20.5 | 15.2 (1.08) / 30.5 (1.07) · 14.9 (1.06) / 29.8 (1.05) | 23.4 (1.06) / 47.4 (1.06) · 23.3 (1.05) / 47.0 (1.05) | 23.0 |

Reading (bounded, 3 repeats each, one node per pair type):

1. **Two additive-looking factors, both host-side.** (a) *Buffer socket*: moving the D2H target buffer from GPU 0's socket to
   the other socket costs the peer an extra ×1.14–1.16 at B=32 on both pair types (same-socket pair 1.13 → 1.29; cross pair
   1.30 → 1.48) and throttles the hog (20.9 → 17.1, 16.7 → 14.5 GB/s) — the DMA then crosses the inter-socket fabric.
   (b) *Pair socket span*: with the buffer local to GPU 0 (the best placement), the cross-socket pair still pays ×1.30 vs ×1.13
   for the same-socket pair, so the EP payload's own host-memory hop across the fabric is what the D2H stream competes with.
   Within a socket the node does not matter (@2 ≈ @3 ≈ unbound).
2. Round 1's ×1.47 is therefore the *worst corner*: cross-socket pair **and** first-touch buffer on the far socket (the
   round-1 node's unbound hog landed on N5 here as well — its cpuset spans nodes 0/1/4). The unbound case is exactly what
   vLLM's CPU offload tier does today (`torch.zeros(pin_memory=True)`, no NUMA policy, `kv_offload/cpu/gpu_worker.py:791`).
3. h2d is ×1.05–1.08 everywhere; buffer placement does not matter for the H2D direction at this rate.
4. Policy reading: NUMA-binding the offload tier to the GPU's socket removes factor (a) (×1.48 → ×1.30 on a cross pair,
   ×1.29 → ×1.13 on a same-socket pair when first touch would have landed far); only pair placement removes factor (b).
   Both matter only when a rank moves bulk KV at multi-GB/s — see the real-mover section for what the connector actually moves.

## Real mover (jobs 1602608 connector ON = `--kv-offloading-size 16` / 1602609 OFF, `sbatch_c26kv.sh` + `measure_kvoffload.py`, `analyze_kvoffload.py`; `raw/c26kv-1602608-1602609.json`, `raw/waves-16026{08,09}.log`)

Both arms landed on **cross-socket pairs** (GPU bus 41 NUMA 1 + bus 81 NUMA 7) but on different nodes (ON 07g6, OFF 06g6);
`plain` baselines agree within 2 % (14.1 vs 13.8 ms at B=8, 22.4 vs 22.2 at B=32). Rank 0 runs the workload (`plain`: B decode
streams; `store`: back-to-back unique 4k-token prompts, 8 output tokens → every full block stored GPU→CPU; `load`: 4k prompts
primed into the CPU tier, then re-requested round robin → CPU→GPU loads instead of recompute); rank 1 always runs B plain
decode streams and moves no KV. 10-s windows, 3 repeats, graph mode, q=512 (chunk steps run eager, `cg_mode NONE`, capture cap 128).

| kind | B | rank-1 ITL p50 OFF → ON (ratio) | rank-1 ITL p95 OFF → ON (ratio) | rank-1 step CUDA p50 on steps where rank 0 ran decode-only, OFF → ON | … where rank 0 ran a 512-token chunk, OFF → ON | rank-0 train: requests / TTFT p50 OFF → ON | KV moved (ON) |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| load (H2D) | 8 | 26.6 → 12.9 (**0.49**) | 91.9 → 26.2 (**0.29**) | 13.3 → 12.9 (0.97) | 45.3 → — (no chunk steps with the connector: full prefix hit) | 15 / 524 ms → 38 / 216 ms | 28.5 GiB, 3.8 GB/s |
| load (H2D) | 32 | 40.3 → 20.5 (**0.51**) | 92.8 → 42.4 (**0.46**) | 20.7 → 20.2 (0.98) | 46.0 → — | 14 / 585 → 23 / 340 ms | 17.2 GiB, 2.3 GB/s |
| plain | 8 | 13.8 → 14.1 (1.02) | 27.7 → 28.3 (1.02) | 13.6 → 13.8 (1.01) | — | — | 0 |
| plain | 32 | 22.2 → 22.4 (1.01) | 45.0 → 45.5 (1.01) | 21.9 → 21.7 (0.99) | — | — | 0 |
| store (D2H) | 8 | 27.9 → 29.9 (1.07) | 94.9 → 105.3 (1.11) | 13.7 → 13.9 (**1.02**) | 50.7 → 53.8 (85.3 in 1 of 3 repeats) | 14 / 602 → 12 / 833 ms | 9.0 GiB, 1.2 GB/s |
| store (D2H) | 32 | 39.5 → 42.5 (1.08) | 98.9 → 178.1 (**1.80**) | 19.8 → 20.0 (**1.01**) | 45.9 → **86.7** (85.9 / 86.4 / 48.3 per repeat) | 13 / 618 → 9 / 903 ms | 6.8 GiB, 0.9 GB/s |

Rank 0's *own* chunk steps (step trace, CUDA-event time): OFF 45–58 ms in every store cell; ON **85–86 ms** in 4 of 6 store
cells (p50 by window third 86.2 / 85.8 / 85.8 — uniform, not bursty), 47.6–48.7 ms in the other two (`store B=8` repeat 0
switched from 85 to 48 ms inside its window; repeat 2 of both B ran entirely at 48). Same 512 tokens, `cg_mode NONE`, 1 request.

Reading (bounded):

1. **The connector's H2D loads are benign for the peer** — decode-only steps ×0.97–0.98 at 2.3–3.8 GB/s of CPU→GPU traffic,
   and because a loaded prompt needs no prefill chunk, rank 1 is *faster* than in `plain` (×0.92). The product comparison
   (load vs recompute) is a 2–3.4× win for the peer's ITL and 2.5× for rank 0's TTFT — the synchronized 512-token chunk
   (G1/24 mechanism) is what the recompute path exports.
2. **The store (D2H) path's cost on the peer is not PCIe arbitration on ordinary steps**: rank 1's CUDA time on lockstep
   steps where rank 0 ran a decode-only step is ×1.01–1.02 with the connector on (0.9–1.6 GB/s average D2H). The entire
   ON−OFF difference (p95 ×1.80 at B=32) sits on the chunk steps, where rank 0's *own* step grows 48 → 86 ms and rank 1 waits.
3. Two readings of the 86-ms chunk step are open and a DP=1 control (jobs 1603014 ON / 1603015 OFF, `sbatch_c26kv1.sh`:
   one engine, no collectives, same trains) discriminates them: **H1** the connector's store path lengthens the chunk step by
   itself (a single-rank cost exported 1:1 by the barrier — a G1-type contagion, not arbitration); **H2** the D2H bursts slow
   the chunk step's EP collectives (512 tokens × 24 layers through host memory across the cross-socket path; the hog data
   gives ×1.3–1.5 for that path at line rate). If H1: the finding is "vLLM's CPU offload store doubles the prefill chunk step"
   (bimodal, to be explained — not `wait_for_save` (a no-op) and not queueing of the sampler's D2H output copy, which runs on a
   separate copy stream after the traced end event; the store copy of the previous chunk is deferred to the next step's start and
   overlaps the next chunk step, so an SM-based memcpy path or GDDR contention is the candidate), an upstream issue with
   a DP amplifier, not a topology problem. If H2: the placement rule of the NUMA round applies to the real connector, but only
   during prefill-heavy store phases (the average store rate, ~1 GB/s, is far below the hog's 15–20 GB/s).
4. Gate reading so far: with the connector on, rank-1 p95 +11 % (B=8) / +80 % (B=32) vs off — passes the +10 % gate on paper,
   but the mechanism behind it must be settled (item 3) before it is counted for task 26; p50 is +7–8 % (engineering band).

## Real mover, step-level re-analysis (round 60, 2026-09-19; `scripts/dp_ep/{chunkseq,modeseries,modescan}.py`; `raw/chunkseq-1602608-dp0.txt`, `raw/modeseries-16026{08,09}.txt`, `raw/modescan-all.txt`)

The 48 → 86 ms chunk step of the connector-ON arm (job 1602608) is **not** a per-copy effect. Reading the rank-0 step
trace step by step (host `t` + `cuda_ms`, cell windows from `kvoffload.json`):

- In the slow store cells **every** eager 512-token step is ~85 ms, including the first chunk of each prompt (78.9 /
  79.5 / 80.3 ms vs 41.4 / 36.9 ms in the fast cells) — a step with no store in flight (the previous prompt's last store,
  96 MiB at the connector's own 10–12.7 GB/s = ~8 ms per store from the `kv_offload_store_*` metrics, finished several
  decode steps earlier). The later chunks add the same +7 ms over the first chunk in both modes (41 → 48, 79 → 86).
- Graph-mode decode steps are identical in both modes (B=32: 20.0 vs 20.1 ms; B=8: 13.8 vs 13.3 ms) → the GPU is not slower.
- The between-step host slack (period − cuda_ms) is 2.9–3.7 ms in slow cells vs 1.4–1.6 ms in fast cells → the worker's
  CPU-only work is slower too.
- The mode is a **time state**: slow 0–53 s, fast 53–91 s, slow 91–170 s, fast 170–189 s, slow 189–262 s, fast 262–311 s
  of the run (`raw/modeseries-1602608.txt`), covering store trains, load-cell priming and the plain-cell priming alike;
  cell 3 (store B=8 r0) flipped slow → fast between two chunks of the same prompt (steps 687 → 688).
- Store cells in the fast state look like the OFF arm: chunk p50 47.6 / 48.7 ms, train 16 / 14 requests, TTFT 521 / 605 ms
  (slow state: 83–86 ms, 9–12 requests, 833–923 ms); the peer's p95 ×1.80 was measured in the slow state.
- Prompts are fresh random tokens from one RNG (no cross-cell repeats), cell 1 ran on an empty tier and cells 15/16 on a
  full one → tier fill / eviction / cache hits are excluded as the differentiator.
- Nothing per step in the connector explains a constant +37 ms: `pre_forward → start_kv_transfers → transfer_async` runs
  inside the tracer window (before `self.model()`), so CPU time there would land in `cuda_ms` of a launch-bound eager
  step, but the first chunk has no transfer to submit; the CPU worker path has no background thread; `cudaHostRegister`
  of the 17.18 GB `/dev/shm` region succeeded (no warning in `server.log`).
- Scan of every results dir with step traces (`raw/modescan-all.txt`, eager `cg NONE` 512-token steps, "slow" = > 65 ms):
  15 runs without the connector (m1/g1/c21/c26/c26numa, ~10 k steps) have slow fractions 0.00–0.05 and p50 44.7–55.6 ms
  (the round-1 node 06g6 sits at 53–56 ms); the OFF arm 1602609 (06g6) 0.13 with 95 flips = scattered single steps plus one
  mixed phase in its first cell (p50 66 ms, `raw/modeseries-1602609.txt`); the ON arm 1602608 (07g6) **0.60 with 10 flips
  = sustained 85-ms runs** on both ranks (dp1 0.59: rank 1's own 32 × 128-token prefills in lockstep).

Reading: a CPU-side slowdown of the worker process (eager steps are launch-bound at ~44–48 ms; ×1.8 slower launching
gives 80–86 ms), present ~60 % of the time in the only run that had the connector, independent of connector activity.
Candidates: worker thread placement (far NUMA node from the GPU, SMT sibling, core shared with a co-located job — Slurm
leaves the cpuset unconfined) or node memory pressure from 2 × 17.18 GB of pinned tmpfs. vLLM 0.29.0 ships `--numa-bind`
(default **off**; numactl `--cpunodebind/--membind` of the GPU worker to its GPU's NUMA node, which also first-touches the
`/dev/shm` offload region locally) — the ON arm ran without it. H1 ("the store path lengthens the chunk step") is
**withdrawn** in its per-copy form; the DP2 ×1.80 p95 is the lockstep export of a rank-0 CPU slow state. Diagnostic job
**1603052** (`sbatch_c26kv3.sh`, 1 GPU, DP=1, one node): arms A connector on / unbound, B connector on / `--numa-bind`,
C connector off, each with `cpusidecar.py` sampling every thread's CPU ticks, last CPU + NUMA node, schedstat run/wait
(run-queue delay = contention), nonvoluntary switches, PSI and meminfo at 0.5 s on the step tracer's monotonic clock.
Pre-registered: A reproduces the slow state in ≥ 1/3 of store cells with either high wait_ms (contention) or a far-node /
other-core placement during slow runs; B removes it if it is placement; C never shows it. If B cures it, the remedy is an
existing flag (evaluation result, not a new mechanism); if A does not reproduce it, the 1602608 state was node-specific
(co-located load) and the real-mover claim reduces to "H2D/D2H at ~1–12 GB/s are benign for the peer (×1.01–1.02)".

## Slow state: the cpuset finding and the placement diagnostic (round 61, 2026-09-19; CPU-only session, 0 free GPUs)

**Correction of a premise.** "Slurm leaves the cpuset unconfined" (rounds 58–60) was wrong: the `taskset -cp $$` line
logged by the NUMA round (`raw/gpus-numa-kv.txt`) shows the job's affinity confined to `--cpus-per-task` hardware
threads allocated as whole cores with their SMT siblings — 1602619 (06g6, GPUs on NUMA 1 and 7): `6-8,36-38,70-72,100-102`
= 3 cores on node 0 + 3 cores on node 4, **none on a GPU's node**; 1602620 (96-thread node, GPUs on NUMA 2 and 1):
`6-8,12-14,54-56,60-62` = 3 cores on node 1 + 3 on node 2 (the GPU nodes, by luck). The site's submit filter caps
`gpu_4090` at 6 CPUs per GPU (`DefCpuPerGPU=6`; a 12-CPU 1-GPU request is rejected), so every DP2 job ran two
EngineCore launch threads, the API server, the client harness, the NCCL SHM proxy threads and (ON arm) the offload
connector's CPU work on **6 cores**, and the 1-GPU jobs (1603014/15/52) run on 3 cores.

Consequences. (1) The candidate mechanism for the sustained ×1.8 eager-step state is now concrete: two busy threads
co-scheduled on SMT siblings of one core inside the cpuset (typical ×1.5–1.8 single-thread slowdown for launch-bound
Python), flipping when the scheduler migrates a thread — the flip pattern, the unchanged graph steps (GPU-bound) and the
inflated host slack all fit; run-queue oversubscription inside the cpuset is the second candidate (the sidecar's
schedstat wait catches that one, a busy sibling it does not — the sidecar now records the sibling map and per-CPU busy
from `/proc/stat`). (2) `--numa-bind` (0.29.0 `utils/numa_utils.py`) **cannot be the remedy inside such a cpuset**:
auto-detection refuses a constrained affinity ("CPU affinity is already constrained… Skipping automatic NUMA binding"
→ `RuntimeError` when `--numa-bind-nodes` is not given), `numactl` must be on PATH (absent on the login node), and
`--cpunodebind=<GPU node>` fails when the cpuset has no CPU on that node → arm B of 1603052 will die at engine start
(the script continues to arm C; A and C remain valid and pick up the upgraded sidecar at run time). (3) Whether the
state is reachable in production depends on the CPU budget per rank — a real deployment knob (Kubernetes CPU limits,
Slurm cpusets), which makes the DP lockstep export of one rank's CPU state a max-over-ranks amplifier.

**Job 1603071** (`sbatch_c26kv4.sh`, 1 GPU, 6 threads = 3 cores, DP=1, numactl-free placement with `pinner.py` /
`cpuplan.py`): arms `A-on-shared` (= 1603052 arm A), `D-on-squeezed` (whole tree + harness on 2 cores),
`B-on-isolated` (EngineCore on its own core, API server on another, harness + sidecar on the third), `C-off-shared`.
Pre-registered: (P1) A or D reproduce the slow state (chunk-step cuda p50 > 65 ms in ≥ 1/5 store cells or ≥ 10 % of
chunk steps) and in slow samples the launch thread's sibling is ≥ 50 % busy or its wait ≥ 10 %; (P2) D's slow
fraction ≥ A's; (P3) B < 5 % slow chunk steps with chunk p50 ≤ A's fast-state p50; (P4) C ≤ 5 %. P1 + P3 = core
sharing inside the cpuset is the mechanism, the remedy a per-rank core budget / pinning (engineering; `--numa-bind`
is not usable here). Analysis: `scripts/dp_ep/analyze_sidecar.py <arm dir>` (validated on a synthetic fixture).

## DP2 production-shaped placement follow-up (round 62, 2026-09-19; CPU-only session, job 1603076 pending)

The 1-GPU diagnostic (1603071) runs one EngineCore on 3 cores; the slow state was observed in the DP2 shape (two
EngineCores + API server + DP coordinator + harness + sidecar + NCCL proxies + connector helpers on 6 cores, job
1602608, N = 1). **Job 1603076** (`sbatch_c26kv5.sh`, 2 GPUs, 12 threads = 6 cores, DP2/EP2 multiport, connector
16 GiB per rank, `--kinds store,plain --batch-sizes 32 --repeats 5`, 1:10 limit) runs four arms in one allocation on
the same node and cpuset, each with the round-61 sidecar:

| arm | connector | placement (`cpuplan.py --dp 2`: ENGINE0/1 = 2 cores each, API = HARN = the remaining 2 cores) |
| --- | --- | --- |
| `A-on-shared` | on | none (= 1602608) |
| `M-on-mainiso` | on | each EngineCore's **main thread** (tid == pid = scheduler + model-launch loop; `UniProcExecutor.execute_model` runs `run_method` in the caller even with `non_block=True`, only the output fetch is deferred) alone on its own core — both SMT threads reserved —, its helper threads (NCCL proxies, ZMQ in/out, connector/copy helpers) on the second core of the pair; API/coordinator/parent + harness/sidecar on the rest (`pinner.py --engine-main-cpus`) |
| `B-on-isolated` | on | each EngineCore (all threads) on its own 2 cores; the rest as above (`pinner.py --engine-cpus "a;b"`, ranks matched by the `VLLM::EngineCore_DP<r>` process title) |
| `C-off-shared` | off | none (= 1602609) |

Pre-registered (chunk step = rank-0 eager step ≥ 256 tokens; slow = cuda p50 > 65 ms in a sustained run,
`modeseries.py`; scored by `scripts/dp_ep/c26kv5_score.py`, validated on 1602608/1602609 where it returns the
round-59 numbers — store p95 ×1.80, chunk p50 ×1.82, slow fraction 0.60 vs 0.13): **(Q1)** A reproduces the slow state
— ≥ 1/5 store cells with chunk p50 > 65 ms or ≥ 10 % slow chunk steps, and rank-1 store-cell p95 ≥ 1.3× **arm C's
store-cell p95** (same workload, connector off, same node); **(Q2)** M < 5 % slow chunk steps and rank-1 store p95
≤ 1.2× C's → the launch thread's core sharing (SMT sibling / run queue) is the mechanism and a per-rank core
reservation the remedy; **(Q3)** B like M → process-level isolation suffices (deployable as a `taskset` per
EngineCore); **(Q4)** C < 5 %. (Correction made before any data existed: the first wording compared store with
*plain* cells inside one arm, but a store cell's rank-1 stream runs against rank 0's prefill train, so that ratio
is the synchronized-chunk cost — ×2.2–3.9 even with the connector off in 1602609 — not the slow state.)
Q1 ∧ ¬Q2 ∧ ¬Q3 → the state is not CPU placement (memory / connector-internal, back to the H1 family); ¬Q1 → node-specific,
N = 1 stands. Analysis per arm: `analyze_sidecar.py <arm> --rank 0 --main-pid <EngineCore_DP0 pid from pinner.txt /
affinity-ready.txt>`, `modeseries.py <arm>`, `analyze_kvoffload.py`. Caveat: `pinner.py` pins after the server is up
(startup and first-touch memory placement unpinned); B and M give an EngineCore's helper threads 2 hardware threads
of one core, so a *slower* M/B than A would itself be informative (intra-process oversubscription, not sibling
sharing of the launch thread). Validated on the login node with a dummy `setproctitle` tree (main → MAIN core,
helpers → REST core per rank; single-list form unchanged for 1603071).

## Caveats (pre-registered)

- The hog is a separate process: it shares the link and root complex like a connector's copies
  but not the worker's copy-engine scheduling; a slowdown of rank 0 itself may come from context
  sharing on GPU 0 (the `idle` and `d2d` controls isolate that) — only the *rank-1* effect with
  `h2d/d2h` > `d2d` counts as PCIe arbitration.
- Graph mode: no EP collective trace (hooks do not run in captured graphs); the collective cost is
  inferred from step CUDA time and ITL.
- ITL p95 in temperature-0 streams is bimodal (≈2×p50, detokenizer chunks) — p50 and the step
  trace are the robust signals; p95 is reported because the gate names it.
