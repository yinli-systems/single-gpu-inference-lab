#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=01:10:00
#SBATCH --job-name=c26kv5-dp2pin
#SBATCH --output=/data/run01/scxi253/inference/logs/c26kv5-%j.out
# Task 26 'slow state', production-shaped DP2/EP2 follow-up of the 1-GPU placement diagnostic (round 62).
# Job 1602608 (DP2, connector ON, cross-socket pair) showed rank 0's eager 512-token chunk step at 85 vs 48 ms in a
# sustained CPU-side slow state (graph steps unchanged, host slack inflated, flipping mid-prompt) that the barrier exports to
# rank 1 (p95 x1.80 at B=32). Both EngineCores, the API server, the DP coordinator, the harness, the sidecar and every
# NCCL proxy / connector helper thread shared one 6-core Slurm cpuset (12 threads = 6 cores + SMT siblings).
# Four arms in ONE 2-GPU allocation, same node, same cpuset, connector = vLLM native CPU offload (16 GiB per rank):
#   A on-shared    no pinning (= 1602608; N=2 for the slow state in the DP2 shape)
#   M on-mainiso   each EngineCore's MAIN thread (scheduler + model-launch loop, tid == pid) alone on its own core (both SMT
#                  threads reserved for it), its helper threads on the second core of the pair; API/coordinator/harness on the rest
#   B on-isolated  each EngineCore (all threads) on its own 2 cores; API/coordinator/parent + harness/sidecar on the remaining 2
#   C off-shared   connector off, no pinning (= 1602609)
# Pre-registered (chunk step = rank-0 eager step with >= 256 tokens; slow = cuda p50 > 65 ms in a sustained run, modeseries.py):
#   (Q1) A reproduces the slow state: >= 1/5 store cells with chunk p50 > 65 ms or >= 10 % slow chunk steps, and rank-1 p95 (B=32 store)
#        >= 1.3x arm C's store p95 (same workload, connector off; store/plain inside one arm = the synchronized-chunk cost, not the
#        signal).  (Q2) M shows < 5 % slow chunk steps and rank-1 store p95 <= 1.2x C's -> the launch thread's core
#        sharing (SMT sibling / run-queue) is the mechanism; a per-rank core reservation is the remedy.  (Q3) B like M -> process-level
#        isolation suffices (the deployable form: taskset per EngineCore).  (Q4) C < 5 % slow.
#   Q1 true + Q2 false + Q3 false -> the slow state is not CPU placement (memory / connector-internal); Q1 false -> node-specific (N=1 stays).
# usage: sbatch sbatch_c26kv5.sh [model] [q] [OFFLOAD_GIB] [harness args]
set -u
W=/data/run01/scxi253/inference
MODEL_NAME=${1:-Qwen1.5-MoE-A2.7B-Chat}; Q=${2:-512}; OFFLOAD=${3:-16}; HARNESS=${4:---kinds store,plain --batch-sizes 32 --repeats 5}
mkdir -p /tmp/scxi253
NEWV=$(tar -xOf $W/venv-vllm.tar venv-vllm/VENV_VERSION 2>/dev/null || echo v1)
CURV=$(cat /tmp/scxi253/venv-vllm/VENV_VERSION 2>/dev/null || echo none)
if [ "$NEWV" != "$CURV" ]; then rm -rf /tmp/scxi253/venv-vllm; tar -C /tmp/scxi253 -xf $W/venv-vllm.tar; fi
[ -d /tmp/scxi253/libfix ] || tar -C /tmp/scxi253 -xf $W/libfix.tar
source /tmp/scxi253/venv-vllm/bin/activate
export LD_LIBRARY_PATH=/tmp/scxi253/libfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=/tmp/scxi253/venv-vllm/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO VLLM_ENGINE_READY_TIMEOUT_S=1800
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/tmp/scxi253/vllm-cache TRITON_CACHE_DIR=/tmp/scxi253/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253/inductor-cache HF_HOME=/tmp/scxi253/hf FLASHINFER_WORKSPACE_BASE=/tmp/scxi253/flashinfer
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME $FLASHINFER_WORKSPACE_BASE
R=$W/results/c26kv5-$SLURM_JOB_ID-$MODEL_NAME-q$Q; mkdir -p $R
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# topology of the allocated pair (bus id + NUMA node; socket = numa // 4 on the 2-socket NPS4 EPYC nodes) and the cpuset
nvidia-smi topo -m > $R/topo.txt 2>&1; nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader > $R/gpus.txt
for b in $(cut -d, -f2 $R/gpus.txt); do b=${b# }; b=0000:${b#*:}; echo "$b numa $(cat /sys/bus/pci/devices/${b,,}/numa_node 2>/dev/null)"; done | tee -a $R/gpus.txt
echo "step gpus $SLURM_STEP_GPUS job gpus $SLURM_JOB_GPUS cpus $(taskset -cp $$ | cut -d: -f2)" | tee -a $R/gpus.txt
lscpu > $R/lscpu.txt 2>&1; lscpu -e > $R/lscpu-e.txt 2>&1; numactl -H > $R/numactl.txt 2>&1; which numactl >> $R/numactl.txt 2>&1
cat /proc/sys/kernel/numa_balancing /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shmem_enabled > $R/sysctl.txt 2>&1
free -g > $R/free-start.txt; uptime >> $R/free-start.txt; ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $R/ps-start.txt
eval $(python $W/lab-scripts/cpuplan.py --dp 2 | tee $R/cpuplan.txt); cat $R/cpuplan.txt
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
nvidia-smi dmon -s ut -d 1 -o T > $R/dmon.log 2>&1 &
DMON=$!
MNS=64
run_arm() {  # <arm-name> <OFFLOAD_GIB> <none|isolate|mainiso>
  local ARM=$1 OFF=$2 PIN=$3 A=$R/$1 EXTRA="" HTS=""; mkdir -p $A/trace
  [ "$OFF" != 0 ] && EXTRA="--kv-offloading-size $OFF --kv-offloading-backend native"
  [ "$PIN" != none ] && HTS="taskset -c $HARN"
  export VLLM_EXP_ITER_TRACE=$A/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$A/trace/step.jsonl; unset VLLM_EXP_EP_TRACE
  local COMMON="--data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend allgather_reducescatter \
    --max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.85 --port 8300 \
    --enable-logging-iteration-details --data-parallel-multi-port-external-lb --api-server-count 1"
  echo "[$ARM] vllm serve $M $COMMON" | tee $A/server.cmd
  vllm serve $M $COMMON > $A/server.log 2>&1 &
  local SPID=$!
  $HTS python $W/lab-scripts/cpusidecar.py --root-pid $SPID --output $A/cpusidecar.jsonl --interval 0.5 > $A/cpusidecar.log 2>&1 &
  local SIDE=$!
  for i in $(seq 1 240); do
    curl -s localhost:8300/health >/dev/null && curl -s localhost:8301/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "[$ARM] server died"; tail -30 $A/server.log; kill $SIDE 2>/dev/null; return 1; }; sleep 5
  done
  if [ "$PIN" = isolate ]; then python $W/lab-scripts/pinner.py --root-pid $SPID --engine-cpus "$ENGINE0;$ENGINE1" --other-cpus $API | tee $A/pinner.txt; fi
  if [ "$PIN" = mainiso ]; then python $W/lab-scripts/pinner.py --root-pid $SPID --engine-cpus "${ENGINE0_REST};${ENGINE1_REST}" --engine-main-cpus "${ENGINE0_MAIN};${ENGINE1_MAIN}" --other-cpus $API | tee $A/pinner.txt; fi
  for p in $(pgrep -P $SPID) $SPID; do echo "pid $p $(cat /proc/$p/comm) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-40) allowed $(grep Cpus_allowed_list /proc/$p/status | cut -f2)"; done | tee $A/affinity-ready.txt
  # per-thread masks of every EngineCore (main thread first) for the record
  for p in $(pgrep -f "EngineCore_DP"); do for t in $(ls /proc/$p/task 2>/dev/null); do echo "pid $p tid $t $(cat /proc/$p/task/$t/comm) allowed $(grep Cpus_allowed_list /proc/$p/task/$t/status | cut -f2)"; done; done > $A/engine-threads.txt 2>/dev/null
  ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $A/ps-ready.txt; free -g > $A/free-ready.txt
  grep -i "numa\|numactl\|affinity\|mmap file\|cudaHostRegister" $A/server.log | head -20 > $A/server-numa.txt
  (cd $W/lab-scripts && $HTS python measure_kvoffload.py --model $M --ports 8300 8301 --output $A/kvoffload.json $HARNESS 2>&1 | tee $A/waves.log)
  curl -s localhost:8300/metrics > $A/metrics-8300.txt; curl -s localhost:8301/metrics > $A/metrics-8301.txt
  ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $A/ps-end.txt
  kill $SPID; sleep 8; pkill -u $USER -f "vllm serve"; sleep 5; kill $SIDE 2>/dev/null
  rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null
  echo "[$ARM] done $(date +%T)"
}
run_arm A-on-shared $OFFLOAD none
run_arm M-on-mainiso $OFFLOAD mainiso
run_arm B-on-isolated $OFFLOAD isolate
run_arm C-off-shared 0 none
kill $DMON 2>/dev/null
free -g > $R/free-end.txt; ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $R/ps-end.txt
echo C26KV5_DONE
