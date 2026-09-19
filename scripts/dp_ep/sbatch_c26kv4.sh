#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --job-name=c26kv4-place
#SBATCH --output=/data/run01/scxi253/inference/logs/c26kv4-%j.out
# Task 26 'slow state' placement diagnostic (round 61). Finding behind it: the Slurm cpuset of every lab job is
# --cpus-per-task threads = whole cores with their SMT siblings (12 threads = 6 cores on the DP2 jobs, e.g. 6-8,36-38 +
# siblings 70-72,100-102 on nodes 0/4 while the GPUs sat on nodes 1/7 in 1602619), so the EngineCore launch thread, the
# API server, the client harness, NCCL proxy threads and the offload connector's CPU work all compete for 6 cores and the
# scheduler can co-locate two busy threads on SMT siblings (~x1.5-1.8 single-thread slowdown = the 48 -> 85 ms eager
# chunk step, flipping when the scheduler migrates). vLLM's --numa-bind cannot be the remedy in such a cpuset: its
# auto-detection refuses a constrained affinity and --cpunodebind to the GPU's node fails when the cpuset has no CPU there.
# Four arms in ONE 1-GPU allocation with 6 threads = 3 cores (the site caps gpu_4090 at 6 CPUs per GPU), DP=1, numactl-free placement:
#   A on-shared    connector on, no pinning (6 threads, = 1603052 arm A)   D on-squeezed  connector on, whole tree + harness on 2 cores
#   B on-isolated  connector on, EngineCore on its own core, API server on another, harness+sidecar on the third (pinner.py)
#   C off-shared   connector off, no pinning
# Pre-registered: (P1) A or D reproduce the slow state (chunk-step cuda p50 > 65 ms in >= 1/5 store cells or >= 10 % of
# chunk steps) and in slow samples the launch thread's SMT sibling is >= 50 % busy (cpu_busy) or its wait% >= 10 %;
# (P2) D's slow fraction >= A's; (P3) B shows < 5 % slow chunk steps and chunk p50 <= A's fast-state p50;
# (P4) C <= 5 %. P1+P3 = core sharing inside the cpuset is the mechanism (remedy: a core budget per rank, not --numa-bind).
# usage: sbatch sbatch_c26kv4.sh [model] [q] [OFFLOAD_GIB] [harness args]
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
R=$W/results/c26kv4-$SLURM_JOB_ID-$MODEL_NAME-q$Q; mkdir -p $R
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi topo -m > $R/topo.txt 2>&1; nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader > $R/gpus.txt
for b in $(cut -d, -f2 $R/gpus.txt); do b=${b# }; b=0000:${b#*:}; echo "$b numa $(cat /sys/bus/pci/devices/${b,,}/numa_node 2>/dev/null)"; done | tee -a $R/gpus.txt
echo "step gpus $SLURM_STEP_GPUS job gpus $SLURM_JOB_GPUS cpus $(taskset -cp $$ | cut -d: -f2)" | tee -a $R/gpus.txt
lscpu > $R/lscpu.txt 2>&1; lscpu -e > $R/lscpu-e.txt 2>&1; numactl -H > $R/numactl.txt 2>&1; which numactl >> $R/numactl.txt 2>&1
cat /proc/sys/kernel/numa_balancing /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shmem_enabled > $R/sysctl.txt 2>&1
free -g > $R/free-start.txt; uptime >> $R/free-start.txt; ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $R/ps-start.txt
eval $(python $W/lab-scripts/cpuplan.py | tee $R/cpuplan.txt); cat $R/cpuplan.txt
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
MNS=64
run_arm() {  # <arm-name> <OFFLOAD_GIB> <none|isolate|squeeze>
  local ARM=$1 OFF=$2 PIN=$3 A=$R/$1 EXTRA="" TS="" HTS=""; mkdir -p $A/trace
  [ "$OFF" != 0 ] && EXTRA="--kv-offloading-size $OFF --kv-offloading-backend native"
  [ "$PIN" = squeeze ] && TS="taskset -c $SQUEEZE" && HTS="taskset -c $SQUEEZE"
  [ "$PIN" = isolate ] && HTS="taskset -c $HARN"
  export VLLM_EXP_ITER_TRACE=$A/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$A/trace/step.jsonl; unset VLLM_EXP_EP_TRACE
  local COMMON="--max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.85 --port 8300 --enable-logging-iteration-details"
  echo "[$ARM] $TS vllm serve $M $COMMON" | tee $A/server.cmd
  $TS vllm serve $M $COMMON > $A/server.log 2>&1 &
  local SPID=$!
  $HTS python $W/lab-scripts/cpusidecar.py --root-pid $SPID --output $A/cpusidecar.jsonl --interval 0.5 > $A/cpusidecar.log 2>&1 &
  local SIDE=$!
  for i in $(seq 1 240); do
    curl -s localhost:8300/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "[$ARM] server died"; tail -30 $A/server.log; kill $SIDE 2>/dev/null; return 1; }; sleep 5
  done
  if [ "$PIN" = isolate ]; then python $W/lab-scripts/pinner.py --root-pid $SPID --engine-cpus $ENGINE --other-cpus $API | tee $A/pinner.txt; fi
  for p in $(pgrep -P $SPID) $SPID; do echo "pid $p $(cat /proc/$p/comm) allowed $(grep Cpus_allowed_list /proc/$p/status | cut -f2)"; done | tee $A/affinity-ready.txt
  ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $A/ps-ready.txt; free -g > $A/free-ready.txt
  grep -i "numa\|numactl\|affinity\|mmap file\|cudaHostRegister" $A/server.log | head -20 > $A/server-numa.txt
  (cd $W/lab-scripts && $HTS python measure_kvoffload.py --model $M --ports 8300 8300 --output $A/kvoffload.json $HARNESS 2>&1 | tee $A/waves.log)
  curl -s localhost:8300/metrics > $A/metrics-8300.txt
  ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $A/ps-end.txt
  kill $SPID; sleep 8; pkill -u $USER -f "vllm serve"; sleep 5; kill $SIDE 2>/dev/null
  rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null
  echo "[$ARM] done"
}
run_arm A-on-shared $OFFLOAD none
run_arm D-on-squeezed $OFFLOAD squeeze
run_arm B-on-isolated $OFFLOAD isolate
run_arm C-off-shared 0 none
free -g > $R/free-end.txt; ps -eo pid,user,pcpu,pmem,psr,comm --sort=-pcpu | head -40 > $R/ps-end.txt
echo C26KV4_DONE
