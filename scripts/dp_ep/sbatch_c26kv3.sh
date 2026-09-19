#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --job-name=c26kv3-mode
#SBATCH --output=/data/run01/scxi253/inference/logs/c26kv3-%j.out
# Task 26 'slow mode' diagnostic (round 60): with the CPU-offload connector on (job 1602608) every eager 512-token chunk
# step ran at ~85 ms instead of ~48 ms in sustained runs (0-53 s, 91-170 s, 189-262 s of the run) regardless of connector
# activity (first chunks, load-priming, plain-cell priming), flipping mid-prompt, with graph-mode decode steps unchanged
# and the between-step CPU slack also inflated (3.7 vs 1.4 ms) -> a CPU-side worker slowdown, never seen in 15 runs without
# the connector. Three arms in ONE 1-GPU allocation (same node, same neighbours), DP=1, each with a CPU sidecar sampling
# every thread's CPU ticks, last CPU + NUMA node, schedstat run/wait (run-queue delay), nonvoluntary switches, PSI, meminfo:
#   A  connector on, default (unbound)        B  connector on + --numa-bind (0.29.0: numactl cpunodebind+membind of the worker,
#   which also first-touches the /dev/shm offload region locally)          C  connector off, default
# Predictions (pre-registered): slow mode reproduces in A (chunk-step cuda p50 > 65 ms in >= 1/3 of store cells) and the
# sidecar shows either high wait_ms (contention) or a far-NUMA/other-core placement during slow runs; B removes it if it is
# placement; C never shows it. usage: sbatch sbatch_c26kv3.sh [model] [q] [OFFLOAD_GIB] [harness args]
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
R=$W/results/c26kv3-$SLURM_JOB_ID-$MODEL_NAME-q$Q; mkdir -p $R
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi topo -m > $R/topo.txt 2>&1; nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader > $R/gpus.txt
for b in $(cut -d, -f2 $R/gpus.txt); do b=${b# }; b=0000:${b#*:}; echo "$b numa $(cat /sys/bus/pci/devices/${b,,}/numa_node 2>/dev/null)"; done | tee -a $R/gpus.txt
echo "step gpus $SLURM_STEP_GPUS job gpus $SLURM_JOB_GPUS cpus $(taskset -cp $$ | cut -d: -f2)" | tee -a $R/gpus.txt
lscpu > $R/lscpu.txt 2>&1; numactl -H > $R/numactl.txt 2>&1; which numactl >> $R/numactl.txt 2>&1
cat /proc/sys/kernel/numa_balancing /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shmem_enabled > $R/sysctl.txt 2>&1
free -g > $R/free-start.txt; uptime >> $R/free-start.txt; ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -25 > $R/ps-start.txt
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
MNS=64
run_arm() {  # <arm-name> <OFFLOAD_GIB> <extra server args>
  local ARM=$1 OFF=$2 EXTRA=$3 A=$R/$1; mkdir -p $A/trace
  [ "$OFF" != 0 ] && EXTRA="$EXTRA --kv-offloading-size $OFF --kv-offloading-backend native"
  export VLLM_EXP_ITER_TRACE=$A/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$A/trace/step.jsonl; unset VLLM_EXP_EP_TRACE
  local COMMON="--max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.85 --port 8300 --enable-logging-iteration-details"
  echo "[$ARM] vllm serve $M $COMMON" | tee $A/server.cmd
  vllm serve $M $COMMON > $A/server.log 2>&1 &
  local SPID=$!
  python $W/lab-scripts/cpusidecar.py --root-pid $SPID --output $A/cpusidecar.jsonl --interval 0.5 > $A/cpusidecar.log 2>&1 &
  local SIDE=$!
  for i in $(seq 1 240); do
    curl -s localhost:8300/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "[$ARM] server died"; tail -30 $A/server.log; kill $SIDE 2>/dev/null; return 1; }; sleep 5
  done
  ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -25 > $A/ps-ready.txt; free -g > $A/free-ready.txt
  grep -i "numa\|numactl\|affinity\|mmap file\|cudaHostRegister" $A/server.log | head -20 > $A/server-numa.txt
  (cd $W/lab-scripts && python measure_kvoffload.py --model $M --ports 8300 8300 --output $A/kvoffload.json $HARNESS 2>&1 | tee $A/waves.log)
  curl -s localhost:8300/metrics > $A/metrics-8300.txt
  kill $SPID; sleep 8; pkill -u $USER -f "vllm serve"; sleep 5; kill $SIDE 2>/dev/null
  rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null
  echo "[$ARM] done"
}
run_arm A-on-unbound $OFFLOAD ""
run_arm B-on-numabind $OFFLOAD "--numa-bind"
run_arm C-off-unbound 0 ""
free -g > $R/free-end.txt; ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -25 > $R/ps-end.txt
echo C26KV3_DONE
