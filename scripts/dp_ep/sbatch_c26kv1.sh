#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --job-name=c26kv1-dp1
#SBATCH --output=/data/run01/scxi253/inference/logs/c26kv1-%j.out
# Task 26 real-mover DP=1 CONTROL: one engine, no DP/EP collectives; the same store/load trains + B decode streams run on the single engine.
# Discriminates H1 (the connector's store path lengthens the 512-token chunk step by itself: 48 -> 86 ms at DP2, job 1602608)
# from H2 (D2H store bursts slow the chunk step's EP collectives on a cross-socket pair). Same OFFLOAD_GIB=16 / 0 arms.
# Run twice: OFFLOAD_GIB=16 (connector on, per rank) and 0 (off); prefix caching ON in both (the connector needs block hashes).
# usage: sbatch sbatch_c26kv.sh <model-dir-name> <eager|graph> <internal|multiport> <q> <OFFLOAD_GIB> [extra server args] [harness args]
set -u
W=/data/run01/scxi253/inference
MODEL_NAME=${1:-Qwen1.5-MoE-A2.7B-Chat}; MODE=${2:-graph}; LB=${3:-multiport}; Q=${4:-512}; OFFLOAD=${5:-16}; EXTRA=${6:-}; HARNESS=${7:-}
[ "$OFFLOAD" != 0 ] && EXTRA="$EXTRA --kv-offloading-size $OFFLOAD --kv-offloading-backend native"
# venv lives on node-local disk at the same path it was built at (/tmp/scxi253/venv-vllm); untar once per node
mkdir -p /tmp/scxi253
# refresh the node-local venv whenever the tarball carries a newer VENV_VERSION
NEWV=$(tar -xOf $W/venv-vllm.tar venv-vllm/VENV_VERSION 2>/dev/null || echo v1)
CURV=$(cat /tmp/scxi253/venv-vllm/VENV_VERSION 2>/dev/null || echo none)
if [ "$NEWV" != "$CURV" ]; then rm -rf /tmp/scxi253/venv-vllm; tar -C /tmp/scxi253 -xf $W/venv-vllm.tar; fi
[ -d /tmp/scxi253/libfix ] || tar -C /tmp/scxi253 -xf $W/libfix.tar
source /tmp/scxi253/venv-vllm/bin/activate
export LD_LIBRARY_PATH=/tmp/scxi253/libfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# FlashInfer JIT needs nvcc: use the pip toolchain (nvcc 13.0.88 + crt/cccl/nvvm) exactly as on the L20
export CUDA_HOME=/tmp/scxi253/venv-vllm/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO VLLM_ENGINE_READY_TIMEOUT_S=1800   # 1602457 died at the 600 s default on a cold node
# the 1 GB home quota: keep every cache/config on node-local disk
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/tmp/scxi253/vllm-cache TRITON_CACHE_DIR=/tmp/scxi253/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253/inductor-cache HF_HOME=/tmp/scxi253/hf FLASHINFER_WORKSPACE_BASE=/tmp/scxi253/flashinfer
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME $FLASHINFER_WORKSPACE_BASE
R=$W/results/c26kv1-$SLURM_JOB_ID-$MODEL_NAME-$MODE-$LB-q$Q-off$OFFLOAD; mkdir -p $R/trace
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# topology of the allocated GPUs (bus id + NUMA node; socket = numa // 4 on the 2-socket NPS4 EPYC nodes) so the pair is never unlogged again
nvidia-smi topo -m > $R/topo.txt 2>&1; nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader > $R/gpus.txt
for b in $(cut -d, -f2 $R/gpus.txt); do b=${b# }; b=0000:${b#*:}; echo "$b numa $(cat /sys/bus/pci/devices/${b,,}/numa_node 2>/dev/null)"; done | tee -a $R/gpus.txt
echo "step gpus $SLURM_STEP_GPUS job gpus $SLURM_JOB_GPUS cpus $(taskset -cp $$ | cut -d: -f2)" | tee -a $R/gpus.txt
# stage model to node-local disk
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
# per-GPU PCIe rx/tx (MB/s) and utilisation at 1 s from nvidia-smi, for the whole job
nvidia-smi dmon -s ut -d 1 -o T > $R/dmon.log 2>&1 &
DMON=$!
# iter/step tracers are valid in both modes (padded_tokens vs num_tokens per rank is the G1 signal);
# the EP collective tracer runs Python hooks that captured graphs skip, so eager only
export VLLM_EXP_ITER_TRACE=$R/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$R/trace/step.jsonl VLLM_EXP_EP_TRACE=$R/trace/ep
[ "$MODE" = graph ] && unset VLLM_EXP_EP_TRACE
MNS=64; [ "$Q" -lt 64 ] && MNS=$Q   # vLLM requires max-num-batched-tokens >= max-num-seqs
COMMON="\
  --max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.85 --port 8300"
COMMON="$COMMON --enable-logging-iteration-details"
[ "$MODE" = eager ] && COMMON="$COMMON --enforce-eager"
LB=internal   # single engine, one port; the harness gets the same port twice (its "rank 1" decode streams share the engine with the train)
echo "vllm serve $M $COMMON" | tee $R/server.cmd
vllm serve $M $COMMON > $R/server.log 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  if [ "$LB" = multiport ]; then curl -s localhost:8300/health >/dev/null && curl -s localhost:8301/health >/dev/null && break
  else curl -s localhost:8300/health >/dev/null && break; fi
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $R/server.log; exit 1; }; sleep 5
done
PORTS="8300 8300"
cd $W/lab-scripts
python measure_kvoffload.py --model $M --ports $PORTS --output $R/kvoffload.json --repeats 3 $HARNESS 2>&1 | tee $R/waves.log
curl -s localhost:8300/metrics > $R/metrics-8300.txt; [ "$LB" = multiport ] && curl -s localhost:8301/metrics > $R/metrics-8301.txt
kill $SPID $DMON; sleep 5; pkill -u $USER -f "vllm serve" ; echo C26KV1_DONE
