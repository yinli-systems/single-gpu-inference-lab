#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --job-name=m1-dpep
#SBATCH --output=/data/run01/scxi253/inference/logs/m1-%j.out
# Campaign M1 (a: eager mechanism trace; b: graph-mode serving A/B) on 2x4090, vLLM 0.29.0 DP2/EP2.
# usage: sbatch sbatch_m1.sh <model-dir-name> <mode: eager|graph> <lb: internal|multiport> <q: max-num-batched-tokens> [extra server args] [harness args]
set -u
W=/data/run01/scxi253/inference
MODEL_NAME=${1:-Qwen1.5-MoE-A2.7B-Chat}; MODE=${2:-eager}; LB=${3:-multiport}; Q=${4:-512}; EXTRA=${5:-}; HARNESS=${6:-}
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
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
# the 1 GB home quota: keep every cache/config on node-local disk
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/tmp/scxi253/vllm-cache TRITON_CACHE_DIR=/tmp/scxi253/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253/inductor-cache HF_HOME=/tmp/scxi253/hf FLASHINFER_WORKSPACE_BASE=/tmp/scxi253/flashinfer
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME $FLASHINFER_WORKSPACE_BASE
R=$W/results/m1-$SLURM_JOB_ID-$MODEL_NAME-$MODE-$LB-q$Q; mkdir -p $R/trace
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# stage model to node-local disk
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
# iter/step tracers are valid in both modes (padded_tokens vs num_tokens per rank is the G1 signal);
# the EP collective tracer runs Python hooks that captured graphs skip, so eager only
export VLLM_EXP_ITER_TRACE=$R/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$R/trace/step.jsonl VLLM_EXP_EP_TRACE=$R/trace/ep
[ "$MODE" = graph ] && unset VLLM_EXP_EP_TRACE
MNS=64; [ "$Q" -lt 64 ] && MNS=$Q   # vLLM requires max-num-batched-tokens >= max-num-seqs
COMMON="--data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.88 --no-enable-prefix-caching --port 8300"
COMMON="$COMMON --enable-logging-iteration-details"
[ "$MODE" = eager ] && COMMON="$COMMON --enforce-eager"
[ "$LB" = multiport ] && COMMON="$COMMON --data-parallel-multi-port-external-lb --api-server-count 1"
echo "vllm serve $M $COMMON" | tee $R/server.cmd
vllm serve $M $COMMON > $R/server.log 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  if [ "$LB" = multiport ]; then curl -s localhost:8300/health >/dev/null && curl -s localhost:8301/health >/dev/null && break
  else curl -s localhost:8300/health >/dev/null && break; fi
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $R/server.log; exit 1; }; sleep 5
done
PORTS="8300"; [ "$LB" = multiport ] && PORTS="8300 8301"
cd $W/lab-scripts
python measure_dp_ep_waves.py --model $M --ports $PORTS --output $R/waves.json --repeats 3 $HARNESS 2>&1 | tee $R/waves.log
curl -s localhost:8300/metrics > $R/metrics-8300.txt; [ "$LB" = multiport ] && curl -s localhost:8301/metrics > $R/metrics-8301.txt
kill $SPID; sleep 5; pkill -u $USER -f "vllm serve" ; echo M1_DONE
