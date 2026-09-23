#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --job-name=c26r3-dpep
#SBATCH --output=/data/run01/scxi253/inference/logs/c26-%j.out
# Task 26 (PCIe arbitration): plain decoders on both DP ranks; a separate process injects bulk H2D/D2H traffic on GPU 0 (rank 0);
# rank 1's ITL / step period is the metric. gpu-memory-utilization 0.85 leaves room for the hog's context + 512 MiB device window.
# round 3 (requester probe: copy-engine vs SM-issued D2H/H2D, same bytes); usage: sbatch sbatch_c26r3.sh <model-dir-name> <mode: eager|graph> <lb: internal|multiport> <q: max-num-batched-tokens> [extra server args] [harness args]
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
# round 2: log the NCCL transport choice (SHM vs P2P) per process without polluting server.log
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET,ENV
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO VLLM_ENGINE_READY_TIMEOUT_S=1800   # 1602457 died at the 600 s default on a cold node
# the 1 GB home quota: keep every cache/config on node-local disk
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/tmp/scxi253/vllm-cache TRITON_CACHE_DIR=/tmp/scxi253/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253/inductor-cache HF_HOME=/tmp/scxi253/hf FLASHINFER_WORKSPACE_BASE=/tmp/scxi253/flashinfer
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME $FLASHINFER_WORKSPACE_BASE
R=$W/results/c26-$SLURM_JOB_ID-$MODEL_NAME-$MODE-$LB-q$Q; mkdir -p $R/trace
export NCCL_DEBUG_FILE=$R/nccl-%h-%p.log
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# stage model to node-local disk
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
# standalone PCIe bandwidth of GPU 0 (no server traffic yet): the reference for the hog's per-burst GB/s inside the cells
cd $W/lab-scripts; for e in ce sm; do for d in h2d d2h d2d; do python pcie_hog.py --dir $d --engine $e --duration 3 --log $R/hog-standalone.jsonl 2>&1 | tail -1; done; done
# per-GPU PCIe rx/tx (MB/s) and utilisation at 1 s from nvidia-smi, for the whole job
nvidia-smi dmon -s ut -d 1 -o T > $R/dmon.log 2>&1 &
DMON=$!
# iter/step tracers are valid in both modes (padded_tokens vs num_tokens per rank is the G1 signal);
# the EP collective tracer runs Python hooks that captured graphs skip, so eager only
export VLLM_EXP_ITER_TRACE=$R/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$R/trace/step.jsonl VLLM_EXP_EP_TRACE=$R/trace/ep
[ "$MODE" = graph ] && unset VLLM_EXP_EP_TRACE
MNS=64; [ "$Q" -lt 64 ] && MNS=$Q   # vLLM requires max-num-batched-tokens >= max-num-seqs
COMMON="--data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --max-model-len 16384 --max-num-seqs $MNS --max-num-batched-tokens $Q $EXTRA --gpu-memory-utilization 0.85 --no-enable-prefix-caching --port 8300"
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
python measure_pcie.py --model $M --ports $PORTS --output $R/pcie.json --hog-dir $R/hog --repeats 3 $HARNESS 2>&1 | tee $R/waves.log
curl -s localhost:8300/metrics > $R/metrics-8300.txt; [ "$LB" = multiport ] && curl -s localhost:8301/metrics > $R/metrics-8301.txt
kill $SPID $DMON; sleep 5; pkill -u $USER -f "vllm serve" ; pkill -u $USER -f pcie_hog.py; echo C26_DONE
