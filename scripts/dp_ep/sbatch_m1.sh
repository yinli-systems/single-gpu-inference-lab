#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --job-name=m1-dpep
#SBATCH --output=/data/run01/scxi253/inference/logs/m1-%j.out
# Campaign M1 (a: eager mechanism trace; b: graph-mode serving A/B) on 2x4090, vLLM 0.29.0 DP2/EP2.
# usage: sbatch sbatch_m1.sh <model-dir-name> <mode: eager|graph> <lb: internal|multiport> <q: max-num-batched-tokens>
set -u
W=/data/run01/scxi253/inference
MODEL_NAME=${1:-Qwen1.5-MoE-A2.7B-Chat}; MODE=${2:-eager}; LB=${3:-multiport}; Q=${4:-512}
# venv lives on node-local disk at the same path it was built at (/tmp/scxi253/venv-vllm); untar once per node
mkdir -p /tmp/scxi253
[ -x /tmp/scxi253/venv-vllm/bin/python ] || tar -C /tmp/scxi253 -xf $W/venv-vllm.tar
[ -d /tmp/scxi253/libfix ] || tar -C /tmp/scxi253 -xf $W/libfix.tar
source /tmp/scxi253/venv-vllm/bin/activate
export LD_LIBRARY_PATH=/tmp/scxi253/libfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
R=$W/results/m1-$SLURM_JOB_ID-$MODEL_NAME-$MODE-$LB-q$Q; mkdir -p $R/trace
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# stage model to node-local disk
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $W/models/$MODEL_NAME $M
export VLLM_EXP_ITER_TRACE=$R/trace/iter.jsonl VLLM_EXP_STEP_TRACE=$R/trace/step.jsonl VLLM_EXP_EP_TRACE=$R/trace/ep
[ "$MODE" = graph ] && unset VLLM_EXP_ITER_TRACE VLLM_EXP_STEP_TRACE VLLM_EXP_EP_TRACE
COMMON="--data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens $Q --gpu-memory-utilization 0.88 --no-enable-prefix-caching --port 8300"
[ "$MODE" = eager ] && COMMON="$COMMON --enforce-eager --enable-logging-iteration-details"
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
python measure_dp_ep_waves.py --model $M --ports $PORTS --output $R/waves.json --repeats 3 2>&1 | tee $R/waves.log
curl -s localhost:8300/metrics > $R/metrics-8300.txt; [ "$LB" = multiport ] && curl -s localhost:8301/metrics > $R/metrics-8301.txt
kill $SPID; sleep 5; pkill -u $USER -f "vllm serve" ; echo M1_DONE
