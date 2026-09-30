#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:35:00
#SBATCH --job-name=sgi-upstream-abort
set -euo pipefail
P=${1:?}; BASE=${2:?}; PATCHED=${3:?}; MODEL=/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
R=$P/upstream-runs/$SLURM_JOB_ID;mkdir -p "$R";scontrol show job "$SLURM_JOB_ID" -o > "$R/slurm.txt";nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$R/hardware.csv"
cleanup(){ RC=$?; trap - EXIT; jobs -pr | xargs -r kill 2>/dev/null || true; echo $RC > "$R/exit.txt"; exit $RC; };trap cleanup EXIT
for MODE in original patched; do
 TREE=$BASE;[ "$MODE" = patched ] && TREE=$PATCHED
 PORT=$("$E/bin/python" - <<'PY'
import socket
s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()
PY
)
 O=$R/$MODE;mkdir "$O"
 export PYTHONPATH=$TREE/python:$P/source/upstream
 export SGI_ABORT_LOG=$O/cleanup.jsonl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
 "$E/bin/python" "$P/source/upstream/abort_server_entry.py" --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --attention-backend flashinfer --dtype bfloat16 --mem-fraction-static 0.80 --context-length 8192 --chunked-prefill-size 1024 --max-running-requests 16 --cuda-graph-max-bs-decode 16 --page-size 16 --skip-tokenizer-init --disable-radix-cache --random-seed 42 --enable-deterministic-inference > "$O/server.log" 2>&1 & SERVER=$!
 "$E/bin/python" "$P/source/upstream/abort_probe.py" --url "http://127.0.0.1:$PORT" --out "$O/results"
 kill -TERM "$SERVER" 2>/dev/null || true;wait "$SERVER" || true
 sha256sum "$TREE/python/sglang/srt/managers/tokenizer_manager.py" > "$O/source.sha256"
done
date -u +%FT%TZ > "$R/complete.txt"
