#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=hogtest
#SBATCH --output=/data/run01/scxi253/inference/logs/hogtest-%j.out
# Standalone check of pcie_hog.py engines (copy engine vs Triton SM kernel) on one idle 4090: bandwidth per direction.
set -u
W=/data/run01/scxi253/inference
mkdir -p /tmp/scxi253
NEWV=$(tar -xOf $W/venv-vllm.tar venv-vllm/VENV_VERSION 2>/dev/null || echo v1)
CURV=$(cat /tmp/scxi253/venv-vllm/VENV_VERSION 2>/dev/null || echo none)
if [ "$NEWV" != "$CURV" ]; then rm -rf /tmp/scxi253/venv-vllm; tar -C /tmp/scxi253 -xf $W/venv-vllm.tar; fi
[ -d /tmp/scxi253/libfix ] || tar -C /tmp/scxi253 -xf $W/libfix.tar
source /tmp/scxi253/venv-vllm/bin/activate
export LD_LIBRARY_PATH=/tmp/scxi253/libfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=/tmp/scxi253/venv-vllm/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config TRITON_CACHE_DIR=/tmp/scxi253/triton-cache HF_HOME=/tmp/scxi253/hf
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $TRITON_CACHE_DIR $HF_HOME
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
LOG=$W/results/hogtest-$SLURM_JOB_ID.jsonl
cd $W/lab-scripts
for e in ce sm; do for d in h2d d2h d2d; do
  echo "== $d / $e"; python pcie_hog.py --dir $d --engine $e --duration 3 --log $LOG 2>&1 | tail -3
done; done
for s in 4 24 48; do echo "== d2h / sm / $s programs"; python pcie_hog.py --dir d2h --engine sm --sm-count $s --duration 2 --log $LOG 2>&1 | tail -2; done
echo "== d2h / sm / cap 8"; python pcie_hog.py --dir d2h --engine sm --rate-gbs 8 --chunk-mb 16 --duration 2 --log $LOG 2>&1 | tail -2
echo HOGTEST_DONE
