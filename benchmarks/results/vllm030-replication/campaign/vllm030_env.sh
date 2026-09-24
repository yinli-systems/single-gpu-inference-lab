# source this before running vLLM 0.30.0 on the L20
export CUDA_HOME=$HOME/inference/venv-vllm030/lib/python3.12/site-packages/nvidia/cu13
export PATH=$HOME/inference/venv-vllm030/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1
