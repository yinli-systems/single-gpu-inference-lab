H100-Qwen3-4B            primary_geometry_one_to_multi    LLMVisor   13.04 ( +13.03)  M2n  2.10  ratio    6.2x
H100-Qwen3-4B            reverse_geometry_multi_to_one    LLMVisor    6.86 (  +1.41)  M2n  1.61  ratio    4.3x
H100-Qwen3-8B            primary_geometry_one_to_multi    LLMVisor   20.08 ( +20.08)  M2n  2.53  ratio    7.9x
H100-Qwen3-8B            reverse_geometry_multi_to_one    LLMVisor    8.86 (  +3.96)  M2n  2.49  ratio    3.6x
H100-Qwen2.5-1.5B-Instruct primary_geometry_one_to_multi    LLMVisor    5.31 (  +5.29)  M2n  1.68  ratio    3.2x
H100-Qwen2.5-1.5B-Instruct reverse_geometry_multi_to_one    LLMVisor   14.49 ( +10.81)  M2n  2.27  ratio    6.4x
H100-Qwen2.5-7B-Instruct primary_geometry_one_to_multi    LLMVisor   18.57 ( +18.57)  M2n  2.04  ratio    9.1x
H100-Qwen2.5-7B-Instruct reverse_geometry_multi_to_one    LLMVisor    5.94 (  +2.19)  M2n  1.81  ratio    3.3x
L20-Qwen3-4B             primary_geometry_one_to_multi    LLMVisor   15.77 ( +11.53)  M2n  3.07  ratio    5.1x
L20-Qwen3-4B             reverse_geometry_multi_to_one    LLMVisor   73.20 ( -70.76)  M2n  2.52  ratio   29.0x
A100-Qwen3-4B            primary_geometry_one_to_multi    LLMVisor    6.88 (  +3.93)  M2n  1.92  ratio    3.6x
A100-Qwen3-4B            reverse_geometry_multi_to_one    LLMVisor   26.64 ( -23.44)  M2n  2.31  ratio   11.6x
A100-Qwen3-8B            primary_geometry_one_to_multi    LLMVisor    7.07 (  +1.52)  M2n  1.66  ratio    4.3x
A100-Qwen3-8B            reverse_geometry_multi_to_one    LLMVisor   36.57 ( -35.14)  M2n  1.72  ratio   21.3x
A100-Qwen2.5-1.5B-Instruct primary_geometry_one_to_multi    LLMVisor    2.95 (  +1.21)  M2n  2.13  ratio    1.4x
A100-Qwen2.5-1.5B-Instruct reverse_geometry_multi_to_one    LLMVisor    6.50 (  -3.46)  M2n  1.48  ratio    4.4x
A100-Qwen2.5-7B-Instruct primary_geometry_one_to_multi    LLMVisor    5.45 (  +0.25)  M2n  1.23  ratio    4.4x
A100-Qwen2.5-7B-Instruct reverse_geometry_multi_to_one    LLMVisor   24.07 ( -22.46)  M2n  1.60  ratio   15.0x
L20-Qwen2.5-1.5B-Instruct primary_geometry_one_to_multi    LLMVisor    5.76 (  -1.12)  M2n  1.75  ratio    3.3x
L20-Qwen2.5-1.5B-Instruct reverse_geometry_multi_to_one    LLMVisor   21.10 ( -18.79)  M2n  2.16  ratio    9.8x
L20-Qwen2.5-7B-Instruct  primary_geometry_one_to_multi    LLMVisor   18.18 ( +17.96)  M2n  2.31  ratio    7.9x
L20-Qwen2.5-7B-Instruct  reverse_geometry_multi_to_one    LLMVisor   52.35 ( -51.60)  M2n  2.55  ratio   20.5x
L20-vllm029              k4k/16k q256/768                   Vidur key A (20480, 656100) B (20480, 656100) same=True  floor 18.24 ms
L20-vllm029              k4k/16k q128/896                   Vidur key A (20480, 819025) B (20480, 819025) same=True  floor 27.11 ms
L20-vllm029              k4k/16k q384/640                   Vidur key A (20480, 556516) B (20480, 556516) same=True  floor 9.29 ms
L20-vllm029              k8k/24k q256/768                   Vidur key A (32768, 656100) B (32768, 656100) same=True  floor 23.55 ms
L20-vllm029              k0/16k q256/768                    Vidur key A (16384, 656100) B (16384, 656100) same=True  floor 27.44 ms
L20-vllm029              k4k/16k q512/512 (control, dW=0)   Vidur key A (20480, 524176) B (20480, 524176) same=True  floor 0.04 ms
A100-vllm029             k4k/16k q256/768                   Vidur key A (20480, 656100) B (20480, 656100) same=True  floor 6.77 ms
A100-vllm029             k4k/16k q128/896                   Vidur key A (20480, 819025) B (20480, 819025) same=True  floor 13.81 ms
A100-vllm029             k4k/16k q384/640                   Vidur key A (20480, 556516) B (20480, 556516) same=True  floor 3.49 ms
A100-vllm029             k8k/24k q256/768                   Vidur key A (32768, 656100) B (32768, 656100) same=True  floor 9.80 ms
A100-vllm029             k0/16k q256/768                    Vidur key A (16384, 656100) B (16384, 656100) same=True  floor 10.39 ms
A100-vllm029             k4k/16k q512/512 (control, dW=0)   Vidur key A (20480, 524176) B (20480, 524176) same=True  floor 0.00 ms
H100-vllm029-Qwen3-4B    k4k/16k q256/768                   Vidur key A (20480, 656100) B (20480, 656100) same=True  floor 3.74 ms
H100-vllm029-Qwen3-4B    k4k/16k q128/896                   Vidur key A (20480, 819025) B (20480, 819025) same=True  floor 3.82 ms
H100-vllm029-Qwen3-4B    k4k/16k q384/640                   Vidur key A (20480, 556516) B (20480, 556516) same=True  floor 2.65 ms
H100-vllm029-Qwen3-4B    k8k/24k q256/768                   Vidur key A (32768, 656100) B (32768, 656100) same=True  floor 5.08 ms
H100-vllm029-Qwen3-4B    k0/16k q256/768                    Vidur key A (16384, 656100) B (16384, 656100) same=True  floor 3.82 ms
H100-vllm029-Qwen3-4B    k4k/16k q512/512 (control, dW=0)   Vidur key A (20480, 524176) B (20480, 524176) same=True  floor 0.06 ms
H100-vllm029-Qwen3-8B    k4k/16k q256/768                   Vidur key A (20480, 656100) B (20480, 656100) same=True  floor 3.80 ms
H100-vllm029-Qwen3-8B    k4k/16k q128/896                   Vidur key A (20480, 819025) B (20480, 819025) same=True  floor 3.84 ms
H100-vllm029-Qwen3-8B    k4k/16k q384/640                   Vidur key A (20480, 556516) B (20480, 556516) same=True  floor 2.70 ms
H100-vllm029-Qwen3-8B    k8k/24k q256/768                   Vidur key A (32768, 656100) B (32768, 656100) same=True  floor 5.25 ms
H100-vllm029-Qwen3-8B    k0/16k q256/768                    Vidur key A (16384, 656100) B (16384, 656100) same=True  floor 3.77 ms
H100-vllm029-Qwen3-8B    k4k/16k q512/512 (control, dW=0)   Vidur key A (20480, 524176) B (20480, 524176) same=True  floor 0.01 ms
H100-sglang-Qwen3-4B     k4k/16k q256/768                   Vidur key A (20480, 656100) B (20480, 656100) same=True  floor 3.84 ms
H100-sglang-Qwen3-4B     k4k/16k q128/896                   Vidur key A (20480, 819025) B (20480, 819025) same=True  floor 3.89 ms
H100-sglang-Qwen3-4B     k4k/16k q384/640                   Vidur key A (20480, 556516) B (20480, 556516) same=True  floor 2.79 ms
H100-sglang-Qwen3-4B     k8k/24k q256/768                   Vidur key A (32768, 656100) B (32768, 656100) same=True  floor 5.32 ms
H100-sglang-Qwen3-4B     k0/16k q256/768                    Vidur key A (16384, 656100) B (16384, 656100) same=True  floor 3.91 ms
H100-sglang-Qwen3-4B     k4k/16k q512/512 (control, dW=0)   Vidur key A (20480, 524176) B (20480, 524176) same=True  floor 0.02 ms
