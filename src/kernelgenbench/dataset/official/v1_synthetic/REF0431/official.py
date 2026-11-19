# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L1072-L1286
from megatron.core.tensor_parallel.layers import RowParallelLinear

run = RowParallelLinear.forward
