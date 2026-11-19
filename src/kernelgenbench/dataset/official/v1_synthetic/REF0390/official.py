# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L351-L374
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L114-L151
from megatron.core.tensor_parallel.mappings import (
    _ReduceScatterToSequenceParallelRegion as official_region,
)

run = official_region.backward
reduce_scatter_forward = official_region.apply
