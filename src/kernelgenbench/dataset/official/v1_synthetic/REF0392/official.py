# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L276-L293
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L56-L76
from megatron.core.tensor_parallel.mappings import (
    _ScatterToSequenceParallelRegion as official_scatter_region,
)





backward = official_scatter_region.backward
scatter_forward = official_scatter_region.apply
