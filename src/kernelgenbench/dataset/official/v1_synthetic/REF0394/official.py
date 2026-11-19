# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py#L9-L92
from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
    reduce_scatter_with_fp32_accumulation as run,
)

__all__ = ["run"]
