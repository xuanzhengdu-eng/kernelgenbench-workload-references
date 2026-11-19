# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L36-L53
from megatron.core.tensor_parallel.mappings import _split_along_last_dim as run

__all__ = ["run"]
