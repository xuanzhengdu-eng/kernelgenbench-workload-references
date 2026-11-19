# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L219-L231
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy as run

__all__ = ["run"]
