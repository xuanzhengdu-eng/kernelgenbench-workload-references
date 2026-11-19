# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/utils.py#L33-L35
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/utils.py#L38-L45
from megatron.core.transformer.utils import (
    get_default_causal_mask,
    get_sliding_window_causal_mask,
)

__all__ = ["get_default_causal_mask", "get_sliding_window_causal_mask"]
