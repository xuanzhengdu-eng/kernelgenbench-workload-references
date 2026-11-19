# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L671-L813
from megatron.core.transformer.moe.moe_utils import (
    topk_routing_with_score_function as run,
)

__all__ = ["run"]
