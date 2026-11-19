# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/experimental_attention_variant/dsa.py#L346-L507
from megatron.core.transformer.experimental_attention_variant.dsa import (
    bwd_fused_indexer_loss_naive as run,
)

__all__ = ["run"]
