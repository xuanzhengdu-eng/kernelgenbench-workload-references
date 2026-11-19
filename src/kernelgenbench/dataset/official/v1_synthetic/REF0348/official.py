# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L553-L730
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L216-L374
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L92-L126
from megatron.core.transformer.multi_latent_attention import MLASelfAttention

run = MLASelfAttention.forward

__all__ = ["MLASelfAttention", "run"]
