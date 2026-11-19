# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1531-L1578
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1580-L1626
from megatron.core.transformer.attention import SelfAttention

ATTENTION = SelfAttention

run = SelfAttention.clip_qk
clip_linear_qkv = SelfAttention._clip_linear_qkv
