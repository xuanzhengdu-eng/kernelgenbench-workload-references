# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1226-L1233
from megatron.core.transformer.attention import SelfAttention

run = SelfAttention._apply_output_gate
