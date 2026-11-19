# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L765-L882
from megatron.core.transformer.attention import Attention

flash_decode_and_prefill = Attention.flash_decode_and_prefill
