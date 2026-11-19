# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/fused_a2a.py#L131-L153
# https://github.com/deepseek-ai/DeepEP/blob/34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72/deep_ep/buffer.py
from megatron.core.transformer.moe.fused_a2a import FusedDispatch

run = FusedDispatch.backward
