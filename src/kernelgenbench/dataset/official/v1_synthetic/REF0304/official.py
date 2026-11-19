# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/experts.py#L250-L268
from megatron.core.transformer.moe.experts import TEGroupedMLP

run = TEGroupedMLP._apply_bias
