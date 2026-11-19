# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_indices_converter.py#L239-L280
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_indices_converter.py#L186-L237
from megatron.core.fusions.fused_indices_converter import IndicesToMultihot

run = IndicesToMultihot.backward
