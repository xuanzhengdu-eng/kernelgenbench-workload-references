# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L154-L176
from megatron.core.fusions.fused_softmax import SoftmaxOne

run = SoftmaxOne.forward
