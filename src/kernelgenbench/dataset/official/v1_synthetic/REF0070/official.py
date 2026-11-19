# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L147-L188
from megatron.core.fusions.fused_bias_swiglu import SwiGLUFunction

run = SwiGLUFunction.backward
