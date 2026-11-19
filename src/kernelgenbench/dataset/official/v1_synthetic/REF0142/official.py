# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layernorm.py#L1108-L1179
from sglang.srt.layers.layernorm import RMSNorm

run = RMSNorm.forward_with_per_tensor_quant_fusion
