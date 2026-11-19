# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/activation.py#L240-L242
from sglang.srt.layers.activation import GeluAndMul

run = GeluAndMul().forward_native
