# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/linear.py#L1749-L1839
from sglang.srt.layers.linear import RowParallelLinear

run = RowParallelLinear.forward
