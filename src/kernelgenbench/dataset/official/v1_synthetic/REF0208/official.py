# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/mrope.py#L255-L268
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/mrope.py#L270-L291
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/rotary_triton.py#L12-L153
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

run = MRotaryEmbedding.forward_cuda

__all__ = ["run"]
