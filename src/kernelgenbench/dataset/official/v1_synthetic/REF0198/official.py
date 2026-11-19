# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L37-L44
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/dsv4/fp4_rope_fake_quant.py#L117-L170
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/quantization/fp4_fake_quant.py#L16-L62
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4 as run

__all__ = ["run"]
