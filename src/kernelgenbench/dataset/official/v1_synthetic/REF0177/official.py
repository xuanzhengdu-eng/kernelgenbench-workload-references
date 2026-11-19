# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/unquant.py#L793-L839
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

run = UnquantizedLinearMethod().apply

__all__ = ["run"]
