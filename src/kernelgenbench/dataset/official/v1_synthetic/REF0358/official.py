# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/concat_mla.py#L54-L72
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/elementwise/concat_mla.cu#L123-L218
from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q as run

__all__ = ["run"]
