# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v4_mhc.py#L479-L546
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/layernorm/hc_combine_norm.py#L55-L80
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/layernorm/mhc.py#L2776-L2797
from sglang.srt.models.deepseek_v4_mhc import combine as run

__all__ = ["run"]
