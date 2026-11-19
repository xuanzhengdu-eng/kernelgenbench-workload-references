# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/marlin_utils_fp4.py#L334-L370
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    deinterleave_moe_mxfp4_w13_for_marlin,
)


def v1_sgl_mxfp4_moe_deinterleave_(layer):
    return deinterleave_moe_mxfp4_w13_for_marlin(layer)
