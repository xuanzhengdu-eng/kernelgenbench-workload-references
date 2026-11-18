"""SGLang NVFP4 Marlin scale conversion (16-bit activation path).

Official source, exact code:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/marlin_utils_fp4.py#L29-L44

The SGLang helper casts the scales to float16, reorders each group of four
values as ``[0, 2, 1, 3]``, multiplies by 2**7, shifts the raw int16 bit
pattern left by one and keeps every second byte as an ``float8_e4m3fn``
scale.  It has no clamp of small values and no caller-supplied rescaling
factor, and it returns a single tensor of the input shape.

vLLM's same-named helper returns ``(scales, scale_factor)`` and clamps values
below 2 to zero, which is a different contract; that variant is the separate
item ``v1_vllm_nvfp4_marlin_scales``.
"""
import torch


def v1_sgl_nvfp4_marlin_scales(marlin_scales):
    x = marlin_scales.to(torch.half)
    x = x.view(-1, 4)[:, [0, 2, 1, 3]].view(x.size(0), -1)
    x = x * (2**7)
    y = (x.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    return y[:, 1::2].contiguous()
