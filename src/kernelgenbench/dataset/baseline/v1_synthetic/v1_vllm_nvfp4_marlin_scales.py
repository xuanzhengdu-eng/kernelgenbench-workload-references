"""vLLM NVFP4 Marlin scale conversion (tuple-returning variant).

Official source, exact code:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py#L58-L122

Same byte-level conversion as the SGLang helper, plus three contract
differences: a caller-supplied power-of-two ``scale_factor`` is honoured (only
values greater than one rescale), ``scale_factor=None`` triggers the upstream
automatic factor so that every non-zero scale satisfies ``scale * 2**7 >= 2``,
values below 2 after the 2**7 multiply are clamped to zero, and the return is
``(scales, scale_factor)``.
"""
import torch

FLOAT4_E2M1_MAX = 6.0


def _compute_scale_factor(marlin_scales, a_dtype=None):
    if a_dtype is not None and a_dtype == torch.half:
        return 1.0
    scaled = marlin_scales.float() * (2**7)
    nonzero = scaled > 0
    if nonzero.any():
        maximum = scaled[nonzero].max()
        if maximum < 448 * (2**7):
            return (448 * (2**7) / maximum).log2().floor().exp2().item()
    return 1.0


def v1_vllm_nvfp4_marlin_scales(marlin_scales, scale_factor=None, a_dtype=None):
    x = marlin_scales.to(torch.half)
    x = x.view(-1, 4)[:, [0, 2, 1, 3]].view(x.size(0), -1)
    if scale_factor is None:
        scale_factor = _compute_scale_factor(x, a_dtype)
    if scale_factor > 1.0:
        x = (x.float() * scale_factor).to(torch.half)
    x = x * (2**7)
    x[x < 2] = 0
    y = (x.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    return y[:, 1::2].contiguous(), scale_factor
