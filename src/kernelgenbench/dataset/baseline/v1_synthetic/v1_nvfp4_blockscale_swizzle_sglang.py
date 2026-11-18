"""Pad and permute FP8 block scales into the 128x4 CUTLASS layout (SGLang).

Official source (one entry, one implementation):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/utils.py#L574-L604

Contract: same zero padding to a 128-row / 4-column tile grid and the same
128x4 block interleave as the vLLM entry, but this body ends with
``.contiguous().to(scale.device)`` and therefore preserves the input device
(a CPU input stays on CPU), and it re-asserts the padded tile shape.  Because the
device-placement contract is observably different, this is a separate Item with
its own reference and test rather than an alias of the vLLM entry.
"""
import torch


def v1_nvfp4_blockscale_swizzle_sglang(scale):
    assert scale.dtype == torch.float8_e4m3fn
    scale_ndim = scale.ndim
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
    assert scale.ndim == 3

    B, M, K = scale.shape

    def round_up_multiple(value, multiple):
        return (value + multiple - 1) // multiple * multiple

    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    padded_scale = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded_scale[:B, :M, :K] = scale

    batches, rows, cols = padded_scale.shape
    assert rows % 128 == 0
    assert cols % 4 == 0
    padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
    swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
    swizzled_scale = swizzled_scale.contiguous().to(scale.device)

    return (
        swizzled_scale.reshape(M_padded, K_padded)
        if scale_ndim == 2
        else swizzled_scale.reshape(B, M_padded, K_padded)
    )
