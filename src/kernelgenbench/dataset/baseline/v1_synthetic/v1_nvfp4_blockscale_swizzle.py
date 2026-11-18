"""Pad and permute FP8 block scales into the 128x4 CUTLASS layout (vLLM).

Official source (one entry, one implementation):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py#L20-L56

Contract: scale is (M, K) or (B, M, K) float8_e4m3fn; M is padded to a multiple
of 128 and K to a multiple of 4 with zeros, then the 128x4 tiles are
block-interleaved.  Rank is preserved.  The pinned body ends with
``.contiguous().cuda()``, so the result always lives on the current CUDA device
even for a CPU input -- that device-placement rule is what distinguishes this
entry from the SGLang one (``v1_nvfp4_blockscale_swizzle_sglang``), which keeps
the input device.
"""
import torch


def v1_nvfp4_blockscale_swizzle(scale):
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape
    M_padded = (M + 127) // 128 * 128
    K_padded = (K + 3) // 4 * 4

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)
