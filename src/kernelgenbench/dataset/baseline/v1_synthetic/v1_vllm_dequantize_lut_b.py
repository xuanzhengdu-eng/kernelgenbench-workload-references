"""Reconstruct LUT-B quantized weights from indices, codebooks, and scales.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/lut_b_utils.py#L203-L258

``packed`` is ``[N/8, K/64, 192]`` or ``[E, N/8, K/64, 192]`` uint8; the
codebook has eight E4M3 entries per tile and ``scales`` is one UE8M0 byte per
tile.  The expert dimension is squeezed for 3D packed input.  The output is a
fresh ``[N, K]`` (or ``[E, N, K]``) tensor in ``out_dtype``.
"""
import torch


def v1_vllm_dequantize_lut_b(packed, codebooks, scales, out_dtype=torch.bfloat16):
    squeeze = packed.ndim == 3
    if squeeze:
        packed = packed.unsqueeze(0)
        codebooks = codebooks.unsqueeze(0)
        scales = scales.unsqueeze(0)
    if packed.ndim != 4 or packed.shape[-1] != 192:
        raise ValueError("invalid packed shape")
    if codebooks.shape != (*packed.shape[:3], 8):
        raise ValueError("invalid LUT-B codebook shape")
    if scales.shape != packed.shape[:3]:
        raise ValueError("invalid LUT-B scale shape")
    e, nt, kt = packed.shape[:3]
    groups = packed.reshape(e, nt, kt, 64, 3).to(torch.int32)
    bits = groups[..., 0] | groups[..., 1] << 8 | groups[..., 2] << 16
    shifts = torch.arange(8, device=packed.device) * 3
    ids = ((bits[..., None] >> shifts) & 7).reshape(e, nt, kt, 512).long()
    books = codebooks.float() * torch.exp2(scales.float() - 127).unsqueeze(-1)
    values = torch.gather(books, -1, ids).reshape(e, nt, kt, 8, 64)
    out = values.permute(0, 1, 3, 2, 4).reshape(e, nt * 8, kt * 64).to(out_dtype)
    return out[0] if squeeze else out
