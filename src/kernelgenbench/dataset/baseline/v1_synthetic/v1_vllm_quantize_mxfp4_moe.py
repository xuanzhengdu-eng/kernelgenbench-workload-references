"""vLLM H800 Triton-path MXFP4 expert weight quantization.

Official Triton kernel, public wrapper and the expert-batched caller that
gives this item its 3-D ``(experts, rows, K)`` contract:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py#L256-L346
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py#L538-L564
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/online/mxfp4.py#L49-L76

The pinned ``mxfp4_quantize`` dispatches to ``downcast_to_mxfp(axis=-1)`` on
H800.  Its per-32 block scale rounds ``max_abs`` to nearest-even in FP32 bits,
then uses ``floor(log2(rounded)) - 2`` clamped to [-127, 127].  E2M1 values are
selected with round-to-nearest-even (sticky bit included) and packed low-first.
The E8M0 scale byte is the top exponent byte of the resulting FP32 power of
two.  The reference returns ``(packed_uint8, scale_bytes_uint8)`` and does not
modify its input.
"""
import torch


def v1_vllm_quantize_mxfp4_moe(weight):
    assert weight.ndim == 3 and weight.dtype in (torch.float16, torch.bfloat16)
    assert all(size > 0 for size in weight.shape) and weight.shape[-1] % 32 == 0
    x = weight.float()
    assert torch.isfinite(x).all() and (x.abs() <= 65536).all()
    assert ((x == 0) | (x.abs() >= 2.0**-120)).all()

    experts, rows, columns = x.shape
    blocks = x.reshape(experts, rows, columns // 32, 32)
    max_abs = blocks.abs().amax(-1).contiguous()
    raw = max_abs.view(torch.int32).to(torch.int64)
    rounded = (raw + 0x200000) & 0x7F800000
    exponent = ((rounded >> 23) - 127 - 2).clamp(-127, 127)
    exponent = torch.where(rounded == 0, torch.full_like(exponent, -127), exponent)
    scale = torch.exp2(exponent.float())
    inverse = torch.where(scale > 0, scale.reciprocal(), torch.zeros_like(scale))

    quantized = blocks * inverse.unsqueeze(-1)
    magnitude = quantized.abs()
    codes = torch.zeros_like(quantized, dtype=torch.uint8)
    for upper, cut in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), 1):
        codes += ((magnitude > cut) | ((magnitude == cut) & (upper % 2 == 0))).to(torch.uint8)
    codes |= torch.signbit(quantized).to(torch.uint8) << 3
    codes = codes.reshape(experts, rows, columns // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).contiguous()
    # The production Triton kernel stores the FP32 exponent byte directly.
    encoded = (
        scale.contiguous().view(torch.int32).bitwise_right_shift(23).bitwise_and(255)
    ).to(torch.uint8)
    return packed, encoded.contiguous()
