"""SGLang fused (optional) residual add + RMSNorm + static per-tensor FP8 quant.

Official python entry (returns ``((fp8, scale, orig_dtype), residual)`` or the
triple when there is no residual):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layernorm.py#L1108-L1179

The fused kernel multiplies by the reciprocal of the static scale and rounds the
normalized value through the activation dtype, so the quantized output can
differ from the plain division by one FP8 code.
"""
import torch


def v1_sgl_add_rmsnorm_static_fp8_quant_(inp, residual, weight, epsilon, static_scale,
                                         post_residual_addition=None):
    assert static_scale.numel() == 1 and static_scale.dtype == torch.float32
    # Mirrors the official method exactly: the post addend only applies when a
    # residual is present, and a separate addend produces a new residual buffer.
    if residual is None:
        target = None
        x = inp.float()
    else:
        target = residual if post_residual_addition is None \
            else residual + post_residual_addition
        target.copy_((inp.float() + target.float()).to(inp.dtype))
        x = target.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    normalized = x * weight.float() * inv
    q = (normalized / static_scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    triple = (q, static_scale, inp.dtype)
    return (triple, target) if target is not None else triple
