"""Deinterleave GPT-OSS gate/up expert rows and their scales and bias in place.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/marlin_utils_fp4.py#L334-L370

Contract (official ``deinterleave_moe_mxfp4_w13_for_marlin``): the layer stores
gate and up rows interleaved as ``[gate0, up0, gate1, up1, ...]`` along dim 1.
The function rewrites ``layer.w13_weight.data``, and when present the scale
(``w13_weight_scale`` or ``w13_weight_scale_inv``) and bias
(``w13_weight_bias`` or ``w13_bias``) tensors, into ``[gate..., up...]`` order.
It returns ``None``, keeps the same tensor objects (``.data`` is reassigned) and
raises ``ValueError`` when a row count is odd or a scale/bias row count differs.
"""
import torch


def v1_sgl_mxfp4_moe_deinterleave_(layer):
    def pick(*keys):
        for key in keys:
            value = getattr(layer, key, None)
            if value is not None:
                return key, value
        return None, None

    weight = layer.w13_weight
    scale_key, scale = pick("w13_weight_scale", "w13_weight_scale_inv")
    bias_key, bias = pick("w13_weight_bias", "w13_bias")
    e, n, k = weight.shape
    if n % 2:
        raise ValueError("expected even row count")
    layer.w13_weight.data = (
        weight.data.view(e, n // 2, 2, k).permute(0, 2, 1, 3).contiguous().view(e, n, k)
    )
    if scale is not None:
        if scale.shape[1] != n:
            raise ValueError("scale row count mismatch")
        getattr(layer, scale_key).data = (
            scale.data.view(e, n // 2, 2, scale.shape[-1])
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, scale.shape[-1])
        )
    if bias is not None:
        if bias.shape[1] != n:
            raise ValueError("bias row count mismatch")
        getattr(layer, bias_key).data = (
            bias.data.view(e, n // 2, 2).permute(0, 2, 1).contiguous().view(e, n)
        )
