"""DeepSpeed normalization + linear (QKV GEMM fallback), with output contracts.

Official source (DeepSpeed, pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L60-L67
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L70-L75

Two official fallbacks are represented:
  * ``qkv_gemm_fallback``: LayerNorm over the last hidden dim, optional beta and
    optional projection bias, then ``matmul(norm, weight.T if transpose else
    weight)``; returns ``[projection, normalized_input]`` (the official fallback returns a list).
  * ``rms_qkv_gemm_fallback``: RMSNorm (no beta), no bias, then the same
    projection; returns ``[projection, normalized_input]`` (the official fallback returns a list).
Both accept 3D inputs and fp32/fp16/bfloat16 tensors; the normalization and
GEMM run in the input dtype, with RMSNorm accumulating the mean square in fp32.
The function returns a tuple and never mutates its inputs.

The legacy Megatron branches of the old unified reference are not part of this
file: their only official analogue is the Transformer-Engine-gated
``InferenceLayerNormColumnParallelLinear`` (source-read, target backend not
available on this pinned Megatron build), recorded separately in the report.
"""
import torch
import torch.nn.functional as F


def v1_norm_linear_unified(x, weight, gamma, beta, bias, eps, norm_type,
                          transpose, backend="deepspeed", tp_group=None,
                          skip_norm_and_all_gather=False, pre_gathered_norm=None,
                          sequence_parallel=False):
    if backend != "deepspeed":
        raise NotImplementedError(
            "legacy Megatron norm-linear branches are quarantined; only the "
            "DeepSpeed QKV GEMM fallbacks are part of this item")
    assert x.ndim == 3 and gamma.ndim == 1 and gamma.numel() == x.shape[-1]
    assert weight.ndim == 2
    if transpose:
        assert weight.shape[1] == x.shape[-1]
        out_features = weight.shape[0]
    else:
        assert weight.shape[0] == x.shape[-1]
        out_features = weight.shape[1]
    if norm_type == "layer":
        assert beta is not None and beta.shape == gamma.shape
        norm = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
    elif norm_type == "rms":
        if beta is not None or bias is not None:
            raise ValueError("the official RMSNorm fallback has no beta or bias")
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        norm = x * torch.rsqrt(variance + eps)
        if gamma.dtype in (torch.float16, torch.bfloat16):
            norm = norm.to(gamma.dtype)
        norm = gamma * norm
    else:
        raise ValueError("unknown normalization")
    out = torch.matmul(norm, weight.t() if transpose else weight)
    if bias is not None:
        assert bias.shape == (out_features,)
        out = out + bias
    return [out, norm]
