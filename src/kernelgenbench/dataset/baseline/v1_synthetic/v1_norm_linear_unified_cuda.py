"""DeepSpeed normalization + linear, native ``transformer_inference`` QKV kernel.

Official source (DeepSpeed, pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1031-L1066
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L60-L67
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L70-L75

This is the same item as ``v1_norm_linear_unified`` but bound to the fused CUDA
kernel that ``QKVGemmOp.forward`` selects on this pinned build
(``qkv_gemm_fp16`` / ``qkv_gemm_bf16`` / ``rms_qkv_gemm_*``), instead of the
torch fallback.  The kernel writes into DeepSpeed's inference workspace, so the
resulting tensors are copies of that workspace.

Contract (identical math to the fallback, fused implementation):
  * LayerNorm variant: ``F.layer_norm(x, (H,), gamma, beta, eps)`` then
    ``matmul(norm, weight.T if transpose else weight)``; optional bias;
    returns ``[projection, normalized_input]``.
  * RMSNorm variant: fp32 mean-square RMSNorm scaled by gamma (no beta, no
    bias), then the same projection; returns ``[projection, normalized_input]``.
  * fp16/bfloat16 inputs and weights; the fused kernel's own rounding is kept.
"""
import torch
import torch.nn.functional as F


def v1_norm_linear_unified_cuda(x, weight, gamma, beta, bias, eps, norm_type,
                                transpose, backend="deepspeed", tp_group=None,
                                skip_norm_and_all_gather=False,
                                pre_gathered_norm=None, sequence_parallel=False):
    if backend != "deepspeed":
        raise NotImplementedError("only the DeepSpeed fused QKV kernel is in scope")
    assert x.ndim == 3 and gamma.shape == (x.shape[-1],)
    assert weight.ndim == 2
    out_features = weight.shape[0] if transpose else weight.shape[1]
    if norm_type == "layer":
        assert beta is not None and beta.shape == gamma.shape
        norm = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
    elif norm_type == "rms":
        if beta is not None or bias is not None:
            raise ValueError("the fused RMSNorm kernel has no beta or bias")
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
