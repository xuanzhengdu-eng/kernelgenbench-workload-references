"""SGLang mHC mix statistics + Sinkhorn split (CUDA/Triton path).

The SGLang caller ``_mix_stats_impl`` flattens ``x`` and, on CUDA, dispatches to
``hc_mix_stats_sinkhorn`` in the pinned SGLang tree.  That kernel computes the
projection and RMS scale, splits the ``(2 + hc_mult) * hc_mult`` outputs into
pre/post/comb, applies the configured sigmoid/softmax/Sinkhorn steps, and returns
three FP32 tensors.  This reference keeps that public tensor contract, including
the fact that the input tensors are not modified.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v4_mhc.py#L346-L466
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/layernorm/mhc.py#L2488-L2568
"""
import torch
import torch.nn.functional as F


def v1_sgl_mhc_mix_stats(x, fn, scale, base, hc_mult, sinkhorn_iters, rms_eps, eps):
    flat = x.flatten(1).float()
    m = hc_mult
    mix = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mix[:, :m] * scale[0] + base[:m]) + eps
    post = 2 * torch.sigmoid(mix[:, m:2 * m] * scale[1] + base[m:2 * m])
    comb = (mix[:, 2 * m:] * scale[2] + base[2 * m:]).view(-1, m, m)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
