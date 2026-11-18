# Source-checked reference for KernelGenBench row 65 (SGLang branch).
"""SGLang gated RMSNorm (FLA ``RMSNorm`` with norm_before_gate=False).

When a gate is supplied, activation is SiLU and the gate multiplies x *before*
the RMS statistic; ``z=None`` is also accepted and then the module is a plain
RMSNorm.  The affine weight is multiplied in fp32 before the output cast.

Official sources (SGLang fixed commit 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L41-L72
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L488-L530
"""
import torch
import torch.nn.functional as F


def v1_rmsnorm_gated_sglang(x, gate, weight, eps, group_size=None):
    gated = x.float() if gate is None else x.float() * F.silu(gate.float())
    if group_size is None:
        normed = gated * torch.rsqrt(gated.square().mean(dim=-1, keepdim=True) + eps)
    else:
        assert gated.shape[-1] % group_size == 0
        grouped = gated.reshape(*gated.shape[:-1], gated.shape[-1] // group_size, group_size)
        normed = (grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)).reshape(
            gated.shape
        )
    return (normed * weight.float()).to(x.dtype)
