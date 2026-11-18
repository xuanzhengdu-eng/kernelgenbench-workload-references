"""SGLang supplied-coefficient mHC boundary: post, pre collapse, input RMSNorm.

Composition of the fixed-version SGLang pieces used by the AMD boundary:
- post:        deepseek_v4_mhc.py L554-L610
- combine:     deepseek_v4_mhc.py L496-L553 (mhc_combine + norm)
- call site:   deepseek_v4_platform_stage.py L645-L663
- fused HIP:   mhc_boundary_hip.py L477-L585 / deepseek_v4_fused_mhc.py L469-L501

This reference starts from supplied post/comb/pre_prev; it does not generate
coefficients or run Sinkhorn.  The HIP fused backend is source-checked but is
not executable on the H800 node; the torch post + Triton mhc_combine path is
the measured backend.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v4_mhc.py#L554-L610
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/layernorm/mhc_boundary_hip.py#L477-L585
"""
import torch


def v1_sgl_mhc_boundary_attn(x, residual, post, comb, pre_prev,
                             norm_weight, norm_eps):
    if x is None:
        r = residual
    else:
        xf = x.float()
        rf = residual.float()
        r = torch.empty_like(residual)
        for k in range(4):
            z = xf * post[:, k, None]
            for j in range(4):
                z = z + rf[:, j, :] * comb[:, j, k, None]
            r[:, k, :] = z.to(residual.dtype)
    if pre_prev is None:
        y = r[:, 0, :].contiguous()
    else:
        y = torch.zeros_like(r[:, 0, :], dtype=torch.float32)
        for k in range(4):
            y = y + r[:, k, :].float() * pre_prev[:, k, None].float()
        y = y.to(residual.dtype)
    z = y.float()
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + norm_eps)
            * norm_weight.float()).to(residual.dtype)
