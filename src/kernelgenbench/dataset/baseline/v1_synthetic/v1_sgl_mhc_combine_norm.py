"""Collapse mHC streams with pre weights, then RMS-normalize.

The measured dispatch is deepseek_v4_mhc._combine (L496-L546): it collapses
with the supplied pre weights (or copy 0 when pre is None), then applies the
RMSNorm.  For bf16 / 4 streams / hidden 5120 / batch <= 96 it calls the fused
Triton hc_combine_norm; otherwise mhc_combine + F.rms_norm.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v4_mhc.py#L496-L546
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/layernorm/hc_combine_norm.py#L55-L80
"""
import torch
def v1_sgl_mhc_combine_norm(streams,pre,norm_weight,eps):
    if pre is None:combined=streams[:,0,:]
    else:combined=(streams.float()*pre.float().unsqueeze(-1)).sum(1).to(streams.dtype)
    f=combined.float();return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)*norm_weight.float()).to(streams.dtype)
