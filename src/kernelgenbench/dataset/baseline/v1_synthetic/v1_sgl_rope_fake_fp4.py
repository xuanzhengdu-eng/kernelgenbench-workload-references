"""SGLang RoPE followed by FP4 fake quantization, with optional position gather.

Fused-backend reference of this item; the torch fallback backend is
``v1_sgl_rope_fake_fp4_torch`` (same item, same contract, other backend).
Official source (pinned commit), the fused branch of ``_rope_fq4`` (lines 37-44; the
function itself starts at line 34):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L37-L44
Official source (pinned commit), the fused kernel ``rope_tail_fake_quant_fp4``:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/dsv4/fp4_rope_fake_quant.py#L117-L170
Official source (pinned commit), the FP4 scale/round definitions the kernel mirrors:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/quantization/fp4_fake_quant.py#L16-L62
"""
import torch
from kernelgenbench.dataset.baseline.v1_synthetic.v1_sgl_rope_tail import v1_sgl_rope_tail
def v1_sgl_rope_fake_fp4(x,freqs,rope_dim,compressed_kv=False,positions=None):
    rotated=v1_sgl_rope_tail(x,freqs if positions is None else freqs[positions],rope_dim)
    block=16 if compressed_kv else 32
    assert x.shape[-1]%block==0
    v=rotated.float().reshape(*x.shape[:-1],-1,block)
    peak=v.abs().amax(-1,keepdim=True)
    if compressed_kv:
        scale=(peak*(1/6)).clamp(2**-9,448).to(torch.float8_e4m3fn).float()
    else:
        z=peak.clamp_min(6*2**-126)*(1/6)
        bits=z.contiguous().view(torch.int32)
        exponent=((bits>>23)&255)-127+((bits&0x7fffff)!=0).to(torch.int32)
        scale=((exponent+127)<<23).view(torch.float32)
    y=(v/scale).clamp(-6,6);mag=y.abs()
    step=torch.where(mag<2,0.5,torch.where(mag<4,1.,2.))
    return (torch.round(mag/step)*step*torch.sign(y)*scale).flatten(-2).to(x.dtype)
