"""SGLang deterministic watermark token selection, torch backend.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/sampling/watermarking/core.py#L287-L391
Murmur-style hash of key, context hash and token ID; optional dual-key
per-request mixing, select argmax(log(U)/probability). Nonpositive probability
gets -inf; all-zero row returns index 0 under torch.argmax. Returns int64.
"""
import torch
_M=0xffffffff

def _rot(x,n):return ((x<<n)|(x>>(32-n)))&_M
def _mix(h,x):
    x=(x*0xcc9e2d51)&_M;x=_rot(x,15);x=(x*0x1b873593)&_M
    return (_rot(h^x,13)*5+0xe6546b64)&_M
def _finish(h):
    h=h^(h>>16);h=(h*0x85ebca6b)&_M;h=h^(h>>13);h=(h*0xc2b2ae35)&_M
    return h^(h>>16)
def _hash(keys,ctx,tokens):
    state=torch.zeros((keys.numel(),tokens.numel()),dtype=torch.int64,device=keys.device);key=keys.view(-1,1)
    for x in (key&_M,(key>>32)&_M,ctx.view(-1,1)&_M,tokens.view(1,-1)&_M):state=_mix(state,x)
    return _finish(state^16)
def v1_sgl_watermark_select(probabilities,context_hashes,keys,keys_b=None,mixing_thresholds=None):
    if (keys_b is None)!=(mixing_thresholds is None):raise ValueError('dual-key inputs')
    tokens=torch.arange(probabilities.shape[-1],device=probabilities.device);hashed=_hash(keys,context_hashes,tokens)
    if keys_b is not None:
        other=_hash(keys_b,context_hashes,tokens);state=torch.zeros_like(keys,dtype=torch.int64)
        for x in (keys&_M,(keys>>32)&_M,keys_b&_M,(keys_b>>32)&_M,context_hashes&_M):state=_mix(state,x)
        choose=_finish(state^20)<mixing_thresholds;hashed=torch.where(choose.view(-1,1),hashed,other)
    lower=torch.log((hashed.to(torch.float32)+.5)/float(1<<32));upper=torch.log1p(-((_M-hashed).to(torch.float32)+.5)/float(1<<32));logu=torch.where(hashed<(1<<31),lower,upper)
    return torch.where(probabilities>0,logu/probabilities.to(torch.float32),-torch.inf).argmax(-1)
