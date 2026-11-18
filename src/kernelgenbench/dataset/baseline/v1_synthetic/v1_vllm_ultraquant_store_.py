"""vLLM UltraQuant per-group FP4 K/V packing into paged cache.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/ultraquant/triton_store.py#L107-L351
"""
import torch
_LEVELS=(-6.,-4.,-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.,4.,6.)
def _wht(x):
    d=x.shape[-1]; ids=torch.arange(d,device=x.device); x=x.float()
    for i in range(d.bit_length()-1):
        bit=1<<i; x=(torch.where((ids & bit)==0,x,-x)+x[...,ids^bit])*0.7071067811865476
    return x
def _pack(x,c):
    d=x.shape[-1]; x=x.reshape(-1,32).float(); m=x.abs().amax(-1); z=m==0
    exp=torch.floor(torch.log2(torch.where(z,1.,m*c))+.5).to(torch.int32)
    div=torch.where(z,1.,torch.exp2(exp.float())); x=x/div[:,None]
    levels=torch.tensor(_LEVELS,device=x.device); mids=(levels[:-1]+levels[1:])/2
    idx=torch.bucketize(x.contiguous(),mids,right=False); bits=torch.where(idx<7,15-idx,idx-7).reshape(-1)
    codes=(bits[::2] | (bits[1::2]<<4)).to(torch.uint8)
    scale=torch.where(z,0,exp+127); scale=torch.where((scale<0)|(scale>255),0,scale).to(torch.uint8)
    return codes,scale
def v1_vllm_ultraquant_store_(key,value,kv_cache,slot_mapping,constant_c=0.156):
    n,h,d=key.shape; bs=kv_cache.shape[1]
    for t in range(n):
        slot=int(slot_mapping[t]);
        if slot<0:continue
        for head in range(h):
            kc,ks=_pack(_wht(key[t,head]),constant_c); vc,vs=_pack(value[t,head],constant_c)
            kv_cache[slot//bs,slot%bs,head,:d+d//16].copy_(torch.cat((kc,ks,vc,vs)))
