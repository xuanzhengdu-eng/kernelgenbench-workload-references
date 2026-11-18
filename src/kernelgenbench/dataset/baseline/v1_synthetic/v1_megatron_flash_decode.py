"""Megatron Attention.flash_decode: RoPE + in-place KV cache + attention.

Inputs: q [Q,B,Hq,D], k/v [T,B,Hkv,D], caches [C,B,Hkv,D], offsets [B]
int32. Hq divisible by Hkv. Both cos and sin must be None or [R,Dr/2], Dr
multiple of 16 and <=D. Finite FP16/BF16. offset+T <= C, valid rope positions.
FlashAttention defaults here are noncausal, no local window, scale=1/sqrt(D):
all Q positions rotate at the same batch offset, while new K rotates at offset+t.
Caches are mutated; return is [B,Q,Hq,D], in query dtype. No backward.
"""
# Source (NVIDIA Megatron fixed commit; Attention.flash_decode wrapper permutes tensors and delegates RoPE, cache update, attention to flash_attn_with_kvcache): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L651-L695
# Source (NVIDIA Megatron fixed commit; decode-only caller branch and cache arguments): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1030-L1036

import math,torch

def v1_megatron_flash_decode(sequence_len_offset,query_layer,key_layer,value_layer,inference_key_memory,inference_value_memory,rotary_cos,rotary_sin,rotary_interleaved=False):
    Q,B,H,D=query_layer.shape;T=key_layer.shape[0];HK=key_layer.shape[2];O=torch.empty((B,Q,H,D),dtype=query_layer.dtype,device=query_layer.device)
    c=None if rotary_cos is None else rotary_cos.to(query_layer.dtype);s=None if rotary_sin is None else rotary_sin.to(query_layer.dtype)
    def rotate(x,pos):
        if c is None:return x
        half=c.shape[-1];v=x[...,:2*half].float()
        if rotary_interleaved:a=v[...,0::2];b=v[...,1::2]
        else:a=v[...,:half];b=v[...,half:]
        cos=c[pos].float();sin=s[pos].float();ra=a*cos-b*sin;rb=a*sin+b*cos
        if rotary_interleaved:r=torch.stack((ra,rb),dim=-1).flatten(-2)
        else:r=torch.cat((ra,rb),dim=-1)
        return torch.cat((r.to(x.dtype),x[...,2*half:]),dim=-1)
    for b in range(B):
        start=int(sequence_len_offset[b]);newk=torch.stack([rotate(key_layer[t,b],start+t) for t in range(T)])
        inference_key_memory[start:start+T,b].copy_(newk);inference_value_memory[start:start+T,b].copy_(value_layer[:,b])
        q=rotate(query_layer[:,b],start).float().transpose(0,1)
        key=inference_key_memory[:start+T,b].float().repeat_interleave(H//HK,dim=1).transpose(0,1)
        value=inference_value_memory[:start+T,b].float().repeat_interleave(H//HK,dim=1).transpose(0,1)
        prob=torch.softmax(torch.matmul(q,key.transpose(1,2))/math.sqrt(D),dim=-1)
        O[b]=torch.matmul(prob,value).transpose(0,1).to(O.dtype)
    return O
