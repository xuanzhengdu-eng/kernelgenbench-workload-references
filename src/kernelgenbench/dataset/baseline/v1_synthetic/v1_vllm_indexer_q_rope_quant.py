"""vLLM DSA indexer Q RoPE + UE8M0-scale FP8 and weight adjustment.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/sparse_attn_indexer.py#L194-L325
"""
import torch
def v1_vllm_indexer_q_rope_quant(positions,q,cos_sin_cache,weights,softmax_scale,head_scale,is_neox):
 assert q.dtype==torch.bfloat16 and q.shape[-1]==128 and cos_sin_cache.shape[-1]==64 and weights.shape==q.shape[:2]
 cs=cos_sin_cache[positions.long()].float();cos=cs[:,:32].unsqueeze(1);sin=cs[:,32:].unsqueeze(1);rot=q[...,:64].float()
 if is_neox:x0,x1=rot[...,:32],rot[...,32:]
 else:x0,x1=rot[...,::2],rot[...,1::2]
 r0=(x0*cos-x1*sin).to(torch.bfloat16).float();r1=(x1*cos+x0*sin).to(torch.bfloat16).float()
 first=torch.cat((r0,r1),-1) if is_neox else torch.stack((r0,r1),-1).flatten(-2)
 values=torch.cat((first,q[...,64:].float()),-1);amax=values.abs().amax(-1).clamp_min(1e-10);scale=torch.exp2(torch.ceil(torch.log2(amax/448)))
 out=(values/scale.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn);out_weight=weights.float()*scale*softmax_scale*head_scale
 return out,out_weight
