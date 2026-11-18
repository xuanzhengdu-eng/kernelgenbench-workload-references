# Source (NVIDIA Megatron MLA forward: q/kv down projections, kv RMSNorm, up projections and RoPE preparation): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L553-L730
# Source (NVIDIA Megatron MLA core attention call, output projection and separate bias return): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L216-L374
# Source (NVIDIA Megatron MLA even/odd (interleaved) RoPE application): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L92-L126
# Removed branch: the former backend='absorbed' path modelled the fixed framework unit-test
# MockCoreAttention algebra (no mask, latent as both key and value). The pinned commit only
# absorbs when cache_mla_latents is set in decode-only inference with Flash MLA, a different
# contract with a different cache/ABI; the mock path is not an official operator and is no
# longer part of this item.
"""Megatron MLA single rank, q_lora_rank=None, standard unfused RoPE.

x[S,B,I], q weight [H*(C+P),I], KV down weight [R+P,I], RMS weight [R],
KV up weight [H*(C+V),R], out weight [O,H*V], separate out bias [O],
frequencies [S,1,1,P], bool mask [B,1,S,S]. FP32, P positive/even.
KV latent uses RMS normalization, RoPE uses MLA even/odd interleave, attention
is masked only by the supplied bool mask (True = -10000 before softmax).
Returns output and the separate bias, which is not added to the output.
Absorbed decode with cache_mla_latents, fusions, query low rank, TP/CP/TE are
out of scope for this item.
"""
import torch

def v1_megatron_mla_full(x,wq,wkv_down,rms_weight,wkv_up,wout,out_bias,freq,mask,heads,content_dim,rope_dim,value_dim,eps,backend='ordinary'):
 S,B,I=x.shape;H=heads;C=content_dim;P=rope_dim;V=value_dim;R=rms_weight.numel()
 if backend!='ordinary':raise ValueError('only the ordinary MLA path is supported')
 if P%2 or not P or wq.shape!=(H*(C+P),I) or wkv_down.shape!=(R+P,I) or wkv_up.shape!=(H*(C+V),R) or wout.shape[1]!=H*V or out_bias.shape!=(wout.shape[0],):raise ValueError('incompatible MLA dimensions')
 q=(x@wq.t()).reshape(S,B,H,C+P);z=x@wkv_down.t();latent,kpos=z.split((R,P),-1)
 latent=latent*torch.rsqrt(latent.square().mean(-1,keepdim=True)+eps)*rms_weight
 qc,qp=q.split((C,P),-1)
 kv=(latent@wkv_up.t()).reshape(S,B,H,C+V);kc,v=kv.split((C,V),-1)
 def rope(z):
  z=torch.cat((z[...,::2],z[...,1::2]),-1);a,b=z.chunk(2,-1);rot=torch.cat((-b,a),-1)
  return z*freq.cos()+rot*freq.sin()
 qp=rope(qp);kp=rope(kpos.unsqueeze(-2)).expand(S,B,H,P)
 q=torch.cat((qc,qp),-1).permute(1,2,0,3);k=torch.cat((kc,kp),-1).permute(1,2,0,3);v=v.permute(1,2,0,3)
 scores=(q@k.transpose(-2,-1))*((C+P)**-.5);probs=scores.masked_fill(mask,-10000.).softmax(-1)
 context=(probs@v).permute(2,0,1,3).contiguous().reshape(S,B,H*V)
 return context@wout.t(),out_bias
