# Source (NVIDIA Megatron pinned block-table attention dispatch: varlen causal prefill and paged decode calls): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L770-L882
# Boundary: dispatch and scale selection follow max_seqlen_q. Local materializes gathered K/V and computes PyTorch scores; upstream delegates to external FlashAttention kernels, whose results/rounding are not inspected.
"""Megatron FA2 paged prefill/decode, standard attention (non-MLA).

q [sum(q_lens),1,Hq,D], k/v [num_pages,256,Hkv,D], block_table [B,M].
cu_seqlens_q/k int32 describe each request; seqlens_k agrees with the latter.
All input buffers read-only. q may be longer than k: causal masking is aligned
bottom-right and a query with no permitted key produces zero. For decode-only,
all q lengths are equal. FA2 prefill uses supplied softmax_scale if not None;
decode uses default 1/sqrt(D) even if a custom value is set on the layer.
FP16/BF16 finite data, Hq divisible by Hkv, valid block IDs, positive key length.
"""
import math,torch

def v1_megatron_paged_attn(q,k,v,max_seqlen_q,max_seqlen_k,cu_seqlens_q,cu_seqlens_k,seqlens_k,block_table,is_decode_only,softmax_scale=None):
    _,_,H,D=q.shape;HK=k.shape[2];B=seqlens_k.numel();out=torch.zeros_like(q)
    offsets=cu_seqlens_q.tolist();lens=seqlens_k.tolist();bs=k.shape[1]
    decode_only=max_seqlen_q <= 1
    scale=D**-0.5 if decode_only or softmax_scale is None else softmax_scale
    for b in range(B):
      lo,hi=offsets[b:b+2];qlen=hi-lo;kn=lens[b]
      pages=block_table[b,:((kn+bs-1)//bs)].long();key=k[pages].reshape(-1,HK,D)[:kn].repeat_interleave(H//HK,dim=1).float().transpose(0,1);value=v[pages].reshape(-1,HK,D)[:kn].repeat_interleave(H//HK,dim=1).float().transpose(0,1)
      query=q[lo:hi,0].float().transpose(0,1);scores=query@key.transpose(1,2)*scale
      valid=torch.arange(kn,device=q.device)[None,:]<=torch.arange(qlen,device=q.device)[:,None]+kn-qlen
      scores=scores.masked_fill(~valid[None,:,:],float('-inf'));has=valid.any(-1);weights=torch.zeros_like(scores)
      weights[:,has]=torch.softmax(scores[:,has],dim=-1);out[lo:hi,0]=(weights@value).transpose(0,1).to(q.dtype)
    return out
