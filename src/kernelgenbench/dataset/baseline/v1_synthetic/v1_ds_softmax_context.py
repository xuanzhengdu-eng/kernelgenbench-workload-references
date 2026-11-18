# Source (DeepSpeed ds_softmax_context QKV transform, cache positions, attention dispatch and cache-prefix outputs): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L439-L568
# Source (SGLang native decode writes K/V then gathers request cache rows for SDPA): https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/torch_native_backend.py#L336-L396
# Boundary: DeepSpeed link is layout=deepspeed context; local uses explicit buffers and PyTorch attention instead of CUDA. SGLang link covers layout=sglang decode S=1 only, not prompt or local mask/ALiBi/window extensions.
"""DeepSpeed softmax_context prompt or single-token decode: QKV, KV write and attention.

qkv [B,S,(H+2*G)*D] fp32/fp16; cache buffers [B,H,capacity,D] and output buffer
[B,S,H*D] same dtype, G divides H, D multiple 8, capacity>=past_len+S. Prompt S>1 requires past_len=0; decode S=1 appends at past_len. No rotary,
optional 2D batch/KV additive mask and [B*H,KV] ALiBi, no quantization; local_attention optionally supplies a window. Writes replicated K/V into the active cache slots; returns aliased output and cache-prefix views.
Scores use norm_factor**2; optional triangular causal mask. Source holds the
buffers inside InferenceContext; explicit buffers are a benchmark ABI adapter.
"""
import torch

def v1_ds_softmax_context(qkv,key_cache,value_cache,output_buffer,heads,kv_heads,norm_factor,triangular,past_len=0,window_size=0,attn_mask=None,alibi=None,layout='deepspeed',cache_loc=None,req_to_token=None,req_pool_indices=None,seq_lens=None):
 if layout=='sglang':
  B,S,width=qkv.shape; H=heads; G=kv_heads; D=key_cache.shape[-1]
  if S!=1 or H%G or width!=(H+2*G)*D or key_cache.shape!=value_cache.shape or key_cache.shape[1]!=G: raise ValueError('invalid SGLang decode dimensions')
  q,k,v=qkv[:,0].split((H*D,G*D,G*D),-1); q=q.reshape(B,H,D);k=k.reshape(B,G,D);v=v.reshape(B,G,D)
  key_cache[cache_loc.long()]=k;value_cache[cache_loc.long()]=v
  for i in range(B):
   ids=req_to_token[req_pool_indices[i].long(),:int(seq_lens[i])].long()
   keys=key_cache[ids].float().repeat_interleave(H//G,dim=1).transpose(0,1);vals=value_cache[ids].float().repeat_interleave(H//G,dim=1).transpose(0,1)
   scores=torch.einsum('hd,hld->hl',q[i].float(),keys)*(norm_factor*norm_factor)
   output_buffer[i,0]=(scores.softmax(-1).unsqueeze(1)@vals).squeeze(1).to(output_buffer.dtype).reshape(-1)
  return output_buffer,key_cache,value_cache
 if layout!='deepspeed': raise ValueError('unknown layout')
 B,S,width=qkv.shape;H=heads;G=kv_heads;D=key_cache.shape[-1];cap=key_cache.shape[2]
 if S<1 or H%G or D%4 or cap<past_len+S or past_len<0 or (S>1 and past_len!=0) or width!=(H+2*G)*D:raise ValueError('invalid attention dimensions')
 if key_cache.shape!=(B,H,cap,D) or value_cache.shape!=key_cache.shape or output_buffer.shape!=(B,S,H*D):raise ValueError('invalid buffers')
 if qkv.dtype not in (torch.float32,torch.float16) or not all(t.dtype==qkv.dtype and t.device==qkv.device and t.is_contiguous() for t in (qkv,key_cache,value_cache,output_buffer)):raise ValueError('contiguous same-dtype fp32/fp16 required')
 if qkv.dtype==torch.float16 and D%8:raise ValueError('fp16 D must be divisible by 8')
 q,k,v=qkv.split((H*D,G*D,G*D),-1);q=q.reshape(B,S,H,D).permute(0,2,1,3);k=k.reshape(B,S,G,D).repeat_interleave(H//G,2).permute(0,2,1,3);v=v.reshape(B,S,G,D).repeat_interleave(H//G,2).permute(0,2,1,3)
 key_cache[:,:,past_len:past_len+S].copy_(k);value_cache[:,:,past_len:past_len+S].copy_(v)
 K=key_cache[:,:,:past_len+S];V=value_cache[:,:,:past_len+S]
 score=((q.float()@K.float().transpose(-2,-1))*(norm_factor*norm_factor)).to(qkv.dtype).float()
 if attn_mask is not None:
  if attn_mask.shape!=(B,past_len+S):raise ValueError('expected [B,KV] mask')
  score=score+attn_mask.float()[:,None,None,:]
 if alibi is not None:
  if alibi.shape!=(B*H,past_len+S):raise ValueError('expected [B*H,KV] alibi')
  score=score+alibi.float().reshape(B,H,1,past_len+S)
 if triangular and S>1:score=score.masked_fill(torch.ones(S,S,device=qkv.device,dtype=torch.bool).triu(1),-torch.inf)
 if window_size>0:
  total=past_len+S;row=torch.arange(S,device=qkv.device);real=row if S==total else row+total;col=torch.arange(total,device=qkv.device)
  stride=torch.where(real>=window_size,real-window_size,-1);stride4=torch.where((real//4)>(window_size//4),real//4-window_size//4,0)
  valid=(col[None,:]>stride[:,None])&((col[None,:]//4)>=stride4[:,None]);score=score.masked_fill(~valid,-torch.inf)
 prob=score.softmax(-1).to(qkv.dtype)
 result=(prob.float()@V.float()).to(qkv.dtype).permute(0,2,1,3).contiguous().reshape(B,S,H*D);output_buffer.copy_(result)
 return output_buffer,K,V
