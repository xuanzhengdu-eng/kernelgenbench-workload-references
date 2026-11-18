# Source (NVIDIA Megatron SelfAttention QKV projection and grouped Q/K/V split): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1380-L1475
# Source (NVIDIA Megatron DotProductAttention GQA expansion, scaled scores, mask-softmax and value reduction): https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/dot_product_attention.py#L142-L250
# Boundary: regular single-rank MHA/GQA math only. Local has no dropout, RoPE, cache, tensor/context parallelism or Transformer Engine backend; mask fill/tolerance are not asserted identical.
"""Megatron SelfAttention single rank, unfused MHA/GQA causal inference math.

x[S,B,I], interleaved QKV weight[G*(H/G+2)*D,I] and bias, output weight
[O,H*D], separate output bias[O], explicit bool causal mask[B,1,S,S] where
True masks. All floating values are float32. QKV storage is grouped as
(q heads of group, k, v) for each group. Returns (output[S,B,O], output_bias),
with bias returned as the original tensor (Megatron skip_bias_add contract).
No dropout, RoPE, layernorm, gate, TP/CP, inference cache, or TE.
"""
import torch

def v1_megatron_self_attention(x,qkv_weight,qkv_bias,out_weight,out_bias,mask,heads,groups,head_dim):
 S,B,I=x.shape;D=head_dim;H=heads;G=groups;Q=H//G
 if H%G or qkv_weight.shape!=(G*(Q+2)*D,I) or qkv_bias.shape!=(G*(Q+2)*D,) or out_weight.shape[1]!=H*D or out_bias.shape!=(out_weight.shape[0],):raise ValueError('incompatible attention dimensions')
 mixed=(x@qkv_weight.t()+qkv_bias).reshape(S,B,G,(Q+2)*D)
 q,k,v=mixed.split((Q*D,D,D),-1);q=q.reshape(S,B,H,D);k=k.repeat_interleave(Q,2);v=v.repeat_interleave(Q,2)
 q,k,v=[z.permute(1,2,0,3) for z in (q,k,v)]
 score=(q@k.transpose(-2,-1))*(D**-.5);score=score.masked_fill(mask,-10000.)
 out=(score.softmax(-1)@v).permute(2,0,1,3).contiguous().reshape(S,B,H*D)
 return out@out_weight.t(),out_bias
