"""DeepSpeed grouped expert MLP, sequential reference path.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_experts.py#L204-L261
"""
import torch
import torch.nn.functional as F

def v1_ds_grouped_expert_mlp(x,w1,w2,w3,num_tokens_per_expert,activation='swiglu',alpha=1.702,limit=7.0):
    counts=num_tokens_per_expert.tolist();out=[];start=0
    for e,n in enumerate(counts):
        tokens=x[start:start+n];gate=F.linear(tokens,w1[e].to(x.dtype));up=F.linear(tokens,w3[e].to(x.dtype))
        if activation=='swiglu':h=F.silu(gate)*up
        elif activation=='geglu_tanh':h=F.gelu(gate,approximate='tanh')*up
        elif activation=='swiglu_clamped':h=F.silu(gate.clamp(max=limit))*up.clamp(-limit,limit)
        elif activation=='swiglu_oai':
            gate=gate.clamp(max=limit);up=up.clamp(-limit,limit);h=(up+1)*(gate*torch.sigmoid(gate*alpha))
        else:raise ValueError('unknown expert activation')
        out.append(F.linear(h,w2[e].to(x.dtype)));start+=n
    out=torch.cat(out,dim=0)
    return torch.cat((out,out.new_zeros((x.shape[0]-start,x.shape[1]))),dim=0)
