# Source (DeepSpeed TopKGate forward dispatch across k=1, k=2 and k>2): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L509-L596
# Source (DeepSpeed top-1 gating and top-2 gating implementation): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L216-L414
# Source (DeepSpeed top-k gating implementation): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L415-L508
# Boundary: local synthetic branches omit jitter/noisy policy, random token selection and no-drop/sparse routes; k=2 and k>=3 have specified tie/capacity assumptions. Links indicate related source branches, not full API equivalence.
"""DeepSpeed TopKGate dense, no jitter/sampling, drop_tokens=True.

x [S,H], weight [E,H], S>0, same-device FP32/FP16/BF16.
used_token, when present, is a same-device binary int64 [S] tensor;
min_capacity is positive and no greater than S. Returns FP32 aux,
FP32 combine [S,E,C], bool dispatch [S,E,C], int64 pre-drop expert counts.
1<=k<=E. k=1 optionally filters used_token and retains raw gate probabilities;
k=1 overflow may choose any of the tied tokens as permitted by torch.topk. k=2 uses top-1-prioritized positional capacity; k>=3 uses score
capacity. For k>=3 ties at selected/capacity boundaries are outside this
contract; k=2 accepts ties with source argmax semantics. Other branches such
as Gumbel sampling, sparse routes and no-drop are excluded.
"""
import torch
import torch.nn.functional as F

def _top2(x,weight,capacity_factor,min_capacity):
    logits=F.linear(x.float(),weight.float());p=F.softmax(logits,dim=1);S,E=p.shape
    cap=max(int(torch.ceil(torch.tensor(capacity_factor*2,dtype=torch.float32)*(S/E)).item()),min_capacity)
    i1=p.argmax(dim=1);m1=F.one_hot(i1,num_classes=E)
    i2=logits.masked_fill(m1.bool(),float('-inf')).argmax(dim=1);m2=F.one_hot(i2,num_classes=E)
    loc1=m1.cumsum(0)-1;loc2=m2.cumsum(0)-1+m1.sum(0,keepdim=True)
    counts=(m1+m2).sum(0).detach();aux=(p.mean(0)*m1.float().mean(0)).mean()*E*E
    m1=m1*(loc1<cap);m2=m2*(loc2<cap)
    l1=(loc1*m1).sum(1);l2=(loc2*m2).sum(1)
    w1=(p*m1).sum(1);w2=(p*m2).sum(1);den=(w1+w2).clamp(min=torch.finfo(p.dtype).eps);w1=w1/den;w2=w2/den
    a=(w1[:,None]*m1.float()).unsqueeze(-1)*F.one_hot(l1,num_classes=cap).float().unsqueeze(1)
    b=(w2[:,None]*m2.float()).unsqueeze(-1)*F.one_hot(l2,num_classes=cap).float().unsqueeze(1)
    combine=a+b
    return aux,combine,combine.bool(),counts

def _topk(x,weight,k,capacity_factor,min_capacity):
    gates=F.softmax(F.linear(x.float(),weight.float()),dim=1)
    S,E=gates.shape;capacity=max(int(torch.ceil(torch.tensor(capacity_factor*k,dtype=torch.float32)*(S/E)).item()),min_capacity)
    top_values,top_idx=torch.topk(gates,k,dim=1)
    selected=torch.zeros_like(gates,dtype=torch.bool).scatter_(1,top_idx,True)
    counts=selected.sum(0).detach()
    aux=(gates.mean(0)*selected.float().mean(0)).mean()*E*E/k
    top_scores=torch.zeros_like(gates).scatter(1,top_idx,top_values)
    chosen=torch.topk(top_scores,k=min(capacity,S),dim=0,sorted=False).indices
    keep=torch.zeros_like(selected).scatter_(0,chosen,True)&selected
    loc=keep.cumsum(0)-1
    masked=gates*keep
    masked=masked/masked.sum(-1,keepdim=True).clamp(min=torch.finfo(masked.dtype).eps)
    slots=F.one_hot(loc*keep,num_classes=capacity).float()
    combine=masked.unsqueeze(-1)*slots
    return aux,combine,combine.bool(),counts

def _top1(x,weight,capacity_factor,min_capacity,used_token):
    logits=F.linear(x.float(),weight.float());p=F.softmax(logits,dim=1);S,E=p.shape
    cap=max(int(torch.ceil(torch.tensor(capacity_factor,dtype=torch.float32)*(S/E)).item()),min_capacity)
    mask=F.one_hot(p.argmax(dim=1),num_classes=E)
    if used_token is not None:mask=mask*used_token[:,None]
    counts=mask.sum(0).detach();aux=(p.mean(0)*mask.float().mean(0)).sum()*E
    chosen=torch.topk(mask,k=min(cap,S),dim=0).indices
    mask=mask*torch.zeros_like(mask).scatter_(0,chosen,1)
    loc=((mask.cumsum(0)-1)*mask).sum(1)
    combined=(p*mask.float()).unsqueeze(-1)*F.one_hot(loc,num_classes=cap).float().unsqueeze(1)
    return aux,combined,combined.bool(),counts

def v1_ds_topkgate_unified(x,weight,k,capacity_factor,min_capacity,used_token=None):
    if not 1<=k<=weight.shape[0]:raise ValueError('k must be between 1 and number of experts')
    if k==1:return _top1(x,weight,capacity_factor,min_capacity,used_token)
    if k==2:return _top2(x,weight,capacity_factor,min_capacity)
    return _topk(x,weight,k,capacity_factor,min_capacity)
