"""DeepSpeed generalized JSD/KL loss over masked response tokens.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/rollout/opsd.py#L71-L133
Student/teacher logits [B,T,V], response mask [B,T]. Optional teacher top-k
and pointwise clipping. Returns (mean, sum, valid count), the source dataclass
unwrapped to a tuple. Explicit source backward is absent; autograd is not split.
"""
import torch
import torch.nn.functional as F

def v1_ds_generalized_jsd(student_logits,teacher_logits,response_mask,beta=.5,temperature=1.,teacher_top_k=None,pointwise_clip=None):
    if student_logits.shape!=teacher_logits.shape or student_logits.ndim!=3:raise ValueError('logit shapes')
    if response_mask.shape!=student_logits.shape[:2]:raise ValueError('mask shape')
    if not 0<=beta<=1 or temperature<=0:raise ValueError('beta or temperature')
    if teacher_top_k is not None and not 0<teacher_top_k<=student_logits.shape[-1]:raise ValueError('top-k')
    if pointwise_clip is not None and pointwise_clip<0:raise ValueError('clip')
    s=student_logits/temperature;t=teacher_logits/temperature
    if teacher_top_k is not None:
        ix=torch.topk(t,teacher_top_k,dim=-1).indices;s=torch.gather(s,-1,ix);t=torch.gather(t,-1,ix)
    sl=F.log_softmax(s,-1);tl=F.log_softmax(t,-1);sp=sl.exp();tp=tl.exp()
    if beta==0:point=tp*(tl-sl)
    elif beta==1:point=sp*(sl-tl)
    else:
        mixture=torch.logaddexp(sl+torch.log1p(torch.tensor(-beta,dtype=s.dtype,device=s.device)),tl+torch.log(torch.tensor(beta,dtype=s.dtype,device=s.device)))
        point=beta*tp*(tl-mixture)+(1-beta)*sp*(sl-mixture)
    if pointwise_clip is not None:point=point.clamp(max=pointwise_clip)
    mask=response_mask.to(point.dtype);total=(point.sum(-1)*mask).sum();count=mask.sum().to(torch.long)
    return total/count.clamp_min(1),total,count
