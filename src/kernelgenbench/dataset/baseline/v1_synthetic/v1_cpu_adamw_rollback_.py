"""DeepSpeed CPU AdamW update rollback on FP32 tensor states.

Source (official fixed commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L279-L325
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L327-L373

Always applies bias correction, even when its bias_correction argument is False.
Updates params, exp_avg and exp_avg_sq in-place and preserves grad. Nonfinite
rolled-back parameter values are replaced by zero; moments are not sanitized.
All tensors contiguous CPU; unsupported dtypes return -1 without mutation.
Optimizer handle and decremented internal local step are not externally observed.
"""
import torch

def v1_cpu_adamw_rollback_(param,grad,exp_avg,exp_avg_sq,step,lr,beta1,beta2,eps,weight_decay,bias_correction):
    ts=(param,grad,exp_avg,exp_avg_sq);assert all(t.device.type=='cpu' and t.is_contiguous() and t.shape==param.shape for t in ts)
    if any(t.dtype!=torch.float32 for t in ts):return -1
    assert step>=1 and 0<beta1<1 and 0<beta2<1
    b1=torch.tensor(beta1,dtype=torch.float32);b2=torch.tensor(beta2,dtype=torch.float32)
    c1=1-b1.pow(step);c2=1-b2.pow(step)
    update=lr*(exp_avg/c1)/((exp_avg_sq/c2).sqrt()+eps)
    p=(param+update)/(1-lr*weight_decay);p=torch.where(torch.isfinite(p),p,torch.zeros_like(p))
    m=(exp_avg-(1-beta1)*grad)/beta1;v=(exp_avg_sq-(1-beta2)*grad*grad)/beta2
    with torch.no_grad():param.copy_(p);exp_avg.copy_(m);exp_avg_sq.copy_(v)
    return 0
