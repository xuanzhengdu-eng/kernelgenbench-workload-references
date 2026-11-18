"""DeepSpeed scalar CPU Adam/AdamW single dense tensor update.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L35-L87
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L251-L277
FP32 internal arithmetic, optional bias correction and coupled Adam or decoupled
AdamW positive weight decay. Mutates parameter, first and second moment buffers;
grad unchanged. Returns C++ status 0. Contiguous CPU tensors; param/grad dtype
FP32/FP16/BF16, both moments FP32 or matching dtype; distinct storage.
This is the scalar dense entry, excluding multi-tensor GPU and sparse/optimizer wrappers.
"""
import torch

def v1_cpu_adam_update_(param,grad,exp_avg,exp_avg_sq,step,lr,beta1,beta2,eps,weight_decay,adamw,bias_correction):
    ts=(param,grad,exp_avg,exp_avg_sq);assert all(t.device.type=='cpu' and t.is_contiguous() and t.shape==param.shape for t in ts)
    assert param.dtype==grad.dtype and param.dtype in (torch.float32,torch.float16,torch.bfloat16)
    assert exp_avg.dtype==exp_avg_sq.dtype and exp_avg.dtype in (torch.float32,param.dtype) and step>=1 and lr>=0 and eps>0 and 0<=beta1<1 and 0<=beta2<1
    p=param.float();g=grad.float();m=exp_avg.float();v=exp_avg_sq.float()
    if weight_decay>0 and not adamw:g=p*weight_decay+g
    m=m*beta1+g*(1-beta1);v=v*beta2+g*g*(1-beta2)
    if bias_correction:
        b1=torch.tensor(beta1,dtype=torch.float32);b2=torch.tensor(beta2,dtype=torch.float32)
        c1=1-b1.pow(step);c2=torch.rsqrt(1-b2.pow(step))
    else:c1=1.;c2=1.
    update=m/(v.sqrt()*c2+eps)*(-lr/c1)
    if weight_decay>0 and adamw:p=p+(-lr*weight_decay)*p
    with torch.no_grad():param.copy_(p+update);exp_avg.copy_(m);exp_avg_sq.copy_(v)
    return 0
