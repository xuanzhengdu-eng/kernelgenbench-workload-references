"""Megatron TensorParallelMuon.orthogonalize (single rank, FP32 / highest).

The orthogonalization is Emerging-Optimizers v0.2.0: normalize by the Frobenius
norm with a 1e-7 floor, run the quintic Newton-Schulz iteration (coefficient sets
cycled), transposing so the smaller dimension is orthogonalized.  Muon then scales
by ``max(rows, cols) ** 0.5`` (spectral), ``(rows/cols) ** 0.5`` (unit_rms_norm) or
``max(1, rows/cols) ** 0.5`` (shape_scaling), times ``extra_scale``.

``qkv`` mirrors the Megatron wrapper's ``split_qkv`` branch: rows are grouped by
query group, each Q/K/V block is orthogonalized on its own and the blocks are
concatenated back.  The output is a new tensor; the gradient and any momentum are
untouched.  Multi-rank TP, BF16 "medium" matmul and other coefficient sets are out
of scope for this item.
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L30-L64
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L114-L207
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L210-L272
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon.py#L136-L162
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/muon.py#L137-L188
"""
import torch
_COEFF=((4.0848,-6.8946,2.9270),(3.9505,-6.3029,2.6377),
 (3.7418,-5.5913,2.3037),(2.8769,-3.1427,1.2046),(2.8366,-3.0525,1.2012))

def v1_megatron_muon_orthogonal(grad,qkv,steps,scale_mode,extra_scale):
    if grad.ndim!=2 or grad.dtype!=torch.float32 or steps<1:raise ValueError('expected 2D fp32 and positive steps')
    if scale_mode not in ('spectral','unit_rms_norm','shape_scaling'):raise ValueError('invalid scale mode')
    def orth(x):
        r,c=x.shape;trans=r>c
        if trans:x=x.t()
        x=x/torch.linalg.vector_norm(x).clamp_min(1e-7)
        for i in range(steps):
            a,b,c0=_COEFF[i%len(_COEFF)];A=x@x.t();B=torch.addmm(A,A,A,alpha=c0,beta=b);x=torch.addmm(x,B,x,beta=a)
        if trans:x=x.t()
        factor=max(r,c)**.5 if scale_mode=='spectral' else ((r/c)**.5 if scale_mode=='unit_rms_norm' else max(1,r/c)**.5)
        return x*factor*extra_scale
    if qkv is None:return orth(grad)
    if len(qkv)!=3 or any(v<=0 for v in qkv) or grad.shape[0]%sum(qkv):raise ValueError('invalid QKV sizes')
    groups=grad.shape[0]//sum(qkv);parts=grad.reshape(groups,sum(qkv),grad.shape[1]).split(qkv,1)
    parts=[orth(v.reshape(-1,grad.shape[1])).reshape(groups,-1,grad.shape[1]) for v in parts]
    return torch.cat(parts,1).reshape_as(grad)
