"""Explicit DeepSpeed grouped GEMM backward, including optional gradients.

Official source (DeepSpeed, pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/group_gemm_triton.py#L430-L459

This is the explicit-tuple form of the autograd backward of
``group_gemm_triton`` (``_GroupGemmFn.backward``).  For each group:
  * grad_a = grad_out @ B          (trans_b=True, B is [N, K])
             grad_out @ B^T        (trans_b=False, B is [K, N])
  * grad_b = grad_out^T @ A        (trans_b=True, produced as [N, K])
             A^T @ grad_out        (trans_b=False, produced as [K, N])
Gradients are accumulated in fp32 and cast back to the input dtype.  The
function returns ``(grad_a_or_None, grad_b_or_None, None, None)``; the final two
Nones stand for ``offs`` and ``trans_b``, which are not differentiable.  Inputs
are not modified.
"""
import torch


def v1_ds_grouped_gemm_backward(mat_a, mat_b, offs, grad_out, trans_b=False,
                                need_grad_a=True, need_grad_b=True):
    assert mat_a.ndim == 2 and mat_b.ndim == 3 and grad_out.ndim == 2
    assert mat_a.dtype == mat_b.dtype == grad_out.dtype
    assert offs.ndim == 1 and offs.numel() == mat_b.shape[0]
    assert int(offs[-1]) == mat_a.shape[0]
    grad_a = []
    grad_b = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        a = mat_a[start:end].float()
        g = grad_out[start:end].float()
        w = mat_b[expert].float()
        if need_grad_a:
            grad_a.append((g @ (w if trans_b else w.T)).to(mat_a.dtype))
        if need_grad_b:
            grad_b.append(((g.T @ a) if trans_b else (a.T @ g)).to(mat_b.dtype))
        start = end
    return (torch.cat(grad_a, dim=0) if need_grad_a else None,
            torch.stack(grad_b) if need_grad_b else None,
            None,
            None)
