"""DeepSpeed CPU Adagrad single dense tensor update.

Source (official fixed commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L20-L55
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L178-L199
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/cpu_adagrad.h#L65-L117
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/simd.h#L34-L49

Contiguous CPU param and grad of equal dtype and shape, separate state of float32
or matching parameter dtype (float32/fp16/bf16).  FP32 math: positive weight
decay adds decay*param to grad; state += grad**2; update = -lr*grad/(sqrt(state)+eps).
Mutates param and state and returns integer 0.

The compiled x86 AVX512 extension splits the tensor at
``SIMD_WIDTH=16``: the vectorized prefix updates param/state but does not write
the update term back to grad; only the scalar tail (``numel % 16`` elements)
replaces grad with the update term, exactly as ``Adagrad_Optimizer::Step_1``
does after ``Step_AVX``.  The reference models that pinned x86 build.  Optimizer
object setup, the sparse branch, the XPU copy branch, and noncontiguous inputs
(which the source silently copies) are out of scope.
"""
import torch

# csrc/includes/simd.h: AVX512 -> SIMD_WIDTH 16.  The pinned H800 host uses the
# AVX512 build; the scalar tail is what Step_1 processes after Step_AVX.
_SIMD_WIDTH = 16


def v1_cpu_adagrad_update_(param, grad, state, step, lr, eps, weight_decay):
    assert param.device.type == grad.device.type == state.device.type == 'cpu'
    assert param.is_contiguous() and grad.is_contiguous() and state.is_contiguous()
    assert param.shape == grad.shape == state.shape and param.dtype == grad.dtype
    assert param.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert state.dtype in (torch.float32, param.dtype) and step >= 1 and lr >= 0 and eps > 0
    p = param.float()
    g = grad.float()
    v = state.float()
    if weight_decay > 0:
        g = g + p * weight_decay
    v = v + g * g
    update = (-lr) * (g / (torch.sqrt(v) + eps))
    rounded = (p.numel() // _SIMD_WIDTH) * _SIMD_WIDTH
    with torch.no_grad():
        param.copy_(p + update)
        state.copy_(v)
        if p.numel() > rounded:
            grad.view(-1)[rounded:].copy_(update.view(-1)[rounded:])
    return 0
