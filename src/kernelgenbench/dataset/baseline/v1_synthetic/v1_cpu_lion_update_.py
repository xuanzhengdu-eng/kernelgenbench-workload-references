"""DeepSpeed scalar CPU Lion single dense tensor update, official C++ contract.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lion/cpu_lion_impl.cpp#L22-L65
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lion/cpu_lion_impl.cpp#L192-L214

The reference reproduces ``ds_lion_step``:
* all arithmetic in FP32; ``param``/``grad`` are contiguous CPU
  FP32/FP16/BF16, ``state`` is FP32 or the parameter dtype;
* ``mix = beta1*state + (1-beta1)*grad``; ``step = -copysign(lr, mix)``;
* with positive weight decay, ``param = param*(1-lr*decay) + step``; otherwise
  ``param = param + step``;
* ``state = beta2*state + (1-beta2)*grad``; ``grad`` is not modified;
* the C++ entry returns status ``0``.

``copysign`` (not ``sign``) is part of the contract: it preserves the direction
of signed zero exactly as the C++ kernel does.
"""
import torch


def v1_cpu_lion_update_(param, grad, state, step, lr, beta1, beta2, weight_decay):
    assert (
        param.device.type == grad.device.type == state.device.type == "cpu"
        and param.is_contiguous()
        and grad.is_contiguous()
        and state.is_contiguous()
    )
    assert param.shape == grad.shape == state.shape and param.dtype == grad.dtype
    assert param.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert state.dtype in (torch.float32, param.dtype)
    assert step >= 1 and lr >= 0 and 0 <= beta1 <= 1 and 0 <= beta2 <= 1
    p = param.float()
    g = grad.float()
    m = state.float()
    mix = m * beta1 + g * (1 - beta1)
    update = -torch.copysign(torch.full_like(mix, lr), mix)
    updated = (p * (1 - lr * weight_decay) + update) if weight_decay > 0 else p + update
    momentum = m * beta2 + g * (1 - beta2)
    with torch.no_grad():
        param.copy_(updated)
        state.copy_(momentum)
    return 0
