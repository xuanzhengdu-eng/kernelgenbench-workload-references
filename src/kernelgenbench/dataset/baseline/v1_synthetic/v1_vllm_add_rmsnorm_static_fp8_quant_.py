"""vLLM fused in-place residual add + RMSNorm + static per-tensor FP8 quant.

Official python binding (result is written into ``out``; returns nothing):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/torch_bindings.cpp#L376-L384
Official CUDA kernel (residual written back, norm rounded through the activation
dtype before the static FP8 conversion):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/layernorm_quant_kernels.cu#L160-L204
"""
import torch


def v1_vllm_add_rmsnorm_static_fp8_quant_(inp, residual, weight, epsilon, static_scale, out):
    assert static_scale.numel() == 1 and static_scale.dtype == torch.float32
    target = residual
    target.copy_((inp.float() + target.float()).to(inp.dtype))
    x = target.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    normalized = (x * inv * weight.float()).to(inp.dtype).float()
    q = (normalized / static_scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    out.copy_(q)
    return None
