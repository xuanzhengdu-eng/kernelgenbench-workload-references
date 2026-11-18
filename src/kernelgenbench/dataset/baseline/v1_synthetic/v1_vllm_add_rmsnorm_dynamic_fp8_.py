"""vLLM fused residual-add + RMSNorm + dynamic per-token FP8 quant (no static scale).

Official python entry:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L435-L451
Official CUDA kernel (residual is added and written back, scale = amax/448 clamped):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu#L12-L110
Official scale computation:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/fused_kernels/layernorm_utils.cuh#L59-L173
"""
import torch


def v1_vllm_add_rmsnorm_dynamic_fp8_(inp, residual, weight, epsilon, scale_ub=None):
    h = inp.shape[-1]
    x = inp.reshape(-1, h).float() + residual.reshape(-1, h).float()
    norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * weight.float()
    amax = norm.abs().amax(dim=-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub.float())
    scale = (amax / 448.0).clamp_min(1.0 / (448.0 * 512.0))
    quant = (norm / scale).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(inp.shape)
    residual.copy_(x.reshape_as(residual))
    return quant, scale
