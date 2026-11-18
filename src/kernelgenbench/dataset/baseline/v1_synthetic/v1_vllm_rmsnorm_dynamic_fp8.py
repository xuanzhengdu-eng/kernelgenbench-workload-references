# Source-checked reference for KernelGenBench row 64.
"""vLLM no-residual dynamic per-token FP8 RMSNorm quant; finite FP32 input.

Official sources (vLLM fixed commit f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L435-L451
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu#L12-L97
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/fused_kernels/layernorm_utils.cuh#L61-L135
"""
import torch


def v1_vllm_rmsnorm_dynamic_fp8(inp, weight, epsilon, scale_ub=None):
    h = inp.shape[-1]
    x = inp.reshape(-1, h).float()
    w = weight.float()
    norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * w
    amax = norm.abs().amax(dim=-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub.float())
    scale = (amax / 448.0).clamp_min(1.0 / (448.0 * 512.0))
    quant = (norm / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(inp.shape)
    return quant, scale
