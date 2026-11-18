"""vLLM dense SiLU-and-mul followed by per-token group FP8 quantization.

Official sources (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L364-L469
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L472-L565

The packed input [M, 2H] is split into gate = input[:, :H] and up = input[:, H:].
After the optional clamp (gate <= clamp_limit, |up| <= clamp_limit), the gate is
silu(gate * alpha) and up is shifted by beta; both are narrowed to the input
dtype before the product, exactly like the Triton kernel. The product is then
quantized per group of `group_size` columns to FP8 E4M3 with a float32 scale.
`use_ue8m0` rounds the scale up to a power of two. When `output` is supplied the
quantized values are written into it; `scales` is always a fresh [M, H/group]
float32 tensor with column-major (1, M) strides.
"""
import torch


def v1_vllm_silu_group_fp8(input, output=None, use_ue8m0=False, eps=1e-10,
                           clamp_limit=None, group_size=128, alpha=1.0, beta=0.0):
    m, n = input.shape
    h = n // 2
    assert m % group_size == 0 and h % group_size == 0
    a, b = input.chunk(2, -1)
    if clamp_limit is not None:
        a = a.float().clamp(max=clamp_limit).to(input.dtype)
        b = b.float().clamp(-clamp_limit, clamp_limit).to(input.dtype)
    glu = (a.float() * torch.sigmoid(a.float() * alpha)).to(input.dtype)
    up = (b.float() + beta).to(input.dtype)
    y = (glu * up).float().reshape(m, h // group_size, group_size)
    scale = y.abs().amax(dim=-1).clamp(min=eps) / 448.0
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (y / scale.unsqueeze(-1)).clamp(-448, 448).reshape(m, h).to(torch.float8_e4m3fn)
    if output is None:
        output = q
    else:
        output.copy_(q)
    scales = torch.empty((h // group_size, m), dtype=torch.float32,
                         device=input.device).t()
    scales.copy_(scale)
    return output, scales
