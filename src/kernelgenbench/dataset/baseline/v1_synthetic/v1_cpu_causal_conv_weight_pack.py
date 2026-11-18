"""Bit-preserving CPU causal-conv weight pack from vLLM.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/sgl-kernels/conv.cpp#L543-L585
and its Python binding:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/torch_bindings.cpp#L602-L604

The x86 build maps ``block_size_n()`` to 32, so a contiguous CPU FP16/BF16
``weight[N, 4]`` with ``N % 32 == 0`` (including ``N == 0``) is copied into the
``[N/32, 2, 32, 2]`` VNNI payload order and returned as contiguous ``[N, 4]``.
Pairs are opaque 16-bit payloads: the values are never converted.  The
``ARM_BF16_SUPPORT`` build uses ``BLOCK_N = 8``; that is a compile-time build
variant of the same C++ entry, not a parameter of the pinned x86 operator.
"""
import torch


def v1_cpu_causal_conv_weight_pack(weight):
    if weight.device.type != "cpu" or not weight.is_contiguous():
        raise ValueError("CPU contiguous weight required")
    if weight.ndim != 2 or weight.shape[1] != 4:
        raise ValueError("expected [N,4]")
    if weight.shape[0] % 32:
        raise ValueError("N must be divisible by 32")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected fp16 or bf16")
    if weight.numel() == 0:
        return torch.empty_like(weight)
    n = weight.shape[0]
    return (
        weight.reshape(n // 32, 32, 2, 2)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(weight.shape)
    )
