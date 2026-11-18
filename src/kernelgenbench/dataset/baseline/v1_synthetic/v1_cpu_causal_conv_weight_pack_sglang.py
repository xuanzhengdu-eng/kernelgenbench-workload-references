"""Bit-preserving CPU causal-conv weight pack from SGLang.

Source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/mamba/conv.cpp#L445-L478
and its Python binding:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp#L951-L952

SGLang's x86 build maps ``block_size_n()`` to 32, so a contiguous CPU FP16/BF16
``weight[N, 4]`` with ``N % 32 == 0`` (including ``N == 0``) is copied into the
``[N/32, 2, 32, 2]`` VNNI payload order and returned as contiguous ``[N, 4]``.
Pairs are opaque 16-bit payloads: the values are never converted.  SGLang
has no ARM ``BLOCK_N = 8`` build branch at this commit; ``BLOCK_N`` is
always ``block_size_n()`` (32 on this x86 host).
"""
import torch


def v1_cpu_causal_conv_weight_pack_sglang(weight):
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
