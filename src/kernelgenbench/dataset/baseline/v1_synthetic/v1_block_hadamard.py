"""vLLM input block-Hadamard rotation (723-row 99).

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/input_hadamard.py#L24-L42

Official entries:
  * ``hadamard_matrix(size, device)``: orthonormal Sylvester matrix built by the
    doubling recursion and divided by ``sqrt(size)``; fp32, ``size`` a power of
    two.
  * ``block_hadamard_rotate(x, hadamard)``: rotates every contiguous
    ``size``-wide block of the last axis in fp32 and returns the input dtype.

The rotation is always orthonormal and the result always carries ``x.dtype``.
The earlier migration flags ``normalized=False`` and ``output_fp32=True`` had no
upstream counterpart (neither vLLM entry exposes them) and were removed.
"""
import torch


def v1_block_hadamard(x, block):
    assert block > 0 and block & (block - 1) == 0, "block must be a power of two"
    assert x.shape[-1] % block == 0, "block must divide the last dimension"
    h = torch.ones((1, 1), device=x.device, dtype=torch.float32)
    while h.shape[0] < block:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    h = h / block ** 0.5
    return (x.float().unflatten(-1, (-1, block)) @ h).flatten(-2).to(x.dtype)
