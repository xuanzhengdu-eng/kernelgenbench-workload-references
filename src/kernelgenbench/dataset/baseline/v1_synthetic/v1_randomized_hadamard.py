"""vLLM deterministic signed Hadamard rotation, forward or inverse order.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/int4_per_token_head.py#L1132-L1167

``single_rht(x, inverse)`` builds a deterministic +/-1 sign vector of length
``block = largest_power_of_2_divisor(d)`` from the fixed CPU seed
``0x9E3779B9`` and applies the unnormalized Hadamard matrix block-diagonally when
``d`` is not a power of two.  Forward is ``H (D x)``, inverse is ``(H x) D``.
On the pinned CUDA build the dispatch selects the hadacore Tensor Core kernel for
every tested ``d``; that kernel rounds the input through bfloat16 and rescales,
so its fp32 output differs from an exact fp32 matmul by up to ~0.1 absolute.
The reference keeps the exact fp32 butterfly/matmul arithmetic; the test allows
for the documented hadacore precision.
"""
import torch


def v1_randomized_hadamard(x, inverse):
    d = x.shape[-1]
    block = d & -d
    gen = torch.Generator(device='cpu').manual_seed(0x9E3779B9)
    signs = (2. * torch.bernoulli(torch.full((block,), .5, device='cpu'),
                                  generator=gen) - 1.).to(x.device)
    h = torch.ones((1, 1), device=x.device, dtype=torch.float32)
    while h.shape[0] < block:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    pieces = x.reshape(*x.shape[:-1], -1, block)
    out = (pieces @ h) * signs if inverse else (pieces * signs) @ h
    return out.reshape(x.shape)
