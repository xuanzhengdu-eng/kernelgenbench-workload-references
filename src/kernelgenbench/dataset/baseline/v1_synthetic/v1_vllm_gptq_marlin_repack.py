"""Repack GPTQ-layout packed int32 weights into the Marlin (GPTQ) tile layout.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L1110-L1116
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/marlin/gptq_marlin_repack.cu#L15-L258

Contract (official ``gptq_marlin_repack``): the GPTQ input is
``[size_k / pack_factor, size_n]`` int32, K-major: the low ``num_bits`` bits of
each word hold consecutive K elements of one output column. ``size_k`` is a
multiple of 16 and ``size_n`` a multiple of 64. Output is int32
``[size_k / 16, size_n * 16 / pack_factor]`` with the per-16-row/8-column tile
order the Marlin GEMM reads. A 3-D input is handled by the official
``gptq_marlin_moe_repack``, which runs the same kernel per expert and stacks the
results; the contract per expert is identical.
"""
import torch


def _repack_expert(b_q_weight, size_k, size_n, num_bits):
    assert b_q_weight.ndim == 2 and b_q_weight.is_contiguous()
    assert b_q_weight.dtype == torch.int32
    assert num_bits in (4, 8)
    assert size_k > 0 and size_k % 16 == 0 and size_n > 0 and size_n % 64 == 0
    pack = 32 // num_bits
    assert b_q_weight.shape == (size_k // pack, size_n)
    words = b_q_weight.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack(
        [(words >> (i * num_bits)) & ((1 << num_bits) - 1) for i in range(pack)], dim=1
    ).reshape(size_k, size_n)
    kt = torch.arange(size_k // 16, device=b_q_weight.device).view(-1, 1, 1, 1)
    nt = torch.arange(size_n // 64, device=b_q_weight.device).view(1, -1, 1, 1)
    tid = torch.arange(32, device=b_q_weight.device).view(1, 1, -1, 1)
    warp = torch.arange(4, device=b_q_weight.device).view(1, 1, 1, -1)
    r = kt * 16 + (tid % 4) * 2
    c = nt * 64 + warp * 16 + tid // 4
    vals = [q[r + off, c] for off in (0, 1, 8, 9)] + [q[r + off, c + 8] for off in (0, 1, 8, 9)]
    if num_bits == 4:
        out = torch.zeros_like(vals[0])
        for i, j in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
            out |= vals[j] << (4 * i)
    else:
        first = torch.zeros_like(vals[0])
        second = torch.zeros_like(vals[0])
        for i, j in enumerate((0, 2, 1, 3)):
            first |= vals[j] << (8 * i)
            second |= vals[4 + j] << (8 * i)
        out = torch.stack((first, second), dim=-1)
    return out.reshape(size_k // 16, size_n * 16 // pack).to(torch.int32).contiguous()


def v1_vllm_gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits):
    if b_q_weight.ndim == 2:
        return _repack_expert(b_q_weight, size_k, size_n, num_bits)
    assert b_q_weight.ndim == 3 and b_q_weight.is_contiguous()
    return torch.stack(
        [_repack_expert(b_q_weight[e], size_k, size_n, num_bits) for e in range(b_q_weight.shape[0])]
    )
