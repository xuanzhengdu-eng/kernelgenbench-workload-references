"""vLLM DeepEP local-to-global route conversion and expert block alignment.

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/fused_globalize_align_block_size.py#L10-L63
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/fused_globalize_align_block_size.cu#L28-L126

``recv_topk_idx`` ``[N, K]`` int64 is globalized in place: local ids in
``[0, local_num_experts)`` that also stay below ``global_num_experts`` after
``rank_expert_offset`` are kept; negative ids, out-of-range ids, and rows beyond
``psum_recv_per_rank[-1]`` become ``-1``.  For each local expert the valid
flattened route indices occupy padded blocks; unused ``sorted_ids`` slots hold
the ``numel`` sentinel.  ``expert_ids`` are local expert ids with ``-1`` for
padding, ``num_tokens_post_pad`` is the padded total, and the optional scatter
index maps valid routes to their original flat position (else ``-1``).
The CUDA source uses atomics, so the order inside one expert block is not part
of the contract.  The implementation is a host-side model of the same formula;
the official CUDA kernel is the independent comparison in the test.
"""
import torch


def v1_globalize_align_expert_routes_(
    recv_topk_idx,
    psum_recv_per_rank,
    rank_expert_offset,
    global_num_experts,
    local_num_experts,
    block_size,
    return_scatter_idx=False,
):
    assert recv_topk_idx.ndim == 2 and recv_topk_idx.dtype == torch.int64
    assert recv_topk_idx.is_contiguous() and recv_topk_idx.shape[1] > 0
    assert psum_recv_per_rank.ndim == 1 and psum_recv_per_rank.dtype == torch.int32
    assert psum_recv_per_rank.numel() > 0
    assert psum_recv_per_rank.device == recv_topk_idx.device
    assert 0 < local_num_experts <= 1024 and block_size > 0
    assert rank_expert_offset >= 0 and global_num_experts > 0

    n, k = recv_topk_idx.shape
    numel = n * k
    maxp = numel + local_num_experts * (block_size - 1)
    if numel < local_num_experts:
        maxp = min(numel * block_size, maxp)
    maxblocks = (maxp + block_size - 1) // block_size
    valid_rows = int(psum_recv_per_rank[-1].item())
    assert 0 <= valid_rows <= n

    src = recv_topk_idx.reshape(-1).clone()
    positions = torch.arange(numel, device=src.device)
    valid = (
        (src >= 0)
        & (src < local_num_experts)
        & (src + rank_expert_offset < global_num_experts)
        & (positions // k < valid_rows)
    )
    sorted_ids = torch.full((maxp,), numel, dtype=torch.int32, device=src.device)
    expert_ids = torch.full((maxblocks,), -1, dtype=torch.int32, device=src.device)
    pos = 0
    for expert in range(local_num_experts):
        ids = torch.nonzero(valid & (src == expert), as_tuple=False).flatten().to(torch.int32)
        count = ids.numel()
        padded = (count + block_size - 1) // block_size * block_size
        sorted_ids[pos : pos + count].copy_(ids)
        if padded:
            expert_ids[pos // block_size : (pos + padded) // block_size] = expert
        pos += padded
    total = torch.tensor([pos], dtype=torch.int32, device=src.device)
    recv_topk_idx.copy_(
        torch.where(valid, src + rank_expert_offset, torch.full_like(src, -1)).reshape(n, k)
    )
    out = (recv_topk_idx, sorted_ids, expert_ids, total)
    if return_scatter_idx:
        scatter = (
            torch.where(valid, positions, torch.full_like(positions, -1))
            .to(torch.int32)
            .view(numel, 1)
        )
        out = (*out, scatter)
    return out
