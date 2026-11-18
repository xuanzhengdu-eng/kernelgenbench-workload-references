"""Stable expert grouping and inverse scatter indices for vLLM CUDA MoE.

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/moe_permute_unpermute_op.cu#L200-L214
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu#L130-L182
Call site:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py#L90-L117

This is the ``moe_prepare_scatter`` contract with caller-owned scratch.  Inputs
are contiguous; ``topk_ids`` is int32 at the native entry (the Python caller
casts int64 first).  Invalid IDs (< 0 or >= n_expert) sort to the sentinel
``n_expert``; ``expert_map[i] == -1`` marks a non-local expert and sorts to
``i + n_expert``; otherwise the mapped local ID is used.  Sorting is stable.
``expert_first_token_offset`` is int64 [n_local_expert + 1] and counts sorted
keys strictly below each local expert index.  ``scatter_idx`` is an int32 view
of length numel() that receives the inverse permutation.  The three diagnostic
scratch buffers receive the preprocessed keys, sorted keys and sorted source
rows for the first numel() entries and are not otherwise specified.  Returns
``(expert_first_token_offset, inverse_view)``.  ``topk_ids`` and
``token_expert_indices`` are not modified.
"""
import torch


def v1_vllm_moe_prepare_scatter(topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, expert_first_token_offset, scatter_idx, sort_workspace, sorted_experts, sorted_rows, topk_ids_for_sort):
    assert topk_ids.ndim == 2 and token_expert_indices.shape == topk_ids.shape
    assert n_expert > 0 and n_local_expert > 0
    size = topk_ids.numel(); shape = topk_ids.shape
    assert expert_first_token_offset.shape == (n_local_expert + 1,) and expert_first_token_offset.dtype == torch.int64
    for x in (scatter_idx, sorted_experts, sorted_rows, topk_ids_for_sort):
        assert x.dtype == torch.int32 and x.numel() >= size and x.is_contiguous()
    assert token_expert_indices.dtype == torch.int32 and sort_workspace.dtype == torch.int8
    inverse = scatter_idx.reshape(-1)[:size].view(shape)
    if size == 0:
        expert_first_token_offset.zero_()
        return expert_first_token_offset, inverse
    raw = topk_ids.to(torch.int32).reshape(-1)
    valid = (raw >= 0) & (raw < n_expert)
    keys = torch.full_like(raw, n_expert)
    if expert_map is None:
        keys[valid] = raw[valid]
    else:
        mapped = expert_map[raw[valid].long()]
        keys[valid] = torch.where(mapped == -1, raw[valid] + n_expert, mapped)
    order = torch.argsort(keys, stable=True)
    skeys = keys[order]; srows = token_expert_indices.reshape(-1)[order]
    topk_ids_for_sort.reshape(-1)[:size].copy_(keys)
    sorted_experts.reshape(-1)[:size].copy_(skeys)
    sorted_rows.reshape(-1)[:size].copy_(srows)
    inverse.reshape(-1).scatter_(0, srows.long(), torch.arange(size, device=raw.device, dtype=torch.int32))
    for i in range(n_local_expert + 1):
        expert_first_token_offset[i] = (skeys < i).sum()
    return expert_first_token_offset, inverse
