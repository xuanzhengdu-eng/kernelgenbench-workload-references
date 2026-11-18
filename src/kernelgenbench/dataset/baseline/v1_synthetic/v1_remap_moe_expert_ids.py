"""Map global MoE top-k IDs to local IDs, preserving invalid/nonlocal markers.

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L262-L305

Contract: ``topk_ids`` is contiguous int32/int64, optionally shaped; ``-1``
marks an invalid route.  ``expert_map`` is a 1-D global -> local map whose
``-1`` marks a non-local expert.  The result is a newly allocated tensor with
the same shape as ``topk_ids`` and the requested int32/int64 ``out_dtype``:
invalid routes become -1, valid routes become ``expert_map[id]`` (including a
mapped -1).  Inputs are not modified.  The source uses a Triton kernel.
"""
import torch


def v1_remap_moe_expert_ids(topk_ids, expert_map, out_dtype=torch.int64):
    assert topk_ids.dtype in (torch.int32, torch.int64) and expert_map.dtype in (torch.int32, torch.int64)
    assert topk_ids.is_contiguous() and expert_map.is_contiguous() and expert_map.ndim == 1 and expert_map.numel() > 0 and expert_map.device == topk_ids.device
    if out_dtype not in (torch.int32, torch.int64):
        raise ValueError('out_dtype must be int32 or int64')
    valid = topk_ids >= 0
    idx = topk_ids.clamp(min=0).long()
    mapped = expert_map[idx].to(out_dtype)
    return torch.where(valid, mapped, torch.full_like(mapped, -1))
