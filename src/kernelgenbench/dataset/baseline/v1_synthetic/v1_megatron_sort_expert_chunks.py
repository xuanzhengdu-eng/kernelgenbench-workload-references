"""Reorder expert chunks and the aligned probabilities together.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L533-L576

Contract (``sort_chunks_by_idxs`` eager branch, ``fused=False``):
- ``input`` is ``[sum(split_sizes), ...]``; ``split_sizes`` are the chunk
  lengths; ``sorted_idxs`` is a permutation of the chunk indices giving the
  output order (the caller guarantees the permutation).
- Output rows are the concatenation of the selected chunks; the optional
  ``probs`` tensor is split by the same sizes and reordered identically.
- Returns ``(output, permuted_probs)``; ``permuted_probs is None`` when
  ``probs is None``. Inputs are not modified; outputs are fresh tensors.
- Empty chunks (size 0) are legal and contribute no rows.
"""
import torch


def v1_megatron_sort_expert_chunks(input, split_sizes, sorted_idxs, probs=None):
    sizes = [int(value) for value in split_sizes.tolist()]
    order = [int(value) for value in sorted_idxs.tolist()]
    assert sum(sizes) == input.shape[0]
    assert sorted(order) == list(range(len(sizes)))
    chunks = torch.split(input, sizes, dim=0)
    output = torch.cat([chunks[index] for index in order], dim=0)
    if probs is None:
        return output, None
    prob_chunks = torch.split(probs, sizes, dim=0)
    return output, torch.cat([prob_chunks[index] for index in order], dim=0)
