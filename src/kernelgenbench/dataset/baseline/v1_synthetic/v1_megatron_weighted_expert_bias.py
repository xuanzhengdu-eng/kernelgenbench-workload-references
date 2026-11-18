"""Grouped expert output bias scaled by each routed token probability.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/experts.py#L250-L268

Contract (``TEGroupedMLP._apply_bias``, a staticmethod called from
``TEGroupedMLP.forward`` with ``tokens_per_expert`` already converted to a
``list[int]``):
- ``intermediate_parallel`` is the grouped FC2 output ``[..., H]``;
  ``bias_parallel`` is ``[num_local_experts, H]`` or ``None``;
  ``permuted_probs`` is ``[N, 1]`` and ``sum(tokens_per_expert) == N``.
- With ``bias_parallel is None`` the input object is returned unchanged
  (identity, no copy).
- Otherwise the rows are split per expert, each expert's bias row is multiplied
  elementwise by that expert's probability slice, the chunks are concatenated
  and the original shape restored. The result is cast to the input dtype.
- Inputs are not modified and no output aliases an input.
"""
import torch


def v1_megatron_weighted_expert_bias(
    intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs
):
    if bias_parallel is None:
        return intermediate_parallel
    counts = [int(value) for value in tokens_per_expert]
    assert len(counts) == bias_parallel.shape[0]
    shape = intermediate_parallel.shape
    assert sum(counts) == intermediate_parallel.numel() // shape[-1]
    chunks = torch.split(intermediate_parallel.reshape(-1, shape[-1]), counts, dim=0)
    prob_chunks = torch.split(permuted_probs, counts, dim=0)
    parts = [
        chunk + bias * probs
        for chunk, bias, probs in zip(chunks, bias_parallel, prob_chunks)
    ]
    return torch.cat(parts, dim=0).view(shape).to(intermediate_parallel.dtype)
