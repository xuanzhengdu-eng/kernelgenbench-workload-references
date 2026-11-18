"""Megatron inference MoE token permutation into aligned expert-major blocks.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/moe/permute.py#L170-L241

Contract (``megatron.core.inference.moe.permute.permute_tokens``):
- ``hidden_states`` is ``[num_tokens, hidden_size]``, ``probs`` and
  ``routing_map`` are ``[num_tokens, topk]``; ``routing_map`` holds global
  expert ids, ``local_expert_start`` is this rank's first global expert.
- The output buffer is sized ``num_tokens * min(topk, num_local_experts) +
  alignment * num_local_experts``. Token counts are rounded up per expert to a
  positive multiple of ``alignment``; zero-count experts keep a zero block.
- Returns ``(permuted_hidden, permuted_probs, permutation_map,
  inclusive_offsets)``. ``permutation_map`` is int32 with ``-1`` in padding
  slots; ``inclusive_offsets`` is the int32 inclusive prefix sum of aligned
  counts.
- The kernel claims a slot per (token, topk) pair with ``atomic_add``, so the
  order of live rows *within* one expert block is not part of the contract.
  Padding payload in ``permuted_hidden``/``permuted_probs`` is uninitialized.
- Neither input is modified. This fixed Megatron entry has no
  ``valid_tokens`` argument: every input token is permuted, unlike the removed
  fork reference which truncated to ``valid_tokens``.
"""
import torch


def v1_megatron_inference_moe_permute(
    hidden_states,
    probs,
    routing_map,
    local_expert_start,
    num_local_experts,
    alignment=1,
):
    num_tokens, hidden_size = hidden_states.shape
    topk = probs.shape[1]
    assert routing_map.shape == probs.shape
    assert num_local_experts > 0 and alignment > 0
    counts = []
    for expert in range(num_local_experts):
        count = int((routing_map == local_expert_start + expert).sum().item())
        if count > 0:
            count = ((count + alignment - 1) // alignment) * alignment
        counts.append(count)
    offsets = []
    running = 0
    for count in counts:
        running += count
        offsets.append(running)
    output_size = num_tokens * min(topk, num_local_experts) + alignment * num_local_experts

    permuted_hidden = torch.empty(
        (output_size, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
    )
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    permutation_map = torch.full((output_size,), -1, dtype=torch.int32, device=probs.device)

    cursor = 0
    for expert in range(num_local_experts):
        for token, slot in torch.nonzero(
            routing_map == local_expert_start + expert, as_tuple=False
        ).tolist():
            permuted_hidden[cursor] = hidden_states[token]
            permuted_probs[cursor] = probs[token, slot]
            permutation_map[cursor] = token
            cursor += 1
        cursor = offsets[expert]

    return (
        permuted_hidden,
        permuted_probs,
        permutation_map,
        torch.tensor(offsets, dtype=torch.int32, device=probs.device),
    )
