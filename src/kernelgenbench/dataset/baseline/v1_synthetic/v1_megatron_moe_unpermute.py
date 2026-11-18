# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b: Megatron training MoE unpermute, non-fused path.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L431-L531

Contract (fused=False): restores `permuted_tokens` to `restore_shape`. With
`probs`, the per-pair probabilities are selected from the expert-major routing
map (drop_and_pad uses the fixed per-expert capacity layout) and applied before
the scatter-add. The result has the dtype of `permuted_tokens`; accumulation
follows the fixed source (scatter_add_, or index_add_ when deterministic
algorithms are enabled). `pad_offsets` is accepted for signature compatibility
with the fixed commit but only the fused Transformer Engine path consumes it.
"""
import torch


def v1_megatron_moe_unpermute(permuted_tokens, sorted_indices, restore_shape,
                              probs=None, routing_map=None, fused=False,
                              drop_and_pad=False, pad_offsets=None):
    assert not fused, "the fused Transformer Engine branch is not this reference"
    assert permuted_tokens.dim() == 2
    input_dtype = permuted_tokens.dtype
    _, hidden = restore_shape
    if probs is not None:
        assert routing_map is not None, "routing_map must be provided to apply probs"
        if drop_and_pad:
            num_experts = routing_map.size(1)
            capacity = sorted_indices.size(0) // num_experts
            probs_t = probs.T.contiguous().view(-1)
            experts = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            picked = (experts * probs.size(0) + sorted_indices.view(num_experts, capacity)).view(-1)
            permuted_probs = probs_t.index_select(0, picked)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous().bool())
        assert permuted_probs.numel() == sorted_indices.numel()
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
    output = torch.zeros(restore_shape, dtype=permuted_tokens.dtype,
                         device=permuted_tokens.device)
    if torch.are_deterministic_algorithms_enabled():
        output.index_add_(0, sorted_indices, permuted_tokens)
    else:
        output.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output.to(dtype=input_dtype)
