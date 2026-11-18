# Source-checked synthetic-workload benchmark reference.
"""OP-c0f3b4fcda53: Megatron training MoE permute, non-fused path.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L302-L429

Contract (fused=False, as in the fixed NVIDIA commit): returns
(permuted_tokens, permuted_probs, sorted_indices, None, tokens_per_expert).
`routing_map` is [num_tokens, num_experts] bool/int. With drop_and_pad=True,
num_out_tokens // num_experts rows are taken per expert (argsort, stable);
otherwise the first num_out_tokens non-zero expert-major entries are used, so
num_out_tokens may be smaller than the number of routed pairs. The last element
is the `tokens_per_expert` object passed in, unchanged. The fused Transformer
Engine path is a separate upstream backend and is not implemented here.
"""
import torch


def _expert_major(routing_map):
    return routing_map.bool().T.contiguous()


def v1_megatron_moe_permute(tokens, routing_map, probs=None, num_out_tokens=None,
                            fused=False, drop_and_pad=False,
                            tokens_per_expert=None, align_size=-1):
    assert not fused, "the fused Transformer Engine branch is not this reference"
    assert tokens.dim() == 2 and routing_map.dim() == 2
    assert routing_map.shape[0] == tokens.shape[0]
    num_tokens, num_experts = routing_map.shape
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        ranked = routing_map.to(torch.int8).T.contiguous().argsort(
            dim=-1, descending=True, stable=True
        )[:, :capacity].contiguous()
        sorted_indices = ranked.view(-1)
        if probs is not None:
            probs_t = probs.T.contiguous().view(-1)
            experts = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            picked = (experts * num_tokens + sorted_indices.view(num_experts, capacity)).view(-1)
            permuted_probs = probs_t.index_select(0, picked)
    else:
        assert num_out_tokens is not None, "num_out_tokens is required for the argsort permute"
        flat_sorted = _expert_major(routing_map).reshape(-1).argsort(descending=True, stable=True)
        flat_sorted = flat_sorted[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens
        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
    permuted_tokens = tokens.index_select(0, sorted_indices)
    return permuted_tokens, permuted_probs, sorted_indices, None, tokens_per_expert
