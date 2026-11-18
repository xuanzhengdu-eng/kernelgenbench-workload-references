# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b: Megatron inference MoE unpermute_tokens.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/moe/permute.py#L271-L291
Triton kernel:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/moe/permute.py#L243-L269

Contract: `expert_output` is [output_size, hidden], `permuted_probs` is fp32
[output_size] and `permutation_map` is int32 [output_size] mapping each row back
to its source token (-1 marks padding and is skipped). Multiple rows may map to
the same token and are summed. The result is a fully zeroed fp32
[num_tokens, hidden] tensor; the kernel accumulates with atomics. The old
fork-only n_used, valid_tokens and out parameters do not exist upstream and are
not part of this reference.
"""
import torch


def v1_megatron_inference_moe_unpermute(expert_output, permuted_probs,
                                        permutation_map, num_tokens):
    assert expert_output.dim() == 2
    assert permuted_probs.dtype == torch.float32
    assert permuted_probs.numel() == expert_output.shape[0]
    assert permutation_map.numel() == expert_output.shape[0]
    output = torch.zeros((int(num_tokens), expert_output.shape[1]),
                         dtype=torch.float32, device=expert_output.device)
    ids = permutation_map.to(torch.int64)
    valid = ids >= 0
    if valid.any():
        values = expert_output[valid].float() * permuted_probs[valid, None]
        output.index_add_(0, ids[valid], values)
    return output
