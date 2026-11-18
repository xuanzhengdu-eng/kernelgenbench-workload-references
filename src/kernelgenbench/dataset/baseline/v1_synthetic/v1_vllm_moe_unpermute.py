# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b (split branch): vLLM moe_unpermute.

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py#L285-L320
Native kernel:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl#L93-L166

Contract: `out` is overwritten in place and None is returned. `topk_weights` is
float32 [num_tokens, topk], `inv_permuted_idx` is int32 [num_tokens, topk] and
`permuted_hidden_states` is [num_tokens * topk, hidden] with the same dtype as
`out`. When `expert_first_token_offset` is given, its last element is the number
of valid rows and larger indices are skipped; otherwise every index is read.
The kernel accumulates in fp32 and requires hidden * element_size % 16 == 0.
"""
import torch


def v1_vllm_moe_unpermute(out, permuted_hidden_states, topk_weights,
                           inv_permuted_idx, expert_first_token_offset=None):
    assert out.dim() == 2 and permuted_hidden_states.dim() == 2
    num_tokens, topk = topk_weights.shape
    hidden = permuted_hidden_states.shape[1]
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == permuted_hidden_states.dtype
    assert topk_weights.dtype == torch.float32
    assert inv_permuted_idx.dtype == torch.int32
    assert (hidden * permuted_hidden_states.element_size()) % 16 == 0
    num_rows = permuted_hidden_states.shape[0]
    num_valid = num_rows
    if expert_first_token_offset is not None:
        num_valid = int(expert_first_token_offset[-1])
    for token in range(num_tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=out.device)
        for slot in range(topk):
            index = int(inv_permuted_idx[token, slot])
            assert index >= 0
            if index < num_valid:
                acc = acc + permuted_hidden_states[index].float() * topk_weights[token, slot]
        out[token].copy_(acc.to(out.dtype))
    return None
