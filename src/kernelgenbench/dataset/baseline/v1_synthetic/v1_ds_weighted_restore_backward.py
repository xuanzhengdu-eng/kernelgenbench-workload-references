# Source-checked synthetic-workload benchmark reference.
"""OP-24a15a06532c: DeepSpeed fused weighted restore backward (explicit helper).

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L287-L349

`_FusedWeightedRestore.backward` returns `(grad_rows, grad_scores, None, None)`.
`combined_rows` is [num_tokens * top_k, hidden], `top_scores` is
[num_tokens, top_k] float32 and `token_indices_sorted` is the forward row
permutation (a complete permutation of 0 .. num_tokens * top_k - 1). For each
top-k slot the official kernel writes the token's gradient into the source row
and reduces the score gradient in fp32. The two trailing None values mirror the
backward signature for the non-tensor inputs.
"""
import torch


def v1_ds_weighted_restore_backward(grad_output, combined_rows, top_scores,
                                    token_indices_sorted, top_k):
    assert grad_output.dim() == 2 and combined_rows.dim() == 2
    num_tokens, hidden = grad_output.shape
    assert top_scores.shape == (num_tokens, top_k)
    assert combined_rows.shape == (num_tokens * top_k, hidden)
    inverse = torch.argsort(token_indices_sorted.to(torch.int64)).view(num_tokens, top_k)
    grad_rows = torch.empty_like(combined_rows)
    grad_scores = torch.empty_like(top_scores)
    for slot in range(top_k):
        source = inverse[:, slot]
        grad_rows[source] = (
            grad_output.float() * top_scores[:, slot, None].float()
        ).to(combined_rows.dtype)
        grad_scores[:, slot] = (
            combined_rows[source].float() * grad_output.float()
        ).sum(-1).to(top_scores.dtype)
    return grad_rows, grad_scores, None, None
