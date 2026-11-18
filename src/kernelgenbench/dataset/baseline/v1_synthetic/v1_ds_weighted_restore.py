# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b: DeepSpeed fused weighted restore forward.

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L456-L469
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L287-L346

Contract: combined_rows is [num_tokens * top_k, hidden], top_scores is
[num_tokens, top_k] float32 and token_indices_sorted is a complete permutation
of 0 .. num_tokens * top_k - 1 (the row permutation produced by sorting routed
assignments). The result is the weighted [batches, seqlen, hidden] tensor with
the same dtype as combined_rows, accumulated in fp32. The vLLM unpermute modes
that shared the old mixed file are separate items; this reference is the
DeepSpeed branch only.
"""
import torch


def v1_ds_weighted_restore(combined_rows, top_scores, token_indices_sorted, top_k, shape):
    batches, seqlen, hidden = shape
    num_tokens = batches * seqlen
    assert top_k > 0 and hidden > 0
    combined_rows = combined_rows.contiguous()
    top_scores = top_scores.contiguous()
    assert combined_rows.shape == (num_tokens * top_k, hidden)
    assert top_scores.shape == (num_tokens, top_k)
    assert top_scores.dtype == torch.float32
    inverse = torch.argsort(token_indices_sorted.to(torch.int64)).view(num_tokens, top_k)
    result = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=combined_rows.device)
    for slot in range(top_k):
        result = result + combined_rows[inverse[:, slot]].float() * top_scores[:, slot, None].float()
    return result.to(combined_rows.dtype).reshape(batches, seqlen, hidden)
