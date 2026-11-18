"""Megatron DSA indexer-loss backward, official ``bwd_fused_indexer_loss_naive``.

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/experimental_attention_variant/dsa.py#L346-L507

The reference reproduces the upstream analytic backward for the fused indexer
loss:
* ``q`` [sq,b,h,d], ``weights`` [sq,b,h], ``k`` [sk,b,d] are indexer tensors
  (bf16 upstream); ``query`` [sq,b,h,d] and ``key`` [sk,b,h,d] are the attention
  tensors whose probabilities form the KL target;
* the forward index scores are recomputed, causal-masked, and optionally
  restricted to ``topk_indices`` [b,sq,K];
* the upstream scalar ``grad_loss`` is propagated through the mean, the KL,
  both softmaxes, the head sum, ReLU, the weighting and both einsums;
* gradients at non-causal and non-top-k positions are zeroed;
* returns ``(dq, dweights, dk)`` cast back to the input indexer dtypes.

The TP all-reduce is the only distributed branch.  There is no separate CUDA
DSA fused-indexer backward in the pinned NVIDIA tree, so this is the official
target entry point.
"""
import torch


def _compute_index_scores(q, weights, k):
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    return index_scores.transpose(0, 1)


def v1_megatron_dsa_indexer_backward(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    index_scores = _compute_index_scores(q, weights, k)  # [b, sq, sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    del query_reshaped, key_reshaped
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    index_scores = index_scores + causal_mask.unsqueeze(0)
    del causal_mask

    if sparse_loss:
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        index_scores = index_scores + index_mask

    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    del attention_scores
    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    del index_scores

    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    del attention_scores_softmax
    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    del attention_scores_sum

    grad_kl_div = grad_loss * loss_coeff
    grad_kl_per_row = grad_kl_div / (b * sq)
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    del attention_scores_normalized

    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    causal_valid_mask = torch.tril(torch.ones((sq, sk), device=q.device, dtype=torch.bool))
    if sparse_loss:
        index_valid_mask = index_mask == 0
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask
        del index_valid_mask
    else:
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    grad_index_scores = grad_index_scores_logits.transpose(0, 1)
    del grad_index_scores_logits
    grad_weighted_scores = grad_index_scores.unsqueeze(2)
    del grad_index_scores

    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)
    del grad_weighted_scores, scores_after_relu
    grad_scores = grad_scores_after_relu * relu_mask.float()
    del grad_scores_after_relu, relu_mask

    grad_q = torch.einsum("sbht,tbd->sbhd", grad_scores, k.float())
    grad_k = torch.einsum("sbht,sbhd->tbd", grad_scores, q.float())
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)
