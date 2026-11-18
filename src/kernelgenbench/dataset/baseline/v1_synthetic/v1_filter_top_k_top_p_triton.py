"""vLLM Triton top-k/top-p logits filter (strict rank semantics).

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_triton.py#L1618-L1710

Unlike ``apply_top_k_top_p_pytorch``, which keeps every value tied with the
k-th largest logit, the Triton kernel ranks values and keeps exactly ``k``
entries per row.  This reference transcribes that rank cut with ``topk`` and
then applies the same top-p cumulative-mass cut to the survivors.  It is a
separate target item from the PyTorch fallback because a tied row produces a
different number of surviving logits.
"""
import torch


def v1_filter_top_k_top_p_triton(logits, k=None, p=None):
    if k is None and p is None:
        return logits
    vocab = logits.shape[-1]
    if k is not None:
        kk = k.to(torch.int64).clamp(1, vocab)
        width = int(kk.max().item())
        _, keep = torch.topk(logits, k=width, dim=-1)
        ranks = torch.arange(width, device=logits.device)[None, :]
        valid = ranks < kk[:, None]
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, keep.clamp_min(0), valid)
        logits = logits.masked_fill(~mask, -float("inf"))
    if p is not None:
        # The cumulative-mass cut removes the low-probability tail, so sort
        # ascending exactly like apply_top_k_top_p_pytorch.
        ordered, indices = logits.sort(dim=-1)
        probs = torch.softmax(ordered, dim=-1)
        cdf = probs.cumsum(dim=-1)
        remove = cdf <= (1 - p.view(-1, 1))
        remove[:, -1] = False
        ordered = ordered.masked_fill(remove, -float("inf"))
        logits = logits.scatter_(1, indices, ordered)
    return logits
