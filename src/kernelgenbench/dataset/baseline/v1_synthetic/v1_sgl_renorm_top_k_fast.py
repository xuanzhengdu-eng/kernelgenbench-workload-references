"""SGLang top-k probability renormalization, fast-kernel reference body.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampling_renorm.py#L73-L84

``probs`` is a nonnegative ``[B, V]`` tensor.  ``top_ks`` is clamped to
``[1, V]``; the largest ``max(top_ks)`` entries are taken per row, entries at
rank ``>= k`` are zeroed, survivors are renormalized and scattered back to the
original column order.  The output is freshly allocated; ``probs`` is not
modified.  The FlashInfer CUDA kernel reads contiguous rows and implements the same
rank cut; this reference is the numerical contract it is compared against.
The kernel keeps the input dtype and requires contiguous rows.
"""
import torch


def v1_sgl_renorm_top_k_fast(probs, top_ks):
    assert probs.ndim == 2 and top_ks.numel() == probs.shape[0]
    vocab = probs.shape[-1]
    ks = top_ks.reshape(-1).to(device=probs.device, dtype=torch.int64).clamp(1, vocab)
    max_top_k = int(ks.max().item())
    topk_probs, topk_indices = torch.topk(probs, k=max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=probs.device)[None, :]
    topk_probs = topk_probs.masked_fill(ranks >= ks[:, None], 0.0)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)
