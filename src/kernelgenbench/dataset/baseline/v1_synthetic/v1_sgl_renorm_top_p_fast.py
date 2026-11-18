"""SGLang top-p probability renormalization, fast-kernel reference body.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampling_renorm.py#L56-L70

``probs`` is a nonnegative ``[B, V]`` tensor whose rows sum to a positive
value.  The reference sorts each row descending, drops every entry whose
strictly-preceding cumulative mass is above ``top_ps`` (so the first entry is
always kept), renormalizes the survivors and scatters them back to the original
column order.  The FlashInfer CUDA kernel implements the same mass cut on contiguous rows;
this reference is the numerical contract it is compared against.  The output
is freshly allocated and ``probs`` is not modified.
"""
import torch


def v1_sgl_renorm_top_p_fast(probs, top_ps):
    assert probs.ndim == 2 and top_ps.numel() == probs.shape[0]
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort = probs_sort.masked_fill((probs_sum - probs_sort) > top_ps.view(-1, 1), 0.0)
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)
