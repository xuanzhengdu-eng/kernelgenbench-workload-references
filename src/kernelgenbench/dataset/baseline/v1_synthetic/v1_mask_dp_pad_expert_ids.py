"""SGLang masks DP-padded MoE top-k rows in place.

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dp_attention.py#L807-L821
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dp_attention.py#L824-L852

The upstream entry reads the per-rank real-token counts and the DP buffer length
from ``_DpGatheredBufferWrapper``.  This reference exposes that state as the
explicit arguments ``counts`` [num_dp_ranks] (int32) and ``max_len`` so the
contract can be tested without a live DP server.

Contract: ``topk_ids`` has shape [num_dp_ranks * max_len, top_k].  Row ``r``
belongs to rank ``r // max_len`` and is a padding row when
``r % max_len >= counts[r // max_len]``; every element of a padding row is set
to -1 in place.  Real rows are unchanged.  Returns None.  When ``counts`` is
None, ``max_len <= 0``, or the row count does not equal
``counts.numel() * max_len`` the layout is not DP-padded and the function is a
no-op.  The upstream kernel uses 32-bit counts and writes only the top_k
columns.
"""
import torch


def v1_mask_dp_pad_expert_ids(topk_ids, counts, max_len):
    if counts is None or max_len <= 0 or topk_ids.shape[0] != counts.shape[0] * max_len:
        return None
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
    topk_ids[rows % max_len >= counts[rows // max_len]] = -1
    return None
