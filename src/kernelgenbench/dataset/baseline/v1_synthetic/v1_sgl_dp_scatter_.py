"""SGLang DP scatter into a caller-allocated local buffer.

Official sources (sgl-project/sglang, pinned commit 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dp_attention.py#L652-L659
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dp_attention.py#L1076-L1104

This reference is the explicit-argument sub-contract of ``dp_scatter``:
  * the destination is preallocated and fully zero-filled first (the padded tail
    of a CUDA-graph batch must be zero),
  * without ``cp_counts`` the source rows are this DP slot's
    ``[dp_start, dp_start+dp_count)`` window of the gathered global buffer,
  * with ``cp_counts`` the source is the CP shard that starts after all previous
    DP slots and all previous CP ranks and is ``cp_counts[cp_rank]`` rows long
    (``_cp_shard_rows``),
  * the return value is ``None`` and the two inputs must not share storage.

``dp_slot`` and ``cp_rank`` replace the runtime ``dp_gather_slot()`` /
``get_parallel().attn_cp_rank`` reads.  Nothing else about the copy is changed.
"""


def v1_sgl_dp_scatter_(local_tokens, global_tokens, dp_start, dp_count, cp_counts=None, cp_rank=0, global_dp_counts=None, dp_slot=0):
    assert local_tokens.is_contiguous() and global_tokens.is_contiguous()
    assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()
    local_tokens.fill_(0)
    if local_tokens.shape[0]:
        if cp_counts is None:
            start, count = dp_start, dp_count
        else:
            start = sum(global_dp_counts[:dp_slot]) + sum(cp_counts[:cp_rank])
            count = cp_counts[cp_rank]
        local_tokens[:count].copy_(global_tokens[start:start + count])
    return None
