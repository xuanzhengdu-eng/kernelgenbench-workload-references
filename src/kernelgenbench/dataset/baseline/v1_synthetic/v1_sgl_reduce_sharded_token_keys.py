"""SGLang sharded-verify token-key reduction (``reduce_sharded_token_keys``).

Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/sharded_verify.py#L649-L690
Key format: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/sharded_verify.py#L173-L186

Every TP rank contributes one int64 key per row (or ``splits`` keys, one per
column range). The high 32 bits hold a monotone score and the low 32 bits hold
``0xffffffff - global_token_id``, so the maximum key across all ranks and splits
selects the winner and breaks ties toward the smaller token. The result is the
token id per row: ``0xffffffff - (best & 0xffffffff)``. Inputs must be contiguous
int64 of shape ``[rows]`` for one split or ``[rows, splits]`` otherwise.
Upstream reaches this shape through a group object (LL/multimem fast paths with
an all-gather fallback); this reference takes a process group and always
performs the all-reduce maximum, which is the same arithmetic.
"""
import torch
import torch.distributed as dist


def v1_sgl_reduce_sharded_token_keys(keys, group, splits):
    if splits <= 0:
        raise ValueError("splits must be positive")
    if keys.dtype != torch.int64:
        raise TypeError("keys must be int64")
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    if splits == 1 and keys.ndim != 1:
        raise ValueError("invalid key shape")
    if splits > 1 and (keys.ndim != 2 or keys.shape[1] != splits):
        raise ValueError("invalid key shape")
    assert keys.shape[0] > 0
    local = keys if splits == 1 else keys.amax(dim=1)
    best = local.clone()
    dist.all_reduce(best, op=dist.ReduceOp.MAX, group=group)
    return 0xffffffff - (best & 0xffffffff)
