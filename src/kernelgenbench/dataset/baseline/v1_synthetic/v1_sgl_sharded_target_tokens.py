"""Distributed greedy token selection over vocabulary-sharded logits.

Official sources:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/sharded_verify.py#L136-L168
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/sharded_verify.py#L634-L689

Each rank takes the max over its valid shard (columns ``[0, valid)``, global
ids ``shard_start + column`` below ``vocab_size``), treats NaN as ``-inf`` and
breaks ties toward the smaller global token id.  It all-reduces the maximum
score and then the minimum matching token id, returning int64 ``[rows]``.
"""
import torch
import torch.distributed as dist


def v1_sgl_sharded_target_tokens(logits, shard_start, valid, vocab_size, group):
    assert logits.ndim == 2 and logits.stride(-1) == 1
    # The official key kernel reads ``min(valid, width)`` columns and masks ids
    # past ``vocab_size``; out-of-range ``valid`` is clamped, not an error.
    rows = logits.shape[0]
    cols = max(0, min(valid, logits.shape[1], max(0, vocab_size - shard_start)))
    x = logits[:, :cols].float()
    x = torch.where(torch.isnan(x), -torch.inf, x) + 0.0
    if cols:
        local_score = x.amax(1)
        ids = torch.arange(shard_start, shard_start + cols, device=logits.device,
                           dtype=torch.int64).expand(rows, cols)
        local_id = torch.where(x == local_score[:, None], ids, 0xffffffff).amin(1)
    else:
        local_score = torch.full((rows,), -torch.inf, device=logits.device)
        local_id = torch.full((rows,), 0xffffffff, device=logits.device, dtype=torch.int64)
    global_score = local_score.clone()
    dist.all_reduce(global_score, op=dist.ReduceOp.MAX, group=group)
    best = torch.where(local_score == global_score, local_id, 0xffffffff)
    dist.all_reduce(best, op=dist.ReduceOp.MIN, group=group)
    return best
