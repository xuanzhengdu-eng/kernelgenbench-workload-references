"""DeepSpeed variable-row gather backward: pad each counted segment, then sum.

Official source (deepspeedai/DeepSpeed, pinned commit 8af59b2e):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_tp_dispatch.py#L283-L339

Sub-contract of ``_AllGatherVariableRows.backward``: the replicated gradient is
split by ``counts``, every segment is zero-padded to ``max_rows`` and summed
over the TP peers; this rank then keeps its own segment, trimmed back to
``counts[rank]``.  Kept behaviour:
  * ``counts`` has one entry per rank and sums to the gradient's first axis;
  * ``max_rows == 0`` returns an empty ``(0, ...)`` tensor without communicating
    (the official early return);
  * the return is the 4-tuple ``(grad, None, None, None)``;
  * the segment is freshly allocated and contiguous.
"""
import torch
import torch.distributed as dist


def v1_deepspeed_gather_reduce_scatter_variable_rows(grad_output, group, counts, max_rows):
    assert len(counts) == group.size(), "counts must have one entry per rank"
    assert sum(counts) == grad_output.shape[0], "counts must cover the gradient rows"
    if max_rows == 0:
        return grad_output.new_empty((0, *grad_output.shape[1:])), None, None, None
    assert max_rows >= max(counts), "max_rows must cover the largest segment"
    total = grad_output.contiguous().clone()
    dist.all_reduce(total, group=group)
    start = sum(counts[:group.rank()])
    out = total.narrow(0, start, counts[group.rank()]).clone(memory_format=torch.contiguous_format)
    return out, None, None, None
