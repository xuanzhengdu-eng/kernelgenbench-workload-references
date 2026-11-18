"""DeepSpeed sequence-loss backward: equal-row reduce-scatter sum.

Official source (deepspeedai/DeepSpeed, pinned commit 8af59b2e):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L107-L131

Sub-contract of ``_GatherSequenceLoss.backward`` with ``sum_gradients=True``:
``dist.reduce_scatter_fn`` performs a SUM reduce-scatter, so this rank keeps the
summed segment of its own rows.  Kept behaviour:
  * rows must divide evenly by the group size (the module's forward gathers
    equal local sequence sizes),
  * the return is the 3-tuple ``(grad, None, None)`` of the official backward,
  * the segment is freshly allocated.
"""
import torch
import torch.distributed as dist


def v1_deepspeed_gather_reduce_scatter_sequence_loss(grad_output, group):
    assert grad_output.shape[0] % group.size() == 0, (
        "local sequence size must be equal across the sequence-parallel group"
    )
    total = grad_output.contiguous().clone()
    dist.all_reduce(total, group=group)
    size = total.shape[0] // group.size()
    out = total.narrow(0, group.rank() * size, size).clone(memory_format=torch.contiguous_format)
    return out, None, None
